import torch
import torch.nn as nn
import torch.nn.functional as F
import util
from model_base import BaseModel
from losses import ATLoss


class DocReModel(BaseModel):
    def __init__(self, config, num_entity_types=0, with_encoder=True, seq_config=None):
        super(DocReModel, self).__init__(config, with_encoder, seq_config)
        self.num_entity_types = num_entity_types

        self.num_re_labels = config['num_re_labels']
        self.max_re_labels = config['max_re_labels']

        self.mention_hidden_size = 2 * self.seq_hidden_size
        self.entity_transform_size = config['entity_transform_size']
        self.num_blocks = config['num_blocks']
        self.block_size = self.entity_transform_size // self.num_blocks
        self.feat_hidden_size = self.num_blocks
        self.entity_hidden_size = self.mention_hidden_size + self.feat_hidden_size * (
            int(config['use_entity_type']) + int(config['use_entity_distance'])
        )

        self.emb_distance = self.make_embedding(20, self.feat_hidden_size)
        self.emb_entity_type = self.make_embedding(num_entity_types, self.feat_hidden_size)

        self.head_transform = self.make_linear(self.entity_hidden_size, self.entity_transform_size)
        self.tail_transform = self.make_linear(self.entity_hidden_size, self.entity_transform_size)
        self.re_bilinear = self.make_linear(self.entity_transform_size * self.block_size, self.num_re_labels)

        self.loss_fct = ATLoss()

    def get_ht(self, doc_len, tokens, entities, entity_types, entity_pairs_h, entity_pairs_t):
        conf, device = self.config, tokens.device
        token_hidden_size = tokens.size()[-1]

        past_total_len = 0
        entity_pair_h_hidden, entity_pair_t_hidden = [], []
        for doc_i, num_doc_tokens in enumerate(doc_len):
            doc_tokens = tokens[past_total_len: past_total_len + num_doc_tokens] 
            past_total_len += num_doc_tokens
            doc_entities = entities[doc_i]
            doc_entity_types = entity_types[doc_i]
            doc_entity_pairs_h = entity_pairs_h[doc_i]
            doc_entity_pairs_t = entity_pairs_t[doc_i]

            if len(doc_entities) == 0:
                continue
            doc_entity_hidden, doc_entity_starts = [], []
            for entity_i, entity in enumerate(doc_entities):
                if len(entity) == 0:
                    doc_entity_hidden.append(torch.zeros(2 * token_hidden_size, dtype=torch.float,
                                                         device=device)) 
                    doc_entity_starts.append(0)
                else:
                    entity_m_starts = torch.tensor([m_s for m_s, m_e in entity], dtype=torch.long, device=device)
                    entity_m_ends = torch.tensor([m_e for m_s, m_e in entity], dtype=torch.long, device=device)
                    entity_mentions = torch.cat([doc_tokens[entity_m_starts], doc_tokens[entity_m_ends]], dim=-1)
                    doc_entity_hidden.append(torch.logsumexp(entity_mentions, dim=0))
                    doc_entity_starts.append(entity[0][0]) 
            doc_entity_hidden = torch.stack(doc_entity_hidden, dim=0) 
            doc_entity_starts = torch.tensor(doc_entity_starts, dtype=torch.long, device=device)
            doc_entity_type_hidden = self.emb_entity_type(torch.tensor([max(0, util.get_most_common(types)) 
                for types in doc_entity_types], device=device)) if conf['use_entity_type'] else None 

            doc_entity_pair_h_hidden = doc_entity_hidden[doc_entity_pairs_h]
            doc_entity_pair_t_hidden = doc_entity_hidden[doc_entity_pairs_t]
            doc_entity_pair_h_dist = doc_entity_starts[doc_entity_pairs_h] - doc_entity_starts[doc_entity_pairs_t]
            dist_is_neg = doc_entity_pair_h_dist < 0
            doc_entity_pair_h_dist = util.bucket_distance(torch.abs(doc_entity_pair_h_dist))
            doc_entity_pair_h_dist[dist_is_neg] *= -1
            doc_entity_pair_t_dist = -doc_entity_pair_h_dist
            doc_entity_pair_h_dist += 10
            doc_entity_pair_t_dist += 10
            doc_entity_pair_h_dist_hidden = self.emb_distance(doc_entity_pair_h_dist)
            doc_entity_pair_t_dist_hidden = self.emb_distance(doc_entity_pair_t_dist)

            doc_h_hidden_final, doc_t_hidden_final = [doc_entity_pair_h_hidden], [doc_entity_pair_t_hidden]
            if self.config['use_entity_type']:
                doc_h_hidden_final.append(self.dropout(doc_entity_type_hidden[doc_entity_pairs_h]))
                doc_t_hidden_final.append(self.dropout(doc_entity_type_hidden[doc_entity_pairs_t]))
            if self.config['use_entity_distance']:
                doc_h_hidden_final.append(self.dropout(doc_entity_pair_h_dist_hidden))
                doc_t_hidden_final.append(self.dropout(doc_entity_pair_t_dist_hidden))
            doc_h_hidden_final = torch.cat(doc_h_hidden_final, dim=-1) 
            doc_t_hidden_final = torch.cat(doc_t_hidden_final, dim=-1)
            entity_pair_h_hidden.append(doc_h_hidden_final)
            entity_pair_t_hidden.append(doc_t_hidden_final)
        entity_pair_h_hidden = torch.cat(entity_pair_h_hidden, dim=0) 
        entity_pair_t_hidden = torch.cat(entity_pair_t_hidden, dim=0)
        return entity_pair_h_hidden, entity_pair_t_hidden

    def get_labels(self, re_logits):
        return self.loss_fct.get_label(re_logits, self.max_re_labels)

    def forward(self, doc_len, tokens=None, input_ids=None, attention_mask=None, token_type_ids=None, is_max_context=None,
                entities=None, entity_types=None, entity_pairs_h=None, entity_pairs_t=None, rel_labels=None, **kwargs):
        if tokens is None:
            tokens = self.encode(doc_len, input_ids, attention_mask, token_type_ids, is_max_context)

        entity_pair_h_hidden, entity_pair_t_hidden = self.get_ht(doc_len, tokens, entities, entity_types,
                                                                 entity_pairs_h, entity_pairs_t)
        entity_pair_h_hidden = torch.tanh(self.head_transform(entity_pair_h_hidden)) 
        entity_pair_t_hidden = torch.tanh(self.tail_transform(entity_pair_t_hidden))

        h_b = entity_pair_h_hidden.view(-1, self.num_blocks, self.block_size)
        t_b = entity_pair_t_hidden.view(-1, self.num_blocks, self.block_size)

        re_logits = self.re_bilinear((self.dropout(h_b).unsqueeze(3) * self.dropout(t_b).unsqueeze(2))
                                     .view(-1, self.entity_transform_size * self.block_size))

        output = re_logits
        if rel_labels is not None:
            rel_labels = torch.cat(rel_labels, dim=0)
            loss = self.loss_fct(re_logits, rel_labels.to(torch.float)).mean()
            return loss, output
        else:
            return output
