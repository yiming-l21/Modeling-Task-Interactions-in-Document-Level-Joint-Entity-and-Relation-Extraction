import torch
import torch.nn as nn
import util
from model_base import BaseModel
import logging
import numpy as np
import torch.nn.functional as F
import losses
import propagation

logger = logging.getLogger(__name__)


class SpanModelBase(BaseModel):
    def __init__(self, config, num_entity_types=0, with_encoder=True, seq_config=None):
        super(SpanModelBase, self).__init__(config, with_encoder, seq_config)
        self.num_entity_types = num_entity_types

        self.max_span_width = config['max_span_width']
        self.max_training_seg = config['max_training_seg']

        self.feat_hidden_size = 8
        self.span_hidden_size = self.seq_hidden_size * (2 * int(config['span_w_boundary']) + int(config['span_w_tokens']))
        if config['use_span_width']:
            self.span_hidden_size += self.feat_hidden_size

        self.emb_span_width = self.make_embedding(self.max_span_width, self.feat_hidden_size)
        self.emb_span_width_prior = self.make_embedding(self.max_span_width, self.feat_hidden_size)

        self.mention_token_attn = self.make_ffnn(self.seq_hidden_size, 0, 1)
        self.span_width_score_ffnn = self.make_ffnn(self.feat_hidden_size, 0, 1)
        self.span_type_ffnn = self.make_ffnn(self.span_hidden_size, 256,
                                             1 + (num_entity_types if config['use_span_type'] else 0))

    @classmethod
    def debatch(cls, doc_len, tokens, tok_start_or_end=None, sent_map=None,
                entities=None, entity_types=None, entity_pairs_h=None, entity_pairs_t=None, rel_labels=None,
                mention_starts=None, mention_ends=None, mention_type=None, mention_cluster_id=None,
                flattened_rel_labels=None):
        docs, past_total_len = [], 0
        for doc_i, num_doc_tokens in enumerate(doc_len):
            doc_tokens = tokens[past_total_len: past_total_len + num_doc_tokens]
            doc_tok_start_or_end, doc_sent_map = tok_start_or_end[doc_i], sent_map[doc_i]
            doc_tok_start_or_end = doc_tok_start_or_end.to(torch.bool)

            doc_entities = entities[doc_i] if entities is not None else None
            doc_entity_types = entity_types[doc_i] if entity_types is not None else None
            doc_entity_pairs_h = entity_pairs_h[doc_i] if entity_pairs_h is not None else None
            doc_entity_pairs_t = entity_pairs_t[doc_i] if entity_pairs_t is not None else None
            doc_rel_labels = rel_labels[doc_i] if rel_labels is not None else None 

            doc_mention_starts = mention_starts[doc_i] if mention_starts is not None else None
            doc_mention_ends = mention_ends[doc_i] if mention_ends is not None else None
            doc_mention_type = mention_type[doc_i] if mention_type is not None else None
            doc_mention_cluster_id = mention_cluster_id[doc_i] if mention_cluster_id is not None else None 
            if doc_mention_cluster_id is not None:
                doc_mention_cluster_id += 1 
                doc_mention_type += 1 

            doc_flattened_rel_labels = flattened_rel_labels[doc_i] if flattened_rel_labels is not None else None

            docs.append({
                'tokens': doc_tokens,
                'tok_start_or_end': doc_tok_start_or_end,
                'sent_map': doc_sent_map,
                'entities': doc_entities,
                'entity_types': doc_entity_types,
                'entity_pairs_h': doc_entity_pairs_h,
                'entity_pairs_t': doc_entity_pairs_t,
                'rel_labels': doc_rel_labels,
                'mention_starts': doc_mention_starts,
                'mention_ends': doc_mention_ends,
                'mention_type': doc_mention_type,
                'mention_cluster_id': doc_mention_cluster_id,
                'flattened_rel_labels': doc_flattened_rel_labels
            })

            past_total_len += num_doc_tokens

        return docs

    def forward(self, doc_len, tokens=None, tok_start_or_end=None, sent_map=None,
                input_ids=None, attention_mask=None, token_type_ids=None, is_max_context=None,
                entities=None, entity_types=None, entity_pairs_h=None, entity_pairs_t=None, rel_labels=None,
                mention_starts=None, mention_ends=None, mention_type=None, mention_cluster_id=None,
                flattened_rel_labels=None):
        if tokens is None:
            tokens = self.encode(doc_len, input_ids, attention_mask, token_type_ids, is_max_context)
        is_training = mention_cluster_id is not None

        doc_returns = []
        for doc in self.debatch(doc_len, tokens, tok_start_or_end=tok_start_or_end, sent_map=sent_map,
                                entities=entities, entity_types=entity_types,
                                entity_pairs_h=entity_pairs_h, entity_pairs_t=entity_pairs_t, rel_labels=rel_labels,
                                mention_starts=mention_starts, mention_ends=mention_ends,
                                mention_type=mention_type, mention_cluster_id=mention_cluster_id,
                                flattened_rel_labels=flattened_rel_labels):
            
            doc_returns.append(self.forward_single(**doc))

        if not is_training:
            return doc_returns
        else:
            loss = sum([doc_loss for doc_loss, _ in doc_returns]) / len(doc_returns)
            return loss, [doc_outputs for _, doc_outputs in doc_returns]

    def extract_spans(self, tokens, tok_start_or_end, sent_map,
                      gold_starts=None, gold_ends=None, gold_types=None, gold_cluster_map=None):
        conf, num_tokens = self.config, tokens.size()[0]
        device = tokens.device

        sentence_indices = sent_map 
        candidate_starts = torch.unsqueeze(torch.arange(0, num_tokens, device=device), 1).repeat(1, self.max_span_width)
        candidate_ends = candidate_starts + torch.arange(0, self.max_span_width, device=device)
        candidate_start_sent_idx = sentence_indices[candidate_starts] 
        candidate_end_sent_idx = sentence_indices[torch.min(candidate_ends, torch.tensor(num_tokens - 1, device=device))] 
        candidate_mask = (candidate_ends < num_tokens) & (candidate_start_sent_idx == candidate_end_sent_idx) 
        candidate_starts, candidate_ends = candidate_starts[candidate_mask], candidate_ends[candidate_mask] 
        candidate_mask = tok_start_or_end[candidate_starts] & tok_start_or_end[candidate_ends] 
        candidate_starts, candidate_ends = candidate_starts[candidate_mask], candidate_ends[candidate_mask] 
        candidate_width_idx = candidate_ends - candidate_starts 
        num_candidates = candidate_starts.size()[0]

        if gold_cluster_map is None:
            candidate_cluster_ids = candidate_type_labels = None
        else:
            same_start = (torch.unsqueeze(gold_starts, 1) == torch.unsqueeze(candidate_starts, 0)) 
            same_end = (torch.unsqueeze(gold_ends, 1) == torch.unsqueeze(candidate_ends, 0))
            same_span = (same_start & same_end).to(torch.float) 
            candidate_cluster_ids = torch.matmul(torch.unsqueeze(gold_cluster_map, 0).to(torch.float), same_span)
            candidate_cluster_ids = torch.squeeze(candidate_cluster_ids.to(torch.long), 0) 
            candidate_type_labels = torch.matmul(torch.unsqueeze(gold_types, 0).to(torch.float), same_span)  
            candidate_type_labels = torch.squeeze(candidate_type_labels.to(torch.long), 0) 

        span_start_emb, span_end_emb = tokens[candidate_starts], tokens[candidate_ends]  
        candidate_emb_list = [span_start_emb, span_end_emb] if conf['span_w_boundary'] else []
        if conf['span_w_tokens']:
            candidate_tokens = torch.unsqueeze(torch.arange(0, num_tokens, device=device), 0).repeat(num_candidates, 1) 
            candidate_tokens_mask = (candidate_tokens >= torch.unsqueeze(candidate_starts, 1)) & (candidate_tokens <= torch.unsqueeze(candidate_ends, 1))
            if conf['mention_heads']:
                token_attn = torch.squeeze(self.mention_token_attn(tokens), -1) 
            else:
                token_attn = torch.ones(num_tokens, dtype=torch.float, device=device) 
            candidate_tokens_attn_raw = torch.log(candidate_tokens_mask.to(torch.float)) + torch.unsqueeze(token_attn, 0) 
            candidate_tokens_attn = F.softmax(candidate_tokens_attn_raw, dim=-1)
            head_attn_emb = torch.matmul(candidate_tokens_attn, tokens)
            candidate_emb_list.append(head_attn_emb)
        if conf['use_span_width']:
            candidate_width_emb = self.emb_span_width(candidate_width_idx) 
            candidate_width_emb = self.dropout(candidate_width_emb)
            candidate_emb_list.append(candidate_width_emb)
        candidate_span_emb = torch.cat(candidate_emb_list, dim=-1) 

        if conf['use_span_type']:
            candidate_type_logits = self.span_type_ffnn(candidate_span_emb)  
            candidate_mention_scores = -candidate_type_logits[:, 0]
        else:
            candidate_type_logits = None
            candidate_mention_scores = torch.squeeze(self.span_type_ffnn(candidate_span_emb), -1) 
        if conf['use_width_prior']:
            width_score = torch.squeeze(self.span_width_score_ffnn(self.emb_span_width_prior.weight), -1)
            candidate_width_score = width_score[candidate_width_idx] 
            candidate_mention_scores += candidate_width_score

        num_top_spans = int(min(conf['max_num_extracted_spans'], conf['top_span_ratio'] * num_tokens))
        if conf['approx_pruning']:
            _, selected_idx = torch.topk(candidate_mention_scores, k=num_top_spans, dim=-1, largest=True, sorted=True)
        else:
            candidate_starts_cpu, candidate_ends_cpu = candidate_starts.tolist(), candidate_ends.tolist()
            candidate_idx_sorted_by_score = torch.argsort(candidate_mention_scores, descending=True).tolist()
            selected_idx_cpu = self.get_top_spans(candidate_idx_sorted_by_score,
                                                  candidate_starts_cpu, candidate_ends_cpu, num_top_spans,
                                                  allow_nested=conf['allow_nested_mentions'])
            assert len(selected_idx_cpu) == num_top_spans
            selected_idx = torch.tensor(selected_idx_cpu, device=device) 
        top_span_starts, top_span_ends = candidate_starts[selected_idx], candidate_ends[selected_idx]
        top_span_emb = candidate_span_emb[selected_idx]
        top_span_mention_scores = candidate_mention_scores[selected_idx]
        top_span_type_logits = candidate_type_logits[selected_idx] if conf['use_span_type'] else None
        top_span_cluster_ids = candidate_cluster_ids[selected_idx] if gold_cluster_map is not None else None

        return (top_span_starts, top_span_ends, top_span_emb, top_span_mention_scores, top_span_type_logits, top_span_cluster_ids), \
               (candidate_starts, candidate_ends, candidate_span_emb, candidate_mention_scores, candidate_type_logits,
                candidate_cluster_ids, candidate_type_labels), selected_idx, num_top_spans


class CorefModel(SpanModelBase):
    def __init__(self, config, num_entity_types=0, with_encoder=True, seq_config=None):
        super(CorefModel, self).__init__(config, num_entity_types, with_encoder, seq_config)

        self.pair_span_hidden_size = self.span_hidden_size
        if config['pair_span_transform']:
            self.pair_span_hidden_size = config['pair_span_hidden_size']
        self.pair_hidden_size = self.pair_span_hidden_size * (2 + int(config['pair_with_similarity']))
        if config['use_antecedent_distance']:
            self.pair_hidden_size += self.feat_hidden_size
        if config['use_speaker_indicator']:
            self.pair_hidden_size += self.feat_hidden_size

        self.emb_top_antecedent_distance = self.make_embedding(10, self.feat_hidden_size)
        self.emb_antecedent_distance_prior = self.make_embedding(10, self.feat_hidden_size)
        self.emb_same_speaker = self.make_embedding(2, self.feat_hidden_size)

        self.antecedent_distance_score_ffnn = self.make_ffnn(self.feat_hidden_size, 0, 1)
        self.pair_span_anaphora_transform = self.make_ffnn(self.span_hidden_size, 0, self.pair_span_hidden_size)
        self.pair_span_antecedent_transform = self.make_ffnn(self.span_hidden_size, 0, self.pair_span_hidden_size)
        self.coarse_bilinear = self.make_ffnn(self.pair_span_hidden_size, 0, self.pair_span_hidden_size)
        self.coref_score_ffnn = self.make_ffnn(self.pair_hidden_size, 512, 1)

        self.gate_ffnn = self.make_ffnn(self.pair_span_hidden_size * 2, 0, self.pair_span_hidden_size)
        self.scalar_gate_ffnn = self.make_ffnn(self.pair_span_hidden_size * 2, 0, 1)

        self.forward_steps = 0
        self.debug = False

    def forward_single(self, tokens, tok_start_or_end, sent_map, speaker_ids=None,
                       mention_starts=None, mention_ends=None, mention_type=None, mention_cluster_id=None, **kwargs):
        """ Coref decoding for single doc. """
        conf, num_tokens = self.config, tokens.size()[0]
        gold_starts, gold_ends, gold_types, gold_cluster_map = mention_starts, mention_ends, mention_type, mention_cluster_id
        device = tokens.device

        (top_span_starts, top_span_ends, top_span_emb, top_span_mention_scores, top_span_type_logits, top_span_cluster_ids), \
        (_, _, _, candidate_mention_scores, candidate_type_logits, candidate_cluster_ids, candidate_type_labels), \
        selected_idx, num_top_spans = self.extract_spans(tokens, tok_start_or_end, sent_map,
                                                         gold_starts, gold_ends, gold_types, gold_cluster_map)

        if conf['pair_span_transform']:
            pair_anaphora_emb = torch.tanh(self.pair_span_anaphora_transform(top_span_emb))  
            pair_antecedent_emb = torch.tanh(self.pair_span_antecedent_transform(top_span_emb))
        else:
            pair_anaphora_emb, pair_antecedent_emb = top_span_emb, top_span_emb

        max_top_antecedents = min(num_top_spans, conf['max_top_antecedents'])
        pairwise_mention_scores = top_span_mention_scores.unsqueeze(1) + top_span_mention_scores.unsqueeze(0)
        span_range = torch.arange(0, num_top_spans, device=device)
        antecedent_offsets = torch.unsqueeze(span_range, 1) - torch.unsqueeze(span_range, 0)
        antecedent_mask = (antecedent_offsets >= 1)
        top_span_speaker_ids = speaker_ids[top_span_starts] if speaker_ids is not None else None
        top_antecedent_scores, top_antecedent_idx, top_antecedent_mask, top_antecedent_offsets, _ = self.do_antecedent_scoring(
            pairwise_mention_scores, pair_anaphora_emb, pair_antecedent_emb, antecedent_offsets, antecedent_mask,
            max_top_antecedents, top_span_speaker_ids)

        if conf['coref_propagation']:
            top_span_emb, (top_antecedent_scores, top_antecedent_idx, top_antecedent_mask, top_antecedent_offsets, pairwise_fast_scores) = \
                self.do_coref_propagation(pairwise_mention_scores, top_span_emb, antecedent_offsets, antecedent_mask, top_span_speaker_ids,
                                          top_antecedent_idx, top_antecedent_scores)

        top_antecedent_scores = torch.cat([torch.zeros(num_top_spans, 1, device=device), top_antecedent_scores], dim=-1)  # [num top spans, max top antecedents + 1]

        outputs = top_span_starts, top_span_ends, top_span_mention_scores, top_span_type_logits, top_antecedent_idx, top_antecedent_scores
        if gold_cluster_map is None:
            return outputs

        loss, (log_marginalized, log_norm) = losses.get_mention_ranking_loss(top_antecedent_scores, top_span_cluster_ids, top_antecedent_idx, top_antecedent_mask)

        if conf['mention_loss_coef']:
            if conf['use_span_type']:
                loss_gold_mention, loss_non_gold_mention = losses.get_mention_type_loss(candidate_type_logits, candidate_type_labels)
                loss_gold_mention *= conf['mention_loss_coef']
                loss_non_gold_mention *= conf['mention_loss_coef']
                loss += loss_gold_mention
                loss += loss_non_gold_mention
            else:
                loss_gold_mention, loss_non_gold_mention = losses.get_mention_score_loss(candidate_mention_scores, candidate_cluster_ids)
                loss_gold_mention *= conf['mention_loss_coef']
                loss_non_gold_mention *= conf['mention_loss_coef']
                loss += loss_gold_mention
                loss += loss_non_gold_mention

        return loss, outputs

    def do_antecedent_scoring(self, pairwise_mention_scores, pair_anaphora_emb, pair_antecedent_emb,
                              antecedent_offsets, antecedent_mask, max_top_antecedents, span_speaker_ids=None):
        conf = self.config
        pairwise_bilinear_scores = self.get_bilinear_pairwise_scores(pair_anaphora_emb, pair_antecedent_emb,
                                                                     antecedent_offsets, antecedent_mask)
        pairwise_fast_scores = pairwise_bilinear_scores + pairwise_mention_scores
        top_pairwise_fast_scores, top_antecedent_idx, top_antecedent_mask, top_antecedent_offsets = \
            self.coarse_antecedent_prune(pairwise_fast_scores, antecedent_mask, antecedent_offsets, max_top_antecedents)

        if not conf['fine_grained']:
            top_antecedent_scores = top_pairwise_fast_scores 
        else:
            top_pairwise_slow_scores = self.get_slow_pairwise_scores(pair_anaphora_emb, pair_antecedent_emb,
                                                                     top_antecedent_idx, top_antecedent_offsets,
                                                                     span_speaker_ids) 
            top_antecedent_scores = top_pairwise_fast_scores + top_pairwise_slow_scores

        return top_antecedent_scores, top_antecedent_idx, top_antecedent_mask, top_antecedent_offsets, pairwise_fast_scores

    def get_bilinear_pairwise_scores(self, pair_anaphora_emb, pair_antecedent_emb, antecedent_offsets, antecedent_mask):
        pairwise_bilinear_scores = torch.matmul(self.coarse_bilinear(pair_anaphora_emb), torch.transpose(pair_antecedent_emb, 0, 1))
        pairwise_bilinear_scores += torch.log(antecedent_mask.to(torch.float)) 

        if self.config['use_antecedent_distance_prior']:
            distance_score = torch.squeeze(self.antecedent_distance_score_ffnn(self.dropout(self.emb_antecedent_distance_prior.weight)), -1) 
            bucketed_distance = util.bucket_distance(antecedent_offsets)  
            antecedent_distance_score = distance_score[bucketed_distance]  
            pairwise_bilinear_scores += antecedent_distance_score
        return pairwise_bilinear_scores

    @classmethod
    def coarse_antecedent_prune(cls, pairwise_scores, antecedent_mask, antecedent_offsets, max_top_antecedents):
        top_pairwise_scores, top_antecedent_idx = torch.topk(pairwise_scores, k=max_top_antecedents, dim=-1)
        top_antecedent_mask = util.batch_select_2d(antecedent_mask, top_antecedent_idx)  
        top_antecedent_offsets = util.batch_select_2d(antecedent_offsets, top_antecedent_idx)
        return top_pairwise_scores, top_antecedent_idx, top_antecedent_mask, top_antecedent_offsets

    def get_slow_pairwise_scores(self, anaphora_emb, antecedent_emb, antecedent_idx, antecedent_offsets, span_speaker_ids=None):
        conf, feat_emb_list = self.config, []
        num_spans, max_top_antecedents = antecedent_idx.size()[:2]
        if conf['use_antecedent_distance']:
            antecedent_distance = util.bucket_distance(antecedent_offsets)  
            antecedent_distance_emb = self.emb_top_antecedent_distance(antecedent_distance) 
            feat_emb_list.append(antecedent_distance_emb)
        if conf['use_speaker_indicator']:
            antecedent_speaker_id = span_speaker_ids[antecedent_idx]
            same_speaker = torch.unsqueeze(span_speaker_ids, 1) == antecedent_speaker_id 
            same_speaker = same_speaker.to(torch.long)
            same_speaker_emb = self.emb_same_speaker(same_speaker)
            feat_emb_list.append(same_speaker_emb)

        anaphora_emb = torch.unsqueeze(anaphora_emb, 1).repeat(1, max_top_antecedents, 1)
        antecedent_emb = antecedent_emb[antecedent_idx]  
        feature_emb = torch.cat(feat_emb_list, dim=-1) 
        pair_emb_list = [anaphora_emb, antecedent_emb, self.dropout(feature_emb)]
        if conf['pair_with_similarity']:
            pair_emb_list.append(anaphora_emb * antecedent_emb)
        pair_emb = torch.cat(pair_emb_list, dim=-1)
        pairwise_slow_scores = torch.squeeze(self.coref_score_ffnn(pair_emb), -1) 
        return pairwise_slow_scores

    def do_coref_propagation(self, pairwise_mention_scores, top_span_emb,
                             antecedent_offsets, antecedent_mask, top_span_speaker_ids,
                             top_antecedent_idx, top_antecedent_scores):
        conf, device = self.config, top_antecedent_idx.device
        num_spans, max_top_antecedents = top_antecedent_idx.size()[:2]
        for prop_i in range(conf['coref_propagation']):
            top_antecedent_emb = top_span_emb[top_antecedent_idx] 
            attended_emb = propagation.propagate_coref(top_span_emb, top_antecedent_emb, top_antecedent_scores,
                                                       void_negative=conf['void_negative'])

            gate_fct = self.scalar_gate_ffnn if conf['scalar_gate'] else self.gate_ffnn
            top_span_emb = self.apply_gate(gate_fct, top_span_emb, attended_emb)

            top_antecedent_scores, top_antecedent_idx, top_antecedent_mask, top_antecedent_offsets, pairwise_fast_scores = \
                self.do_antecedent_scoring(pairwise_mention_scores, top_span_emb, top_span_emb,
                                           antecedent_offsets, antecedent_mask, max_top_antecedents, top_span_speaker_ids)

            if prop_i == conf['coref_propagation'] - 1:
                return top_span_emb, (top_antecedent_scores, top_antecedent_idx, top_antecedent_mask, top_antecedent_offsets, pairwise_fast_scores)

    @classmethod
    def apply_gate(cls, gate_ffnn, old_emb, new_emb):
        gate = gate_ffnn(torch.cat([old_emb, new_emb], dim=-1)) 
        gate = torch.sigmoid(gate)
        return gate * old_emb + (1 - gate) * new_emb

    @classmethod
    def get_top_spans(cls, candidate_idx_sorted, candidate_starts, candidate_ends, num_top_spans, allow_nested=True):
        selected_candidate_idx = []
        start_to_max_end, end_to_min_start = {}, {} 
        selected_token_idx = set() 
        for i, candidate_idx in enumerate(candidate_idx_sorted):
            if len(selected_candidate_idx) >= num_top_spans:
                break
            span_start_idx = candidate_starts[candidate_idx]
            span_end_idx = candidate_ends[candidate_idx]
            if allow_nested:
                cross_overlap = False
                for token_idx in range(span_start_idx, span_end_idx + 1):
                    max_end = start_to_max_end.get(token_idx, -1)
                    if token_idx > span_start_idx and max_end > span_end_idx:
                        cross_overlap = True
                        break
                    min_start = end_to_min_start.get(token_idx, -1)
                    if token_idx < span_end_idx and 0 <= min_start < span_start_idx:
                        cross_overlap = True
                        break
                if not cross_overlap:
                    selected_candidate_idx.append(candidate_idx)
                    max_end = start_to_max_end.get(span_start_idx, -1)
                    if span_end_idx > max_end:
                        start_to_max_end[span_start_idx] = span_end_idx
                    min_start = end_to_min_start.get(span_end_idx, -1)
                    if min_start == -1 or span_start_idx < min_start:
                        end_to_min_start[span_end_idx] = span_start_idx
            else:
                span_start_idx = candidate_starts[candidate_idx]
                span_end_idx = candidate_ends[candidate_idx]
                overlap = False
                for token_idx in range(span_start_idx, span_end_idx + 1):
                    if token_idx in selected_token_idx:
                        overlap = True
                        break
                if not overlap:
                    selected_candidate_idx.append(candidate_idx)
                    for token_idx in range(span_start_idx, span_end_idx + 1):
                        selected_token_idx.add(token_idx)
        selected_candidate_idx = sorted(selected_candidate_idx,
                                        key=lambda idx: (candidate_starts[idx], candidate_ends[idx]))
        if len(selected_candidate_idx) < num_top_spans: 
            selected_candidate_idx += ([selected_candidate_idx[0]] * (num_top_spans - len(selected_candidate_idx)))
        return selected_candidate_idx

    @classmethod
    def get_span_types(cls, span_type_logits):
        if span_type_logits is None:
            return None
        span_types = span_type_logits.argmax(axis=-1) 
        span_types -= 1 
        return span_types.tolist()

    @classmethod
    def get_predicted_antecedents(cls, antecedent_idx, antecedent_scores):
        predicted_antecedents = []
        for i, idx in enumerate(np.argmax(antecedent_scores, axis=1) - 1):
            if idx < 0:
                predicted_antecedents.append(-1)
            else:
                predicted_antecedents.append(antecedent_idx[i][idx])
        return predicted_antecedents

    @classmethod
    def get_predicted_clusters(cls, span_starts, span_ends, span_mention_scores, span_types,
                               antecedent_idx, antecedent_scores, subtoken_map,
                               predicted_clusters, predicted_clusters_subtok, predicted_clusters_idx, predicted_types,
                               mention_to_cluster_id,
                               start_token_idx=0, context_mention_starts=None, context_mention_ends=None,
                               allow_singletons=True):
        def is_singleton(idx):
            return (span_types[idx] >= 0) if span_types is not None else (span_mention_scores[idx] > 0)

        predicted_antecedents = cls.get_predicted_antecedents(antecedent_idx, antecedent_scores)

        mentions = set() 
        if context_mention_starts is not None:
            for m_s, m_e in zip(context_mention_starts, context_mention_ends):
                mentions.add((m_s, m_e))
                assert (subtoken_map[m_s], subtoken_map[m_e]) in mention_to_cluster_id
        for i, predicted_idx in enumerate(predicted_antecedents):
            mention = span_starts[i], span_ends[i]
            mention_tok = subtoken_map[mention[0]], subtoken_map[mention[1]] 
            if span_starts[i] < start_token_idx:
                continue

            if predicted_idx < 0:
                if allow_singletons and is_singleton(i) > 0:
                    cluster_id = len(predicted_clusters)
                    predicted_clusters.append([mention_tok])
                    predicted_clusters_subtok.append([mention])
                    predicted_clusters_idx.append([i])
                    predicted_types.append([span_types[i] if span_types else None])
                    mention_to_cluster_id[mention_tok] = cluster_id
                    mentions.add(mention)
                continue

            antecedent = span_starts[predicted_idx], span_ends[predicted_idx]
            antecedent_tok = subtoken_map[antecedent[0]], subtoken_map[antecedent[1]]
            antecedent_cluster_id = mention_to_cluster_id.get(antecedent_tok, -1)
            if antecedent[0] < start_token_idx:
                assert antecedent_cluster_id != -1 
            if antecedent_cluster_id == -1: 
                antecedent_cluster_id = len(predicted_clusters)
                predicted_clusters.append([antecedent_tok])
                predicted_clusters_subtok.append([antecedent])
                predicted_clusters_idx.append([predicted_idx])
                predicted_types.append([span_types[predicted_idx] if span_types else None])
                mention_to_cluster_id[antecedent_tok] = antecedent_cluster_id

            predicted_clusters[antecedent_cluster_id].append(mention_tok)
            predicted_clusters_subtok[antecedent_cluster_id].append(mention)
            predicted_clusters_idx[antecedent_cluster_id].append(i)
            predicted_types[antecedent_cluster_id].append(span_types[i] if span_types else None)
            mention_to_cluster_id[mention_tok] = antecedent_cluster_id

            mentions.add(antecedent)
            mentions.add(mention)

        mentions = sorted(list(mentions))
        mention_starts = [s for s, e in mentions]
        mention_ends = [e for s, e in mentions]

        return mention_starts, mention_ends
