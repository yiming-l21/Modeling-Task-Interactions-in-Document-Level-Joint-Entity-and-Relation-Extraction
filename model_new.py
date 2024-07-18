import propagation
from model_coref import CorefModel
import torch
import torch.nn as nn
import util
import logging
import numpy as np
import torch.nn.functional as F
import losses
import math

logger = logging.getLogger(__name__)


class NewJointModel(CorefModel):
    def __init__(self, config, num_entity_types=0):
        super(NewJointModel, self).__init__(config, num_entity_types)

        self.num_re_labels = config['num_re_labels']
        self.max_re_labels = config['max_re_labels']
        self.rel_emb_hidden_size = 32

        self.re_mention_hidden_size = config['re_mention_hidden_size'] if config['re_transform_mention'] else self.span_hidden_size
        self.re_head_transform = self.make_linear(self.span_hidden_size, self.re_mention_hidden_size)
        self.re_tail_transform = self.make_linear(self.span_hidden_size, self.re_mention_hidden_size)
        self.re_head_prior = self.make_linear(self.re_mention_hidden_size, self.num_re_labels, bias=False)
        self.re_tail_prior = self.make_linear(self.re_mention_hidden_size, self.num_re_labels, bias=False)
        self.re_fast_bilinear = self.make_linear(self.re_mention_hidden_size, self.re_mention_hidden_size * self.num_re_labels)
        self.re_fast_bilinear_propagation = self.make_linear(self.re_mention_hidden_size, self.re_mention_hidden_size * self.num_re_labels)
        self.re_num_blocks = config['re_num_blocks']
        self.re_block_size = self.re_mention_hidden_size // self.re_num_blocks
        self.re_slow_bilinear = self.make_linear(self.re_mention_hidden_size * self.re_block_size, self.num_re_labels)
        self.re_scoring = self.get_re_logits_fast if config['re_fast_bilinear'] else self.get_re_logits_slow
        self.re_loss_fct = losses.ATLoss()

        self.dygie_transform = self.make_linear(self.num_re_labels, self.re_mention_hidden_size, bias=False)
        self.dygie_gate = self.make_linear(self.re_mention_hidden_size * 2, self.re_mention_hidden_size)
        self.rel_dependent_transform = self.make_linear(self.re_mention_hidden_size, self.re_mention_hidden_size * self.num_re_labels)
        self.rel_independent_transform = self.make_linear(self.re_mention_hidden_size + self.rel_emb_hidden_size, self.re_mention_hidden_size)
        self.rel_emb = self.make_embedding(self.num_re_labels, self.rel_emb_hidden_size)
        self.re_rel_attn = self.make_ffnn(2 * self.re_mention_hidden_size, 0, 1)
        self.coref_add_bilinear = self.make_linear(self.re_mention_hidden_size, self.re_mention_hidden_size, bias=False)
        self.r2c_metric_ffnn = self.make_ffnn(self.num_re_labels * 3, self.num_re_labels, 1)
        self.r2c_dist_coef = self.make_linear(self.num_re_labels, 1, bias=False)
        self.r2c_scale = self.make_scalar(config['r2c_learned_scale_init'])

        self.debug = True

    def get_re_labels(self, re_logits):
        return self.re_loss_fct.get_label(re_logits, self.max_re_labels)

    def forward_single(self, tokens, tok_start_or_end=None, sent_map=None, speaker_ids=None,
                       mention_starts=None, mention_ends=None, mention_type=None, mention_cluster_id=None,
                       flattened_rel_labels=None, **kwargs):
        """ Decoding for a single doc. """
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
        if not conf['r2c_only'] or not conf['r2c_coef']:
            top_antecedent_scores, top_antecedent_idx, top_antecedent_mask, top_antecedent_offsets, pairwise_fast_scores = self.do_antecedent_scoring(
                pairwise_mention_scores, pair_anaphora_emb, pair_antecedent_emb, antecedent_offsets, antecedent_mask, max_top_antecedents, top_span_speaker_ids)
        if conf['coref_propagation']:
            top_span_emb, (top_antecedent_scores, top_antecedent_idx, top_antecedent_mask, top_antecedent_offsets, pairwise_fast_scores) = \
                self.do_coref_propagation(pairwise_mention_scores, top_span_emb, antecedent_offsets, antecedent_mask, top_span_speaker_ids,
                                          top_antecedent_idx, top_antecedent_scores)

        if conf['re_transform_mention']:
            if conf['re_distinguish_ht']:
                re_head_emb = self.dropout(torch.tanh(self.re_head_transform(top_span_emb)))
                re_tail_emb = self.dropout(torch.tanh(self.re_tail_transform(top_span_emb)))
            else:
                re_head_emb = re_tail_emb = self.dropout(torch.tanh(self.re_head_transform(top_span_emb)))
        else:
            re_head_emb = re_tail_emb = top_span_emb

        re_pair_logits = self.re_scoring(self.re_fast_bilinear, re_head_emb, re_tail_emb) 
        if conf['re_add_prior']:
            head_priors = self.re_head_prior(re_head_emb) 
            tail_priors = self.re_tail_prior(re_tail_emb)
            re_priors = (head_priors.unsqueeze(1) + tail_priors.unsqueeze(0)).view(-1, tail_priors.size()[-1])
            re_pair_logits += re_priors

        if conf['re_dygie']:
            for dygie_i in range(conf['re_dygie']):
                attended_emb = propagation.dygie(re_head_emb, re_tail_emb, re_pair_logits, self.dygie_transform, with_atloss=True)
                span_emb = self.apply_gate(self.dygie_gate, re_head_emb, attended_emb)
                re_head_emb = re_tail_emb = span_emb

                re_pair_logits = self.re_scoring(self.re_fast_bilinear, re_head_emb, re_tail_emb) 

        if conf['re_propagation']:
            new_re_pair_logits, (re_head_emb, re_tail_emb) = self.do_re_propagation_rel_dependent(re_head_emb, re_tail_emb, re_pair_logits)
            if conf['re_propagation_update_scores']:
                if conf['re_propagation_only_last']:
                    re_pair_logits = new_re_pair_logits
                    if conf['re_add_prior']:
                        re_pair_logits += re_priors 
                else:
                    re_pair_logits += new_re_pair_logits 

        if conf['coref_additional'] and conf['re_propagation']:
            top_antecedent_scores, top_antecedent_idx, top_antecedent_mask, top_antecedent_offsets, pairwise_fast_scores = \
                self.redo_antecedent_scoring(re_head_emb, re_tail_emb, pairwise_fast_scores, antecedent_mask, antecedent_offsets, max_top_antecedents)
        if conf['r2c_coef']:
            metric_ffnn = self.r2c_metric_ffnn if conf['r2c_learned_metric'] else None
            dist_coef = self.r2c_dist_coef if conf['r2c_fixed_metric_coef'] else None
            r2c_distances = self.re_to_coref(top_span_emb, re_pair_logits.detach() if conf['r2c_detach'] else re_pair_logits,
                                             metric_ffnn, dist_coef, node_attn=None,
                                             with_atloss=True, fixed_metric=conf['r2c_fixed_metric']) 
            r2c_distances *= (self.r2c_scale if conf['r2c_learned_scale'] else conf['r2c_learned_scale_init'])

            if conf['r2c_only']:
                pairwise_fast_scores = pairwise_mention_scores - r2c_distances
            else:
                pairwise_fast_scores -= r2c_distances
            top_antecedent_scores, top_antecedent_idx, top_antecedent_mask, top_antecedent_offsets = \
                self.coarse_antecedent_prune(pairwise_fast_scores, antecedent_mask, antecedent_offsets, max_top_antecedents)
        top_antecedent_scores = torch.cat([torch.zeros(num_top_spans, 1, device=device), top_antecedent_scores], dim=-1)  # [num top spans, max top antecedents + 1]

        outputs = top_span_starts, top_span_ends, top_span_mention_scores, top_span_type_logits, \
                  top_antecedent_idx, top_antecedent_scores, re_pair_logits
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
        loss_coref = loss * conf['coref_loss_coef']
        re_pair_labels = self.adapt_re_gold_labels(top_span_cluster_ids, flattened_rel_labels)
        loss_re = self.re_loss_fct(re_pair_logits, re_pair_labels.to(torch.float)).mean()
        loss = loss_coref + loss_re
        if conf['r2c_coef']:
            loss_r2c = losses.get_r2c_loss(top_span_cluster_ids, antecedent_mask, r2c_distances, margin=conf['r2c_margin'])
            if conf['r2c_learned_scale_reg'] and conf['r2c_learned_scale']:
                loss_r2c += conf['r2c_learned_scale_reg'] / self.r2c_scale
            loss_r2c *= conf['r2c_coef']
            loss += loss_r2c

        return loss, outputs

    def get_re_logits_fast(self, bilinear, re_head_emb, re_tail_emb):
        intermediate = bilinear(re_head_emb).view(-1, self.re_mention_hidden_size, self.num_re_labels).permute(2, 0, 1) 
        target = torch.transpose(re_tail_emb, 0, 1).unsqueeze(0) 
        re_logits = torch.matmul(intermediate, target) 
        re_logits = re_logits.permute(1, 2, 0).contiguous() 
        re_logits = re_logits.view(-1, re_logits.size()[-1]) 
        return re_logits

    def get_re_logits_slow(self, placeholder, re_head_emb, re_tail_emb):
        num_spans = re_head_emb.size()[0]
        pair_h_hidden = re_head_emb.repeat_interleave(num_spans, dim=0)
        pair_t_hidden = re_tail_emb.repeat(num_spans, 1)
        h_b = pair_h_hidden.view(-1, self.re_num_blocks, self.re_block_size)
        t_b = pair_t_hidden.view(-1, self.re_num_blocks, self.re_block_size)
        re_logits = self.re_slow_bilinear((h_b.unsqueeze(3) * t_b.unsqueeze(2))
                                          .view(-1, self.re_mention_hidden_size * self.re_block_size))
        re_logits = re_logits.view(num_spans, num_spans, -1) 
        re_logits = re_logits.view(-1, re_logits.size()[-1])
        return re_logits

    def do_re_propagation_rel_dependent(self, head_emb, tail_emb, re_logits):
        conf = self.config
        for prop_i in range(conf['re_propagation']):
            mention_transform, rel_emb = None, None
            if conf['re_propagation_transform']:
                if conf['re_propagation_rel_emb']:
                    mention_transform, rel_emb = self.rel_independent_transform, self.dropout(self.rel_emb.weight)
                else:
                    mention_transform = self.rel_dependent_transform
            attended_head_emb, attended_tail_emb = propagation.propagate_re(head_emb, tail_emb, re_logits,
                mention_transform, rel_emb, void_negative=conf['re_propagation_void_negative'], with_atloss=True)
            rel_reduction = self.apply_re_rel_attention if conf['re_propagation_rel_attention'] else self.apply_re_rel_reduce
            new_head_emb = rel_reduction(head_emb, self.dropout(attended_tail_emb), self.re_rel_attn)
            new_tail_emb = rel_reduction(tail_emb, self.dropout(attended_head_emb), self.re_rel_attn)
            head_emb, tail_emb = self.dropout(new_head_emb), self.dropout(new_tail_emb) 
            if conf['re_propagation_update_scores']:
                bilinear = self.re_fast_bilinear if self.config['re_propagation_same_bilinear'] else self.re_fast_bilinear_propagation
                re_logits = self.get_re_logits_fast(bilinear, head_emb, tail_emb)
                re_logits = re_logits.view(-1, re_logits.size()[-1]) 

        return re_logits, (head_emb, tail_emb)

    @classmethod
    def apply_re_rel_attention(cls, span_emb, span_rel_emb, rel_attn):
        (num_labels, num_spans), device = span_rel_emb.size()[0:2], span_rel_emb.device
        span_rel_emb = span_rel_emb.transpose(0, 1) 
        attn_emb = torch.cat([span_rel_emb, span_emb.unsqueeze(1).repeat(1, num_labels, 1)], dim=-1)
        attentions = rel_attn(attn_emb).squeeze(-1) 
        attentions = torch.cat([torch.ones(num_spans, 1, device=device), attentions], dim=-1)
        attentions += torch.log((attentions > 0).to(torch.float)) 
        attentions = F.softmax(attentions, dim=-1)

        attn_emb = torch.cat([span_emb.unsqueeze(1), span_rel_emb], dim=1)  
        attended_emb = attn_emb * attentions.unsqueeze(-1)  
        attended_emb = attended_emb.sum(dim=1, keepdims=False) 
        return attended_emb

    @classmethod
    def apply_re_rel_reduce(cls, span_emb, span_rel_emb, placeholder):
        reduced_emb = span_emb + span_rel_emb.mean(dim=0) 
        return reduced_emb

    @classmethod
    def adapt_re_gold_labels(cls, span_cluster_ids, flattened_rel_labels):
        num_spans = span_cluster_ids.size()[0]
        num_clusters = math.isqrt(flattened_rel_labels.size()[0])
        matrix_rel_labels = flattened_rel_labels.view(num_clusters, num_clusters, -1) 

        pair_cluster_ids_x = span_cluster_ids.repeat_interleave(num_spans, dim=0)
        pair_cluster_ids_y = span_cluster_ids.repeat(num_spans)
        re_pair_labels = matrix_rel_labels[pair_cluster_ids_x, pair_cluster_ids_y]
        return re_pair_labels

    def redo_antecedent_scoring(self, pair_anaphora_emb, pair_antecedent_emb, pairwise_fast_scores,
                                antecedent_mask, antecedent_offsets, max_top_antecedents):
        bilinear = self.coarse_bilinear if self.config['coref_same_bilinear'] else self.coref_add_bilinear
        pairwise_bilinear_scores_1 = torch.matmul(bilinear(pair_anaphora_emb),
                                                  torch.transpose(pair_antecedent_emb, 0, 1))
        pairwise_bilinear_scores_2 = torch.matmul(bilinear(pair_antecedent_emb),
                                                  torch.transpose(pair_anaphora_emb, 0, 1))
        new_pairwise_bilinear_scores = (pairwise_bilinear_scores_1 + pairwise_bilinear_scores_2) / 2

        pairwise_fast_scores += new_pairwise_bilinear_scores
        top_pairwise_fast_scores, top_antecedent_idx, top_antecedent_mask, top_antecedent_offsets = \
            self.coarse_antecedent_prune(pairwise_fast_scores, antecedent_mask, antecedent_offsets, max_top_antecedents)
        return top_pairwise_fast_scores, top_antecedent_idx, top_antecedent_mask, top_antecedent_offsets, pairwise_fast_scores

    def re_to_coref(self, span_emb, re_pair_logits, metric_ffnn=None, dist_coef=None, node_attn=None, with_atloss=True, fixed_metric='l1'):
        num_spans, conf = span_emb.size()[0], self.config
        if with_atloss:
            re_pair_logits_th = re_pair_logits[:, 0:1]
            re_pair_logits -= re_pair_logits_th
        if conf['r2c_normalize']:
            re_pair_logits = F.softmax(re_pair_logits, dim=-1)
        re_pair_logits = re_pair_logits.view(num_spans, num_spans, -1) 
        indices = torch.arange(0, num_spans, dtype=torch.long, device=re_pair_logits.device)
        pair_idx_x = indices.repeat_interleave(num_spans, dim=0)
        pair_idx_y = indices.repeat(num_spans)

        def get_distance(re_pair_logits, metric_ffnn, dist_coef):
            re_pair_logits_sum = F.relu(re_pair_logits).sum(dim=-1).sum(dim=0)
            _, node_idx = torch.topk(re_pair_logits_sum, k=min(conf['r2c_max_nodes'], re_pair_logits_sum.size()[-1]), dim=-1)
            re_pair_logits = re_pair_logits[:, node_idx, :]
            pair_logits_x, pair_logits_y = re_pair_logits[pair_idx_x], re_pair_logits[pair_idx_y] 
            if metric_ffnn is not None:
                pair_logits_all = [pair_logits_x, pair_logits_y, pair_logits_x - pair_logits_y]
                pair_logits_all = torch.cat(pair_logits_all, dim=-1)
                dist = metric_ffnn(pair_logits_all).squeeze(-1)
                dist = F.elu(dist) + 1 
            else:
                if fixed_metric == 'cosine':
                    dist = F.cosine_similarity(pair_logits_x, pair_logits_y, dim=-1)
                else:
                    if fixed_metric == 'l2':
                        dist = (pair_logits_x - pair_logits_y) ** 2
                    else:
                        dist = torch.abs(pair_logits_x - pair_logits_y)
                    if dist_coef is None:
                        dist = dist.sum(dim=-1)
                    else:
                        dist_coef = F.softmax(dist_coef.weight.squeeze(0), dim=0)
                        dist = torch.matmul(dist, dist_coef) 

            dist = dist.sum(dim=-1)
            if fixed_metric == 'l2':
                dist = torch.sqrt(dist)
            return dist

        dist_head = get_distance(re_pair_logits, metric_ffnn, dist_coef)
        re_pair_logits = re_pair_logits.transpose(0, 1).contiguous()
        dist_tail = get_distance(re_pair_logits, metric_ffnn, dist_coef)
        dist_final = dist_head + dist_tail
        dist_final = dist_final.view(num_spans, num_spans)
        return dist_final
