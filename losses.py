import torch
import torch.nn as nn
import torch.nn.functional as F
import util


class ATLoss(nn.Module):
    """ Adapted threshold loss from Zhou et al., 2021. """

    def __init__(self):
        super().__init__()

    def forward(self, logits, labels):
        # TH label
        th_label = torch.zeros_like(labels)
        th_label[:, 0] = 1.0
        labels[:, 0] = 0.0

        p_mask = labels + th_label
        n_mask = 1 - labels

        logit1 = logits - (1 - p_mask) * 1e30  # Total: P & th
        loss1 = -(F.log_softmax(logit1, dim=-1) * labels).sum(dim=-1)  # Maximize likelihood of P

        logit2 = logits - (1 - n_mask) * 1e30  # Total: th & N
        loss2 = -(F.log_softmax(logit2, dim=-1) * th_label).sum(dim=-1)  # Maximize likelihood of th

        # Sum two parts
        loss = loss1 + loss2
        return loss

    @classmethod
    def get_label(cls, logits, max_labels=None):
        th_logit = logits[:, 0].unsqueeze(1)
        mask = (logits > th_logit)
        if max_labels:
            top_logits, _ = torch.topk(logits, max_labels, sorted=True, dim=-1)
            mask &= logits >= top_logits[:, -1].unsqueeze(1)
        output = mask.to(torch.long)
        output[:, 0] = (output.sum(dim=-1) == 0).to(torch.long)
        return output


def get_mention_ranking_loss(antecedent_scores, span_cluster_ids, antecedent_idx, antecedent_mask):
    antecedent_cluster_ids = span_cluster_ids[antecedent_idx]
    antecedent_cluster_ids += (antecedent_mask.to(torch.long) - 1) * 100000
    same_gold_cluster_indicator = (antecedent_cluster_ids == torch.unsqueeze(span_cluster_ids, 1))
    non_dummy_indicator = torch.unsqueeze(span_cluster_ids > 0, 1)
    pairwise_labels = same_gold_cluster_indicator & non_dummy_indicator
    dummy_antecedent_labels = torch.logical_not(pairwise_labels.any(dim=-1, keepdims=True))
    antecedent_gold_labels = torch.cat([dummy_antecedent_labels, pairwise_labels], dim=-1)

    log_marginalized_antecedent_scores = torch.logsumexp(antecedent_scores + torch.log(antecedent_gold_labels.to(torch.float)), dim=-1)
    log_norm = torch.logsumexp(antecedent_scores, dim=-1) 
    loss = torch.sum(log_norm - log_marginalized_antecedent_scores)
    return loss, (log_marginalized_antecedent_scores, log_norm)


def get_mention_score_loss(mention_scores, labels):
    gold_mention_nll = -nn.LogSigmoid()(mention_scores[labels > 0])
    gold_mention_nll = gold_mention_nll[gold_mention_nll < 4.6]

    non_gold_mention_prob = 1 - torch.sigmoid(mention_scores[labels == 0])
    non_gold_mention_prob = non_gold_mention_prob[non_gold_mention_prob > 1e-2]
    non_gold_mention_nll = -torch.log(non_gold_mention_prob)
    num_sampling = min(gold_mention_nll.size()[0], non_gold_mention_nll.size()[0]) * 2
    gold_mention_nll = util.random_select(gold_mention_nll, num_sampling)
    non_gold_mention_nll = util.random_select(non_gold_mention_nll, num_sampling)

    loss_gold_mention = gold_mention_nll.sum()
    loss_non_gold_mention = non_gold_mention_nll.sum()
    return loss_gold_mention, loss_non_gold_mention


def get_mention_type_loss(mention_type_logits, labels):
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    nll = loss_fct(mention_type_logits, labels)
    loss_gold_mention = nll[labels > 0].sum()
    loss_non_gold_mention = util.random_select(nll[labels == 0], (labels > 0).sum() * 2).sum()

    return loss_gold_mention, loss_non_gold_mention


def get_r2c_loss(span_cluster_ids, antecedent_mask, r2c_distances, margin=1):
    num_spans = span_cluster_ids.size()[0]
    r2c_mask = (span_cluster_ids != 0).unsqueeze(1).repeat(1, num_spans)
    r2c_mask |= (span_cluster_ids != 0).unsqueeze(0)
    r2c_mask &= antecedent_mask
    r2c_labels = (span_cluster_ids.unsqueeze(1) == span_cluster_ids.unsqueeze(0)).to(torch.float)
    r2c_labels = r2c_labels[r2c_mask]
    r2c_distances = r2c_distances[r2c_mask]
    loss = r2c_labels * torch.pow(r2c_distances, 2) + \
           (1 - r2c_labels) * torch.pow(torch.clamp(margin - r2c_distances, min=0), 2)
    r2c_labels_pos = r2c_labels > 0
    loss_pos = loss[r2c_labels_pos]
    loss_neg = util.random_select(loss[r2c_labels == 0], 1 + r2c_labels_pos.sum() * 2)
    loss = loss_pos.sum() + loss_neg.sum()
    loss /= (1 + r2c_labels_pos.sum() * 3)
    return loss
