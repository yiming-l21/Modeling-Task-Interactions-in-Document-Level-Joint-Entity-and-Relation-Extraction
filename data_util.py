from transformers import RobertaTokenizer, RobertaTokenizerFast
from collections import defaultdict
import util
import numpy as np
import util_io
from seq_encoding import encode_long_sequence
import torch
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


def merge_clusters(clusters, subtoks=None):
    merged_clusters = []
    merged2orig = {}
    for orig_i, cluster in enumerate(clusters):
        existing_i = None
        for mention in cluster:
            for merged_i, merged_cluster in enumerate(merged_clusters):
                if mention in merged_cluster:
                    existing_i = merged_i
                    break
            if existing_i is not None:
                break
        if existing_i is not None:
            merged2orig[existing_i].append(orig_i)
            merged_clusters[existing_i].update(cluster)
        else:
            merged2orig[len(merged2orig)] = [orig_i]
            merged_clusters.append(set(cluster))

    return [tuple(cluster) for cluster in merged_clusters], merged2orig


def get_doc_docred(inst, tokenizer, ner2id, rel2id, is_training):
    subtoks, subtok_tok_end, subtok_sent_end = [], [], []
    subtok_to_tok, tok_to_subtok = [], {}
    for sent_i, sent in enumerate(inst['sents']):
        if len(sent) > 70:
            logger.info(f'Long sentence w/ {len(sent)} tokens: {sent}')
        for token_i, token in enumerate(sent):
            tok_to_subtok[(sent_i, token_i)] = len(subtoks)
            if isinstance(tokenizer, (RobertaTokenizer, RobertaTokenizerFast)):
                token = ' ' + token
            subs = tokenizer.tokenize(token)

            subtoks += subs
            subtok_tok_end += [0] * len(subs)
            subtok_tok_end[-1] = 1
            subtok_sent_end += [0] * len(subs)
            subtok_to_tok += [(sent_i, token_i)] * len(subs)
        tok_to_subtok[(sent_i, len(sent))] = len(subtoks)
        subtok_sent_end[-1] = 1
    entities, entity_types = [], []
    mention2types = {}
    for entity_i, entity in enumerate(inst['vertexSet']):
        mentions, types = [], []
        for m in entity:
            m_s = tok_to_subtok[(m['sent_id'], m['pos'][0])]
            m_e = tok_to_subtok[(m['sent_id'], m['pos'][1])] - 1
            if (m_s, m_e) not in mentions:
                mentions.append((m_s, m_e))
                types.append(ner2id[m['type']])
                mention2types[(m_s, m_e)] = types[-1]
        sorted_indices = util.argsort(mentions)
        entities.append([mentions[idx] for idx in sorted_indices])
        entity_types.append([types[idx] for idx in sorted_indices])

    pos_pairs = defaultdict(lambda: [0] * len(rel2id))
    if 'labels' in inst:
        for rel in inst['labels']:
            pos_pairs[(rel['h'], rel['t'])][rel2id[rel['r']]] = 1
    def create_neg_rel_label():
        rel_label = [0] * len(rel2id)
        rel_label[rel2id['Na']] = 1
        return rel_label
    entity_pairs, rel_labels = [], []
    num_pos_pairs, num_neg_pairs = 0, 0
    for h in range(len(entities)):
        for t in range(len(entities)):
            if True or h != t:
                if (h, t) in pos_pairs:
                    rel_label = pos_pairs[(h, t)]
                    num_pos_pairs += 1
                else:
                    rel_label = create_neg_rel_label()
                    num_neg_pairs += 1
                entity_pairs.append((h, t))
                rel_labels.append(rel_label)
    assert len(entity_pairs) == len(entities)**2

    merged_clusters = [[(subtok_to_tok[m_s], subtok_to_tok[m_e]) for m_s, m_e in entity] for entity in entities]
    merged_clusters, merged2orig = merge_clusters(merged_clusters)
    merged_cluster_types = []
    mention_starts, mention_ends, mention_cluster_id, mention_type = [], [], [], []
    for cluster_i, cluster in enumerate(merged_clusters):
        cluster_types = []
        for m_s_orig, m_e_orig in cluster:
            m_e_orig = m_e_orig[0], m_e_orig[1] + 1
            m_s, m_e = tok_to_subtok[m_s_orig], tok_to_subtok[m_e_orig] - 1
            mention_starts.append(m_s)
            mention_ends.append(m_e)
            mention_type.append(mention2types[(m_s, m_e)])
            mention_cluster_id.append(cluster_i)
            cluster_types.append(mention_type[-1])
        merged_cluster_types.append(tuple(cluster_types))
    mentions = [(m_s, m_e) for m_s, m_e in zip(mention_starts, mention_ends)]
    sorted_indices = util.argsort(mentions)
    mention_starts = np.array(mention_starts)[sorted_indices].tolist()
    mention_ends = np.array(mention_ends)[sorted_indices].tolist()
    mention_type = np.array(mention_type)[sorted_indices].tolist()
    mention_cluster_id = np.array(mention_cluster_id)[sorted_indices].tolist()
    flattened_rel_labels = []
    for merged_h in range(len(merged2orig) + 1):
        for merged_t in range(len(merged2orig) + 1):
            if merged_h == 0 or merged_t == 0:
                rel_label = create_neg_rel_label()
            else:
                orig_h, orig_t = merged2orig[merged_h - 1][0], merged2orig[merged_t - 1][0]
                if (orig_h, orig_t) in pos_pairs:
                    rel_label = pos_pairs[(orig_h, orig_t)]
                else:
                    rel_label = create_neg_rel_label()
            flattened_rel_labels.append(rel_label)

    doc = {
        'title': inst['title'],
        'sents': inst['sents'],
        'subtoks': subtoks,
        'subtok_tok_end': subtok_tok_end,
        'subtok_sent_end': subtok_sent_end,
        'subtok_to_tok': subtok_to_tok,
        'tok_to_subtok': tok_to_subtok,
        'entities': entities,
        'entity_types': entity_types,
        'entity_pairs': entity_pairs,
        'rel_labels': rel_labels,
        'mention_starts': mention_starts,
        'mention_ends': mention_ends,
        'mention_type': mention_type,
        'mention_cluster_id': mention_cluster_id,
        'clusters': merged_clusters,
        'cluster_types': merged_cluster_types,
        'flattened_rel_labels': flattened_rel_labels
    }
    return doc, (num_pos_pairs, num_neg_pairs)


def get_doc_dwie(inst, tokenizer, ner2id, rel2id, is_training):
    to_return = get_doc_docred(inst, tokenizer, ner2id, rel2id, is_training)
    return to_return


def get_all_docs(dataset_name, file_path, tokenizer, ner2id, rel2id, is_training):
    if dataset_name in ['docred', 'dwie','envdocred','redocred','envredocred']:
        instances = util_io.read_json(file_path)
    if dataset_name == 'docred' or dataset_name == 'envdocred' or dataset_name == 'redocred' or dataset_name == 'envredocred':
        get_doc = get_doc_docred
    elif dataset_name == 'dwie':
        get_doc = get_doc_dwie
    else:
        raise ValueError(dataset_name)
    docs, total_pos_pairs, total_neg_pairs = [], 0, 0
    for inst in tqdm(instances, desc='Docs'):
        doc, (num_pos, num_neg) = get_doc(inst, tokenizer, ner2id, rel2id, is_training)
        docs.append(doc)
        total_pos_pairs += num_pos
        total_neg_pairs += num_neg
    return docs, (total_pos_pairs, total_neg_pairs)


def convert_docs_to_features(dataset_name, docs, tokenizer, max_seq_len, overlapping, is_training, max_training_seg,
                             show_example=False):
    short_example_shown, long_example_shown = False, False
    features = []

    for doc_i, doc in enumerate(tqdm(docs, desc='Features')):
        feature, is_max_context = encode_long_sequence(doc['title'], tokenizer, doc['subtoks'],
                                                       max_seq_len=max_seq_len, overlapping=overlapping,
                                                       constraints=(doc['subtok_sent_end'], doc['subtok_tok_end']))

        num_seg, m_i = len(feature['input_ids']), len(doc['mention_ends'])
        if is_training and num_seg > max_training_seg:
            num_training_subtoks = sum(util.flatten(is_max_context[:max_training_seg]))
            is_max_context = is_max_context[:max_training_seg]
            feature['input_ids'] = feature['input_ids'][:max_training_seg]
            feature['attention_mask'] = feature['attention_mask'][:max_training_seg]
            feature['token_type_ids'] = feature['token_type_ids'][:max_training_seg]
            while m_i > 0 and doc['mention_ends'][m_i - 1] >= num_training_subtoks:
                m_i -= 1
            truncated_entities = []
            for entity in doc['entities']:
                truncated_entity = [(m_s, m_e) for m_s, m_e in entity if m_e < num_training_subtoks]
                truncated_entities.append(truncated_entity)
            truncated_entity_pairs, truncated_rel_labels = [], []
            for (h, t), rel_label in zip(doc['entity_pairs'], doc['rel_labels']):
                if truncated_entities[h] and truncated_entities[t]:
                    truncated_entity_pairs.append((h, t))
                    truncated_rel_labels.append(rel_label)
            feature['entities'] = truncated_entities
            feature['entity_pairs'] = truncated_entity_pairs
            feature['rel_labels'] = truncated_rel_labels
            feature['subtok_tok_end'] = doc['subtok_tok_end'][:num_training_subtoks]
            feature['subtok_sent_end'] = doc['subtok_sent_end'][:num_training_subtoks]
        else:
            feature['entities'] = doc['entities']
            feature['entity_pairs'] = doc['entity_pairs']
            feature['rel_labels'] = doc['rel_labels']
            feature['subtok_tok_end'] = doc['subtok_tok_end']
            feature['subtok_sent_end'] = doc['subtok_sent_end']

        feature['is_max_context'] = is_max_context
        feature['entity_types'] = doc['entity_types']
        entity_pairs_h = [pair[0] for pair in feature['entity_pairs']]
        entity_pairs_t = [pair[1] for pair in feature['entity_pairs']]
        feature['entity_pairs_h'] = entity_pairs_h
        feature['entity_pairs_t'] = entity_pairs_t
        del feature['entity_pairs']
        feature['mention_starts'] = doc['mention_starts'][:m_i]
        feature['mention_ends'] = doc['mention_ends'][:m_i]
        feature['mention_type'] = doc['mention_type'][:m_i]
        feature['mention_cluster_id'] = doc['mention_cluster_id'][:m_i]
        feature['flattened_rel_labels'] = doc['flattened_rel_labels']

        def get_sent_map(is_end):
            mapping, offset = [], 0
            for idx_is_end in is_end:
                mapping.append(offset)
                if idx_is_end:
                    offset += 1
            return mapping

        def get_tok_start_or_end(tok_end):
            start_or_end = tok_end[:] 
            for i in range(len(start_or_end) - 1)[::-1]:
                if start_or_end[i]:
                    start_or_end[i + 1] = 1
            start_or_end[0] = 1
            return start_or_end

        feature['doc_len'] = sum(util.flatten(is_max_context))
        feature['tok_start_or_end'] = get_tok_start_or_end(feature.pop('subtok_tok_end')) 
        feature['sent_map'] = get_sent_map(feature.pop('subtok_sent_end'))
        feature = {
            'title': doc['title'],
            'input_ids': feature['input_ids'],
            'attention_mask': feature['attention_mask'],
            'token_type_ids': feature['token_type_ids'],
            'is_max_context': feature['is_max_context'],
            'doc_len': feature['doc_len'],
            'tok_start_or_end': torch.tensor(feature['tok_start_or_end'], dtype=torch.long),
            'sent_map': torch.tensor(feature['sent_map'], dtype=torch.long),
            'mention_starts': torch.tensor(feature['mention_starts'], dtype=torch.long),
            'mention_ends': torch.tensor(feature['mention_ends'], dtype=torch.long),
            'mention_type': torch.tensor(feature['mention_type'], dtype=torch.long),
            'mention_cluster_id': torch.tensor(feature['mention_cluster_id'], dtype=torch.long),
            'flattened_rel_labels': torch.tensor(feature['flattened_rel_labels'], dtype=torch.long),
            'entities': feature['entities'], 
            'entity_types': feature['entity_types'],
            'entity_pairs_h': torch.tensor(feature['entity_pairs_h'], dtype=torch.long),
            'entity_pairs_t': torch.tensor(feature['entity_pairs_t'], dtype=torch.long),
            'rel_labels': torch.tensor(feature['rel_labels'], dtype=torch.long),
        }
        features.append(feature)

    return features
