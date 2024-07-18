import os
from os.path import join, exists
import ujson as json
import numpy as np
from functools import cached_property, lru_cache
import torch
import logging
import pickle

logger = logging.getLogger(__name__)


class DocredEvaluator:
    def __init__(self, dataset_dir, ner2id, rel2id):
        self.dataset_dir = dataset_dir
        self.ner2id = ner2id
        self.rel2id = rel2id

        self.id2ner = {i: t for t, i in ner2id.items()}
        self.id2rel = {i: t for t, i in rel2id.items()}

    @cached_property
    def fact_dir(self):
        fact_dir = join(self.dataset_dir, 'ref')
        os.makedirs(fact_dir, exist_ok=True)
        return fact_dir

    @lru_cache()
    def get_train_facts(self, raw_file_name):
        fact_file_path = raw_file_name[raw_file_name.find("train_"):]
        fact_file_path = os.path.join(self.fact_dir, fact_file_path.replace(".json", ".fact"))
        if os.path.exists(fact_file_path):
            print(f"Loading {fact_file_path}")
            with open(fact_file_path, 'rb') as f:
                fact_in_train = pickle.load(f)
        else:
            fact_in_train = set()
            with open(join(self.dataset_dir, raw_file_name), 'r') as f:
                orig_data = json.load(f)
            for data in orig_data:
                vertexSet = data['vertexSet']
                for label in data['labels']:
                    rel = label['r']
                    for n1 in vertexSet[label['h']]:
                        for n2 in vertexSet[label['t']]:
                            fact_in_train.add((n1['name'], n2['name'], rel))
            with open(fact_file_path, 'wb') as f:
                pickle.dump(fact_in_train, f, protocol=4)

        return fact_in_train

    @lru_cache()
    def get_gold(self, partition):
        with open(join(self.dataset_dir, f'{partition}.json'), 'r') as f:
            truth = json.load(f)

        gold = {}
        tot_evidences = 0
        titleset = set([])
        title2vectexSet = {}

        for x in truth:
            title = x['title']
            titleset.add(title)

            vertexSet = x['vertexSet']
            title2vectexSet[title] = vertexSet

            for label in x['labels']:
                r = label['r']
                h_idx = label['h']
                t_idx = label['t']
                gold[(title, r, h_idx, t_idx)] = set(label['evidence'])
                tot_evidences += len(label['evidence'])

        return gold, tot_evidences, titleset, title2vectexSet

    def to_official(self, all_preds, features):
        all_entity_pair_h, all_entity_pair_t, all_titles = [], [], []
        for f in features:
            all_entity_pair_h.append(f['entity_pairs_h'])
            all_entity_pair_t.append(f['entity_pairs_t'])
            all_titles += [f['title']] * f['entity_pairs_h'].shape[0]
        all_entity_pair_h = torch.cat(all_entity_pair_h, dim=-1).tolist()
        all_entity_pair_t = torch.cat(all_entity_pair_t, dim=-1).tolist()
        ans = []
        for entity_pair_i in range(all_preds.shape[0]):
            if all_entity_pair_h[entity_pair_i] == all_entity_pair_t[entity_pair_i]:
                continue
            rel_ids = np.nonzero(all_preds[entity_pair_i])[0].tolist()
            for rel_id in rel_ids:
                if rel_id != 0:
                    ans.append({
                        'title': all_titles[entity_pair_i],
                        'h_idx': all_entity_pair_h[entity_pair_i],
                        't_idx': all_entity_pair_t[entity_pair_i],
                        'r': self.id2rel[rel_id]
                    })
        return ans

    def official_evaluate(self, ans, partition='dev'):
        fact_in_train_annotated = self.get_train_facts('train_revised.json')
        fact_in_train_distant = self.get_train_facts('train_distant.json')
        gold, tot_evidences, titleset, title2vectexSet = self.get_gold(partition)
        ans.sort(key=lambda x: (x['title'], x['h_idx'], x['t_idx'], x['r']))
        if ans:
            submission_answer = [ans[0]]
            for i in range(1, len(ans)):
                x = ans[i]
                y = ans[i - 1]
                if (x['title'], x['h_idx'], x['t_idx'], x['r']) != (y['title'], y['h_idx'], y['t_idx'], y['r']):
                    submission_answer.append(ans[i])
        else:
            submission_answer = []
        correct_re = 0
        correct_evidence = 0
        pred_evi = 0

        correct_in_train_annotated = 0
        correct_in_train_distant = 0
        titleset2 = set([])
        for x in submission_answer:
            title = x['title']
            h_idx = x['h_idx']
            t_idx = x['t_idx']
            r = x['r']
            titleset2.add(title)
            if title not in title2vectexSet:
                continue
            vertexSet = title2vectexSet[title]

            if 'evidence' in x:
                evi = set(x['evidence'])
            else:
                evi = set([])
            pred_evi += len(evi)

            if (title, r, h_idx, t_idx) in gold:
                correct_re += 1
                gold_evi = gold[(title, r, h_idx, t_idx)]
                correct_evidence += len(gold_evi & evi)
                in_train_annotated = in_train_distant = False
                for n1 in vertexSet[h_idx]:
                    for n2 in vertexSet[t_idx]:
                        if not in_train_annotated and (n1['name'], n2['name'], r) in fact_in_train_annotated:
                            in_train_annotated = True
                        if not in_train_distant and (n1['name'], n2['name'], r) in fact_in_train_distant:
                            in_train_distant = True

                if in_train_annotated:
                    correct_in_train_annotated += 1
                if in_train_distant:
                    correct_in_train_distant += 1

        re_p = 1.0 * correct_re / len(submission_answer) * 100 if submission_answer else 0
        re_r = 1.0 * correct_re / len(gold) * 100
        if re_p + re_r == 0:
            re_f1 = 0
        else:
            re_f1 = 2.0 * re_p * re_r / (re_p + re_r)

        evi_p = 1.0 * correct_evidence / pred_evi * 100 if pred_evi > 0 else 0
        evi_r = 1.0 * correct_evidence / tot_evidences * 100
        if evi_p + evi_r == 0:
            evi_f1 = 0
        else:
            evi_f1 = 2.0 * evi_p * evi_r / (evi_p + evi_r)

        re_p_ignore_train_annotated = 1.0 * (correct_re - correct_in_train_annotated) / (
                len(submission_answer) - correct_in_train_annotated + 1e-5) * 100 if submission_answer else 0
        re_p_ignore_train = 1.0 * (correct_re - correct_in_train_distant) / (
                len(submission_answer) - correct_in_train_distant + 1e-5) * 100 if submission_answer else 0

        if re_p_ignore_train_annotated + re_r == 0:
            re_f1_ignore_train_annotated = 0
        else:
            re_f1_ignore_train_annotated = 2.0 * re_p_ignore_train_annotated * re_r / (
                        re_p_ignore_train_annotated + re_r)

        if re_p_ignore_train + re_r == 0:
            re_f1_ignore_train = 0
        else:
            re_f1_ignore_train = 2.0 * re_p_ignore_train * re_r / (re_p_ignore_train + re_r)

        return (re_p, re_r, re_f1), (evi_p, evi_r, evi_f1), \
               (re_p_ignore_train_annotated, re_r, re_f1_ignore_train_annotated), \
               (re_p_ignore_train, re_r, re_f1_ignore_train)


class DwieEvaluator:
    def __init__(self, dataset_dir, ner2id, rel2id):
        self.dataset_dir = dataset_dir
        self.ner2id = ner2id
        self.rel2id = rel2id

        self.id2ner = {i: t for t, i in ner2id.items()}
        self.id2rel = {i: t for t, i in rel2id.items()}

    @lru_cache()
    def get_gold(self, partition):
        with open(join(self.dataset_dir, f'{partition}.json'), 'r') as f:
            truth = json.load(f)

        gold = {}
        tot_evidences = 0
        titleset = set([])
        title2vectexSet = {}

        for x in truth:
            title = x['title']
            titleset.add(title)

            vertexSet = x['vertexSet']
            title2vectexSet[title] = vertexSet

            for label in x['labels']:
                r = label['r']
                h_idx = label['h']
                t_idx = label['t']
                gold[(title, r, h_idx, t_idx)] = set(label['evidence'])
                tot_evidences += len(label['evidence'])

        return gold, tot_evidences, titleset, title2vectexSet

    def to_official(self, all_preds, features):
        all_entity_pair_h, all_entity_pair_t, all_titles = [], [], []
        for f in features:
            all_entity_pair_h.append(f['entity_pairs_h'])
            all_entity_pair_t.append(f['entity_pairs_t'])
            all_titles += [f['title']] * f['entity_pairs_h'].shape[0]
        all_entity_pair_h = torch.cat(all_entity_pair_h, dim=-1).tolist()
        all_entity_pair_t = torch.cat(all_entity_pair_t, dim=-1).tolist()

        ans = []
        for entity_pair_i in range(all_preds.shape[0]):
            if all_entity_pair_h[entity_pair_i] == all_entity_pair_t[entity_pair_i]:
                continue
            rel_ids = np.nonzero(all_preds[entity_pair_i])[0].tolist()
            for rel_id in rel_ids:
                if rel_id != 0:
                    ans.append({
                        'title': all_titles[entity_pair_i],
                        'h_idx': all_entity_pair_h[entity_pair_i],
                        't_idx': all_entity_pair_t[entity_pair_i],
                        'r': self.id2rel[rel_id]
                    })
        return ans

    def official_evaluate(self, ans, partition='dev'):
        gold, tot_evidences, titleset, title2vectexSet = self.get_gold(partition)
        ans.sort(key=lambda x: (x['title'], x['h_idx'], x['t_idx'], x['r']))
        if ans:
            submission_answer = [ans[0]]
            for i in range(1, len(ans)):
                x = ans[i]
                y = ans[i - 1]
                if (x['title'], x['h_idx'], x['t_idx'], x['r']) != (y['title'], y['h_idx'], y['t_idx'], y['r']):
                    submission_answer.append(ans[i])
        else:
            submission_answer = []

        correct_re = 0
        for x in submission_answer:
            title = x['title']
            h_idx = x['h_idx']
            t_idx = x['t_idx']
            r = x['r']
            if title not in title2vectexSet:
                continue

            if (title, r, h_idx, t_idx) in gold:
                correct_re += 1

        re_p = 1.0 * correct_re / len(submission_answer) * 100 if submission_answer else 0
        re_r = 1.0 * correct_re / len(gold) * 100
        if re_p + re_r == 0:
            re_f1 = 0
        else:
            re_f1 = 2.0 * re_p * re_r / (re_p + re_r)

        return (re_p, re_r, re_f1), None, None, None
