import torch

import util
from run_re import DocReRunner, main
from run_pipeline import PipelineRunner
from model_new import NewJointModel
import numpy as np
import math
import logging

logger = logging.getLogger(__name__)


class NewRunner(DocReRunner):
    def __init__(self, config_name, gpu_id, **kwargs):
        super(NewRunner, self).__init__(config_name, gpu_id, **kwargs)

        ner2id, rel2id = self.data.get_label_types(self.dataset_name)
        self.id2ner = {i: t for t, i in ner2id.items()}
        self.id2rel = {i: t for t, i in rel2id.items()}

    def initialize_model(self, init_suffix=None):
        model = NewJointModel(self.config, num_entity_types=len(self.id2ner))
        if init_suffix:
            self.load_model_checkpoint(model, init_suffix)
        return model

    def evaluate(self, model, dataset_name, partition, docs, features, tb_writer=None, step=0, do_eval=True, re_agg_union=False):
        coref_return = self.evaluate_coref(model, dataset_name, partition, docs, features, tb_writer, step, do_eval)
        if do_eval:
            coref_eval_score, metrics, (coref_predictions, re_predictions) = coref_return
        else:
            coref_predictions, re_predictions = coref_return
        re_pred_official = []
        for doc_i, ((_, _, cluster_idx, _), re_pair_logits) in enumerate(zip(coref_predictions, re_predictions)):
            doc_re_pred = []
            num_spans = math.isqrt(re_pair_logits.shape[0])
            re_pair_logits = re_pair_logits.view()
            re_pair_logits.shape = (num_spans, num_spans, -1)

            for h in range(len(cluster_idx)):
                h_m = np.array(cluster_idx[h])
                for t in range(len(cluster_idx)):
                    if h == t: 
                        continue
                    t_m = np.array(cluster_idx[t])
                    pair_indices_h = h_m.repeat(len(cluster_idx[t]))
                    pair_indices_t = np.tile(t_m, len(cluster_idx[h]))
                    pair_logits = re_pair_logits[pair_indices_h, pair_indices_t]
                    if re_agg_union:
                        pair_logits -= pair_logits[:, 0:1]
                        pair_logits = (pair_logits > 1e-6).any(axis=0).astype(float) 

                    else:
                        pair_logits = np.mean(pair_logits, axis=0, keepdims=False) 
                    doc_re_pred.append(pair_logits)
            if not doc_re_pred:
                continue
            doc_re_pred = np.stack(doc_re_pred) 
            doc_re_pred = model.get_re_labels(torch.from_numpy(doc_re_pred)).numpy()

            pair_i = 0
            for h in range(len(cluster_idx)):
                for t in range(len(cluster_idx)):
                    if h == t:
                        continue
                    rel_ids = np.nonzero(doc_re_pred[pair_i])[0].tolist()
                    for rel_id in rel_ids:
                        if rel_id != 0: 
                            re_pred_official.append({
                                'title': features[doc_i]['title'],
                                'h_idx': h,
                                't_idx': t,
                                'r': self.id2rel[rel_id]
                            })
                    pair_i += 1
            assert pair_i == doc_re_pred.shape[0]

        predidx2goldidx = PipelineRunner.build_index_mapping(features, coref_predictions)
        PipelineRunner.convert_official_re_predictions(re_pred_official, predidx2goldidx)
        results = coref_predictions, re_pred_official
        if not do_eval:
            return results

        re_eval_score, re_metrics = self.get_re_metrics(dataset_name, partition, re_pred_official)
        self.log_metrics(re_metrics)
        metrics.update(re_metrics)
        print(metrics)

        return re_eval_score, metrics, results


if __name__ == '__main__':
    main(NewRunner, partition_config='dev', do_eval=True)
