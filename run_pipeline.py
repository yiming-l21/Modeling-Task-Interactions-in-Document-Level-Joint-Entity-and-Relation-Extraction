from run_re import DocReRunner
from run_coref import CorefRunner
import sys
import logging
import util
from run_base import BaseRunner
from functools import cached_property
import torch
from copy import deepcopy
from util_io import read_pickle, write_pickle
from os.path import join, exists
import os

logger = logging.getLogger(__name__)


class PipelineRunner(BaseRunner):

    def __init__(self, config_name, gpu_id):
        super(PipelineRunner, self).__init__(config_name, gpu_id)

    @cached_property
    def coref_runner(self):
        return CorefRunner(self.config['coref_config_name'], gpu_id)

    @cached_property
    def re_runner(self):
        return DocReRunner(self.config['re_config_name'], gpu_id)

    @cached_property
    def coref_model(self):
        return self.coref_runner.initialize_model(self.config['coref_model_suffix'])

    @cached_property
    def re_model(self):
        return self.re_runner.initialize_model(self.config['re_model_suffix'])

    @cached_property
    def data(self):
        return self.re_runner.data

    @classmethod
    def build_index_mapping(cls, features, coref_predictions):
        predidx2goldidx = {}

        for doc_i, (_, predicted_clusters_subtok, _, _) in enumerate(coref_predictions):
            title = features[doc_i]['title']
            gold2idx = {util.tuplize_cluster(entity): entity_i
                        for entity_i, entity in enumerate(features[doc_i]['entities'])}
            num_nongold = 0
            for entity_i, pred_entity in enumerate(predicted_clusters_subtok):
                mapped = gold2idx.get(util.tuplize_cluster(pred_entity), None)
                if mapped is None:
                    mapped = len(gold2idx) + num_nongold
                    num_nongold += 1
                predidx2goldidx[(title, entity_i)] = mapped
        return predidx2goldidx

    @classmethod
    def convert_features(cls, features, coref_predictions):
        converted_features = []
        for feature, (_, predicted_clusters_subtok, _, predicted_types) in zip(features, coref_predictions):
            feature = deepcopy(feature)
            feature['entities'] = predicted_clusters_subtok
            feature['entity_types'] = predicted_types

            feature['entity_pairs_h'] = []
            feature['entity_pairs_t'] = []
            for h in range(len(predicted_clusters_subtok)):
                for t in range(len(predicted_clusters_subtok)):
                    if h != t:
                        feature['entity_pairs_h'].append(h)
                        feature['entity_pairs_t'].append(t)
            feature['entity_pairs_h'] = torch.tensor(feature['entity_pairs_h'], dtype=torch.long)
            feature['entity_pairs_t'] = torch.tensor(feature['entity_pairs_t'], dtype=torch.long)
            feature['rel_labels'] = None

            converted_features.append(feature)
        return converted_features

    @classmethod
    def convert_official_re_predictions(self, re_pred_official, predidx2goldidx):
        for inst in re_pred_official:
            inst['h_idx'] = predidx2goldidx[(inst['title'], inst['h_idx'])]
            inst['t_idx'] = predidx2goldidx[(inst['title'], inst['t_idx'])]

    def evaluate(self, dataset_name, partition, do_eval=True):
        eval_docs, eval_features = self.data.get_data(dataset_name, partition)
        return self.evaluate_pipeline(self.re_runner, self.coref_model, self.re_model,
                                      dataset_name, partition, eval_docs, eval_features, do_eval=do_eval)

    @classmethod
    def evaluate_pipeline(cls, re_runner, coref_model, re_model, dataset_name, partition, docs, features, do_eval=True):
        coref_return = re_runner.evaluate_coref(coref_model, dataset_name, partition, docs, features, do_eval=do_eval)
        if do_eval:
            coref_eval_score, metrics, coref_predictions = coref_return
        else:
            coref_predictions = coref_return

        converted_features = cls.convert_features(features, coref_predictions)
        re_pred_official, _ = re_runner.evaluate_re(re_model, dataset_name, partition, docs, converted_features, do_eval=False)
        predidx2goldidx = cls.build_index_mapping(features, coref_predictions)
        cls.convert_official_re_predictions(re_pred_official, predidx2goldidx)
        results = coref_predictions, re_pred_official
        print(do_eval)
        if not do_eval:
            return results

        re_eval_score, re_metrics = re_runner.get_re_metrics(dataset_name, partition, re_pred_official)
        cls.log_metrics(re_metrics)
        metrics.update(re_metrics)

        return re_eval_score, metrics, results

    def get_pipeline_results_path(self, dataset_name, partition):
        save_dir = join(self.config['log_dir'], 'results')
        os.makedirs(save_dir, exist_ok=True)
        return join(save_dir, f'results_{dataset_name}_{partition}.bin')

    def save_pipeline_results(self, dataset_name, partition, results):
        save_path = self.get_pipeline_results_path(dataset_name, partition)
        write_pickle(results, save_path)
        logger.info(f'Saved results to {save_path}')

    def load_pipeline_results(self, dataset_name, partition):
        save_path = self.get_pipeline_results_path(dataset_name, partition)
        if exists(save_path):
            results = read_pickle(save_path)
            logger.info(f'Loaded results from {save_path}')
            return results
        else:
            return None

    def initialize_model(self, init_suffix=None):
        pass


if __name__ == '__main__':
    config_name, gpu_id = sys.argv[1], int(sys.argv[2])
    partition_config = 'test'#'dev' if len(sys.argv) == 3 else 'test'
    print(partition_config)
    runner = PipelineRunner(config_name, gpu_id)
    returned = runner.evaluate(runner.dataset_name, runner.partition[partition_config], do_eval=True)
