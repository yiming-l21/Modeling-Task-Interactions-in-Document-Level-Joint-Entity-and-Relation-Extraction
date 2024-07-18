from run_re import DocReRunner, main
from run_coref import CorefRunner
from run_pipeline import PipelineRunner
from model_joint import JointModel
import logging

logger = logging.getLogger(__name__)


class JointRunner(DocReRunner):
    def __init__(self, config_name, gpu_id, **kwargs):
        super(JointRunner, self).__init__(config_name, gpu_id, **kwargs)

        self.coref_runner = CorefRunner(self.config['coref_config_name'], gpu_id, save_log=False)
        self.re_runner = DocReRunner(self.config['re_config_name'], gpu_id, save_log=False)

        self.coref_config = self.coref_runner.config
        self.re_config = self.re_runner.config

    def initialize_model(self, init_suffix=None):
        ner2id, rel2id = self.data.get_label_types(self.dataset_name)

        model = JointModel(self.config, self.coref_config, self.re_config, num_entity_types=len(ner2id))
        if init_suffix:
            self.load_model_checkpoint(model, init_suffix)
        return model

    def evaluate(self, model, dataset_name, partition, docs, features, tb_writer=None, step=0, do_eval=True):
        coref_model, re_model = model.coref_model, model.re_model
        coref_model.seq_encoder = re_model.seq_encoder = model.seq_encoder
        pipeline_return = PipelineRunner.evaluate_pipeline(self.re_runner, coref_model, re_model,
                                                           dataset_name, partition, docs, features, do_eval=do_eval)
        coref_model.seq_encoder = re_model.seq_encoder = None
        return pipeline_return


if __name__ == '__main__':
    main(JointRunner, partition_config='dev')
