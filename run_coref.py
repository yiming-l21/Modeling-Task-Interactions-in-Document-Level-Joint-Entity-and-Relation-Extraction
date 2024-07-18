import logging
from model_coref import CorefModel
from run_re import DocReRunner, main

logger = logging.getLogger(__name__)


class CorefRunner(DocReRunner):
    def __init__(self, config_name, gpu_id=None, **kwargs):
        super(CorefRunner, self).__init__(config_name, gpu_id, **kwargs)

    def initialize_model(self, init_suffix=None):
        ner2id, rel2id = self.data.get_label_types(self.dataset_name)

        model = CorefModel(self.config, num_entity_types=len(ner2id))
        if init_suffix!=None:
            self.load_model_checkpoint(model, init_suffix)
        return model

    def evaluate(self, model, dataset_name, partition, docs, features, tb_writer=None, step=0, do_eval=True):
        return self.evaluate_coref(model, dataset_name, partition, docs, features, tb_writer, step, do_eval)


if __name__ == '__main__':
    main(CorefRunner)
