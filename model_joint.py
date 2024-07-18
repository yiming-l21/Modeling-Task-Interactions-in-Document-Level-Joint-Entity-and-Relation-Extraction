from model_base import BaseModel
import logging
from model_re import DocReModel
from model_coref import CorefModel

logger = logging.getLogger(__name__)


class JointModel(BaseModel):
    def __init__(self, config, coref_config, re_config, num_entity_types=0):
        super(JointModel, self).__init__(config, with_encoder=True)
        self.num_entity_types = num_entity_types

        self.coref_model = CorefModel(coref_config, num_entity_types, with_encoder=False, seq_config=self.seq_config)
        self.re_model = DocReModel(re_config, num_entity_types, with_encoder=False, seq_config=self.seq_config)

        self.forward_steps = 0
        self.debug = False

    def forward(self, doc_len, tok_start_or_end=None, sent_map=None,
                input_ids=None, attention_mask=None, token_type_ids=None, is_max_context=None,
                entities=None, entity_types=None, entity_pairs_h=None, entity_pairs_t=None, rel_labels=None,
                mention_starts=None, mention_ends=None, mention_type=None, mention_cluster_id=None, **kwargs):
        """ Only for training; use gold for both COREF and RE. """
        conf = self.config
        tokens = self.encode(doc_len, input_ids, attention_mask, token_type_ids, is_max_context)
        coref_return = self.coref_model(doc_len, tokens, mention_starts=mention_starts, mention_ends=mention_ends,
                                        mention_type=mention_type, mention_cluster_id=mention_cluster_id,
                                        tok_start_or_end=tok_start_or_end, sent_map=sent_map)
        loss_coref, outputs_coref = coref_return
        loss_re, outputs_re = self.re_model(doc_len, tokens, entities=entities, entity_types=entity_types,
                                            entity_pairs_h=entity_pairs_h, entity_pairs_t=entity_pairs_t,
                                            rel_labels=rel_labels)
        loss_coref *= conf['coref_loss_coef']
        loss = loss_coref + loss_re
        outputs = (outputs_coref, outputs_re)
        return loss, outputs
