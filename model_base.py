import util
import torch
import torch.nn as nn
import torch.nn.init as init
import logging
from collections import Iterable

logger = logging.getLogger(__name__)


class BaseModel(nn.Module):
    def __init__(self, config, with_encoder=True, seq_config=None):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config['dropout_rate'])
        self.seq_encoder = util.get_transformer_encoder(config) if with_encoder else None
        self.seq_config = self.seq_encoder.config if with_encoder else seq_config
        self.seq_hidden_size = self.seq_config.hidden_size

    @classmethod
    def make_scalar(cls, init_val):
        param = nn.Parameter(torch.tensor(init_val, dtype=torch.float))
        return param

    @classmethod
    def make_embedding(cls, dict_size, hidden_size, std=None):
        emb = nn.Embedding(dict_size, hidden_size)
        if std:
            init.normal_(emb.weight, std=std)
        return emb

    @classmethod
    def make_linear(cls, in_features, out_features, bias=True, std=None):
        linear = nn.Linear(in_features, out_features, bias)
        if std:
            init.normal_(linear.weight, std=std)
        if bias:
            init.zeros_(linear.bias)
        return linear

    def make_ffnn(self, feat_size, hidden_size, output_size):
        if not hidden_size or hidden_size == [0]:
            return self.make_linear(feat_size, output_size)

        if not isinstance(hidden_size, Iterable):
            hidden_size = [hidden_size]
        ffnn = [self.make_linear(feat_size, hidden_size[0]), nn.GELU(), self.dropout]
        for i in range(1, len(hidden_size)):
            ffnn += [self.make_linear(hidden_size[i - 1], hidden_size[i]), nn.GELU(), self.dropout]
        ffnn.append(self.make_linear(hidden_size[-1], output_size))
        return nn.Sequential(*ffnn)

    def get_params(self, named=False):
        seq_encoder_param, task_param = [], []
        for name, param in self.named_parameters():
            if 'seq_encoder' in name:
                seq_encoder_param.append((name, param) if named else param)
            else:
                task_param.append((name, param) if named else param)
        return seq_encoder_param, task_param

    def freeze_encoder(self):
        seq_encoder_param, _ = self.get_params()
        for p in seq_encoder_param:
            p.requires_grad = False

    def encode(self, doc_len, input_ids=None, attention_mask=None, token_type_ids=None, is_max_context=None):
        conf, (num_seq, seq_len) = self.config, input_ids.size()[:2]
        inputs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids,
                  'output_attentions': False, 'output_hidden_states': False, 'return_dict': True}
        seq_hidden = self.seq_encoder(**inputs)['last_hidden_state']
        tokens = self.dropout(seq_hidden[is_max_context.to(torch.bool)])
        return tokens
