from transformers import AutoTokenizer
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.roberta.modeling_roberta import RobertaModel
import torch
import operator
import random
import logging
from collections import Counter
import numpy as np

logger = logging.getLogger(__name__)


def flatten(l):
    return [e for ll in l for e in ll]


def tuplize_cluster(c):
    return tuple(tuple(m) for m in c)


def rindex(l, v):
    return len(l) - operator.indexOf(reversed(l), v) - 1


def get_most_common(l):
    data = Counter(l)
    return data.most_common(1)[0][0]


def argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)


def flatten_matrix_pos(x1, x2, matrix_size):
    return x1 * matrix_size + x2


def batch_select_2d(tensor, idx):
    assert idx.dim() == 2
    assert tensor.size()[0] == idx.size()[0]
    dim0, dim1 = tensor.size()[:2]

    tensor = tensor.view(dim0 * dim1, *tensor.size()[2:])
    idx_offset = torch.unsqueeze(torch.arange(0, dim0, device=tensor.device) * dim1, 1)
    new_idx = idx + idx_offset
    selected = tensor[new_idx]

    return selected


def random_select(tensor, num_selection):
    if tensor.size()[0] > num_selection:
        rand_idx = torch.randperm(tensor.size()[0])[:num_selection]
        return tensor[rand_idx]
    else:
        return tensor


def bucket_distance(offsets):
    logspace_distance = torch.log2(offsets.to(torch.float)).to(
        torch.long) + 3 
    identity_mask = (offsets <= 4).to(torch.long)
    combined_distance = identity_mask * offsets + (1 - identity_mask) * logspace_distance
    combined_distance = torch.clamp(combined_distance, 0, 9)
    return combined_distance


def set_seed(seed, set_gpu=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if set_gpu and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(seed)


def get_transformer_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer'], use_fast=False) 

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token
    return tokenizer


def get_transformer_encoder(config):
    if config['model_type'] == 'bert':
        return BertModel.from_pretrained(config['pretrained'])
    elif config['model_type'] == 'roberta':
        return RobertaModel.from_pretrained(config['pretrained'])
    else:
        raise ValueError(config['model_type'])
