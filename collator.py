import util
from dataclasses import dataclass
import torch
from torch import Tensor


@dataclass
class FeatureCollator:
    tokenizer: None
    device: torch.device('cpu')

    def __post_init__(self):
        assert self.tokenizer.padding_side == 'right'
        self.ignored_keys = {'title'}

    def __call__(self, features):
        collated = {
            'input_ids': util.flatten([f['input_ids'] for f in features]),
            'attention_mask': util.flatten([f['attention_mask'] for f in features]),
            'token_type_ids': util.flatten([f['token_type_ids'] for f in features])
        }
        collated = self.tokenizer.pad(collated, padding=True, pad_to_multiple_of=8)
        num_seg, seg_len = len(collated['input_ids']), len(collated['input_ids'][0])
        collated['is_max_context'] = util.flatten([f['is_max_context'] for f in features])
        for seg_i in range(num_seg):
            collated['is_max_context'][seg_i] = collated['is_max_context'][seg_i] +\
                                                [0] * (seg_len - len(collated['is_max_context'][seg_i]))
        collated = {
            'input_ids': torch.tensor(collated['input_ids'], dtype=torch.long, device=self.device),
            'attention_mask': torch.tensor(collated['attention_mask'], dtype=torch.long, device=self.device),
            'token_type_ids': torch.tensor(collated['token_type_ids'], dtype=torch.long, device=self.device),
            'is_max_context': torch.tensor(collated['is_max_context'], dtype=torch.long, device=self.device)
        }
        others = {feat_attr: [(f[feat_attr].to(self.device) if isinstance(f[feat_attr], Tensor) else f[feat_attr])
                              for f in features]
                  for feat_attr in features[0].keys() if feat_attr not in collated and feat_attr not in self.ignored_keys}
        collated.update(others)

        return collated
