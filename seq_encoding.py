import util
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


def encode_long_sequence(seq_id, tokenizer, subtoks, max_seq_len, overlapping=0, constraints=None):
    assert max_seq_len <= tokenizer.model_max_length
    assert overlapping < max_seq_len // 2
    num_seq_added = tokenizer.model_max_length - tokenizer.max_len_single_sentence

    segments, is_max_context = [], []
    till_idx = 0 
    while till_idx < len(subtoks):
        seg_start = max(0, till_idx - overlapping) 
        seg_end = min(len(subtoks), seg_start + max_seq_len - num_seq_added)
        if constraints:
            for c_i, constraint in enumerate(constraints):
                while seg_end > seg_start and not constraint[seg_end - 1]:
                    seg_end -= 1

                if seg_end <= seg_start and c_i != len(constraints) - 1:
                    logger.info(f'{seq_id}: try to segment by next constraint {c_i + 1}')
                elif seg_end <= seg_start:
                    raise RuntimeError(f'{seq_id}: cannot split segment by neither constraints')
                else:
                    break
        if seg_end == len(subtoks):
            seg_start = max(0, len(subtoks) - max_seq_len + num_seq_added)
        segments.append(subtoks[seg_start: seg_end])
        is_max_context.append([0] * (till_idx - seg_start) + [1] * (seg_end - till_idx))
        till_idx = seg_end

    longest_seg_len = max([len(seg) for seg in segments]) + num_seq_added
    encoded_segments = []
    for seg_i, segment in enumerate(segments):
        encoded = tokenizer.prepare_for_model(tokenizer.convert_tokens_to_ids(segment),
                                              padding='max_length', max_length=longest_seg_len,
                                              return_attention_mask=True, return_token_type_ids=True,
                                              return_special_tokens_mask=True)
        encoded_segments.append(encoded)
        special_tokens_mask = encoded.pop('special_tokens_mask')
        num_left_special = special_tokens_mask.index(0)
        num_right_special = longest_seg_len - util.rindex(special_tokens_mask, 0) - 1
        is_max_context[seg_i] = [0] * num_left_special + is_max_context[seg_i] + [0] * num_right_special
    encoded_segments = {
        'input_ids': [encoded_seg['input_ids'] for encoded_seg in encoded_segments],
        'attention_mask': [encoded_seg['attention_mask'] for encoded_seg in encoded_segments],
        'token_type_ids': [encoded_seg['token_type_ids'] for encoded_seg in encoded_segments],
    }
    assert len(subtoks) == sum(util.flatten(is_max_context))
    return encoded_segments, is_max_context
