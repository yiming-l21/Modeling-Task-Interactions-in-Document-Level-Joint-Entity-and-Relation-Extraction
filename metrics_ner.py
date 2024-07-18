from collections import defaultdict


def get_prf(num_tps, num_preds, num_golds):
    precision = (num_tps * 1.0 / num_preds * 100) if num_preds != 0 else 0
    recall = (num_tps * 1.0 / num_golds * 100) if num_golds != 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if precision != 0 or recall != 0 else 0
    return precision, recall, f1


class NerEvaluator:
    def __init__(self, ner2id):
        self.ner2id = ner2id

    def _collect_stats(self, predicted_spans, predicted_types, gold_spans, gold_types):
        all_preds, all_golds = set(), set()
        num_preds, num_golds = defaultdict(int), defaultdict(int)

        for span, span_type in zip(predicted_spans, predicted_types):
            all_preds.add((span, span_type))
            num_preds[span_type] += 1

        for span, span_type in zip(gold_spans, gold_types):
            all_golds.add((span, span_type))
            num_golds[span_type] += 1

        all_tps = all_preds & all_golds
        num_tps = defaultdict(int)
        for span, span_type in all_tps:
            num_tps[span_type] += 1

        return num_tps, num_preds, num_golds

    def evaluate(self, predicted_spans, predicted_types, gold_spans, gold_types):
        num_tps, num_preds, num_golds = self._collect_stats(predicted_spans, predicted_types, gold_spans, gold_types)
        type2prf = {ner_type: get_prf(num_tps[ner_id], num_preds[ner_id], num_golds[ner_id])
                    for ner_type, ner_id in self.ner2id.items()}
        total_prf = get_prf(sum(num_tps.values()), sum(num_preds.values()), sum(num_golds.values()))
        return total_prf, type2prf


class MeEvaluator:
    def __init__(self):
        pass

    def _collect_stats(self, predicted_spans, gold_spans):
        all_preds, all_golds = set(predicted_spans), set(gold_spans)
        num_preds, num_golds = len(all_preds), len(all_golds)

        all_tps = all_preds & all_golds
        num_tps = len(all_tps)

        return num_tps, num_preds, num_golds

    def evaluate(self, predicted_spans, gold_spans):
        num_tps, num_preds, num_golds = self._collect_stats(predicted_spans, gold_spans)
        total_prf = get_prf(num_tps, num_preds, num_golds)
        return total_prf
