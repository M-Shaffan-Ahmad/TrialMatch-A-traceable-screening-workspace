"""Strict structured-output validation and fixed-label criterion/evidence metrics."""
import json
import re

from trialmatch.data import LABELS, sentence_map

INVALID = '__invalid__'
ALL_LABELS = sorted(set().union(*LABELS.values()))


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def parse_prediction(text, criterion_type, note):
    """Accept plain JSON or one enclosing JSON fence. Never salvage prose or use gold."""
    result = {'valid': False, 'label': INVALID, 'evidence_sentence_ids': [],
              'normalization': 'none', 'error': None}
    try:
        stripped = text.strip()
        match = re.fullmatch(r'```(?:json)?\s*(.*?)\s*```', stripped, flags=re.DOTALL)
        if match:
            stripped = match.group(1)
            result['normalization'] = 'stripped_markdown_fence'
        value = json.loads(stripped, object_pairs_hook=_unique_object)
        if not isinstance(value, dict) or set(value) != {'label', 'evidence_sentence_ids'}:
            raise ValueError('Expected exactly label and evidence_sentence_ids')
        label = value['label']
        if not isinstance(label, str) or label not in LABELS[criterion_type]:
            raise ValueError('Label is not permitted for this criterion type')
        ids = value['evidence_sentence_ids']
        if not isinstance(ids, list) or any(type(i) is not int or i < 0 for i in ids):
            raise ValueError('Evidence IDs must be nonnegative integers')
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicate evidence IDs')
        if set(ids) - sentence_map(note).keys():
            raise ValueError('Cited evidence ID is absent from the source note')
        result.update(valid=True, label=label, evidence_sentence_ids=sorted(ids))
    except (ValueError, TypeError, KeyError) as exc:
        result['error'] = str(exc)
    return result


def prediction_record(example, raw_output, error=None, generation=None):
    parsed = parse_prediction(raw_output, example['input']['criterion_type'], example['input']['note'])
    if error is not None:
        parsed.update(valid=False, label=INVALID, evidence_sentence_ids=[], error=error)
    sentences = sentence_map(example['input']['note'])
    evidence = [{'sentence_id': sid, 'text': sentences[sid]}
                for sid in parsed['evidence_sentence_ids']]
    if not parsed['valid']:
        explanation = 'Structured prediction failed validation; manual review is required.'
    else:
        explanation = f'Model prediction: {parsed["label"]}. '
        explanation += ('Cited source text: ' + ' '.join(f'[{x["sentence_id"]}] {x["text"]}' for x in evidence)
                        if evidence else 'No evidence sentences were cited; absence of a citation is not proof of eligibility or ineligibility.')
        explanation += ' This is a source-evidence summary, not an independently verified medical rationale.'
    return {
        'annotation_id': example['annotation_id'], 'patient_id': example['patient_id'],
        'trial_id': example['trial_id'], 'criterion_type': example['input']['criterion_type'],
        'criterion_text': example['input']['criterion_text'],
        'gold_label': example['target']['label'],
        'gold_evidence_sentence_ids': example['target']['evidence_sentence_ids'],
        'raw_output': raw_output, **parsed, 'evidence': evidence, 'explanation': explanation,
        'generation': generation or {},
    }


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def classification_metrics(records, labels):
    # Invalid outputs occupy an additional prediction column, not a gold class.
    columns = list(labels) + [INVALID]
    matrix = [[0 for _ in columns] for _ in labels]
    for record in records:
        gold = record['gold_label']
        predicted = record['label'] if record['valid'] else INVALID
        if gold not in labels or predicted not in columns:
            raise ValueError('Unexpected gold or predicted category')
        matrix[labels.index(gold)][columns.index(predicted)] += 1
    per_class = {}
    for i, label in enumerate(labels):
        tp = matrix[i][i]
        fp = sum(row[i] for row in matrix) - tp
        fn = sum(matrix[i]) - tp
        per_class[label] = {'precision': _ratio(tp, tp+fp), 'recall': _ratio(tp, tp+fn),
                            'f1': _ratio(2*tp, 2*tp+fp+fn), 'support': sum(matrix[i])}
    return {
        'n': len(records), 'accuracy': _ratio(sum(matrix[i][i] for i in range(len(labels))), len(records)),
        'macro_f1': _ratio(sum(x['f1'] for x in per_class.values()), len(labels)),
        'per_class': per_class,
        'confusion_matrix': {'gold_rows': list(labels), 'prediction_columns': columns, 'counts': matrix},
        'macro_policy': 'Fixed applicable label set; zero-support class F1 is 0. Invalid outputs count as incorrect.',
    }


def evidence_metrics(records):
    tp = fp = fn = 0
    example_scores = []
    both_empty = invalid = 0
    for record in records:
        gold = set(record['gold_evidence_sentence_ids'])
        if record['valid']:
            predicted = set(record['evidence_sentence_ids'])
        else:
            # Standard set-based micro counts: malformed output supplies no usable
            # citations. Count it separately and force its per-example scores to 0.
            predicted = set()
            invalid += 1
        a, b, c = len(gold & predicted), len(predicted-gold), len(gold-predicted)
        tp += a
        fp += b
        fn += c
        if not record['valid']:
            example_scores.append((0.0, 0.0, 0.0))
        elif not gold and not predicted:
            both_empty += 1
            example_scores.append((1.0, 1.0, 1.0))
        elif not gold or not predicted:
            example_scores.append((0.0, 0.0, 0.0))
        else:
            example_scores.append((_ratio(a,a+b), _ratio(a,a+c), _ratio(2*a,2*a+b+c)))
    no_evidence = tp+fp+fn == 0
    return {
        'n': len(records), 'tp': tp, 'fp': fp, 'fn': fn,
        'micro': {'precision': None if no_evidence else _ratio(tp,tp+fp),
                  'recall': None if no_evidence else _ratio(tp,tp+fn),
                  'f1': None if no_evidence else _ratio(2*tp,2*tp+fp+fn)},
        'macro': {name: _ratio(sum(row[i] for row in example_scores),len(example_scores))
                  for i,name in enumerate(['precision','recall','f1'])},
        'both_empty_examples': both_empty, 'invalid_examples': invalid,
        'micro_no_evidence_condition': no_evidence,
        'empty_policy': 'Per example: both empty=1, exactly one empty=0. All-empty micro=undefined/null.',
        'invalid_policy': 'Invalid output: macro=0; micro uses empty predicted set (all gold citations become false negatives). Invalid rate is reported separately; no fabricated citations are added.',
    }


def evaluate_records(records):
    if not records:
        raise ValueError('Cannot evaluate an empty set')
    ids = [r['annotation_id'] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate annotation IDs in evaluation')
    def score(subset, labels):
        return {'classification': classification_metrics(subset, labels),
                'evidence': evidence_metrics(subset),
                'invalid_outputs': sum(not r['valid'] for r in subset)}
    return {'overall': score(records, ALL_LABELS),
            'by_criterion_type': {kind: score([r for r in records if r['criterion_type']==kind], sorted(labels))
                                  for kind,labels in LABELS.items()}}
