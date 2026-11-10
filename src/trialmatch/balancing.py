"""Capped inverse-square-root example weights computed on accepted training rows only."""
from collections import Counter
import math


def class_weights(examples, method='none', cap=3.0):
    if method not in {'none','inverse_sqrt_capped'}:
        raise ValueError('Unknown class weighting method')
    if not examples or not math.isfinite(cap) or cap < 1:
        raise ValueError('Training rows required and cap must be finite and at least 1')
    keys = [(e['input']['criterion_type'],e['target']['label']) for e in examples]
    counts = Counter(keys)
    maximum = {kind:max(count for (t,_),count in counts.items() if t==kind)
               for kind,_ in counts}
    raw = {key:(min(cap,math.sqrt(maximum[key[0]]/count)) if method!='none' else 1.0)
           for key,count in counts.items()}
    average = sum(raw[key] for key in keys)/len(keys)
    weights = [raw[key]/average for key in keys]
    summary = {
        'method':method,'cap_on_relative_weight':cap,'source':'accepted training examples only',
        'normalization':'mean example weight = 1; batch loss divided by weighted target-token count',
        'sampling':'Each original accepted example occurs once per epoch; no oversampling or dropped rows.',
        'strata':{f'{kind}::{label}':{'count':count,'raw_weight':raw[(kind,label)],
                  'normalized_weight':raw[(kind,label)]/average}
                  for (kind,label),count in sorted(counts.items())},
    }
    return weights,summary
