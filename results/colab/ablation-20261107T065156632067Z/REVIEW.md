# Ablation review - November 7, 2026

Archive integrity passed. Ten prediction/metric stages recomputed exactly; annotation IDs,
expert labels/evidence, original split manifests and data hashes match local validation.
The shared untouched-base predictions are byte-identical. Controlled-comparison checks
passed for source/model/runtime, training settings and example/token/update budgets.
Both runs used A100 40 GB, passed adapter-reload validation and recorded no test-set use.

| Recipe | Selected epoch | Correct / 206 | Accuracy | Macro-F1 | Evidence micro-F1 |
|---|---:|---:|---:|---:|---:|
| Original v1, historical T4 | 3 | 145 | 0.7039 | 0.4119 | 0.5323 |
| A: v2, unweighted | 3 | 152 | 0.7379 | 0.4478 | 0.6298 |
| B: v2, weighted | 2 | 153 | 0.7427 | 0.4420 | 0.4751 |

A is the strongest current candidate under the declared macro-F1-first selection rule.
B gains one correct classification, but loses 0.0058 macro-F1 and 0.1546 evidence
micro-F1 versus A. Evidence precision falls from 0.6167 to 0.3482 while recall rises
from 0.6435 to 0.7478: B cites 247 sentences versus A's 120, with 161 false-positive
citations versus 46. The tested weighting recipe did not achieve its intended rare-class
benefit; both selected adapters have zero recall for `excluded` and `not included`.
B emits one `not included` prediction, but it is a false positive. Do not describe B as
better based only on accuracy, or claim all forms of class balancing fail.

Prompt v2's untouched-base responses have zero invalid JSON (original v1 had 163).
Its base accuracy is still only 18.45% and macro-F1 20.59%, indicating that valid JSON
alone is not sufficient reasoning. It predicts `excluded` for all 140 exclusion criteria;
fine-tuning changes this bias, but still misses the rare positive exclusion cases.
A improves over the historical v1 result, including evidence F1, but source/hardware
changed, so this is not a strict same-hardware estimate of the prompt-only causal effect.
The A/B comparison itself does control those factors.

The selected A adapter was imported into
`checkpoints/candidates/prompt-v2-unweighted-epoch-03`; its base revision, prompt version,
file hashes and validation metrics are recorded in `configs/model_candidate.json`.
It remains a development candidate, not a final tested or clinically validated model.
Preserve all runs and document remaining minority-label failures as limitations/error cases.

Next implementation priority: use A to implement explicit cached/uncached greedy decoding,
tiled attention and speculative-decoding correctness/benchmarks, then integrate the app.
Continue to reserve final test evaluation until decisions are frozen. Additional validation
experiments, if any, should be limited and justified; these eight patients are already used
for checkpoint and recipe selection. Do not claim statistical significance from this run.
