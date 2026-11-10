# Validation comparison

Selected fine-tuned epoch: 3. Test set was not used.

| Metric | Untouched base | Selected adapter |
|---|---:|---:|
| Accuracy | 0.0049 | 0.7039 |
| Classification macro-F1 | 0.0098 | 0.4119 |
| Evidence micro-F1 | 0.0585 | 0.5323 |
| Evidence macro-F1 | 0.0365 | 0.7396 |
| Invalid structured outputs | 163 | 0 |

Macro-F1 uses the fixed six-category overall label set, including zero-support classes. Per-type/per-class metrics and confusion matrices are in comparison.json. These are development results, not final held-out test results.
