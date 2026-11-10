# Validation comparison

Selected fine-tuned epoch: 3. Test set was not used.

| Metric | Untouched base | Selected adapter |
|---|---:|---:|
| Accuracy | 0.1845 | 0.7379 |
| Classification macro-F1 | 0.2059 | 0.4478 |
| Evidence micro-F1 | 0.1100 | 0.6298 |
| Evidence macro-F1 | 0.1074 | 0.7936 |
| Invalid structured outputs | 0 | 0 |

Macro-F1 uses the fixed six-category overall label set, including zero-support classes. Per-type/per-class metrics and confusion matrices are in comparison.json. These are development results, not final held-out test results.
