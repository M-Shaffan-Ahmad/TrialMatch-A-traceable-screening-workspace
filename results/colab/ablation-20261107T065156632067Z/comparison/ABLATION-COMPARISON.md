# Prompt v2 and class-balancing ablation

Validation only. Same model, prompt, dataset, runtime, example exposure and update budget.

| Metric | A: prompt v2 | B: v2 + weighting | B minus A |
|---|---:|---:|---:|
| Accuracy | 0.7379 | 0.7427 | 0.0049 |
| Classification macro-F1 | 0.4478 | 0.4420 | -0.0058 |
| Evidence micro-F1 | 0.6298 | 0.4751 | -0.1546 |
| Evidence macro-F1 | 0.7936 | 0.7206 | -0.0730 |
| Invalid outputs | 0 | 0 | 0 |

| Label | Support | A recall | B recall | A F1 | B F1 |
|---|---:|---:|---:|---:|---:|
| excluded | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| included | 35 | 0.7143 | 0.7714 | 0.7812 | 0.7826 |
| not applicable | 12 | 0.4167 | 0.4167 | 0.4762 | 0.4348 |
| not enough information | 53 | 0.5849 | 0.5283 | 0.5905 | 0.5773 |
| not excluded | 101 | 0.9010 | 0.9208 | 0.8387 | 0.8571 |
| not included | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

Each run selects its own epoch using the same validation rule. This compares checkpoint-selected training recipes; validation selection can be optimistic. No test results are claimed.

Training loss values optimize different objectives and should not be compared as identical quantities; each log also records unweighted token loss.

Historical original prompt-v1 run: accuracy 0.7039, macro-F1 0.4119, hardware Tesla T4. New A/B hardware: NVIDIA A100-SXM4-40GB. The historical comparison is contextual; hardware/source differences must be disclosed.
