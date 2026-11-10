# Assignment requirements

Source: `GenAI_Assignment#2.pdf`, seven pages. Deadline: **November 10, 2026**.
The exact cutoff time is not stated. There is a two-hour grace period after the official
cutoff; work after that is not evaluated. Missing the demo receives no marks.
The heading says 100 marks, but the categories and final total sum to **135**; clarify with
the instructor. This checklist follows every category rather than assuming normalization.

## What you are building

A patient note plus one inclusion/exclusion criterion goes into a locally executable,
open-weight decoder Transformer. It predicts the original expert category, cites original
patient sentence IDs, shows exact supporting text, and provides a grounded explanation.
The final app processes several criteria and exports a readable report. Any patient-trial
summary is an explicit rule-based aggregation, not a separately labeled prediction task.

## Tasks and evidence

| Work | Required evidence |
|---|---|
| Research / architecture (15) | At least 5 credible sources; at least 2 decoder candidates; parameters, architecture, tokenizer, context, licensing, memory and hardware justification; rejected choices and failures. |
| Data / adaptation (20) | Only the mandatory TrialGPT criterion dataset; fixed patient split; no GPT answer leakage; executable fine-tuning and checkpoint; curves/logs; identical before/after evaluation. |
| Eligibility / evidence (20) | Original labels including insufficient information and not applicable; accuracy, macro-F1, per-class P/R/F1 and confusion matrices overall and separately for inclusion/exclusion; micro/macro evidence P/R/F1 with empty-set policy; at least 8 distinct error analyses. |
| KV caching (15) | Explicit token-by-token cached and uncached forward passes; greedy token equality across at least 3 input/output length settings; prefill/decode timing, tokens/s and peak memory; warmup, synchronization, repeated runs. Explain position, mask, cache growth and numerical differences. |
| Tiled attention (15) | Own stable scaled-dot-product reference and tiled online-softmax implementation; causal masks, multiple tensor shapes, max/mean numerical error; honest timing/memory comparison. A fused library call alone is insufficient. |
| Speculation (15) | Smaller compatible draft model; own greedy draft/verify/accept/reject/cache synchronization flow; equality with target greedy; acceptance statistics, latency, throughput, memory, speedup or slowdown. |
| Browser app (20) | Patient input, entered/selected criteria, real model predictions, exact evidence and missing information, preliminary aggregate, inference comparison, readable report export, invalid-input recovery and educational notice. Accessible through a browser without notebook/IDE use. |
| Report / defense (15) | IEEE LaTeX paper, complete methods and interpreted results; GitHub source and setup, weights link, manifest, logs, 5-7 minute demo video and AI-use appendix; explain/change code during live defense. |

## Non-negotiable experimental rules

- Sort unique patient IDs, shuffle with `numpy.random.default_rng(42)`, then split 37/8/8.
- Every row for a patient stays in one split. Save patient IDs and annotation counts.
- Source dataset's `training` boolean is NOT the assignment split.
- Inputs: patient note, criterion type, criterion text only. Expert labels/evidence are targets.
- Never use GPT-generated labels, sentences, explanations, or explanation-correctness fields as features or supervision.
- Tune on validation only; use final test only after choices are frozen, for the same base and adapted model comparison.
- Document truncation and removed evidence; prefer rejecting/logging overlength input initially.
- Absence of evidence is not proof of either inclusion or exclusion.
- Neither notebook-only work nor simply running the original TrialGPT repository meets the assignment.
- Free compute and open weights must suffice. LoRA is optional; CUDA-only attention kernels are not compulsory.
- Correct measured speculative decoding can earn implementation credit even if it is slower.

## Submission checklist

- [ ] IEEE report PDF and LaTeX source-project link.
- [ ] GitHub repository with all reproducible scripts, dependencies and app source.
- [ ] Trained adapter/weights download link and exact base-model revision.
- [ ] Fixed split manifest, preprocessing provenance and evaluation/benchmark logs.
- [ ] Reproducible training/evaluation/startup instructions.
- [ ] 5-7 minute recorded app demonstration and video link.
- [ ] AI/library-use acknowledgment and live defense readiness.
