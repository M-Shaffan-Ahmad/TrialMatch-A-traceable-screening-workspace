# Validation diagnosis - November 7, 2026

## Verified observations

The original dataset hashes and split manifest match the local project. Every reported
baseline/epoch metric recomputed exactly from the saved predictions; saved gold targets
also match local validation records. The run used 206 validation annotations, no test data,
and selected epoch 3 using the declared macro-F1 rule. Adapter reload verification passed.

Selected adapter: 145/206 correct (70.39%), classification macro-F1 0.411876,
evidence micro-F1 0.532258 and evidence macro-F1 0.739594, with zero invalid outputs.

| Label | Train examples | Validation examples | Selected F1 | Predictions |
|---|---:|---:|---:|---:|
| excluded | 8 | 2 | 0.0000 | 0 |
| not included | 13 | 3 | 0.0000 | 0 |
| not applicable | 28 | 12 | 0.4167 | 12 |
| not enough information | 174 | 53 | 0.5385 | 51 |
| included | 88 | 35 | 0.6774 | 27 |
| not excluded | 292 | 101 | 0.8387 | 116 |

Macro-F1 averages the six class F1 values equally, so the two zero classes materially
lower the score even though they have only five validation examples between them.
Every class has validation support; this is not a zero-support-label averaging artifact.
The training set is strongly imbalanced, and the model never emits the two rare outcomes.
Epoch 1 predicted `not excluded` for all 140 exclusion criteria. Epoch 3 improved, but
still mapped 17/26 exclusion `not enough information` cases to `not excluded`.

## Concrete failures

- Annotation 41, exclusion criterion `Respiratory distress`: patient sentence 4 explicitly
  states respiratory distress. The model cites sentence 4 correctly but predicts
  `not excluded` instead of `excluded`. Evidence retrieval alone did not solve label polarity.
- Annotation 31, inclusion criterion fever for more than one week: sentence 0 says two days.
  The model cites that sentence but predicts insufficient information rather than the
  expert `not included` label. This is a duration/contradiction and category error.

## Formatting and the original baseline

The untouched base got only 1/206 correct under the frozen strict-output policy. 163/206
responses failed JSON parsing because label strings were not quoted. Those failures were
not token-limit cutoffs. The tuned model fixed formatting (0 invalid responses), so part
of the apparent gain is output-schema adaptation, not pure clinical reasoning.
A post-hoc diagnostic that only quotes bare allowed labels is recorded separately in
`diagnosis.json`. It does not replace or modify the official baseline, parser or main scores.
Even repaired label syntax does not establish that the base model reasons correctly.

## Learning curve and interpretation

Token-weighted training loss decreased approximately 0.1947 -> 0.0802 -> 0.0516.
Validation macro-F1 increased 0.3514 -> 0.3524 -> 0.4119. Validation accuracy was
0.7184 -> 0.7184 -> 0.7039, so epoch 3 was selected for broader class performance,
not highest accuracy. Falling teacher-forced training loss does not measure held-out
classification/evidence quality and is dominated by repeated JSON/schema tokens as
well as target labels and citations. These curves alone do not prove overfitting.
There are only 37 distinct training patients and eight validation patients, so the
criterion rows are correlated and rare-class estimates have very small support.

Evidence micro-F1 uses actual citation overlaps. Many valid both-empty citation sets
raise macro evidence scores (101/206 both empty), so the higher macro evidence score
must not be presented as evidence localization success across positive-evidence cases.

## Next controlled experiments (not yet run)

1. Preserve this original run and strict parser as experiment 1.
2. Clarify label semantics explicitly in a versioned prompt (especially exclusion polarity,
   absent information versus explicit contradiction, and not-applicable criteria), with
   an exact quoted JSON example. Re-evaluate the untouched base and re-train under the
   same new prompt for a fair paired comparison; do not quietly retrofit the old baseline.
3. On training only, test one mild class-balanced sampler/weighting variant. Keep every
   original training annotation available, cap oversampling and report the change. Compare
   minority-class recall and overall macro-F1 against the original unweighted run.
4. Inspect validation errors and compare lower-LR or additional-epoch variants only as
   documented ablations. Improvement is not guaranteed; do not select on test data.
5. Record the actual hardware: this full run used a Tesla T4, although the earlier smoke
   run used A100. A different GPU affects runtime and numerical behavior; it does not
   itself explain the observed majority-label prediction pattern. Verify the assigned
   runtime before describing experiments as A100 results.

The input prompt supplies allowed labels but no explicit category definitions or JSON
example. Its role in errors is a hypothesis to test, alongside observed label imbalance;
limited training and small-model reasoning capacity are plausible contributing factors,
not measured causal conclusions from this single run.
