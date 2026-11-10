# Prompt v2 and mild class-weighting experiments

Run `notebooks/03_colab_prompt_balance.ipynb` with the newest
`output/trialmatch-colab-starter.zip`. This is a new experiment pair, not a resume or
replacement of the original prompt-v1 run. Default `REQUIRE_A100=True` checks that
Colab actually assigned the user's requested A100 before either long run starts. It can
be explicitly disabled for other CUDA GPUs; compulsory code does not require paid compute.

## Preserved original evidence

The uploaded original full run is preserved under
`results/colab/training-20261107T052336764258Z/`. Its untouched source ZIP was saved to
`output/archive/experiment-01-source.zip` before changing implementation. That source
hash matched the uploaded run. Its metrics remain unchanged: selected accuracy 0.7039,
macro-F1 0.4119 and evidence micro-F1 0.5323. The actual hardware was a Tesla T4.
`docs/reference-experiment-01.json` contains historical summary metrics and provenance,
not patient examples. The parser and scoring implementation are unchanged in these ablations.
Do not use the updated source to resume the old run; source/protocol checks will reject it.

## A: clearer prompt, unweighted training

`configs/experiments/02_prompt_v2.toml` uses prompt v2 and ordinary assistant-token loss.
The prompt defines inclusion and exclusion label meanings explicitly, distinguishes
contradiction, missing information and non-applicability, checks polarity and temporal
conditions, and shows one exact quoted JSON syntax example. It limits citations to
original directly relevant sentence IDs. The syntax example is explicitly not a case
answer and contains no benchmark targets. Its possible label-priming effect is a limitation.

The original v1 prompt remains the API default. An explicit version is passed identically
to supervised encoding and validation generation; no target fields enter either prompt.
The rendered system texts are saved as `prompt_protocol.json` in every run. The real
pinned tokenizer encoded all 603 train examples under v2 without truncation (maximum
786 tokens versus the 2048-token limit). No validation/test data determined class weights.

A measures its own untouched-base validation with v2, then trains a fresh adapter.
The original base's invalid-JSON score must not be silently replaced by this new baseline.
A versus the historical v1 run is contextual evidence; report prompt/source and any hardware
changes instead of claiming an isolated causal effect of prompting.

## B: same v2 prompt plus training-only class weights

`configs/experiments/03_prompt_v2_balanced.toml` changes only the training loss weights
(and descriptive experiment ID) relative to A. It starts from the same pretrained base,
not A's adapter or the earlier smoke adapter. No rows are duplicated or dropped.

For accepted training examples, count each `(criterion_type, expert_label)` stratum.
Within each criterion type define

    raw_weight(type, label) = min(3, sqrt(max_class_count_for_type / class_count))

Normalize the example weights to mean 1. The cap refers to the relative rare/common
weight before normalization; normalization preserves those ratios. Compute the weights
once from accepted train rows, and publish counts/raw/normalized values in
`class_weighting.json`. Unrepresented classes get no invented examples.

For each accumulated optimizer update, use

    loss = sum_i(weight_i * sum_supervised_tokens(cross_entropy_i))
           / sum_i(weight_i * supervised_token_count_i)

Prompt and padding labels remain -100; causal shifting is identical to ordinary SFT.
This weights all supervised assistant tokens for an example, including label and evidence,
not vocabulary classes. It is not a replacement for missing clinical knowledge or a
promise of better performance. Both weighted and unweighted token loss are logged;
raw optimized training losses across A/B are different objectives.

Every original accepted example appears once per epoch, in the same seeded order in A
and B. Epoch count, batch/accumulation, LR schedule, LoRA modules/rank, dropout, generation,
parser, metrics, validation and checkpoint selection are held constant. With 603 accepted
examples, micro-batch 2, accumulation 4 and three epochs, each run has 228 optimizer
updates. Balance does not buy extra training exposure or optimization steps.

## Shared baseline and comparison checks

B may reuse only A's completed untouched-base validation. Reuse requires identical pinned
model/data/source/runtime provenance and the same prompt, max lengths, generation batch
and seed. Cached predictions/metrics are verified by the evaluator; adapter weights are
never reused. A mismatched old prompt or different GPU runtime fails closed.

Each recipe selects its own epoch using validation macro-F1, then evidence micro-F1,
then earliest epoch. Each selected adapter is reloaded and checked against its saved
validation outputs. These are checkpoint-selected recipe comparisons, not fixed-epoch
weight comparisons, and selection can make validation performance optimistic.

`scripts/compare_ablations.py` requires passed no-test statuses, the same provenance,
identical settings apart from weighting/ID, equal base metrics and equal optimizer/example/
supervised-token budgets. It reports overall metrics, invalid outputs and per-class recall/F1.
The final comparison makes no significance claim: there are only eight validation patients
and very few rare-class validation records. Do not tune on the test set or alter gold labels.

## Colab execution and resume

1. Open notebook 03, select A100, upload the newest source ZIP and mount Drive.
2. Recreate the pinned data and run the synthetic/local correctness checks.
3. Leave `RESUME_EXPERIMENT_ROOT` empty for a new timestamped A/B parent directory.
4. Run A and B sequentially in the same runtime. Each child has independent checkpoints,
   provenance, validation predictions, curves, selected adapter and reload checks.
5. Inspect `comparison/ABLATION-COMPARISON.md` and download the combined review ZIP.

To resume, upload the identical bundle, retain the same settings/runtime identity, and
paste the A/B **parent** directory into `RESUME_EXPERIMENT_ROOT`. The plan guards bundle
and config identity; child runs guard code, data and actual runtime identity. Each child
resumes its last complete epoch. A completed child is verified/reused rather than retrained.
The original prompt-v1 parent is not a valid resume path for notebook 03.

The combined export contains both selected adapters, logs, metrics, predictions and curves.
It excludes nonselected adapters and optimizer states; keep the Drive parent to resume.
A score decrease must be recorded, not hidden. This preparation creates no full-model
A/B performance results; execute notebook 03 to obtain them.

## Verification

Local tests cover v1 preservation, v2 input/target separation, type-specific weight caps and
normalization, all-one-weight equivalence to the original loss (including real tiny Qwen),
weighted micro/full-batch equivalence, baseline-reuse guards, comparison confound checks,
and exact weighted-v2 interruption/resume versus uninterrupted tiny-model training.
The test split is not opened by this development workflow.

## Implementation reference

Unreduced cross-entropy and ignored targets use the official PyTorch semantics:
https://docs.pytorch.org/docs/2.15/generated/torch.nn.CrossEntropyLoss.html .
The weighting/normalization formula above is this project's documented experimental choice.
