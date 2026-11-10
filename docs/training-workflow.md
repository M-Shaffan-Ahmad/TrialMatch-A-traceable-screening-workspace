# Baseline validation and full LoRA training

Use `notebooks/02_colab_train_evaluate.ipynb` and the **updated**
`output/trialmatch-colab-starter.zip`. The notebook explicitly uploads the new source
bundle, even if an older setup project remains in the same Colab session.

## Experiment

1. Recreate the pinned official dataset and required patient split.
2. Load the original pinned Qwen2.5-1.5B-Instruct base model, not the smoke adapter.
3. Evaluate the base on all 206 validation examples using library greedy generation.
4. Train on training examples only with assistant-target-only cross-entropy.
5. Save a complete adapter/optimizer checkpoint after each epoch and evaluate on the
   same validation examples, with identical prompts and generation settings.
6. Select the epoch with greatest overall validation classification macro-F1; break ties
   using evidence micro-F1, then keep the earliest epoch. The base is reported separately,
   not silently substituted when fine-tuning underperforms.
7. Reload the selected adapter on a fresh pinned base and repeat validation. Verify raw
   generated outputs and metrics match its epoch evaluation before marking success.
8. Save comparisons, curves, prediction/evidence records and the selected adapter pointer.

This development workflow has no test-set flag. It opens only `train.jsonl` and
`validation.jsonl`, verifies their IDs/counts against the manifest and records file hashes.
Final held-out test evaluation is still a later, separately gated task.

## Initial training settings

- 3 epochs, seed 42, micro-batch 2, accumulation 4: nominal effective batch 8.
- AdamW, learning rate 2e-4, weight decay 0.01, gradient norm clipped to 1.0.
- Linear warm-up over ceil(10% of total optimizer updates), then linear decay to zero.
  The first update uses 1/warmup_steps of peak LR; the first decay update is at peak LR.
- LoRA rank 16, alpha 32, dropout 0.05; attention Q/K/V/O and MLP gate/up/down projections.
- BF16 on supported hardware; otherwise FP32, recorded in provenance.
- Gradient checkpointing during training. No cache reuse during training.
- Total supervised sequence limit 2048; reject/log overlength train examples, never truncate.
- Validation generation batch 4, greedy, max 192 new tokens. Overlength validation prompts
  remain in the denominator as invalid predictions; no evaluation examples are dropped.

The batch increase from the one-step check is conservative but not yet A100-tested with
this full workflow. If it OOMs, preserve the failed logs and start a new run with a smaller
batch; adjust accumulation to preserve effective batch size. Do not change settings inside
an existing run and call it a resume. No class reweighting/oversampling is used initially;
inspect minority-class validation results before proposing a documented ablation.

Training batches use right padding with labels -100 on prompts/padding. Generation uses
left padding; the training collator is independent of tokenizer padding_side. Every epoch
shuffles all training examples using default_rng(seed+epoch_index). The final short batch
is included. Gradient accumulation weights each micro-batch by its count of shifted
supervised tokens, so a partial update is not incorrectly divided by the nominal group size.

## Parsing and metrics (freeze for the before/after comparison)

- Accept one JSON object or one enclosing ```json fence. Record fence normalization.
- Require exactly `label` and `evidence_sentence_ids`; reject extra/duplicate keys, trailing
  prose, wrong-type labels, duplicate/boolean/string/negative IDs and nonexistent citations.
- The parser uses the note and criterion type, never expert targets. Targets enter scoring only.
- Any invalid structured response is an incorrect classification (`__invalid__` prediction
  column). It receives no partial classification credit even if its label text was plausible.
- Report overall accuracy and macro-F1 over the fixed six-category union; per-type macro-F1
  uses the applicable four-category set. Report every class P/R/F1 and support; zero-support
  classes have F1=0. Confusion matrices have an extra invalid-prediction column.
- Evidence micro P/R/F1 pools actual sentence-ID TP/FP/FN. Macro averages per-example
  P/R/F1. For valid responses: both sets empty gives per-example 1, exactly one empty gives 0.
- Invalid responses have an empty usable evidence prediction and per-example scores 0;
  all gold evidence becomes FN in micro counts. If micro has no evidence in either set,
  report null/undefined, not a perfect score. Always report invalid counts separately.
- All rows remain in scoring, including generation cutoffs, invalid JSON and rejected prompts.
- Predictions include raw output, original IDs, exact evidence text and a clearly labeled
  source-evidence summary. That summary is templated, not an expert-authored or independently
  verified medical rationale. Missing evidence is never converted into proof of eligibility.

No unsupported rationale targets or GPT-generated dataset fields are added to training.
The library greedy implementation is a common baseline; it does not satisfy the later
custom cache/speculative-decoding requirements by itself. Timing fields here are diagnostic
validation durations, not warm-up/repeated inference benchmarks.

## Artifacts and resuming

Every run goes to a new timestamped Drive folder. Important files:

- `run.json`: exact settings, source-code hash, data hashes, pinned model and runtime identity.
- `baseline/`: raw/validated predictions, metrics and diagnostic duration.
- `epoch-XX/`: adapter/tokenizer, optimizer/scheduler/RNG states, update logs and validation.
- `selected_adapter.json`: relative path and base revision for the chosen adapter.
- `selected-adapter-validation/` and `checkpoint_reload.json`: actual reload verification.
- `comparison.json`, `COMPARISON.md`, `training_history.jsonl`, PNG/SVG learning curves.
- `status.json`, `failure.txt` when relevant, and notebook `console.log`.

To resume, upload the same bundle in notebook 02, keep settings identical, and set
`RESUME_RUN_PATH` to the earlier run folder. A complete-epoch checkpoint restores adapter,
optimizer, LR scheduler and torch RNG state. An unfinished epoch repeats from its start;
progress lines for that partial attempt are replaced. Completed validation outputs are
recomputed from saved predictions and verified before reuse. Changed config, source code,
data or recorded runtime identity causes resume to refuse mixing experiments.

The exporter includes the selected adapter and all readable run logs/plots, excluding other
adapters and optimizer `.pt` files. Keep the original Drive folder for future resumes;
the portable ZIP is not a full resume archive. Do not delete failed-run evidence.

## Local verification and remaining execution

Synthetic/local tests verify exact metrics on hand-calculated examples, invalid JSON and
citation handling, masking, padding/accumulation, and a complete tiny-Qwen workflow. An
interruption/resume experiment verifies exact adapter weights against uninterrupted CPU
training; a fresh selected-adapter reload is also verified. Notebook cells and portable
packaging are checked without fabricating execution outputs.

The successful uploaded A100 one-step run supports feasibility. Full-size baseline scoring,
three-epoch training, learning curves and final validation results remain to be measured
by running notebook 02 in the user's Colab account.

## Primary implementation references

- https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/text_generation
- https://huggingface.co/docs/peft/v0.17.0/package_reference/peft_model
- https://huggingface.co/docs/transformers/v4.57.1/en/chat_templating
