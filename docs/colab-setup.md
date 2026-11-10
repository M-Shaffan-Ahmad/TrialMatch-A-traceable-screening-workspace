# Colab setup and one-step feasibility

## Files to use

- `notebooks/01_colab_setup.ipynb`: upload/open this notebook in Google Colab.
- `output/trialmatch-colab-starter.zip`: upload this companion archive when cell 1 asks.

The starter ZIP contains source/configuration, not downloaded weights, raw data, prepared
patient examples, credentials or the local Python environment. It needs no GitHub remote.
Rebuild it after source changes with `python scripts/package_colab.py`.

## Run sequence

1. Open Google Colab, choose File -> Upload notebook, and select the notebook.
2. Choose Runtime -> Change runtime type -> GPU, selecting A100 when available.
3. Run cells in order. Upload the starter ZIP when prompted.
4. Allow the optional Drive mount so run artifacts persist under `MyDrive/TrialMatch/results`.
5. The notebook installs the starter stack, recreates the exact dataset split from its
   immutable commit, runs integrity tests, and launches the feasibility script.
6. A successful run produces `status.json` with `status: passed`. Keep or download the
   run folder. If it fails, preserve `status.json`, `failure.txt` and `console.log`.

If package installation requests a runtime restart, restart and rerun the cells. The uploaded
project remains on disk in the same session. A deleted runtime requires uploading it again.
A missing GPU is a setup error: select a GPU runtime rather than train on the notebook CPU.
The assigned GPU name, VRAM and BF16 support are detected, not assumed from the plan.

## What is checked

- Both selected model revisions are immutable and recorded; tokenizer-only preflight
  already pinned them in `configs/model_revisions.json`.
- Full vocabulary/added-token mappings, special-token semantics, chat templates and
  representative encodings agree. No model weights were required for the local check.
- Both models load and run a short synthetic greedy generation. This is a library smoke
  check, not an implementation of cached or speculative decoding.
- Only the training split is opened by the feasibility script. No validation/test scoring.
- Supervision masks prompt tokens with `-100`; only assistant JSON and its termination
  are trained. Overlength examples are rejected and logged, never silently truncated.
- Exactly one optimizer step changes a LoRA weight with finite loss and gradients.
- A saved adapter reloads on the same base revision and reproduces synthetic greedy tokens.

BF16 is selected if supported; otherwise the script uses FP32. Training batch size is one.
The draft is unloaded before training; the target uses gradient checkpointing and LoRA
on attention and MLP projection layers. These settings are for a feasibility check, not
selected full-training hyperparameters. No 4-bit library or paid API is required.

## Saved artifacts

Each run has a new timestamped directory containing environment and resolved package
versions, model revisions, the split manifest, tokenizer checks, training-only length/rejection
statistics, the one-step loss/update/memory record, adapter reload checks and console logs.
The `smoke-adapter/NOT_FINAL_CHECKPOINT.md` marker identifies the one-step adapter.
Base weights remain in the temporary Colab cache; use the pinned download method to reload.
The optional result ZIP includes logs and the small adapter, not the base weights.

## Validation status

Locally verified on CPU with the pinned Transformers/PEFT stack: 12 unit tests, including
assistant masking, overlength rejection, incompatible-tokenizer detection, a real optimizer
update and tiny-Qwen adapter save/reload. The actual selected model tokenizers also passed.
Notebook code cells were syntax checked; they contain no execution outputs. A100 model
loading, full-size optimizer step, Drive integration and Colab installation remain to be
verified by running the supplied notebook in the user's account.

The data-preparation manifest retains its original NumPy-version provenance. Recreating
it on another NumPy version is accepted only when every split ID/count and source hash
is identical; a changed split or dataset still fails.

## Technical references

- Chat templating: https://huggingface.co/docs/transformers/v4.57.1/en/chat_templating
- Qwen2 implementation: https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/qwen2
- PEFT LoRA: https://huggingface.co/docs/peft/v0.17.0/developer_guides/lora

Next: base-model validation evaluation and full training scripts, after this GPU check passes.

## Follow-up status (November 7, 2026)

The user's uploaded A100 40 GB feasibility run passed; see
`results/colab/feasibility-20261107T050049399113Z/REVIEW.md`. Continue with
`notebooks/02_colab_train_evaluate.ipynb` and the updated source ZIP.
