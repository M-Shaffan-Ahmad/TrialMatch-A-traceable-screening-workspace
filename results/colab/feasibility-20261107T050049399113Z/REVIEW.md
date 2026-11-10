# Colab feasibility review - November 7, 2026

The uploaded run passed its recorded checks. Its dataset manifest and model revision
pins exactly match the project's originals. Archive integrity passed. Small logs were
imported for reproducibility; the 73.9 MB one-step adapter remains in the source archive.

## Verified from the run

- NVIDIA A100-SXM4-40GB, 39.49 GiB reported device capacity, BF16.
- Both selected pretrained models loaded and generated synthetic outputs.
- All tokenizer compatibility checks passed.
- All 603 training examples fit: median 259, p95 455.9, maximum 578 tokens;
  zero truncation or rejection at the configured 2048-token limit.
- One optimizer update on annotation 43: 217 input tokens, 19 supervised tokens,
  loss 1.277868, 18,464,768 trainable LoRA parameters and a nonzero weight change.
- Step time 0.759 seconds; peak allocated GPU memory 3.403 GiB for this step.
- Saved/reloaded adapter reproduced the 16 generated tokens on the synthetic prompt.
- The log states no test set use; the supplied script reads training examples only.

## Interpretation and limits

This establishes that the initial stack runs on the actual selected hardware. It does not
establish model accuracy, convergence, maximum-length batch capacity or decoding speedups.
One-step training loss is not an eligibility score. Preserve this adapter as a smoke artifact,
not as the final trained checkpoint. Full training must start from the pinned pretrained base.
The TensorFlow GPU-driver warning did not prevent the PyTorch CUDA run; the GPU identity,
training update and memory measurements show that the required PyTorch path succeeded.

Training labels are imbalanced: `excluded` has 8 examples and `not included` has 13,
compared with 292 `not excluded`. Keep per-class and macro-F1 reporting central and
investigate minority-class failures using validation data, without changing the test split.

Next: implement base-model validation and full adapter training with validation-based
checkpoint selection, persistent logs and before/after comparisons on identical examples.
Test evaluation remains reserved until choices are frozen. Record the measured Colab
stack as run provenance; do not assume later sessions supply identical versions.
