# Selected models and initial design

## Data flow

Official dataset -> immutable snapshot -> safe field projection -> fixed patient split ->
chat input + JSON expert target -> base validation -> supervised adapter training ->
validation selection -> frozen base/adapted final test -> browser inference/report.

The user selected Google Colab for training and benchmarks and reports A100 access
through their Colab plan. Record the actual assigned GPU, VRAM and runtime versions
at the start of each session. Keep the compulsory workflow runnable on free compute
with conservative settings; the reported A100 access is an additional available resource. The application will run
locally using the downloaded model and adapter. Streamlit is the proposed interface so
one Python model service can support forms, evidence views, benchmarks and report export.
The final user must interact entirely in the browser after documented startup.

## Selected model pair and candidate comparison

| Candidate | Parameters | Architecture | License | Context | Estimated weight memory only |
|---|---:|---|---|---:|---|
| Qwen2.5-1.5B-Instruct | 1.54B | Decoder, RoPE, GQA (12 Q / 2 KV heads), 28 layers | Apache-2.0 | 32,768 | ~3.08 GB FP16/BF16, ~6.16 GB FP32 |
| Qwen2.5-0.5B-Instruct | 0.49B | Decoder, RoPE, GQA (14 Q / 2 KV heads), 24 layers | Apache-2.0 | 32,768 | ~0.98 GB FP16/BF16, ~1.96 GB FP32 |

Primary sources, checked November 6, 2026:
- https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
- https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct

These decimal estimates are parameter count times bytes per weight, not total runtime
or training requirements. Add activations, gradients, optimizer state, allocator overhead,
and KV cache. Both models together need ~4.06 GB in FP16 weights alone. The current
laptop has approximately 16 GB system RAM and no `nvidia-smi` executable was found;
CUDA capability on the laptop remains unverified until PyTorch is installed. Training
and GPU benchmarks will use the user-reported Colab A100; local work covers data
preparation, development and small correctness tests. Colab hardware has not been
inspected from this workspace, and its VRAM capacity is not assumed.

Selected by the user: 1.5B target and 0.5B draft. LoRA remains the proposed adaptation
method. This balances feasibility and instruction
following; clinical quality remains unmeasured. The 0.5B model is also a low-resource target
alternative. Tokenizer-only preflight passed on the selected immutable revisions: complete vocabulary
and added-token mappings, special-token IDs/maps, chat templates and encoding probes
agreed. Results are in `results/tokenizer-preflight/`; model commits are pinned in
`configs/model_revisions.json`. GPU feasibility and clinical quality remain unverified.

Start with 2048 total training tokens and reject/log overlength examples. Inspect train and
validation lengths before deciding whether to increase this; never silently drop gold evidence.
Hyperparameters in `configs/experiment.toml` are starting values, not selected results.

## Implementations still required

- Training: assistant-target-only cross-entropy, padding labels masked, adapter save/reload.
- Uncached greedy reference and explicit cached forward loop with positions/masks.
- Own reference and tiled stable online-softmax attention (separate correctness module).
- Greedy speculative decoder with accepted-prefix verification, correction and cache crop.
- Criterion/evidence metrics, parse-failure accounting and eight-error analysis.
- Browser screening, exact evidence mapping, measured comparison and report export.
- Explicit conservative aggregate: blocked if any inclusion is `not included` or exclusion
  is `excluded`; unresolved if any is `not enough information`; otherwise potentially
  eligible if at least one applicable criterion exists. All-not-applicable remains unassessed.
  This proposed rule must be documented and tested; it is not a clinical decision.

Do not present scaffolding or synthetic output as measured model performance.
