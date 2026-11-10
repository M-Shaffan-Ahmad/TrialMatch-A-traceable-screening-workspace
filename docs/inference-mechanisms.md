# Explicit inference implementations and benchmark protocol

Use notebook 04, the updated source ZIP, and `trialmatch-selected-adapter.zip`.
The selected candidate remains prompt v2, unweighted epoch 3. Base and draft model
commits, adapter hashes and its prompt protocol are fixed in configuration. Benchmarks
use synthetic text only and do not open any patient dataset files.

## Cached and uncached greedy

`src/trialmatch/decoding.py` calls model forward directly; no `generate()` is used.
The uncached reference reprocesses the entire prefix after each token with no KV cache.
The cached path prefills a DynamicCache, then feeds just the previously selected token.
Each call explicitly supplies absolute position_ids/cache_position and an attention mask
of length past+new. The model adds its causal mask, including within multi-token blocks.
The supported API is one unpadded sequence; padded batched decoding is not implemented.
Cache length grows by one processed token per step. Ordinary generation returns with the
last selected token not yet fed, so cache length is input+generated-1. The returned cache
is diagnostic only, not a reusable state object. No gradients/dropout are active.

Prefill is synchronized model work through the first argmax. Decode is remaining time;
ordinary decode throughput counts generated-1. End-to-end throughput counts every token
and includes prefill/Python overhead. Cache-related matrix shapes can change floating
point order: an argmax mismatch, especially near a tie, is logged as failure and must be
investigated rather than ignored. Full checkpoint checks are still a Colab execution step.

## Greedy speculation

The state invariant is that both caches contain the entire committed prefix, with logits
for the next token. The smaller draft predicts and forwards up to K candidates. One
multi-token target call verifies the block. Candidate 0 is compared against saved prefix
logits; candidate j against logits after candidate j-1. Matching tokens are accepted until
the first rejection. Both caches are cropped to the accepted prefix, discarding every
rejected suffix. The target correction token is fed to both models. On complete acceptance,
the target supplies a bonus token and both models process it. This also keeps the draft
synchronized after correction. EOS/length limits apply to committed tokens, including the
accepted prefix; no bonus is appended past the requested budget. Final cache lengths are
input+generated for this fully synchronized implementation. The tokenizer mapping is
verified before any token comparison. Acceptance counts drafted candidates, never bonus
or correction tokens. Raw round statistics are saved.

This educational implementation prioritizes explicit state correctness. Correction/bonus
processing performs extra model calls, Python launch overhead matters, and a 0.5B draft
may have low acceptance against the adapted target. A speedup is neither guaranteed nor
required by the assignment. Do not compare it to uncached decoding alone: report speedup
or slowdown against ordinary cached greedy, including both model prefills and residency.

## Tiled attention and online softmax

`src/trialmatch/attention.py` implements a dense stable-softmax reference and a tiled
variant for [batch, heads, sequence, channels] Q/K/V. GQA heads must be expanded by the
caller. Causal masks compare absolute query positions (query_offset+row) with key positions,
including rectangular cache-style shapes. Inputs must have at least one allowed key per row.

For each query tile the algorithm maintains row maxima m, exponential normalizers l,
and unnormalized value accumulators a. For a new key tile S, compute m'=max(m,max(S)),
rescale old state by exp(m-m'), then update l'=exp(m-m')*l+sum(exp(S-m')) and
 a'=exp(m-m')*a+exp(S-m')@V. Output is a/l. Entirely masked tile rows are handled without
-inf minus -inf NaNs. FP16/BF16/FP32 inputs accumulate in FP32; FP64 stays FP64.
This avoids an entire N-by-N score/probability matrix, but still has quadratic arithmetic.
It is an inference-focused educational module, not a custom fused forward/backward kernel.

It is tested independently, not inserted into the pretrained model. All model decoding
modes use the same eager model backend. The assignment's conceptual tiled attention
implementation and model cache/speculation experiments are distinct reported components.
A Python tiled implementation may be slower because many small operations lack the
fusion and GPU scheduling of FlashAttention; no fused-kernel speed claims are made.

## Measurements

Default GPU protocol: two warmups, five repeats, synchronize before/after each timed call.
Garbage collection and empty_cache occur outside timed sections. Fixed decode work uses
128/32, 512/64, 1024/128 input/output lengths with EOS disabled. A separate synthetic
screening prompt verifies EOS-aware outputs across all modes. Attention benchmarks use
128/512/1024 lengths with FP32; controlled correctness includes causal/noncausal and
rectangular shapes, plus local extreme-logit FP64 tests. Max/mean errors and honest raw
runtimes are saved. Tolerances are atol=rtol=2e-5 for GPU FP32 attention comparisons.

PyTorch peak allocated and peak reserved bytes, pre-call allocated baseline and incremental
peak allocation are logged. These describe the device allocator, not driver/process total
memory. Target and draft remain resident in all decoding modes to make allocation baselines
comparable; a target-only deployment would have a smaller resident baseline. Raw mode order
is fixed and disclosed; thermal/order effects and finite repeats remain limitations.
Model weights are not downloaded by local tests; their actual benchmark execution happens
in the user's Colab account. Failure status and partial results persist for inspection.

## Verification and remaining work

Local tiny-Qwen tests compare uncached, cached and speculative output tokens against each
other and ordinary library greedy. Tests include multiple lengths, rejection, complete
acceptance, EOS, budget boundaries and independent draft weights/architecture. Attention
is compared to both the reference and PyTorch SDPA; causal future perturbations cannot
change earlier outputs. No full-model GPU results are implied by these CPU tests.

The browser application, held-out final evaluation, final paper and demo remain work.
This notebook specifically completes the implementation/preparation of tasks 3-5, with
full-model measured evidence still required from Colab.

Primary references:
- https://huggingface.co/docs/transformers/v4.57.1/en/cache_explanation
- https://huggingface.co/docs/transformers/v4.57.1/en/internal/generation_utils
- https://arxiv.org/abs/2205.14135
- https://arxiv.org/abs/2211.17192
