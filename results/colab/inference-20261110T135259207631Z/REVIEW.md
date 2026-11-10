# Verified inference review - November 10, 2026

All 45 measured decoder records agree token-for-token with uncached greedy for their
respective fixed-work case; the separate EOS-aware screening prompt also agrees across
all three custom decoders. Raw timings/medians, acceptance counts, attention measurements
and candidate identity were independently checked against the exported logs. The archive
is intact. The actual GPU was NVIDIA L4 (~22.03 GiB), BF16, not A100.

## Repeated fixed-work results (2 warmups, 5 repetitions)

| Input / output tokens | Uncached seconds | Cached seconds | Speculative seconds | Cache speedup | Speculative vs cached |
|---|---:|---:|---:|---:|---:|
| 128 / 32 | 2.6025 | 2.5353 | 2.2684 | 1.03x | 1.12x |
| 512 / 64 | 5.7220 | 5.0002 | 4.3338 | 1.14x | 1.15x |
| 1024 / 128 | 27.3058 | 10.1472 | 8.7691 | 2.69x | 1.16x |

The fixed-work prompt repeats a short synthetic paragraph and the models continue that
repetition. All drafted candidates were accepted in these cases. These are controlled
length benchmarks, not representative clinical-reasoning throughput or broad acceptance
estimates. Model attention backend is eager, and both models remain resident for all modes.
At 1024/128, peak allocated memory was ~4.21 GiB cached versus ~4.51 GiB speculative;
the allocator measurement does not include all driver/process memory.

## EOS-aware synthetic screening (single diagnostic timing)

The model emitted 16 tokens including EOS; all three modes matched. Cached greedy took
1.2661 s and speculative took 1.6192 s: speculation was ~27.9% slower (0.782x speedup).
Draft acceptance was 12/18 = 66.7%, with both partial rejection and zero-acceptance rounds.
This demonstrates the actual rejection/correction path on the selected checkpoint.
These timings are single diagnostics, not a repeated screening-performance estimate.
Report both the fixed-work gain and the screening slowdown honestly.

## Attention

All controlled causal, noncausal and rectangular-mask cases passed. Maximum absolute
error across the logged FP32 comparisons was 5.96e-7, below atol/rtol 2e-5.
For sequence length 1024, dense reference took 1.263 ms versus tiled 305.160 ms.
The educational tiled implementation is much slower because it launches many Python/
PyTorch operations, despite avoiding the full attention matrix. It is tested standalone,
not patched into the pretrained model, and must not be described as a fused FlashAttention
speedup. The raw allocator peaks are available in attention.json for memory analysis.

## Status and next work

This provides selected-checkpoint GPU evidence for the explicit KV cache, tiled attention
and speculative implementations under the stated conditions. It does not establish
clinical accuracy beyond the earlier validation results or a general speculative speedup.
The fixed test set remains unused. Next: browser screening/report export, integration of
these modes and live comparison, then final held-out evaluation after configuration freeze,
IEEE report, reproducibility packaging and demo. Repeated screening benchmarks may be
added for app-representative performance, without tuning on held-out patients.
