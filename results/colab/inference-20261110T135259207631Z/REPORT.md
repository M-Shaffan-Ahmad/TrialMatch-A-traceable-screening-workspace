# Inference benchmark

All fixed-length and EOS-aware custom decoding token checks passed. Synthetic inputs only; test set unused.

| Input/output tokens | Mode | Median seconds | Tokens/s | Prefill seconds | Decode seconds |
|---|---|---:|---:|---:|---:|
| 128/32 | uncached | 2.6025 | 12.30 | 0.0829 | 2.5200 |
| 128/32 | cached | 2.5353 | 12.62 | 0.0837 | 2.4511 |
| 128/32 | speculative | 2.2684 | 14.11 | 0.1198 | 2.1461 |
| 512/64 | uncached | 5.7220 | 11.18 | 0.0871 | 5.6349 |
| 512/64 | cached | 5.0002 | 12.80 | 0.0863 | 4.9137 |
| 512/64 | speculative | 4.3338 | 14.77 | 0.1235 | 4.2100 |
| 1024/128 | uncached | 27.3058 | 4.69 | 0.2047 | 27.1011 |
| 1024/128 | cached | 10.1472 | 12.61 | 0.1866 | 9.9506 |
| 1024/128 | speculative | 8.7691 | 14.60 | 0.2592 | 8.5089 |

Both models remain resident in every decoding timing for comparable memory baselines. Speculative prefill includes both models; ordinary prefill includes target only. Decode throughput, raw repetitions, acceptance and allocator peaks are in JSON. No speedup is assumed. Educational tiled attention is a standalone correctness/performance implementation, not a fused model backend.
