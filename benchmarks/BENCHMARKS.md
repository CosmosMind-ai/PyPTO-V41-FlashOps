# Benchmark protocol and results

## Protocol

- **Hardware**: Ascend 910C (`a2a3`), CANN ≥ 9.0.
- **Timing**: median effective per-round device latency, 50 measured rounds after 5
  warmup rounds, on otherwise-idle devices. Official-comparison runs use the same
  card, the same protocol, and the same golden toolchain on both sides.
- **Correctness**: every reported number comes from a build that first passed an
  independent PyTorch golden at the kernel's documented tolerances, then a **sealed
  holdout** (fresh shapes not visible during search) with dual gates — correctness
  and latency. `latency_us` is the geometric mean of per-shape medians where a
  campaign evaluates multiple shapes.
- **Baseline**: "decomposition" numbers are same-DSL (PyPTO) unfused reference
  decompositions of the same operator, measured under the same protocol.

## Results

### mega_mhc (bf16, geometric mean of T=2048 / T=4096)

| Build | Latency |
|---|---|
| Same-DSL decomposition | 43.8 ms |
| This kernel (decision) | 66.6 µs |
| This kernel (sealed holdout) | **61.1 µs** |
| Official PyPTO `hc.py` | not runnable on 910C (a5-only extern C++) |

### engram (teaching width: EMBED=768, HIDDEN=1024, VOCAB=4096)

| T | This kernel | Official `l2_engram` | Speedup |
|---|---|---|---|
| 64 | 57.6 µs | 204.2 µs | 3.5× |
| 256 | 119.6 µs | 497.8 µs | 4.2× |
| 1024 | 337.3 µs | 1428.8 µs | 4.2× |

Official at native V4.1 width (D=5120): does not compile on 910C (vector-buffer
overflow: 197,120 B > 188,416 B).

### hierindex (bf16, S=131072)

| Build | Latency |
|---|---|
| Same-DSL decomposition | 269 ms |
| This kernel (decision) | 73.8 µs |
| This kernel (sealed holdout) | **83.8 µs** |
| Official DSpark `decode_indexer` | 180.8 µs / 128 queries — different operator class (compressed-cache, speculative-batch); complementary |

### swa_replay (B=32, ROWS=4096, D=7168→1024 + RoPE)

| Build | Latency |
|---|---|
| This kernel (decision) | 474.5 µs |
| This kernel (sealed holdout) | **465.6 µs** |
| Official PyPTO `decode_swa` | not runnable on the 910C PyPTO stack (newer API) |

## Notes

- All measurements were taken with no other workload on the card; shared-machine
  readings were discarded.
- Holdout shapes differ from search shapes; the sealed number — not the decision
  number — is the headline figure for each kernel.
