# hierindex — exact hierarchical sparse indexer

One self-contained PyPTO kernel implementing an exact hierarchical sparse indexer
for one decode query on Ascend 910C. This is exact selection, not approximate
sparse attention.

## Operator

`H = 8` heads, `DI = 128`, `BLOCK = 64`, `TOPB = 256`, `TOPK = 512`:

```text
s_i  = Σ_h w_h · dot(q_h, k_i)        # every KV position, FP32 scores
bm_j = max of scores in block j       # S/64 block maxima
pool = positions in the top-256 blocks
(top_val, top_idx) = descending top-512 inside the pool
```

BF16 inputs are promoted to FP32 before score arithmetic; accumulation, workspaces
and `top_val` stay FP32; `top_idx` is INT32 with exact ordered (tie-stable) indices.

## Interface

`build({"S": S, "dtype": "fp32"|"bf16"})` — `S` a multiple of 64, at least 32768.

| tensor | shape | dtype |
|---|---|---|
| `q` | [8, 128] | fp32 / bf16 |
| `kc` | [S, 128] | fp32 / bf16 |
| `w` | [8] | fp32 / bf16 |
| `ws_scr` | [1, S] | fp32 (observable score workspace) |
| `ws_pool` | [1, S] | fp32 (0/1 selected-block mask) |
| `ws_bm` | [1, S/64] | fp32 (block maxima, selected blocks offset by −1e30) |
| `top_val` | [1, 512] | fp32 |
| `top_idx` | [1, 512] | int32 |

Tolerances: `rtol = 0`, `atol ≤ 1e-4` — exact on INT32 indices.

## Measured (Ascend 910C, S=131072)

| Stage | Latency |
|---|---|
| Decision | 73.8 µs |
| Sealed holdout | **83.8 µs** (PASS) |

Against a same-DSL decomposition of the full pipeline (269 ms) this is ~3200×.
The official DSpark `decode_indexer` (180.8 µs for 128 compressed-cache queries) is
a different operator class — compressed-cache, speculative-batch indexing — so the
two are complementary rather than competing implementations.
