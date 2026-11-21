# engram — Engram conditional memory forward core

One self-contained PyPTO kernel implementing the Engram conditional-memory forward
core (one layer, one residual stream) on Ascend 910C.

## Operator

`NH = 24` heads, `DH = 32`, `EMBED = 768`, `HIDDEN = 1024`, `VOCAB = 4096` rows per
head table, `eps = 1e-6`:

```text
emb[t]  = concat_h table[h*VOCAB + hash_ids[t,h]]        # gather, IDs supplied
key     = emb @ wk;  value = emb @ wv                    # fp32 accumulation
kn      = key    * rsqrt(mean(key²,    -1) + eps)
qn      = hidden * rsqrt(mean(hidden², -1) + eps)
g       = Σ(kn · qn, -1) / sqrt(HIDDEN)
gate    = sigmoid(sign(g) * sqrt(|g| + eps))             # signed-sqrt, then sigmoid
out     = gate * value
```

Hash computation, convolution, and residual addition are outside this operator.

## Interface

`build({"T": T, "dtype": "fp32"|"bf16"})` — any positive `T`; invalid (padded) rows
are masked and never affect valid-row statistics.

| tensor | shape | dtype |
|---|---|---|
| `hash_ids` | [T, 24] | int32 |
| `head_off` | [1, 24] | int32 |
| `table` | [24·4096, 32] | fp32 / bf16 |
| `wk`, `wv` | [768, 1024] | fp32 / bf16 |
| `hidden` | [T, 1024] | fp32 / bf16 |
| `out` | [T, 1024] | fp32 / bf16 |

Tolerances: bf16 `(2e-2, 2e-2)`; fp32 `(1e-4, 1e-4)`.

## Measured (Ascend 910C, vs official `l2_engram`, same shapes, same golden)

| T | This kernel | Official PyPTO | Speedup |
|---|---|---|---|
| 64 | 57.6 µs | 204.2 µs | **3.5×** |
| 256 | 119.6 µs | 497.8 µs | **4.2×** |
| 1024 | 337.3 µs | 1428.8 µs | **4.2×** |

The official kernel cannot run at native V4.1 width on 910C at all: its vector
buffer requirement (197,120 B) exceeds the a2a3 limit (188,416 B). Numbers above use
the reduced teaching width both sides share.
