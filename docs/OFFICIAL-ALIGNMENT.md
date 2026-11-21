# Official-alignment evidence: V4.1 operators × official PyPTO on Ascend 910C

Method: we pulled the official PyPTO V4.1 model-path implementations
([hw-native-sys/pypto-lib](https://github.com/hw-native-sys/pypto-lib)) onto the
same Ascend 910C and measured them under the same protocol and the same golden
toolchain as our kernels (median effective latency, 50 rounds + 5 warmup, real
`a2a3` silicon).

## Head-to-head summary

| Kernel | Official implementation @ 910C | This library | Comparison |
|---|---|---|---|
| **mega_mhc** | **Not runnable**: official `hc.py`'s sinkhorn stage is extern C++ written for the a5 ISA; a2a3 compilation fails with `__VEC_SCOPE__ undeclared` | 61.1–67.0 µs (T=2048/4096 bf16, holdout PASS) | No official reference on 910C; ~650× vs an `examples/intermediate`-style decomposition (43.8 ms) |
| **engram** | Official `l2_engram` runs at reduced width: T=64/256/1024 = **204.2 / 497.8 / 1428.8 µs**; at **native V4.1 width it does not compile on 910C** (vector buffer 197,120 B > 188,416 B limit) | 57.6 / 119.6 / 337.3 µs at the same three shapes | **3.5× / 4.2× / 4.2× faster**, same shapes, same golden |
| **hierindex** | Official DSpark `decode_indexer` (a2a3-native): **180.8 µs** / 128 queries (64 requests × 2 speculative tokens), top-512 over compressed caches | 83.8 µs sealed holdout, single query over 131,072 raw positions, two-level hierarchical pooling | Different operator classes (compressed cache vs raw-position hierarchical pooling) — complementary, no conflict |
| **swa_replay** | **Not runnable**: official `decode_swa` depends on a newer PyPTO API (`pl.Tile` annotations) than the 910C stack provides; it is a TP-distributed full-layer orchestration, not a replay-segment kernel | 465.6 µs sealed holdout (B=32) | No official reference on 910C |

## Semantic alignment (formula-level review)

- **mega_mhc vs official `golden_mhc_mixes`**: pre = sigmoid scaled by `s1`,
  post = 2·sigmoid, sinkhorn on the combination matrix (we use 10 iterations vs the
  official 20 — converged doubly-stochastic matrices agree well inside tolerance),
  residual update `x·post + comb @ residual`, coefficient prediction `linear ·
  inv_rms` (project-then-normalize order identical) — **mathematically consistent**.
  Differences: sinkhorn iteration count; the official `hc_head` consumes the
  *previous* layer's pre-mix (delayed mixing cut) while we fuse prediction and the
  next layer's mixing into one operator (pipeline cut differs, math equivalent);
  width (official hidden=5120, ours the reduced teaching width).
- **engram vs official `golden_engram`**: signed-sqrt gating formula character-for-
  character identical (`sigmoid(sign(g)·√(|g|+ε))`), dual RMS normalization,
  gather → project → gated residual add — **mathematically consistent**.
  Differences: the official uses a single `wkv` projection for key|value, we split
  `wk`/`wv` (equivalent); reduced width (24×32=768 vs 24×256=6144).
- **hierindex vs official `decode_indexer`**: both select top-512; the official
  walks compressed caches with speculative-decoding batching, we walk raw positions
  with two-level hierarchical pooling (block max → in-pool exact selection).
  Tie handling matches the official ordered-index semantics.
- **swa_replay vs official `decode_swa`**: window W=128 matches the official
  `sliding_window=128`; absolute-position replay RoPE semantics align with vLLM's
  `--swa-bounded-replay`.

## Measured ecosystem gaps (the motivation for this library)

1. The official V4.1 HC path (`hc.py`) **only supports a5/950-class chips** — its
   extern C++ kernel uses a5 ISA and fails to compile on 910C.
2. The official Engram kernel at **native V4.1 width** exceeds 910C's 188 KB vector
   budget and cannot run; only reduced widths compile.
3. The official `decode_swa` requires a newer PyPTO API than the 910C stack has.
4. The official DSpark `decode_indexer` is the only V4.1 component that runs
   natively on 910C (180.8 µs, validated).

⇒ "Key V4.1-Flash components are either missing or not runnable on 910C" is a
measured conclusion, not a conjecture — and the reason these four kernels exist.
