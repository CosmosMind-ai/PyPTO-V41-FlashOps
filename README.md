# PyPTO-V41-FlashOps

**English** | [中文](README.zh-CN.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Platform](https://img.shields.io/badge/Platform-Ascend%20910C-blue)
![CANN](https://img.shields.io/badge/CANN-%E2%89%A5%209.0-orange)

> Part of the **CosmosMind PyPTO kernel series** for Ascend — all kernels driven by [MetaRSI](https://www.cosmosmind.ai/research/metarsi-v1).

DeepSeek V4.1-Flash–style operators for **Ascend 910C**, written in
[PyPTO](https://github.com/hw-native-sys/pypto): four fused kernels covering the
modern linear-attention / sparse-attention serving stack — multi-head context mixing
(Mega-mHC), conditional memory (Engram), exact hierarchical sparse indexing
(HierIndex), and sliding-window bounded-replay KV projection with RoPE (SWA-Replay).

**Why this library exists.** We measured the official PyPTO V4.1 model path on 910C
(`a2a3`) and found that key components either do not exist for this chip or do not
run on it: the official HC sinkhorn path is written for the a5 ISA and fails to
compile on 910C; the official Engram kernel exceeds the 910C vector-buffer budget at
native model width; the official `decode_swa` path depends on a newer PyPTO API than
the 910C stack provides. This library fills those gaps with kernels that compile and
run on 910C today, each gated by an independent golden reference and a sealed
holdout. Full evidence: [`docs/OFFICIAL-ALIGNMENT.md`](docs/OFFICIAL-ALIGNMENT.md).

## Kernels

| Kernel | Operator | Sealed-holdout latency | Official PyPTO @ 910C |
|---|---|---|---|
| [`mega_mhc`](kernels/mega_mhc/) | Fused mHC layer step: Sinkhorn combination matrix + gated residual update + next-mix prediction (M=4, D=256) | **61.1 µs** @ T=2048/4096 bf16 | Not runnable — official `hc.py` uses a5-only extern C++ |
| [`engram`](kernels/engram/) | Engram conditional memory: hash-gather → dual projection → dual RMS norm → signed-sqrt gate | **57.6 / 119.6 / 337.3 µs** @ T=64/256/1024 | 204.2 / 497.8 / 1428.8 µs → **3.5× / 4.2× / 4.2× faster** |
| [`hierindex`](kernels/hierindex/) | Exact hierarchical sparse indexer: FP32 scores → block maxima → top-256 blocks → exact ordered top-512 | **83.8 µs** @ S=131072 | Complementary class — official `decode_indexer` works on compressed caches |
| [`swa_replay`](kernels/swa_replay/) | SWA bounded-replay KV projection (D=7168→1024) + absolute-position RoPE | **465.6 µs** @ B=32 (4096 rows) | Not runnable — official `decode_swa` needs a newer PyPTO API |

All latencies are median effective per-round device time on a real Ascend 910C
(`a2a3`), measured under the protocol in [`benchmarks/BENCHMARKS.md`](benchmarks/BENCHMARKS.md).
Every kernel passed a **sealed holdout** (fresh shapes, dual gates: correctness +
latency) against an independent PyTorch golden reference.

## Layout

```
kernels/
  mega_mhc/kernel.py    # fused mHC layer step
  engram/kernel.py      # Engram conditional-memory forward core
  hierindex/kernel.py   # exact hierarchical sparse indexer
  swa_replay/kernel.py  # SWA bounded-replay KV projection + RoPE
benchmarks/BENCHMARKS.md
docs/OFFICIAL-ALIGNMENT.md
```

## Usage

Each kernel is a single self-contained `kernel.py` in pypto-lib
`examples/intermediate/` style, exposing the standard build contract:

```python
fn, specs, golden_fn, rtol, atol = build(shape)
```

- `fn` — a `@pl.jit` kernel or built `@pl.program`, compiled through PyPTO;
- `specs` — `golden.TensorSpec` / `ScalarSpec` entries matching the kernel
  signature, with input distributions;
- `golden_fn` — an independent PyTorch reference that fills outputs in place;
- `rtol`, `atol` — validation tolerances.

Requirements: Ascend 910C (`a2a3`), CANN ≥ 9.0, `pypto`, and `pypto-lib` (its
`golden` package backs the validated launch path), `torch`. Example:

```python
import importlib.util

spec = importlib.util.spec_from_file_location("engram", "kernels/engram/kernel.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

fn, specs, golden_fn, rtol, atol = mod.build({"T": 256, "dtype": "bf16"})
# compile + dispatch fn through the pypto-lib golden runner, then validate
# outputs against golden_fn at (rtol, atol)
```

See each kernel's README for the exact operator semantics, tensor interface, and
supported shape/dtype domain. Unsupported shapes raise — kernels never silently
compute outside their contract.

## Status

v0.1.0 — all four kernels pass their correctness gates and sealed holdouts on 910C.
Interfaces follow the V4.1 operator semantics; where an official PyPTO counterpart
exists and runs on 910C, we report head-to-head numbers
([`docs/OFFICIAL-ALIGNMENT.md`](docs/OFFICIAL-ALIGNMENT.md)). This is a kernel
library, not a full model path: paged-KV scatter, hash computation, and host-side
orchestration are intentionally out of scope.

## License

[MIT](LICENSE)
