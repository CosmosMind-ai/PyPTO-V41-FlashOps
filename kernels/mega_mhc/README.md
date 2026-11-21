# mega_mhc — fused mHC layer step

One self-contained PyPTO kernel implementing a complete unshifted mHC layer step
(mixed-head context, M=4 residual copies, D=256 per copy) on Ascend 910C.

## Operator

Per token `t`, independently (`M3 = 2M + M² = 24`, `SINK = 10` Sinkhorn iterations):

```text
post  = 2 * sigmoid(input_mixes[:, M:2M] * scale[1] + base[:, M:2M])
comb  = sinkhorn(reshape(input_mixes[:, 2M:] * scale[2] + base[:, 2M:], [T,M,M]))
updated[t,m,:] = post[t,m] * x[t,:] + Σ_n comb[t,m,n] * residual_in[t,n,:]
flat  = reshape(updated, [T, M*D])
inv_rms = rsqrt(mean(flat², -1) + 1e-6)
next_mixes = (flat @ (fn * normw)ᵀ) * inv_rms
next_pre = sigmoid(next_mixes[:, :M] * scale[0] + base[:, :M])
h_next[t,:] = Σ_m next_pre[t,m] * updated[t,m,:]
residual_out = flat
```

The residual mixing orientation is `comb[t,m,n] * residual_in[t,n,d]`; RMS spans the
full flattened residual; `h_next` uses coefficients predicted from the *updated*
residual.

## Interface

`build({"T": T, "dtype": "fp32"|"bf16"})` — `T` a positive multiple of 8.

| tensor | shape | dtype (bf16 mode) | role |
|---|---|---|---|
| `x` | [T, 256] | bf16 | input |
| `residual` | [T, 1024] | bf16 | in/out |
| `input_mixes` | [T, 24] | fp32 | input |
| `fn` | [24, 1024] | fp32 | input |
| `normw` | [1, 24] | fp32 | input |
| `mhc_scale` | [3] | fp32 | input |
| `mhc_base` | [1, 24] | fp32 | input |
| `next_mixes` | [T, 24] | fp32 | output |
| `h_next` | [T, 256] | bf16 | output |

## Measured (Ascend 910C, sealed holdout)

| Shape | Latency |
|---|---|
| T=2048/4096 bf16 (decision) | 66.6 µs |
| Sealed holdout | **61.1 µs** (PASS) |

The official PyPTO HC path (`hc.py`) is not runnable on 910C — its sinkhorn stage is
extern C++ written for the a5 ISA (`__VEC_SCOPE__`), which fails to compile on a2a3.
Against an `examples/intermediate`-style decomposition of the same operator (43.8 ms)
this kernel is ~650× faster.
