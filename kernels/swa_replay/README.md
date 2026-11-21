# swa_replay — SWA bounded-replay KV projection + absolute-position RoPE

One self-contained PyPTO kernel implementing sliding-window bounded-replay KV
projection with absolute-position RoPE on Ascend 910C.

## Operator

`W = 128` window, `ROWS = B × 128`, `D = 7168`, `N = 1024 = 8 heads × 128`:

```text
projected_key = hidden @ weight_k            # fp32 accumulation
value_out     = hidden @ weight_v
key_out = [k_low·cos − k_high·sin, k_high·cos + k_low·sin]   # per head, 64-lane halves
```

Cosine/sine coefficients have shape `[ROWS, 64]`, broadcast across the eight heads,
and encode **absolute** positions: they are consumed unchanged — never restarted at
request/window boundaries, never regenerated. This matches the bounded-replay
semantics of vLLM's `--swa-bounded-replay` with `sliding_window = 128`. Paged-KV
scatter is outside this operator.

## Interface

`build({"B": B, "dtype": "fp32"|"bf16"})` — any positive `B`, including grouped and
residual rows.

| tensor | shape | dtype |
|---|---|---|
| `hidden` | [B·128, 7168] | fp32 / bf16 |
| `weight_k`, `weight_v` | [7168, 1024] | fp32 / bf16 |
| `cos`, `sin` | [B·128, 64] | fp32 |
| `key_out` | [B·128, 1024] | fp32 |
| `value_out` | [B·128, 1024] | fp32 |

Tolerances: bf16 `(2e-2, 2e-2)`; fp32 `(1e-4, 1e-4)`.

## Measured (Ascend 910C, B=32 → 4096 rows)

| Stage | Latency |
|---|---|
| Decision | 474.5 µs |
| Sealed holdout | **465.6 µs** (PASS) |

The official PyPTO `decode_swa` path is not runnable on the 910C software stack —
it depends on a newer PyPTO API (`pl.Tile` annotations) than the stack provides, and
it is a TP-distributed full-layer orchestration rather than a replay-segment kernel.
