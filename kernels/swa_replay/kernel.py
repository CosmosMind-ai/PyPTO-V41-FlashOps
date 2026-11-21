"""Paired KV projection with shape-derived dispatch and lane-local RoPE."""
import pypto.language as pl

ROWS = pl.dynamic("ROWS")
D = 7168
N = 1024
DH = 128
ROW_TILE = 128
K_TILE = 256


@pl.jit
def swa_replay_bf16(
    hidden: pl.Tensor[[ROWS, D], pl.BF16],
    weight_k: pl.Tensor[[D, N], pl.BF16],
    weight_v: pl.Tensor[[D, N], pl.BF16],
    cosine: pl.Tensor[[ROWS, DH // 2], pl.FP32],
    sine: pl.Tensor[[ROWS, DH // 2], pl.FP32],
    key_out: pl.Out[pl.Tensor[[ROWS, N], pl.FP32]],
    value_out: pl.Out[pl.Tensor[[ROWS, N], pl.FP32]],
):
    coefficient_rows = pl.tensor.dim(cosine, 0)
    paired_rows = coefficient_rows // 2
    cosine_pairs = pl.reshape(cosine, [paired_rows, DH])
    sine_pairs = pl.reshape(sine, [paired_rows, DH])
    complete_group_rows = ROWS // 512 * 512
    for row_group_start in pl.parallel(0, complete_group_rows, 512):
        for head_start in pl.parallel(0, N, DH * 2):
            raw_key_part0 = pl.create_tensor([512, DH * 2], dtype=pl.FP16)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="key_partial0") as producer_tid_part0:
                for row_owner_part0 in pl.range(4):
                    row_start_part0 = row_group_start + row_owner_part0 * 128
                    key_acc_part0 = pl.tile.create([128, DH * 2], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc)
                    for macro_index_part0 in pl.pipeline(14, stage=2):
                        reduction_start_part0 = (macro_index_part0 + 0) * 256
                        hidden_panel_part0 = pl.load(hidden, [row_start_part0, reduction_start_part0], [128, 256], target_memory=pl.MemorySpace.Mat)
                        weight_panel_part0 = pl.load(weight_k, [reduction_start_part0, head_start], [256, DH * 2], target_memory=pl.MemorySpace.Mat)
                        for pair_offset_part0 in pl.range(0, 256, 64):
                            hidden_micro_even_part0 = pl.tile.extract(hidden_panel_part0, 0, pair_offset_part0, [128, 32], target_memory=pl.MemorySpace.Left)
                            weight_micro_even_part0 = pl.tile.extract(weight_panel_part0, pair_offset_part0, 0, [32, DH * 2], target_memory=pl.MemorySpace.Right)
                            hidden_micro_odd_part0 = pl.tile.extract(hidden_panel_part0, 0, pair_offset_part0 + 32, [128, 32], target_memory=pl.MemorySpace.Left)
                            weight_micro_odd_part0 = pl.tile.extract(weight_panel_part0, pair_offset_part0 + 32, 0, [32, DH * 2], target_memory=pl.MemorySpace.Right)
                            key_acc_part0 = pl.tile.matmul_acc(key_acc_part0, hidden_micro_even_part0, weight_micro_even_part0, init_cond=((macro_index_part0 == 0) and (pair_offset_part0 == 0)))
                            key_acc_part0 = pl.tile.matmul_acc(key_acc_part0, hidden_micro_odd_part0, weight_micro_odd_part0)
                    raw_key_part0 = pl.store(key_acc_part0, [row_owner_part0 * 128, 0], raw_key_part0)
            raw_key_part1 = pl.create_tensor([512, DH * 2], dtype=pl.FP16)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="key_partial1") as producer_tid_part1:
                for row_owner_part1 in pl.range(4):
                    row_start_part1 = row_group_start + row_owner_part1 * 128
                    key_acc_part1 = pl.tile.create([128, DH * 2], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc)
                    for macro_index_part1 in pl.pipeline(14, stage=2):
                        reduction_start_part1 = (macro_index_part1 + 14) * 256
                        hidden_panel_part1 = pl.load(hidden, [row_start_part1, reduction_start_part1], [128, 256], target_memory=pl.MemorySpace.Mat)
                        weight_panel_part1 = pl.load(weight_k, [reduction_start_part1, head_start], [256, DH * 2], target_memory=pl.MemorySpace.Mat)
                        for pair_offset_part1 in pl.range(0, 256, 64):
                            hidden_micro_even_part1 = pl.tile.extract(hidden_panel_part1, 0, pair_offset_part1, [128, 32], target_memory=pl.MemorySpace.Left)
                            weight_micro_even_part1 = pl.tile.extract(weight_panel_part1, pair_offset_part1, 0, [32, DH * 2], target_memory=pl.MemorySpace.Right)
                            hidden_micro_odd_part1 = pl.tile.extract(hidden_panel_part1, 0, pair_offset_part1 + 32, [128, 32], target_memory=pl.MemorySpace.Left)
                            weight_micro_odd_part1 = pl.tile.extract(weight_panel_part1, pair_offset_part1 + 32, 0, [32, DH * 2], target_memory=pl.MemorySpace.Right)
                            key_acc_part1 = pl.tile.matmul_acc(key_acc_part1, hidden_micro_even_part1, weight_micro_even_part1, init_cond=((macro_index_part1 == 0) and (pair_offset_part1 == 0)))
                            key_acc_part1 = pl.tile.matmul_acc(key_acc_part1, hidden_micro_odd_part1, weight_micro_odd_part1)
                    raw_key_part1 = pl.store(key_acc_part1, [row_owner_part1 * 128, 0], raw_key_part1)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="key_rope", deps=[producer_tid_part0, producer_tid_part1]) as consumer_tid:
                for head_index in pl.range(2):
                    for vector_offset in pl.pipeline(0, 512, 32, stage=3):
                        vector_row_start = row_group_start + vector_offset
                        key_head_part0_loaded = pl.load(raw_key_part0, [vector_offset, head_index * DH], [32, DH], target_memory=pl.MemorySpace.Vec)
                        key_head_part0 = pl.cast(key_head_part0_loaded, pl.FP32)
                        key_head_part1_loaded = pl.load(raw_key_part1, [vector_offset, head_index * DH], [32, DH], target_memory=pl.MemorySpace.Vec)
                        key_head_part1 = pl.cast(key_head_part1_loaded, pl.FP32)
                        key_head = pl.add(key_head_part0, key_head_part1)
                        key_low = pl.tile.slice(key_head, [32, DH // 2], [0, 0])
                        key_high = pl.tile.slice(key_head, [32, DH // 2], [0, DH // 2])
                        cosine_pair_tile = pl.load(cosine_pairs, [vector_row_start // 2, 0], [16, DH], target_memory=pl.MemorySpace.Vec)
                        cosine_tile = pl.tile.reshape(cosine_pair_tile, [32, DH // 2])
                        sine_pair_tile = pl.load(sine_pairs, [vector_row_start // 2, 0], [16, DH], target_memory=pl.MemorySpace.Vec)
                        sine_tile = pl.tile.reshape(sine_pair_tile, [32, DH // 2])
                        rotated_low = pl.sub(pl.mul(key_low, cosine_tile), pl.mul(key_high, sine_tile))
                        rotated_high = pl.add(pl.mul(key_high, cosine_tile), pl.mul(key_low, sine_tile))
                        rotated_head = pl.concat(rotated_low, rotated_high)
                        key_out = pl.store(rotated_head, [vector_row_start, head_start + head_index * DH], key_out)
    for row_start_tail in pl.parallel(complete_group_rows, ROWS, 128):
        for head_start_tail in pl.parallel(0, N, DH * 2):
            raw_key_tail = pl.create_tensor([128, DH * 2], dtype=pl.FP32)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="tail_key_producer") as producer_tid_tail:
                key_acc_tail = pl.tile.create([128, DH * 2], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc)
                for macro_index_tail in pl.pipeline(D // 256, stage=2):
                    reduction_start_tail = macro_index_tail * 256
                    hidden_panel_tail = pl.load(hidden, [row_start_tail, reduction_start_tail], [128, 256], target_memory=pl.MemorySpace.Mat)
                    weight_panel_tail = pl.load(weight_k, [reduction_start_tail, head_start_tail], [256, DH * 2], target_memory=pl.MemorySpace.Mat)
                    for pair_offset_tail in pl.range(0, 256, 64):
                        hidden_micro_even_tail = pl.tile.extract(hidden_panel_tail, 0, pair_offset_tail, [128, 32], target_memory=pl.MemorySpace.Left)
                        weight_micro_even_tail = pl.tile.extract(weight_panel_tail, pair_offset_tail, 0, [32, DH * 2], target_memory=pl.MemorySpace.Right)
                        hidden_micro_odd_tail = pl.tile.extract(hidden_panel_tail, 0, pair_offset_tail + 32, [128, 32], target_memory=pl.MemorySpace.Left)
                        weight_micro_odd_tail = pl.tile.extract(weight_panel_tail, pair_offset_tail + 32, 0, [32, DH * 2], target_memory=pl.MemorySpace.Right)
                        key_acc_tail = pl.tile.matmul_acc(key_acc_tail, hidden_micro_even_tail, weight_micro_even_tail, init_cond=((macro_index_tail == 0) and (pair_offset_tail == 0)))
                        key_acc_tail = pl.tile.matmul_acc(key_acc_tail, hidden_micro_odd_tail, weight_micro_odd_tail)
                raw_key_tail = pl.store(key_acc_tail, [0, 0], raw_key_tail)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="tail_key_rope", deps=[producer_tid_tail]) as consumer_tid_tail:
                for head_index_tail in pl.range(2):
                    for vector_offset_tail in pl.pipeline(0, 128, 32, stage=3):
                        vector_row_start_tail = row_start_tail + vector_offset_tail
                        key_head_tail = pl.load(raw_key_tail, [vector_offset_tail, head_index_tail * DH], [32, DH], target_memory=pl.MemorySpace.Vec)
                        key_low_tail = pl.tile.slice(key_head_tail, [32, DH // 2], [0, 0])
                        key_high_tail = pl.tile.slice(key_head_tail, [32, DH // 2], [0, DH // 2])
                        cosine_pair_tile_tail = pl.load(cosine_pairs, [vector_row_start_tail // 2, 0], [16, DH], target_memory=pl.MemorySpace.Vec)
                        cosine_tile_tail = pl.tile.reshape(cosine_pair_tile_tail, [32, DH // 2])
                        sine_pair_tile_tail = pl.load(sine_pairs, [vector_row_start_tail // 2, 0], [16, DH], target_memory=pl.MemorySpace.Vec)
                        sine_tile_tail = pl.tile.reshape(sine_pair_tile_tail, [32, DH // 2])
                        rotated_low_tail = pl.sub(pl.mul(key_low_tail, cosine_tile_tail), pl.mul(key_high_tail, sine_tile_tail))
                        rotated_high_tail = pl.add(pl.mul(key_high_tail, cosine_tile_tail), pl.mul(key_low_tail, sine_tile_tail))
                        rotated_head_tail = pl.concat(rotated_low_tail, rotated_high_tail)
                        key_out = pl.store(rotated_head_tail, [vector_row_start_tail, head_start_tail + head_index_tail * DH], key_out)
    for row_start in pl.parallel(0, ROWS, 128):
        for head_start in pl.parallel(0, N, DH * 2):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="separate_value"):
                value_acc = pl.tile.create([128, DH * 2], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc)
                for macro_index in pl.pipeline(D // 256, stage=2):
                    reduction_start = macro_index * 256
                    hidden_panel = pl.load(hidden, [row_start, reduction_start], [128, 256], target_memory=pl.MemorySpace.Mat)
                    weight_panel = pl.load(weight_v, [reduction_start, head_start], [256, DH * 2], target_memory=pl.MemorySpace.Mat)
                    for pair_offset in pl.range(0, 256, 64):
                        hidden_micro_even = pl.tile.extract(hidden_panel, 0, pair_offset, [128, 32], target_memory=pl.MemorySpace.Left)
                        weight_micro_even = pl.tile.extract(weight_panel, pair_offset, 0, [32, DH * 2], target_memory=pl.MemorySpace.Right)
                        hidden_micro_odd = pl.tile.extract(hidden_panel, 0, pair_offset + 32, [128, 32], target_memory=pl.MemorySpace.Left)
                        weight_micro_odd = pl.tile.extract(weight_panel, pair_offset + 32, 0, [32, DH * 2], target_memory=pl.MemorySpace.Right)
                        value_acc = pl.tile.matmul_acc(value_acc, hidden_micro_even, weight_micro_even, init_cond=((macro_index == 0) and (pair_offset == 0)))
                        value_acc = pl.tile.matmul_acc(value_acc, hidden_micro_odd, weight_micro_odd)
                value_out = pl.store(value_acc, [row_start, head_start], value_out)
    return key_out, value_out


@pl.jit
def swa_replay_fp32(
    hidden: pl.Tensor[[ROWS, D], pl.FP32],
    weight_k: pl.Tensor[[D, N], pl.FP32],
    weight_v: pl.Tensor[[D, N], pl.FP32],
    cosine: pl.Tensor[[ROWS, DH // 2], pl.FP32],
    sine: pl.Tensor[[ROWS, DH // 2], pl.FP32],
    key_out: pl.Out[pl.Tensor[[ROWS, N], pl.FP32]],
    value_out: pl.Out[pl.Tensor[[ROWS, N], pl.FP32]],
):
    for row_start in pl.parallel(0, ROWS, 32):
        for head_start in pl.parallel(0, N, DH):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="paired_kv_rope"):
                key_acc = pl.create_tensor([32, DH], dtype=pl.FP32)
                value_acc = pl.create_tensor([32, DH], dtype=pl.FP32)
                for reduction_index in pl.pipeline(D // 128, stage=2):
                    reduction_start = reduction_index * 128
                    hidden_tile = pl.slice(hidden, [32, 128], [row_start, reduction_start])
                    key_weight_tile = pl.slice(weight_k, [128, DH], [reduction_start, head_start])
                    value_weight_tile = pl.slice(weight_v, [128, DH], [reduction_start, head_start])
                    key_acc = pl.matmul_acc(key_acc, hidden_tile, key_weight_tile, init_cond=(reduction_index == 0))
                    value_acc = pl.matmul_acc(value_acc, hidden_tile, value_weight_tile, init_cond=(reduction_index == 0))
                key_low = pl.slice(key_acc, [32, DH // 2], [0, 0])
                key_high = pl.slice(key_acc, [32, DH // 2], [0, DH // 2])
                cosine_tile = pl.slice(cosine, [32, DH // 2], [row_start, 0])
                sine_tile = pl.slice(sine, [32, DH // 2], [row_start, 0])
                rotated_low = pl.sub(pl.mul(key_low, cosine_tile), pl.mul(key_high, sine_tile))
                rotated_high = pl.add(pl.mul(key_high, cosine_tile), pl.mul(key_low, sine_tile))
                key_out[row_start:row_start + 32, head_start:head_start + DH // 2] = rotated_low
                key_out[row_start:row_start + 32, head_start + DH // 2:head_start + DH] = rotated_high
                value_out[row_start:row_start + 32, head_start:head_start + DH] = value_acc
    return key_out, value_out


def build(shape):
    import torch
    from golden import TensorSpec

    rows = shape["B"] * 128
    dtype_name = shape.get("dtype", "fp32")
    if dtype_name not in ("fp32", "bf16"):
        raise ValueError("dtype must be fp32 or bf16")
    input_dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float32

    def normal(*dimensions, scale=0.02):
        return lambda: torch.randn(*dimensions) * scale

    def positions():
        return torch.rand(rows, DH // 2) * 100.0

    specs = [
        TensorSpec("hidden", [rows, D], input_dtype, init_value=normal(rows, D, scale=0.5)),
        TensorSpec("weight_k", [D, N], input_dtype, init_value=normal(D, N)),
        TensorSpec("weight_v", [D, N], input_dtype, init_value=normal(D, N)),
        TensorSpec("cosine", [rows, DH // 2], torch.float32, init_value=lambda: torch.cos(positions())),
        TensorSpec("sine", [rows, DH // 2], torch.float32, init_value=lambda: torch.sin(positions())),
        TensorSpec("key_out", [rows, N], torch.float32),
        TensorSpec("value_out", [rows, N], torch.float32),
    ]

    def golden(tensors):
        projected_key = (tensors["hidden"].float() @ tensors["weight_k"].float()).view(rows, 8, DH)
        key_low = projected_key[..., :DH // 2]
        key_high = projected_key[..., DH // 2:]
        cosine = tensors["cosine"].float().view(rows, 1, DH // 2)
        sine = tensors["sine"].float().view(rows, 1, DH // 2)
        tensors["key_out"][:] = torch.cat([key_low * cosine - key_high * sine, key_high * cosine + key_low * sine], dim=-1).view(rows, N)
        tensors["value_out"][:] = tensors["hidden"].float() @ tensors["weight_v"].float()

    if dtype_name == "bf16":
        return swa_replay_bf16, specs, golden, 2e-2, 2e-2
    return swa_replay_fp32, specs, golden, 1e-4, 1e-4
