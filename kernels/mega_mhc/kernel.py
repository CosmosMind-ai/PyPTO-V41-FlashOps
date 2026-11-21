import pypto.language as pl
FUSED_OWNER_LIMIT = 24

M = 4
D = 256
M3 = 2 * M + M * M
SINK = 10
PRE_EPS = 1e-06
POST_MULT = 2.0
RMS_EPS = 1e-06
T = pl.dynamic('T')
FD = M * D



@pl.jit
def mega_mhc_plane_8(x: pl.Tensor[[T, D], pl.FP32], residual: pl.InOut[pl.Tensor[[T, FD], pl.FP32]], input_mixes: pl.Tensor[[T, M3], pl.FP32], fn: pl.Tensor[[M3, FD], pl.FP32], normw: pl.Tensor[[1, M3], pl.FP32], mhc_scale: pl.Tensor[[3], pl.FP32], mhc_base: pl.Tensor[[1, M3], pl.FP32], next_mixes: pl.Out[pl.Tensor[[T, M3], pl.FP32]], h_next: pl.Out[pl.Tensor[[T, D], pl.FP32]]):
    coefficient_planes = pl.create_tensor([M3, T], dtype=pl.FP32)
    for task_index in pl.spmd(T // 8, name_hint='mhc_plane_8'):
        token_start = task_index * 8
        scale_pre = pl.read(mhc_scale, [0])
        scale_post = pl.read(mhc_scale, [1])
        scale_comb = pl.read(mhc_scale, [2])
        scaled_post = pl.mul(input_mixes[token_start:token_start + 8, 0:M3], scale_post)
        scaled_comb = pl.mul(input_mixes[token_start:token_start + 8, 0:M3], scale_comb)
        coefficient_indices = pl.cast(pl.arange(0, [1, M3], dtype=pl.INT32), pl.FP32)
        post_selector0 = pl.maximum(pl.add(pl.neg(pl.abs(pl.sub(coefficient_indices, 4.0))), 1.0), 0.0)
        post_raw0 = pl.add(pl.row_sum(pl.col_expand_mul(scaled_post, post_selector0)), pl.read(mhc_base, [0, 4]))
        post0 = pl.mul(pl.recip(pl.add(pl.exp(pl.neg(post_raw0)), 1.0)), POST_MULT)
        post_selector1 = pl.maximum(pl.add(pl.neg(pl.abs(pl.sub(coefficient_indices, 5.0))), 1.0), 0.0)
        post_raw1 = pl.add(pl.row_sum(pl.col_expand_mul(scaled_post, post_selector1)), pl.read(mhc_base, [0, 5]))
        post1 = pl.mul(pl.recip(pl.add(pl.exp(pl.neg(post_raw1)), 1.0)), POST_MULT)
        post_selector2 = pl.maximum(pl.add(pl.neg(pl.abs(pl.sub(coefficient_indices, 6.0))), 1.0), 0.0)
        post_raw2 = pl.add(pl.row_sum(pl.col_expand_mul(scaled_post, post_selector2)), pl.read(mhc_base, [0, 6]))
        post2 = pl.mul(pl.recip(pl.add(pl.exp(pl.neg(post_raw2)), 1.0)), POST_MULT)
        post_selector3 = pl.maximum(pl.add(pl.neg(pl.abs(pl.sub(coefficient_indices, 7.0))), 1.0), 0.0)
        post_raw3 = pl.add(pl.row_sum(pl.col_expand_mul(scaled_post, post_selector3)), pl.read(mhc_base, [0, 7]))
        post3 = pl.mul(pl.recip(pl.add(pl.exp(pl.neg(post_raw3)), 1.0)), POST_MULT)
        comb_selector00 = pl.maximum(pl.add(pl.neg(pl.abs(pl.sub(coefficient_indices, 8.0))), 1.0), 0.0)
        comb00 = pl.add(pl.row_sum(pl.col_expand_mul(scaled_comb, comb_selector00)), pl.read(mhc_base, [0, 8]))
        comb_selector01 = pl.maximum(pl.add(pl.neg(pl.abs(pl.sub(coefficient_indices, 9.0))), 1.0), 0.0)
        comb01 = pl.add(pl.row_sum(pl.col_expand_mul(scaled_comb, comb_selector01)), pl.read(mhc_base, [0, 9]))
        comb_selector02 = pl.maximum(pl.add(pl.neg(pl.abs(pl.sub(coefficient_indices, 10.0))), 1.0), 0.0)
        comb02 = pl.add(pl.row_sum(pl.col_expand_mul(scaled_comb, comb_selector02)), pl.read(mhc_base, [0, 10]))
        comb_selector03 = pl.maximum(pl.add(pl.neg(pl.abs(pl.sub(coefficient_indices, 11.0))), 1.0), 0.0)
        comb03 = pl.add(pl.row_sum(pl.col_expand_mul(scaled_comb, comb_selector03)), pl.read(mhc_base, [0, 11]))
        row_max0 = pl.maximum(pl.maximum(comb00, comb01), pl.maximum(comb02, comb03))
        comb00 = pl.exp(pl.sub(comb00, row_max0))
        comb01 = pl.exp(pl.sub(comb01, row_max0))
        comb02 = pl.exp(pl.sub(comb02, row_max0))
        comb03 = pl.exp(pl.sub(comb03, row_max0))
        row_sum0 = pl.add(pl.add(comb00, comb01), pl.add(comb02, comb03))
        comb00 = pl.add(pl.div(comb00, row_sum0), PRE_EPS)
        comb01 = pl.add(pl.div(comb01, row_sum0), PRE_EPS)
        comb02 = pl.add(pl.div(comb02, row_sum0), PRE_EPS)
        comb03 = pl.add(pl.div(comb03, row_sum0), PRE_EPS)
        comb_selector10 = pl.maximum(pl.add(pl.neg(pl.abs(pl.sub(coefficient_indices, 12.0))), 1.0), 0.0)
        comb10 = pl.add(pl.row_sum(pl.col_expand_mul(scaled_comb, comb_selector10)), pl.read(mhc_base, [0, 12]))
        comb_selector11 = pl.maximum(pl.add(pl.neg(pl.abs(pl.sub(coefficient_indices, 13.0))), 1.0), 0.0)
        comb11 = pl.add(pl.row_sum(pl.col_expand_mul(scaled_comb, comb_selector11)), pl.read(mhc_base, [0, 13]))
        comb_selector12 = pl.maximum(pl.add(pl.neg(pl.abs(pl.sub(coefficient_indices, 14.0))), 1.0), 0.0)
        comb12 = pl.add(pl.row_sum(pl.col_expand_mul(scaled_comb, comb_selector12)), pl.read(mhc_base, [0, 14]))
        comb_selector13 = pl.maximum(pl.add(pl.neg(pl.abs(pl.sub(coefficient_indices, 15.0))), 1.0), 0.0)
        comb13 = pl.add(pl.row_sum(pl.col_expand_mul(scaled_comb, comb_selector13)), pl.read(mhc_base, [0, 15]))
        row_max1 = pl.maximum(pl.maximum(comb10, comb11), pl.maximum(comb12, comb13))
        comb10 = pl.exp(pl.sub(comb10, row_max1))
        comb11 = pl.exp(pl.sub(comb11, row_max1))
        comb12 = pl.exp(pl.sub(comb12, row_max1))
        comb13 = pl.exp(pl.sub(comb13, row_max1))
        row_sum1 = pl.add(pl.add(comb10, comb11), pl.add(comb12, comb13))
        comb10 = pl.add(pl.div(comb10, row_sum1), PRE_EPS)
        comb11 = pl.add(pl.div(comb11, row_sum1), PRE_EPS)
        comb12 = pl.add(pl.div(comb12, row_sum1), PRE_EPS)
        comb13 = pl.add(pl.div(comb13, row_sum1), PRE_EPS)
        comb_selector20 = pl.maximum(pl.add(pl.neg(pl.abs(pl.sub(coefficient_indices, 16.0))), 1.0), 0.0)
        comb20 = pl.add(pl.row_sum(pl.col_expand_mul(scaled_comb, comb_selector20)), pl.read(mhc_base, [0, 16]))
        comb_selector21 = pl.maximum(pl.add(pl.neg(pl.abs(pl.sub(coefficient_indices, 17.0))), 1.0), 0.0)
        comb21 = pl.add(pl.row_sum(pl.col_expand_mul(scaled_comb, comb_selector21)), pl.read(mhc_base, [0, 17]))
        comb_selector22 = pl.maximum(pl.add(pl.neg(pl.abs(pl.sub(coefficient_indices, 18.0))), 1.0), 0.0)
        comb22 = pl.add(pl.row_sum(pl.col_expand_mul(scaled_comb, comb_selector22)), pl.read(mhc_base, [0, 18]))
        comb_selector23 = pl.maximum(pl.add(pl.neg(pl.abs(pl.sub(coefficient_indices, 19.0))), 1.0), 0.0)
        comb23 = pl.add(pl.row_sum(pl.col_expand_mul(scaled_comb, comb_selector23)), pl.read(mhc_base, [0, 19]))
        row_max2 = pl.maximum(pl.maximum(comb20, comb21), pl.maximum(comb22, comb23))
        comb20 = pl.exp(pl.sub(comb20, row_max2))
        comb21 = pl.exp(pl.sub(comb21, row_max2))
        comb22 = pl.exp(pl.sub(comb22, row_max2))
        comb23 = pl.exp(pl.sub(comb23, row_max2))
        row_sum2 = pl.add(pl.add(comb20, comb21), pl.add(comb22, comb23))
        comb20 = pl.add(pl.div(comb20, row_sum2), PRE_EPS)
        comb21 = pl.add(pl.div(comb21, row_sum2), PRE_EPS)
        comb22 = pl.add(pl.div(comb22, row_sum2), PRE_EPS)
        comb23 = pl.add(pl.div(comb23, row_sum2), PRE_EPS)
        comb_selector30 = pl.maximum(pl.add(pl.neg(pl.abs(pl.sub(coefficient_indices, 20.0))), 1.0), 0.0)
        comb30 = pl.add(pl.row_sum(pl.col_expand_mul(scaled_comb, comb_selector30)), pl.read(mhc_base, [0, 20]))
        comb_selector31 = pl.maximum(pl.add(pl.neg(pl.abs(pl.sub(coefficient_indices, 21.0))), 1.0), 0.0)
        comb31 = pl.add(pl.row_sum(pl.col_expand_mul(scaled_comb, comb_selector31)), pl.read(mhc_base, [0, 21]))
        comb_selector32 = pl.maximum(pl.add(pl.neg(pl.abs(pl.sub(coefficient_indices, 22.0))), 1.0), 0.0)
        comb32 = pl.add(pl.row_sum(pl.col_expand_mul(scaled_comb, comb_selector32)), pl.read(mhc_base, [0, 22]))
        comb_selector33 = pl.maximum(pl.add(pl.neg(pl.abs(pl.sub(coefficient_indices, 23.0))), 1.0), 0.0)
        comb33 = pl.add(pl.row_sum(pl.col_expand_mul(scaled_comb, comb_selector33)), pl.read(mhc_base, [0, 23]))
        row_max3 = pl.maximum(pl.maximum(comb30, comb31), pl.maximum(comb32, comb33))
        comb30 = pl.exp(pl.sub(comb30, row_max3))
        comb31 = pl.exp(pl.sub(comb31, row_max3))
        comb32 = pl.exp(pl.sub(comb32, row_max3))
        comb33 = pl.exp(pl.sub(comb33, row_max3))
        row_sum3 = pl.add(pl.add(comb30, comb31), pl.add(comb32, comb33))
        comb30 = pl.add(pl.div(comb30, row_sum3), PRE_EPS)
        comb31 = pl.add(pl.div(comb31, row_sum3), PRE_EPS)
        comb32 = pl.add(pl.div(comb32, row_sum3), PRE_EPS)
        comb33 = pl.add(pl.div(comb33, row_sum3), PRE_EPS)
        column_sum0 = pl.add(pl.add(pl.add(comb00, comb10), pl.add(comb20, comb30)), PRE_EPS)
        comb00 = pl.div(comb00, column_sum0)
        comb10 = pl.div(comb10, column_sum0)
        comb20 = pl.div(comb20, column_sum0)
        comb30 = pl.div(comb30, column_sum0)
        column_sum1 = pl.add(pl.add(pl.add(comb01, comb11), pl.add(comb21, comb31)), PRE_EPS)
        comb01 = pl.div(comb01, column_sum1)
        comb11 = pl.div(comb11, column_sum1)
        comb21 = pl.div(comb21, column_sum1)
        comb31 = pl.div(comb31, column_sum1)
        column_sum2 = pl.add(pl.add(pl.add(comb02, comb12), pl.add(comb22, comb32)), PRE_EPS)
        comb02 = pl.div(comb02, column_sum2)
        comb12 = pl.div(comb12, column_sum2)
        comb22 = pl.div(comb22, column_sum2)
        comb32 = pl.div(comb32, column_sum2)
        column_sum3 = pl.add(pl.add(pl.add(comb03, comb13), pl.add(comb23, comb33)), PRE_EPS)
        comb03 = pl.div(comb03, column_sum3)
        comb13 = pl.div(comb13, column_sum3)
        comb23 = pl.div(comb23, column_sum3)
        comb33 = pl.div(comb33, column_sum3)
        for iteration in pl.range(SINK - 1):
            row_sum0 = pl.add(pl.add(pl.add(comb00, comb01), pl.add(comb02, comb03)), PRE_EPS)
            comb00 = pl.div(comb00, row_sum0)
            comb01 = pl.div(comb01, row_sum0)
            comb02 = pl.div(comb02, row_sum0)
            comb03 = pl.div(comb03, row_sum0)
            row_sum1 = pl.add(pl.add(pl.add(comb10, comb11), pl.add(comb12, comb13)), PRE_EPS)
            comb10 = pl.div(comb10, row_sum1)
            comb11 = pl.div(comb11, row_sum1)
            comb12 = pl.div(comb12, row_sum1)
            comb13 = pl.div(comb13, row_sum1)
            row_sum2 = pl.add(pl.add(pl.add(comb20, comb21), pl.add(comb22, comb23)), PRE_EPS)
            comb20 = pl.div(comb20, row_sum2)
            comb21 = pl.div(comb21, row_sum2)
            comb22 = pl.div(comb22, row_sum2)
            comb23 = pl.div(comb23, row_sum2)
            row_sum3 = pl.add(pl.add(pl.add(comb30, comb31), pl.add(comb32, comb33)), PRE_EPS)
            comb30 = pl.div(comb30, row_sum3)
            comb31 = pl.div(comb31, row_sum3)
            comb32 = pl.div(comb32, row_sum3)
            comb33 = pl.div(comb33, row_sum3)
            column_sum0 = pl.add(pl.add(pl.add(comb00, comb10), pl.add(comb20, comb30)), PRE_EPS)
            comb00 = pl.div(comb00, column_sum0)
            comb10 = pl.div(comb10, column_sum0)
            comb20 = pl.div(comb20, column_sum0)
            comb30 = pl.div(comb30, column_sum0)
            column_sum1 = pl.add(pl.add(pl.add(comb01, comb11), pl.add(comb21, comb31)), PRE_EPS)
            comb01 = pl.div(comb01, column_sum1)
            comb11 = pl.div(comb11, column_sum1)
            comb21 = pl.div(comb21, column_sum1)
            comb31 = pl.div(comb31, column_sum1)
            column_sum2 = pl.add(pl.add(pl.add(comb02, comb12), pl.add(comb22, comb32)), PRE_EPS)
            comb02 = pl.div(comb02, column_sum2)
            comb12 = pl.div(comb12, column_sum2)
            comb22 = pl.div(comb22, column_sum2)
            comb32 = pl.div(comb32, column_sum2)
            column_sum3 = pl.add(pl.add(pl.add(comb03, comb13), pl.add(comb23, comb33)), PRE_EPS)
            comb03 = pl.div(comb03, column_sum3)
            comb13 = pl.div(comb13, column_sum3)
            comb23 = pl.div(comb23, column_sum3)
            comb33 = pl.div(comb33, column_sum3)
        input_tile = x[token_start:token_start + 8, 0:D]
        original0 = residual[token_start:token_start + 8, 0 * D:1 * D]
        original1 = residual[token_start:token_start + 8, 1 * D:2 * D]
        original2 = residual[token_start:token_start + 8, 2 * D:3 * D]
        original3 = residual[token_start:token_start + 8, 3 * D:4 * D]
        updated0 = pl.row_expand_mul(input_tile, post0)
        updated0 = pl.add(updated0, pl.row_expand_mul(original0, comb00))
        updated0 = pl.add(updated0, pl.row_expand_mul(original1, comb01))
        updated0 = pl.add(updated0, pl.row_expand_mul(original2, comb02))
        updated0 = pl.add(updated0, pl.row_expand_mul(original3, comb03))
        updated1 = pl.row_expand_mul(input_tile, post1)
        updated1 = pl.add(updated1, pl.row_expand_mul(original0, comb10))
        updated1 = pl.add(updated1, pl.row_expand_mul(original1, comb11))
        updated1 = pl.add(updated1, pl.row_expand_mul(original2, comb12))
        updated1 = pl.add(updated1, pl.row_expand_mul(original3, comb13))
        updated2 = pl.row_expand_mul(input_tile, post2)
        updated2 = pl.add(updated2, pl.row_expand_mul(original0, comb20))
        updated2 = pl.add(updated2, pl.row_expand_mul(original1, comb21))
        updated2 = pl.add(updated2, pl.row_expand_mul(original2, comb22))
        updated2 = pl.add(updated2, pl.row_expand_mul(original3, comb23))
        updated3 = pl.row_expand_mul(input_tile, post3)
        updated3 = pl.add(updated3, pl.row_expand_mul(original0, comb30))
        updated3 = pl.add(updated3, pl.row_expand_mul(original1, comb31))
        updated3 = pl.add(updated3, pl.row_expand_mul(original2, comb32))
        updated3 = pl.add(updated3, pl.row_expand_mul(original3, comb33))
        square_sum = pl.row_sum(pl.add(pl.add(pl.mul(updated0, updated0), pl.mul(updated1, updated1)), pl.add(pl.mul(updated2, updated2), pl.mul(updated3, updated3))))
        mean_square = pl.add(pl.mul(square_sum, 1.0 / FD), RMS_EPS)
        inverse_rms = pl.rsqrt(mean_square)
        inverse_rms = pl.mul(inverse_rms, pl.add(pl.mul(pl.mul(mean_square, pl.mul(inverse_rms, inverse_rms)), -0.5), 1.5))
        hidden = pl.full([8, D], dtype=pl.FP32, value=0.0)
        for coefficient in pl.range(M3):
            norm_weight = pl.read(normw, [0, coefficient])
            weighted_fn0 = pl.mul(fn[coefficient:coefficient + 1, 0 * D:1 * D], norm_weight)
            partial0 = pl.row_sum(pl.col_expand_mul(updated0, weighted_fn0))
            weighted_fn1 = pl.mul(fn[coefficient:coefficient + 1, 1 * D:2 * D], norm_weight)
            partial1 = pl.row_sum(pl.col_expand_mul(updated1, weighted_fn1))
            weighted_fn2 = pl.mul(fn[coefficient:coefficient + 1, 2 * D:3 * D], norm_weight)
            partial2 = pl.row_sum(pl.col_expand_mul(updated2, weighted_fn2))
            weighted_fn3 = pl.mul(fn[coefficient:coefficient + 1, 3 * D:4 * D], norm_weight)
            partial3 = pl.row_sum(pl.col_expand_mul(updated3, weighted_fn3))
            next_coefficient = pl.mul(pl.add(pl.add(partial0, partial1), pl.add(partial2, partial3)), inverse_rms)
            coefficient_planes[coefficient:coefficient + 1, token_start:token_start + 8] = pl.reshape(next_coefficient, [1, 8])
            if coefficient < M:
                pre_raw = pl.add(pl.mul(next_coefficient, scale_pre), pl.read(mhc_base, [0, coefficient]))
                next_pre = pl.add(pl.recip(pl.add(pl.exp(pl.neg(pre_raw)), 1.0)), PRE_EPS)
                if coefficient == 0:
                    hidden = pl.add(hidden, pl.row_expand_mul(updated0, next_pre))
                elif coefficient == 1:
                    hidden = pl.add(hidden, pl.row_expand_mul(updated1, next_pre))
                elif coefficient == 2:
                    hidden = pl.add(hidden, pl.row_expand_mul(updated2, next_pre))
                elif coefficient == 3:
                    hidden = pl.add(hidden, pl.row_expand_mul(updated3, next_pre))
        next_mixes[token_start:token_start + 8, 0:M3] = pl.transpose(coefficient_planes[0:M3, token_start:token_start + 8], 0, 1)
        h_next[token_start:token_start + 8, 0:D] = hidden
        residual[token_start:token_start + 8, 0 * D:1 * D] = updated0
        residual[token_start:token_start + 8, 1 * D:2 * D] = updated1
        residual[token_start:token_start + 8, 2 * D:3 * D] = updated2
        residual[token_start:token_start + 8, 3 * D:4 * D] = updated3
    return h_next

@pl.jit
def mega_mhc_bridge_tail_bf16_8(x: pl.Tensor[[T, D], pl.BF16], residual: pl.InOut[pl.Tensor[[T, FD], pl.BF16]], input_mixes: pl.Tensor[[T, M3], pl.FP32], fn: pl.Tensor[[M3, FD], pl.FP32], normw: pl.Tensor[[1, M3], pl.FP32], mhc_scale: pl.Tensor[[3], pl.FP32], mhc_base: pl.Tensor[[1, M3], pl.FP32], next_mixes: pl.Out[pl.Tensor[[T, M3], pl.FP32]], h_next: pl.Out[pl.Tensor[[T, D], pl.BF16]]):
    updated_fp32 = pl.create_tensor([T, FD], dtype=pl.FP32)
    rms_scratch = pl.create_tensor([1, T], dtype=pl.FP32)
    for task_index in pl.spmd(T // 8, name_hint='mhc_plane_8'):
        token_start = task_index * 8
        scale_pre = pl.read(mhc_scale, [0])
        scale_post = pl.read(mhc_scale, [1])
        scale_comb = pl.read(mhc_scale, [2])
        mix_planes = pl.transpose(input_mixes[token_start:token_start + 8, 0:M3], 0, 1)
        post_raw0 = pl.add(pl.mul(pl.reshape(mix_planes[4:5, 0:8], [8, 1]), scale_post), pl.read(mhc_base, [0, 4]))
        post0 = pl.mul(pl.recip(pl.add(pl.exp(pl.neg(post_raw0)), 1.0)), POST_MULT)
        post_raw1 = pl.add(pl.mul(pl.reshape(mix_planes[5:6, 0:8], [8, 1]), scale_post), pl.read(mhc_base, [0, 5]))
        post1 = pl.mul(pl.recip(pl.add(pl.exp(pl.neg(post_raw1)), 1.0)), POST_MULT)
        post_raw2 = pl.add(pl.mul(pl.reshape(mix_planes[6:7, 0:8], [8, 1]), scale_post), pl.read(mhc_base, [0, 6]))
        post2 = pl.mul(pl.recip(pl.add(pl.exp(pl.neg(post_raw2)), 1.0)), POST_MULT)
        post_raw3 = pl.add(pl.mul(pl.reshape(mix_planes[7:8, 0:8], [8, 1]), scale_post), pl.read(mhc_base, [0, 7]))
        post3 = pl.mul(pl.recip(pl.add(pl.exp(pl.neg(post_raw3)), 1.0)), POST_MULT)
        comb00 = pl.add(pl.mul(pl.reshape(mix_planes[8:9, 0:8], [8, 1]), scale_comb), pl.read(mhc_base, [0, 8]))
        comb01 = pl.add(pl.mul(pl.reshape(mix_planes[9:10, 0:8], [8, 1]), scale_comb), pl.read(mhc_base, [0, 9]))
        comb02 = pl.add(pl.mul(pl.reshape(mix_planes[10:11, 0:8], [8, 1]), scale_comb), pl.read(mhc_base, [0, 10]))
        comb03 = pl.add(pl.mul(pl.reshape(mix_planes[11:12, 0:8], [8, 1]), scale_comb), pl.read(mhc_base, [0, 11]))
        row_max0 = pl.maximum(pl.maximum(comb00, comb01), pl.maximum(comb02, comb03))
        comb00 = pl.exp(pl.sub(comb00, row_max0))
        comb01 = pl.exp(pl.sub(comb01, row_max0))
        comb02 = pl.exp(pl.sub(comb02, row_max0))
        comb03 = pl.exp(pl.sub(comb03, row_max0))
        row_sum0 = pl.add(pl.add(comb00, comb01), pl.add(comb02, comb03))
        comb00 = pl.add(pl.div(comb00, row_sum0), PRE_EPS)
        comb01 = pl.add(pl.div(comb01, row_sum0), PRE_EPS)
        comb02 = pl.add(pl.div(comb02, row_sum0), PRE_EPS)
        comb03 = pl.add(pl.div(comb03, row_sum0), PRE_EPS)
        comb10 = pl.add(pl.mul(pl.reshape(mix_planes[12:13, 0:8], [8, 1]), scale_comb), pl.read(mhc_base, [0, 12]))
        comb11 = pl.add(pl.mul(pl.reshape(mix_planes[13:14, 0:8], [8, 1]), scale_comb), pl.read(mhc_base, [0, 13]))
        comb12 = pl.add(pl.mul(pl.reshape(mix_planes[14:15, 0:8], [8, 1]), scale_comb), pl.read(mhc_base, [0, 14]))
        comb13 = pl.add(pl.mul(pl.reshape(mix_planes[15:16, 0:8], [8, 1]), scale_comb), pl.read(mhc_base, [0, 15]))
        row_max1 = pl.maximum(pl.maximum(comb10, comb11), pl.maximum(comb12, comb13))
        comb10 = pl.exp(pl.sub(comb10, row_max1))
        comb11 = pl.exp(pl.sub(comb11, row_max1))
        comb12 = pl.exp(pl.sub(comb12, row_max1))
        comb13 = pl.exp(pl.sub(comb13, row_max1))
        row_sum1 = pl.add(pl.add(comb10, comb11), pl.add(comb12, comb13))
        comb10 = pl.add(pl.div(comb10, row_sum1), PRE_EPS)
        comb11 = pl.add(pl.div(comb11, row_sum1), PRE_EPS)
        comb12 = pl.add(pl.div(comb12, row_sum1), PRE_EPS)
        comb13 = pl.add(pl.div(comb13, row_sum1), PRE_EPS)
        comb20 = pl.add(pl.mul(pl.reshape(mix_planes[16:17, 0:8], [8, 1]), scale_comb), pl.read(mhc_base, [0, 16]))
        comb21 = pl.add(pl.mul(pl.reshape(mix_planes[17:18, 0:8], [8, 1]), scale_comb), pl.read(mhc_base, [0, 17]))
        comb22 = pl.add(pl.mul(pl.reshape(mix_planes[18:19, 0:8], [8, 1]), scale_comb), pl.read(mhc_base, [0, 18]))
        comb23 = pl.add(pl.mul(pl.reshape(mix_planes[19:20, 0:8], [8, 1]), scale_comb), pl.read(mhc_base, [0, 19]))
        row_max2 = pl.maximum(pl.maximum(comb20, comb21), pl.maximum(comb22, comb23))
        comb20 = pl.exp(pl.sub(comb20, row_max2))
        comb21 = pl.exp(pl.sub(comb21, row_max2))
        comb22 = pl.exp(pl.sub(comb22, row_max2))
        comb23 = pl.exp(pl.sub(comb23, row_max2))
        row_sum2 = pl.add(pl.add(comb20, comb21), pl.add(comb22, comb23))
        comb20 = pl.add(pl.div(comb20, row_sum2), PRE_EPS)
        comb21 = pl.add(pl.div(comb21, row_sum2), PRE_EPS)
        comb22 = pl.add(pl.div(comb22, row_sum2), PRE_EPS)
        comb23 = pl.add(pl.div(comb23, row_sum2), PRE_EPS)
        comb30 = pl.add(pl.mul(pl.reshape(mix_planes[20:21, 0:8], [8, 1]), scale_comb), pl.read(mhc_base, [0, 20]))
        comb31 = pl.add(pl.mul(pl.reshape(mix_planes[21:22, 0:8], [8, 1]), scale_comb), pl.read(mhc_base, [0, 21]))
        comb32 = pl.add(pl.mul(pl.reshape(mix_planes[22:23, 0:8], [8, 1]), scale_comb), pl.read(mhc_base, [0, 22]))
        comb33 = pl.add(pl.mul(pl.reshape(mix_planes[23:24, 0:8], [8, 1]), scale_comb), pl.read(mhc_base, [0, 23]))
        row_max3 = pl.maximum(pl.maximum(comb30, comb31), pl.maximum(comb32, comb33))
        comb30 = pl.exp(pl.sub(comb30, row_max3))
        comb31 = pl.exp(pl.sub(comb31, row_max3))
        comb32 = pl.exp(pl.sub(comb32, row_max3))
        comb33 = pl.exp(pl.sub(comb33, row_max3))
        row_sum3 = pl.add(pl.add(comb30, comb31), pl.add(comb32, comb33))
        comb30 = pl.add(pl.div(comb30, row_sum3), PRE_EPS)
        comb31 = pl.add(pl.div(comb31, row_sum3), PRE_EPS)
        comb32 = pl.add(pl.div(comb32, row_sum3), PRE_EPS)
        comb33 = pl.add(pl.div(comb33, row_sum3), PRE_EPS)
        column_sum0 = pl.add(pl.add(pl.add(comb00, comb10), pl.add(comb20, comb30)), PRE_EPS)
        comb00 = pl.div(comb00, column_sum0)
        comb10 = pl.div(comb10, column_sum0)
        comb20 = pl.div(comb20, column_sum0)
        comb30 = pl.div(comb30, column_sum0)
        column_sum1 = pl.add(pl.add(pl.add(comb01, comb11), pl.add(comb21, comb31)), PRE_EPS)
        comb01 = pl.div(comb01, column_sum1)
        comb11 = pl.div(comb11, column_sum1)
        comb21 = pl.div(comb21, column_sum1)
        comb31 = pl.div(comb31, column_sum1)
        column_sum2 = pl.add(pl.add(pl.add(comb02, comb12), pl.add(comb22, comb32)), PRE_EPS)
        comb02 = pl.div(comb02, column_sum2)
        comb12 = pl.div(comb12, column_sum2)
        comb22 = pl.div(comb22, column_sum2)
        comb32 = pl.div(comb32, column_sum2)
        column_sum3 = pl.add(pl.add(pl.add(comb03, comb13), pl.add(comb23, comb33)), PRE_EPS)
        comb03 = pl.div(comb03, column_sum3)
        comb13 = pl.div(comb13, column_sum3)
        comb23 = pl.div(comb23, column_sum3)
        comb33 = pl.div(comb33, column_sum3)
        for iteration in pl.range(SINK - 1):
            row_sum0 = pl.add(pl.add(pl.add(comb00, comb01), pl.add(comb02, comb03)), PRE_EPS)
            comb00 = pl.div(comb00, row_sum0)
            comb01 = pl.div(comb01, row_sum0)
            comb02 = pl.div(comb02, row_sum0)
            comb03 = pl.div(comb03, row_sum0)
            row_sum1 = pl.add(pl.add(pl.add(comb10, comb11), pl.add(comb12, comb13)), PRE_EPS)
            comb10 = pl.div(comb10, row_sum1)
            comb11 = pl.div(comb11, row_sum1)
            comb12 = pl.div(comb12, row_sum1)
            comb13 = pl.div(comb13, row_sum1)
            row_sum2 = pl.add(pl.add(pl.add(comb20, comb21), pl.add(comb22, comb23)), PRE_EPS)
            comb20 = pl.div(comb20, row_sum2)
            comb21 = pl.div(comb21, row_sum2)
            comb22 = pl.div(comb22, row_sum2)
            comb23 = pl.div(comb23, row_sum2)
            row_sum3 = pl.add(pl.add(pl.add(comb30, comb31), pl.add(comb32, comb33)), PRE_EPS)
            comb30 = pl.div(comb30, row_sum3)
            comb31 = pl.div(comb31, row_sum3)
            comb32 = pl.div(comb32, row_sum3)
            comb33 = pl.div(comb33, row_sum3)
            column_sum0 = pl.add(pl.add(pl.add(comb00, comb10), pl.add(comb20, comb30)), PRE_EPS)
            comb00 = pl.div(comb00, column_sum0)
            comb10 = pl.div(comb10, column_sum0)
            comb20 = pl.div(comb20, column_sum0)
            comb30 = pl.div(comb30, column_sum0)
            column_sum1 = pl.add(pl.add(pl.add(comb01, comb11), pl.add(comb21, comb31)), PRE_EPS)
            comb01 = pl.div(comb01, column_sum1)
            comb11 = pl.div(comb11, column_sum1)
            comb21 = pl.div(comb21, column_sum1)
            comb31 = pl.div(comb31, column_sum1)
            column_sum2 = pl.add(pl.add(pl.add(comb02, comb12), pl.add(comb22, comb32)), PRE_EPS)
            comb02 = pl.div(comb02, column_sum2)
            comb12 = pl.div(comb12, column_sum2)
            comb22 = pl.div(comb22, column_sum2)
            comb32 = pl.div(comb32, column_sum2)
            column_sum3 = pl.add(pl.add(pl.add(comb03, comb13), pl.add(comb23, comb33)), PRE_EPS)
            comb03 = pl.div(comb03, column_sum3)
            comb13 = pl.div(comb13, column_sum3)
            comb23 = pl.div(comb23, column_sum3)
            comb33 = pl.div(comb33, column_sum3)
        input_tile = pl.cast(x[token_start:token_start + 8, 0:D], pl.FP32)
        original0 = pl.cast(residual[token_start:token_start + 8, 0 * D:1 * D], pl.FP32)
        original1 = pl.cast(residual[token_start:token_start + 8, 1 * D:2 * D], pl.FP32)
        original2 = pl.cast(residual[token_start:token_start + 8, 2 * D:3 * D], pl.FP32)
        original3 = pl.cast(residual[token_start:token_start + 8, 3 * D:4 * D], pl.FP32)
        updated0 = pl.row_expand_mul(input_tile, post0)
        updated0 = pl.add(updated0, pl.row_expand_mul(original0, comb00))
        updated0 = pl.add(updated0, pl.row_expand_mul(original1, comb01))
        updated0 = pl.add(updated0, pl.row_expand_mul(original2, comb02))
        updated0 = pl.add(updated0, pl.row_expand_mul(original3, comb03))
        updated1 = pl.row_expand_mul(input_tile, post1)
        updated1 = pl.add(updated1, pl.row_expand_mul(original0, comb10))
        updated1 = pl.add(updated1, pl.row_expand_mul(original1, comb11))
        updated1 = pl.add(updated1, pl.row_expand_mul(original2, comb12))
        updated1 = pl.add(updated1, pl.row_expand_mul(original3, comb13))
        updated2 = pl.row_expand_mul(input_tile, post2)
        updated2 = pl.add(updated2, pl.row_expand_mul(original0, comb20))
        updated2 = pl.add(updated2, pl.row_expand_mul(original1, comb21))
        updated2 = pl.add(updated2, pl.row_expand_mul(original2, comb22))
        updated2 = pl.add(updated2, pl.row_expand_mul(original3, comb23))
        updated3 = pl.row_expand_mul(input_tile, post3)
        updated3 = pl.add(updated3, pl.row_expand_mul(original0, comb30))
        updated3 = pl.add(updated3, pl.row_expand_mul(original1, comb31))
        updated3 = pl.add(updated3, pl.row_expand_mul(original2, comb32))
        updated3 = pl.add(updated3, pl.row_expand_mul(original3, comb33))
        square_sum = pl.row_sum(pl.add(pl.add(pl.mul(updated0, updated0), pl.mul(updated1, updated1)), pl.add(pl.mul(updated2, updated2), pl.mul(updated3, updated3))))
        mean_square = pl.add(pl.mul(square_sum, 1.0 / FD), RMS_EPS)
        inverse_rms = pl.rsqrt(mean_square)
        inverse_rms = pl.mul(inverse_rms, pl.add(pl.mul(pl.mul(mean_square, pl.mul(inverse_rms, inverse_rms)), -0.5), 1.5))
        rms_scratch[0:1, token_start:token_start + 8] = pl.reshape(inverse_rms, [1, 8])
        updated_fp32[token_start:token_start + 8, 0 * D:1 * D] = updated0
        residual[token_start:token_start + 8, 0 * D:1 * D] = pl.cast(updated0, pl.BF16, mode="rint")
        updated_fp32[token_start:token_start + 8, 1 * D:2 * D] = updated1
        residual[token_start:token_start + 8, 1 * D:2 * D] = pl.cast(updated1, pl.BF16, mode="rint")
        updated_fp32[token_start:token_start + 8, 2 * D:3 * D] = updated2
        residual[token_start:token_start + 8, 2 * D:3 * D] = pl.cast(updated2, pl.BF16, mode="rint")
        updated_fp32[token_start:token_start + 8, 3 * D:4 * D] = updated3
        residual[token_start:token_start + 8, 3 * D:4 * D] = pl.cast(updated3, pl.BF16, mode="rint")
    for token_start in pl.parallel(0, T, 8):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="bridge_cube_projection_0"):
            norm_row = pl.slice(normw, [1, 32], [0, 0], valid_shape=[1, 24], clamp=True)
            accumulator = pl.create_tensor([16, 32], dtype=pl.FP32)
            for hidden_block in pl.range(FD // 128):
                hidden_start = hidden_block * 128
                updated_chunk = pl.slice(updated_fp32, [16, 128], [token_start, hidden_start], valid_shape=[8, 128], clamp=True)
                weight_chunk = pl.slice(fn, [32, 128], [0, hidden_start], valid_shape=[24, 128], clamp=True)
                accumulator = pl.matmul_acc(accumulator, updated_chunk, weight_chunk, b_trans=True, init_cond=(hidden_block == 0))
            rms_tile = pl.reshape(pl.slice(rms_scratch, [1, 16], [0, token_start], valid_shape=[1, 8], clamp=True), [16, 1])
            predicted = pl.row_expand_mul(pl.col_expand_mul(accumulator, norm_row), rms_tile)
            next_mixes[token_start:token_start + 8, 0:24] = predicted[0:8, 0:24]
            scale_pre = pl.read(mhc_scale, [0])
            predicted_planes = pl.transpose(predicted, 0, 1)
            pre_coefficient0 = pl.reshape(predicted_planes[0:1, 0:8], [8, 1])
            pre_raw0 = pl.add(pl.mul(pre_coefficient0, scale_pre), pl.read(mhc_base, [0, 0]))
            next_pre0 = pl.add(pl.recip(pl.add(pl.exp(pl.neg(pre_raw0)), 1.0)), PRE_EPS)
            pre_coefficient1 = pl.reshape(predicted_planes[1:2, 0:8], [8, 1])
            pre_raw1 = pl.add(pl.mul(pre_coefficient1, scale_pre), pl.read(mhc_base, [0, 1]))
            next_pre1 = pl.add(pl.recip(pl.add(pl.exp(pl.neg(pre_raw1)), 1.0)), PRE_EPS)
            pre_coefficient2 = pl.reshape(predicted_planes[2:3, 0:8], [8, 1])
            pre_raw2 = pl.add(pl.mul(pre_coefficient2, scale_pre), pl.read(mhc_base, [0, 2]))
            next_pre2 = pl.add(pl.recip(pl.add(pl.exp(pl.neg(pre_raw2)), 1.0)), PRE_EPS)
            pre_coefficient3 = pl.reshape(predicted_planes[3:4, 0:8], [8, 1])
            pre_raw3 = pl.add(pl.mul(pre_coefficient3, scale_pre), pl.read(mhc_base, [0, 3]))
            next_pre3 = pl.add(pl.recip(pl.add(pl.exp(pl.neg(pre_raw3)), 1.0)), PRE_EPS)
            for hidden_block in pl.range(D // 128):
                hidden_start = hidden_block * 128
                updated_copy0 = updated_fp32[token_start:token_start + 8, 0 * D + hidden_start:0 * D + hidden_start + 128]
                hidden = pl.row_expand_mul(updated_copy0, next_pre0)
                updated_copy1 = updated_fp32[token_start:token_start + 8, 1 * D + hidden_start:1 * D + hidden_start + 128]
                hidden = pl.add(hidden, pl.row_expand_mul(updated_copy1, next_pre1))
                updated_copy2 = updated_fp32[token_start:token_start + 8, 2 * D + hidden_start:2 * D + hidden_start + 128]
                hidden = pl.add(hidden, pl.row_expand_mul(updated_copy2, next_pre2))
                updated_copy3 = updated_fp32[token_start:token_start + 8, 3 * D + hidden_start:3 * D + hidden_start + 128]
                hidden = pl.add(hidden, pl.row_expand_mul(updated_copy3, next_pre3))
                h_next[token_start:token_start + 8, hidden_start:hidden_start + 128] = pl.cast(hidden, pl.BF16, mode="rint")
    return h_next

@pl.jit
def mega_mhc_bridge_compact32(x: pl.Tensor[[T, D], pl.BF16], residual: pl.InOut[pl.Tensor[[T, FD], pl.BF16]], input_mixes: pl.Tensor[[T, M3], pl.FP32], fn: pl.Tensor[[M3, FD], pl.FP32], normw: pl.Tensor[[1, M3], pl.FP32], mhc_scale: pl.Tensor[[3], pl.FP32], mhc_base: pl.Tensor[[1, M3], pl.FP32], next_mixes: pl.Out[pl.Tensor[[T, M3], pl.FP32]], h_next: pl.Out[pl.Tensor[[T, D], pl.BF16]]):
    updated_fp32 = pl.create_tensor([T, FD], dtype=pl.FP32)
    rms_scratch = pl.create_tensor([1, T], dtype=pl.FP32)
    for task_index in pl.spmd(T // 32, name_hint="macro_producer_32"):
        macro_start = task_index * 32
        scale_pre = pl.read(mhc_scale, [0])
        scale_post = pl.read(mhc_scale, [1])
        scale_comb = pl.read(mhc_scale, [2])
        mix_planes = pl.transpose(input_mixes[macro_start:macro_start + 32, 0:M3], 0, 1)
        post_raw0 = pl.add(pl.mul(pl.reshape(mix_planes[4:5, 0:32], [1, 32]), scale_post), pl.read(mhc_base, [0, 4]))
        post0_planes = pl.mul(pl.recip(pl.add(pl.exp(pl.neg(post_raw0)), 1.0)), POST_MULT)
        post_raw1 = pl.add(pl.mul(pl.reshape(mix_planes[5:6, 0:32], [1, 32]), scale_post), pl.read(mhc_base, [0, 5]))
        post1_planes = pl.mul(pl.recip(pl.add(pl.exp(pl.neg(post_raw1)), 1.0)), POST_MULT)
        post_raw2 = pl.add(pl.mul(pl.reshape(mix_planes[6:7, 0:32], [1, 32]), scale_post), pl.read(mhc_base, [0, 6]))
        post2_planes = pl.mul(pl.recip(pl.add(pl.exp(pl.neg(post_raw2)), 1.0)), POST_MULT)
        post_raw3 = pl.add(pl.mul(pl.reshape(mix_planes[7:8, 0:32], [1, 32]), scale_post), pl.read(mhc_base, [0, 7]))
        post3_planes = pl.mul(pl.recip(pl.add(pl.exp(pl.neg(post_raw3)), 1.0)), POST_MULT)
        comb00_planes = pl.add(pl.mul(pl.reshape(mix_planes[8:9, 0:32], [1, 32]), scale_comb), pl.read(mhc_base, [0, 8]))
        comb01_planes = pl.add(pl.mul(pl.reshape(mix_planes[9:10, 0:32], [1, 32]), scale_comb), pl.read(mhc_base, [0, 9]))
        comb02_planes = pl.add(pl.mul(pl.reshape(mix_planes[10:11, 0:32], [1, 32]), scale_comb), pl.read(mhc_base, [0, 10]))
        comb03_planes = pl.add(pl.mul(pl.reshape(mix_planes[11:12, 0:32], [1, 32]), scale_comb), pl.read(mhc_base, [0, 11]))
        row_max0 = pl.maximum(pl.maximum(comb00_planes, comb01_planes), pl.maximum(comb02_planes, comb03_planes))
        comb00_planes = pl.exp(pl.sub(comb00_planes, row_max0))
        comb01_planes = pl.exp(pl.sub(comb01_planes, row_max0))
        comb02_planes = pl.exp(pl.sub(comb02_planes, row_max0))
        comb03_planes = pl.exp(pl.sub(comb03_planes, row_max0))
        row_sum0 = pl.add(pl.add(comb00_planes, comb01_planes), pl.add(comb02_planes, comb03_planes))
        comb00_planes = pl.add(pl.div(comb00_planes, row_sum0), PRE_EPS)
        comb01_planes = pl.add(pl.div(comb01_planes, row_sum0), PRE_EPS)
        comb02_planes = pl.add(pl.div(comb02_planes, row_sum0), PRE_EPS)
        comb03_planes = pl.add(pl.div(comb03_planes, row_sum0), PRE_EPS)
        comb10_planes = pl.add(pl.mul(pl.reshape(mix_planes[12:13, 0:32], [1, 32]), scale_comb), pl.read(mhc_base, [0, 12]))
        comb11_planes = pl.add(pl.mul(pl.reshape(mix_planes[13:14, 0:32], [1, 32]), scale_comb), pl.read(mhc_base, [0, 13]))
        comb12_planes = pl.add(pl.mul(pl.reshape(mix_planes[14:15, 0:32], [1, 32]), scale_comb), pl.read(mhc_base, [0, 14]))
        comb13_planes = pl.add(pl.mul(pl.reshape(mix_planes[15:16, 0:32], [1, 32]), scale_comb), pl.read(mhc_base, [0, 15]))
        row_max1 = pl.maximum(pl.maximum(comb10_planes, comb11_planes), pl.maximum(comb12_planes, comb13_planes))
        comb10_planes = pl.exp(pl.sub(comb10_planes, row_max1))
        comb11_planes = pl.exp(pl.sub(comb11_planes, row_max1))
        comb12_planes = pl.exp(pl.sub(comb12_planes, row_max1))
        comb13_planes = pl.exp(pl.sub(comb13_planes, row_max1))
        row_sum1 = pl.add(pl.add(comb10_planes, comb11_planes), pl.add(comb12_planes, comb13_planes))
        comb10_planes = pl.add(pl.div(comb10_planes, row_sum1), PRE_EPS)
        comb11_planes = pl.add(pl.div(comb11_planes, row_sum1), PRE_EPS)
        comb12_planes = pl.add(pl.div(comb12_planes, row_sum1), PRE_EPS)
        comb13_planes = pl.add(pl.div(comb13_planes, row_sum1), PRE_EPS)
        comb20_planes = pl.add(pl.mul(pl.reshape(mix_planes[16:17, 0:32], [1, 32]), scale_comb), pl.read(mhc_base, [0, 16]))
        comb21_planes = pl.add(pl.mul(pl.reshape(mix_planes[17:18, 0:32], [1, 32]), scale_comb), pl.read(mhc_base, [0, 17]))
        comb22_planes = pl.add(pl.mul(pl.reshape(mix_planes[18:19, 0:32], [1, 32]), scale_comb), pl.read(mhc_base, [0, 18]))
        comb23_planes = pl.add(pl.mul(pl.reshape(mix_planes[19:20, 0:32], [1, 32]), scale_comb), pl.read(mhc_base, [0, 19]))
        row_max2 = pl.maximum(pl.maximum(comb20_planes, comb21_planes), pl.maximum(comb22_planes, comb23_planes))
        comb20_planes = pl.exp(pl.sub(comb20_planes, row_max2))
        comb21_planes = pl.exp(pl.sub(comb21_planes, row_max2))
        comb22_planes = pl.exp(pl.sub(comb22_planes, row_max2))
        comb23_planes = pl.exp(pl.sub(comb23_planes, row_max2))
        row_sum2 = pl.add(pl.add(comb20_planes, comb21_planes), pl.add(comb22_planes, comb23_planes))
        comb20_planes = pl.add(pl.div(comb20_planes, row_sum2), PRE_EPS)
        comb21_planes = pl.add(pl.div(comb21_planes, row_sum2), PRE_EPS)
        comb22_planes = pl.add(pl.div(comb22_planes, row_sum2), PRE_EPS)
        comb23_planes = pl.add(pl.div(comb23_planes, row_sum2), PRE_EPS)
        comb30_planes = pl.add(pl.mul(pl.reshape(mix_planes[20:21, 0:32], [1, 32]), scale_comb), pl.read(mhc_base, [0, 20]))
        comb31_planes = pl.add(pl.mul(pl.reshape(mix_planes[21:22, 0:32], [1, 32]), scale_comb), pl.read(mhc_base, [0, 21]))
        comb32_planes = pl.add(pl.mul(pl.reshape(mix_planes[22:23, 0:32], [1, 32]), scale_comb), pl.read(mhc_base, [0, 22]))
        comb33_planes = pl.add(pl.mul(pl.reshape(mix_planes[23:24, 0:32], [1, 32]), scale_comb), pl.read(mhc_base, [0, 23]))
        row_max3 = pl.maximum(pl.maximum(comb30_planes, comb31_planes), pl.maximum(comb32_planes, comb33_planes))
        comb30_planes = pl.exp(pl.sub(comb30_planes, row_max3))
        comb31_planes = pl.exp(pl.sub(comb31_planes, row_max3))
        comb32_planes = pl.exp(pl.sub(comb32_planes, row_max3))
        comb33_planes = pl.exp(pl.sub(comb33_planes, row_max3))
        row_sum3 = pl.add(pl.add(comb30_planes, comb31_planes), pl.add(comb32_planes, comb33_planes))
        comb30_planes = pl.add(pl.div(comb30_planes, row_sum3), PRE_EPS)
        comb31_planes = pl.add(pl.div(comb31_planes, row_sum3), PRE_EPS)
        comb32_planes = pl.add(pl.div(comb32_planes, row_sum3), PRE_EPS)
        comb33_planes = pl.add(pl.div(comb33_planes, row_sum3), PRE_EPS)
        column_sum0 = pl.add(pl.add(pl.add(comb00_planes, comb10_planes), pl.add(comb20_planes, comb30_planes)), PRE_EPS)
        comb00_planes = pl.div(comb00_planes, column_sum0)
        comb10_planes = pl.div(comb10_planes, column_sum0)
        comb20_planes = pl.div(comb20_planes, column_sum0)
        comb30_planes = pl.div(comb30_planes, column_sum0)
        column_sum1 = pl.add(pl.add(pl.add(comb01_planes, comb11_planes), pl.add(comb21_planes, comb31_planes)), PRE_EPS)
        comb01_planes = pl.div(comb01_planes, column_sum1)
        comb11_planes = pl.div(comb11_planes, column_sum1)
        comb21_planes = pl.div(comb21_planes, column_sum1)
        comb31_planes = pl.div(comb31_planes, column_sum1)
        column_sum2 = pl.add(pl.add(pl.add(comb02_planes, comb12_planes), pl.add(comb22_planes, comb32_planes)), PRE_EPS)
        comb02_planes = pl.div(comb02_planes, column_sum2)
        comb12_planes = pl.div(comb12_planes, column_sum2)
        comb22_planes = pl.div(comb22_planes, column_sum2)
        comb32_planes = pl.div(comb32_planes, column_sum2)
        column_sum3 = pl.add(pl.add(pl.add(comb03_planes, comb13_planes), pl.add(comb23_planes, comb33_planes)), PRE_EPS)
        comb03_planes = pl.div(comb03_planes, column_sum3)
        comb13_planes = pl.div(comb13_planes, column_sum3)
        comb23_planes = pl.div(comb23_planes, column_sum3)
        comb33_planes = pl.div(comb33_planes, column_sum3)
        for iteration in pl.range(SINK - 1):
            row_sum0 = pl.add(pl.add(pl.add(comb00_planes, comb01_planes), pl.add(comb02_planes, comb03_planes)), PRE_EPS)
            comb00_planes = pl.div(comb00_planes, row_sum0)
            comb01_planes = pl.div(comb01_planes, row_sum0)
            comb02_planes = pl.div(comb02_planes, row_sum0)
            comb03_planes = pl.div(comb03_planes, row_sum0)
            row_sum1 = pl.add(pl.add(pl.add(comb10_planes, comb11_planes), pl.add(comb12_planes, comb13_planes)), PRE_EPS)
            comb10_planes = pl.div(comb10_planes, row_sum1)
            comb11_planes = pl.div(comb11_planes, row_sum1)
            comb12_planes = pl.div(comb12_planes, row_sum1)
            comb13_planes = pl.div(comb13_planes, row_sum1)
            row_sum2 = pl.add(pl.add(pl.add(comb20_planes, comb21_planes), pl.add(comb22_planes, comb23_planes)), PRE_EPS)
            comb20_planes = pl.div(comb20_planes, row_sum2)
            comb21_planes = pl.div(comb21_planes, row_sum2)
            comb22_planes = pl.div(comb22_planes, row_sum2)
            comb23_planes = pl.div(comb23_planes, row_sum2)
            row_sum3 = pl.add(pl.add(pl.add(comb30_planes, comb31_planes), pl.add(comb32_planes, comb33_planes)), PRE_EPS)
            comb30_planes = pl.div(comb30_planes, row_sum3)
            comb31_planes = pl.div(comb31_planes, row_sum3)
            comb32_planes = pl.div(comb32_planes, row_sum3)
            comb33_planes = pl.div(comb33_planes, row_sum3)
            column_sum0 = pl.add(pl.add(pl.add(comb00_planes, comb10_planes), pl.add(comb20_planes, comb30_planes)), PRE_EPS)
            comb00_planes = pl.div(comb00_planes, column_sum0)
            comb10_planes = pl.div(comb10_planes, column_sum0)
            comb20_planes = pl.div(comb20_planes, column_sum0)
            comb30_planes = pl.div(comb30_planes, column_sum0)
            column_sum1 = pl.add(pl.add(pl.add(comb01_planes, comb11_planes), pl.add(comb21_planes, comb31_planes)), PRE_EPS)
            comb01_planes = pl.div(comb01_planes, column_sum1)
            comb11_planes = pl.div(comb11_planes, column_sum1)
            comb21_planes = pl.div(comb21_planes, column_sum1)
            comb31_planes = pl.div(comb31_planes, column_sum1)
            column_sum2 = pl.add(pl.add(pl.add(comb02_planes, comb12_planes), pl.add(comb22_planes, comb32_planes)), PRE_EPS)
            comb02_planes = pl.div(comb02_planes, column_sum2)
            comb12_planes = pl.div(comb12_planes, column_sum2)
            comb22_planes = pl.div(comb22_planes, column_sum2)
            comb32_planes = pl.div(comb32_planes, column_sum2)
            column_sum3 = pl.add(pl.add(pl.add(comb03_planes, comb13_planes), pl.add(comb23_planes, comb33_planes)), PRE_EPS)
            comb03_planes = pl.div(comb03_planes, column_sum3)
            comb13_planes = pl.div(comb13_planes, column_sum3)
            comb23_planes = pl.div(comb23_planes, column_sum3)
            comb33_planes = pl.div(comb33_planes, column_sum3)
        for update_block in pl.range(32 // 32):
            update_offset = update_block * 32
            token_start = macro_start + update_offset
            post0 = pl.reshape(post0_planes[0:1, update_offset:update_offset + 32], [32, 1])
            post1 = pl.reshape(post1_planes[0:1, update_offset:update_offset + 32], [32, 1])
            post2 = pl.reshape(post2_planes[0:1, update_offset:update_offset + 32], [32, 1])
            post3 = pl.reshape(post3_planes[0:1, update_offset:update_offset + 32], [32, 1])
            comb00 = pl.reshape(comb00_planes[0:1, update_offset:update_offset + 32], [32, 1])
            comb01 = pl.reshape(comb01_planes[0:1, update_offset:update_offset + 32], [32, 1])
            comb02 = pl.reshape(comb02_planes[0:1, update_offset:update_offset + 32], [32, 1])
            comb03 = pl.reshape(comb03_planes[0:1, update_offset:update_offset + 32], [32, 1])
            comb10 = pl.reshape(comb10_planes[0:1, update_offset:update_offset + 32], [32, 1])
            comb11 = pl.reshape(comb11_planes[0:1, update_offset:update_offset + 32], [32, 1])
            comb12 = pl.reshape(comb12_planes[0:1, update_offset:update_offset + 32], [32, 1])
            comb13 = pl.reshape(comb13_planes[0:1, update_offset:update_offset + 32], [32, 1])
            comb20 = pl.reshape(comb20_planes[0:1, update_offset:update_offset + 32], [32, 1])
            comb21 = pl.reshape(comb21_planes[0:1, update_offset:update_offset + 32], [32, 1])
            comb22 = pl.reshape(comb22_planes[0:1, update_offset:update_offset + 32], [32, 1])
            comb23 = pl.reshape(comb23_planes[0:1, update_offset:update_offset + 32], [32, 1])
            comb30 = pl.reshape(comb30_planes[0:1, update_offset:update_offset + 32], [32, 1])
            comb31 = pl.reshape(comb31_planes[0:1, update_offset:update_offset + 32], [32, 1])
            comb32 = pl.reshape(comb32_planes[0:1, update_offset:update_offset + 32], [32, 1])
            comb33 = pl.reshape(comb33_planes[0:1, update_offset:update_offset + 32], [32, 1])
            square_acc = pl.full([1, 32], dtype=pl.FP32, value=0.0)
            for hidden_block in pl.range(D // 128):
                hidden_start = hidden_block * 128
                input_tile = pl.cast(x[token_start:token_start + 32, hidden_start:hidden_start + 128], pl.FP32)
                original0 = pl.cast(residual[token_start:token_start + 32, 0 * D + hidden_start:0 * D + hidden_start + 128], pl.FP32)
                original1 = pl.cast(residual[token_start:token_start + 32, 1 * D + hidden_start:1 * D + hidden_start + 128], pl.FP32)
                original2 = pl.cast(residual[token_start:token_start + 32, 2 * D + hidden_start:2 * D + hidden_start + 128], pl.FP32)
                original3 = pl.cast(residual[token_start:token_start + 32, 3 * D + hidden_start:3 * D + hidden_start + 128], pl.FP32)
                updated0 = pl.row_expand_mul(input_tile, post0)
                updated0 = pl.add(updated0, pl.row_expand_mul(original0, comb00))
                updated0 = pl.add(updated0, pl.row_expand_mul(original1, comb01))
                updated0 = pl.add(updated0, pl.row_expand_mul(original2, comb02))
                updated0 = pl.add(updated0, pl.row_expand_mul(original3, comb03))
                updated1 = pl.row_expand_mul(input_tile, post1)
                updated1 = pl.add(updated1, pl.row_expand_mul(original0, comb10))
                updated1 = pl.add(updated1, pl.row_expand_mul(original1, comb11))
                updated1 = pl.add(updated1, pl.row_expand_mul(original2, comb12))
                updated1 = pl.add(updated1, pl.row_expand_mul(original3, comb13))
                updated2 = pl.row_expand_mul(input_tile, post2)
                updated2 = pl.add(updated2, pl.row_expand_mul(original0, comb20))
                updated2 = pl.add(updated2, pl.row_expand_mul(original1, comb21))
                updated2 = pl.add(updated2, pl.row_expand_mul(original2, comb22))
                updated2 = pl.add(updated2, pl.row_expand_mul(original3, comb23))
                updated3 = pl.row_expand_mul(input_tile, post3)
                updated3 = pl.add(updated3, pl.row_expand_mul(original0, comb30))
                updated3 = pl.add(updated3, pl.row_expand_mul(original1, comb31))
                updated3 = pl.add(updated3, pl.row_expand_mul(original2, comb32))
                updated3 = pl.add(updated3, pl.row_expand_mul(original3, comb33))
                square_sum = pl.row_sum(pl.add(pl.add(pl.mul(updated0, updated0), pl.mul(updated1, updated1)), pl.add(pl.mul(updated2, updated2), pl.mul(updated3, updated3))))
                square_acc = pl.add(square_acc, pl.reshape(square_sum, [1, 32]))
                updated_fp32[token_start:token_start + 32, 0 * D + hidden_start:0 * D + hidden_start + 128] = updated0
                residual[token_start:token_start + 32, 0 * D + hidden_start:0 * D + hidden_start + 128] = pl.cast(updated0, pl.BF16, mode="rint")
                updated_fp32[token_start:token_start + 32, 1 * D + hidden_start:1 * D + hidden_start + 128] = updated1
                residual[token_start:token_start + 32, 1 * D + hidden_start:1 * D + hidden_start + 128] = pl.cast(updated1, pl.BF16, mode="rint")
                updated_fp32[token_start:token_start + 32, 2 * D + hidden_start:2 * D + hidden_start + 128] = updated2
                residual[token_start:token_start + 32, 2 * D + hidden_start:2 * D + hidden_start + 128] = pl.cast(updated2, pl.BF16, mode="rint")
                updated_fp32[token_start:token_start + 32, 3 * D + hidden_start:3 * D + hidden_start + 128] = updated3
                residual[token_start:token_start + 32, 3 * D + hidden_start:3 * D + hidden_start + 128] = pl.cast(updated3, pl.BF16, mode="rint")
            square_sum = pl.reshape(square_acc, [32, 1])
            mean_square = pl.add(pl.mul(square_sum, 1.0 / FD), RMS_EPS)
            inverse_rms = pl.rsqrt(mean_square)
            inverse_rms = pl.mul(inverse_rms, pl.add(pl.mul(pl.mul(mean_square, pl.mul(inverse_rms, inverse_rms)), -0.5), 1.5))
            rms_scratch[0:1, token_start:token_start + 32] = pl.reshape(inverse_rms, [1, 32])
    for token_start in pl.parallel(0, T, 128):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="bridge_cube_projection_0"):
            norm_row = pl.slice(normw, [1, 32], [0, 0], valid_shape=[1, 24], clamp=True)
            first_updated = pl.slice(updated_fp32, [128, 128], [token_start, 0], valid_shape=[128, 128])
            first_weight = pl.slice(fn, [32, 128], [0, 0], valid_shape=[24, 128], clamp=True)
            accumulator = pl.matmul(first_updated, first_weight, b_trans=True)
            for hidden_block in pl.range(1, FD // 128):
                hidden_start = hidden_block * 128
                updated_chunk = pl.slice(updated_fp32, [128, 128], [token_start, hidden_start], valid_shape=[128, 128])
                weight_chunk = pl.slice(fn, [32, 128], [0, hidden_start], valid_shape=[24, 128], clamp=True)
                accumulator = pl.matmul_acc(accumulator, updated_chunk, weight_chunk, b_trans=True)
            rms_tile = pl.reshape(pl.slice(rms_scratch, [1, 128], [0, token_start], valid_shape=[1, 128]), [128, 1])
            predicted = pl.row_expand_mul(pl.col_expand_mul(accumulator, norm_row), rms_tile)
            next_mixes[token_start:token_start + 128, 0:24] = predicted[0:128, 0:24]
            scale_pre = pl.read(mhc_scale, [0])
            predicted_planes = pl.transpose(predicted, 0, 1)
            pre_coefficient0 = pl.reshape(predicted_planes[0:1, 0:128], [128, 1])
            pre_raw0 = pl.add(pl.mul(pre_coefficient0, scale_pre), pl.read(mhc_base, [0, 0]))
            next_pre0 = pl.add(pl.recip(pl.add(pl.exp(pl.neg(pre_raw0)), 1.0)), PRE_EPS)
            pre_coefficient1 = pl.reshape(predicted_planes[1:2, 0:128], [128, 1])
            pre_raw1 = pl.add(pl.mul(pre_coefficient1, scale_pre), pl.read(mhc_base, [0, 1]))
            next_pre1 = pl.add(pl.recip(pl.add(pl.exp(pl.neg(pre_raw1)), 1.0)), PRE_EPS)
            pre_coefficient2 = pl.reshape(predicted_planes[2:3, 0:128], [128, 1])
            pre_raw2 = pl.add(pl.mul(pre_coefficient2, scale_pre), pl.read(mhc_base, [0, 2]))
            next_pre2 = pl.add(pl.recip(pl.add(pl.exp(pl.neg(pre_raw2)), 1.0)), PRE_EPS)
            pre_coefficient3 = pl.reshape(predicted_planes[3:4, 0:128], [128, 1])
            pre_raw3 = pl.add(pl.mul(pre_coefficient3, scale_pre), pl.read(mhc_base, [0, 3]))
            next_pre3 = pl.add(pl.recip(pl.add(pl.exp(pl.neg(pre_raw3)), 1.0)), PRE_EPS)
            for hidden_block in pl.range(D // 128):
                hidden_start = hidden_block * 128
            for hidden_block in pl.range(D // 128):
                hidden_start = hidden_block * 128
                updated_copy0 = updated_fp32[token_start:token_start + 128, 0 * D + hidden_start:0 * D + hidden_start + 128]
                hidden = pl.row_expand_mul(updated_copy0, next_pre0)
                updated_copy1 = updated_fp32[token_start:token_start + 128, 1 * D + hidden_start:1 * D + hidden_start + 128]
                hidden = pl.add(hidden, pl.row_expand_mul(updated_copy1, next_pre1))
                updated_copy2 = updated_fp32[token_start:token_start + 128, 2 * D + hidden_start:2 * D + hidden_start + 128]
                hidden = pl.add(hidden, pl.row_expand_mul(updated_copy2, next_pre2))
                updated_copy3 = updated_fp32[token_start:token_start + 128, 3 * D + hidden_start:3 * D + hidden_start + 128]
                hidden = pl.add(hidden, pl.row_expand_mul(updated_copy3, next_pre3))
                h_next[token_start:token_start + 128, hidden_start:hidden_start + 128] = pl.cast(hidden, pl.BF16, mode="rint")
    return h_next



@pl.jit
def mega_mhc_bridge_compact64(x: pl.Tensor[[T, D], pl.BF16], residual: pl.InOut[pl.Tensor[[T, FD], pl.BF16]], input_mixes: pl.Tensor[[T, M3], pl.FP32], fn: pl.Tensor[[M3, FD], pl.FP32], normw: pl.Tensor[[1, M3], pl.FP32], mhc_scale: pl.Tensor[[3], pl.FP32], mhc_base: pl.Tensor[[1, M3], pl.FP32], next_mixes: pl.Out[pl.Tensor[[T, M3], pl.FP32]], h_next: pl.Out[pl.Tensor[[T, D], pl.BF16]]):
    updated_fp32 = pl.create_tensor([T, FD], dtype=pl.FP32)
    rms_scratch = pl.create_tensor([1, T], dtype=pl.FP32)
    for task_index in pl.spmd(T // 64, name_hint="macro_producer_64"):
        macro_start = task_index * 64
        scale_pre = pl.read(mhc_scale, [0])
        scale_post = pl.read(mhc_scale, [1])
        scale_comb = pl.read(mhc_scale, [2])
        mix_planes = pl.transpose(input_mixes[macro_start:macro_start + 64, 0:M3], 0, 1)
        post_raw0 = pl.add(pl.mul(pl.reshape(mix_planes[4:5, 0:64], [1, 64]), scale_post), pl.read(mhc_base, [0, 4]))
        post0_planes = pl.mul(pl.recip(pl.add(pl.exp(pl.neg(post_raw0)), 1.0)), POST_MULT)
        post_raw1 = pl.add(pl.mul(pl.reshape(mix_planes[5:6, 0:64], [1, 64]), scale_post), pl.read(mhc_base, [0, 5]))
        post1_planes = pl.mul(pl.recip(pl.add(pl.exp(pl.neg(post_raw1)), 1.0)), POST_MULT)
        post_raw2 = pl.add(pl.mul(pl.reshape(mix_planes[6:7, 0:64], [1, 64]), scale_post), pl.read(mhc_base, [0, 6]))
        post2_planes = pl.mul(pl.recip(pl.add(pl.exp(pl.neg(post_raw2)), 1.0)), POST_MULT)
        post_raw3 = pl.add(pl.mul(pl.reshape(mix_planes[7:8, 0:64], [1, 64]), scale_post), pl.read(mhc_base, [0, 7]))
        post3_planes = pl.mul(pl.recip(pl.add(pl.exp(pl.neg(post_raw3)), 1.0)), POST_MULT)
        comb00_planes = pl.add(pl.mul(pl.reshape(mix_planes[8:9, 0:64], [1, 64]), scale_comb), pl.read(mhc_base, [0, 8]))
        comb01_planes = pl.add(pl.mul(pl.reshape(mix_planes[9:10, 0:64], [1, 64]), scale_comb), pl.read(mhc_base, [0, 9]))
        comb02_planes = pl.add(pl.mul(pl.reshape(mix_planes[10:11, 0:64], [1, 64]), scale_comb), pl.read(mhc_base, [0, 10]))
        comb03_planes = pl.add(pl.mul(pl.reshape(mix_planes[11:12, 0:64], [1, 64]), scale_comb), pl.read(mhc_base, [0, 11]))
        row_max0 = pl.maximum(pl.maximum(comb00_planes, comb01_planes), pl.maximum(comb02_planes, comb03_planes))
        comb00_planes = pl.exp(pl.sub(comb00_planes, row_max0))
        comb01_planes = pl.exp(pl.sub(comb01_planes, row_max0))
        comb02_planes = pl.exp(pl.sub(comb02_planes, row_max0))
        comb03_planes = pl.exp(pl.sub(comb03_planes, row_max0))
        row_sum0 = pl.add(pl.add(comb00_planes, comb01_planes), pl.add(comb02_planes, comb03_planes))
        comb00_planes = pl.add(pl.div(comb00_planes, row_sum0), PRE_EPS)
        comb01_planes = pl.add(pl.div(comb01_planes, row_sum0), PRE_EPS)
        comb02_planes = pl.add(pl.div(comb02_planes, row_sum0), PRE_EPS)
        comb03_planes = pl.add(pl.div(comb03_planes, row_sum0), PRE_EPS)
        comb10_planes = pl.add(pl.mul(pl.reshape(mix_planes[12:13, 0:64], [1, 64]), scale_comb), pl.read(mhc_base, [0, 12]))
        comb11_planes = pl.add(pl.mul(pl.reshape(mix_planes[13:14, 0:64], [1, 64]), scale_comb), pl.read(mhc_base, [0, 13]))
        comb12_planes = pl.add(pl.mul(pl.reshape(mix_planes[14:15, 0:64], [1, 64]), scale_comb), pl.read(mhc_base, [0, 14]))
        comb13_planes = pl.add(pl.mul(pl.reshape(mix_planes[15:16, 0:64], [1, 64]), scale_comb), pl.read(mhc_base, [0, 15]))
        row_max1 = pl.maximum(pl.maximum(comb10_planes, comb11_planes), pl.maximum(comb12_planes, comb13_planes))
        comb10_planes = pl.exp(pl.sub(comb10_planes, row_max1))
        comb11_planes = pl.exp(pl.sub(comb11_planes, row_max1))
        comb12_planes = pl.exp(pl.sub(comb12_planes, row_max1))
        comb13_planes = pl.exp(pl.sub(comb13_planes, row_max1))
        row_sum1 = pl.add(pl.add(comb10_planes, comb11_planes), pl.add(comb12_planes, comb13_planes))
        comb10_planes = pl.add(pl.div(comb10_planes, row_sum1), PRE_EPS)
        comb11_planes = pl.add(pl.div(comb11_planes, row_sum1), PRE_EPS)
        comb12_planes = pl.add(pl.div(comb12_planes, row_sum1), PRE_EPS)
        comb13_planes = pl.add(pl.div(comb13_planes, row_sum1), PRE_EPS)
        comb20_planes = pl.add(pl.mul(pl.reshape(mix_planes[16:17, 0:64], [1, 64]), scale_comb), pl.read(mhc_base, [0, 16]))
        comb21_planes = pl.add(pl.mul(pl.reshape(mix_planes[17:18, 0:64], [1, 64]), scale_comb), pl.read(mhc_base, [0, 17]))
        comb22_planes = pl.add(pl.mul(pl.reshape(mix_planes[18:19, 0:64], [1, 64]), scale_comb), pl.read(mhc_base, [0, 18]))
        comb23_planes = pl.add(pl.mul(pl.reshape(mix_planes[19:20, 0:64], [1, 64]), scale_comb), pl.read(mhc_base, [0, 19]))
        row_max2 = pl.maximum(pl.maximum(comb20_planes, comb21_planes), pl.maximum(comb22_planes, comb23_planes))
        comb20_planes = pl.exp(pl.sub(comb20_planes, row_max2))
        comb21_planes = pl.exp(pl.sub(comb21_planes, row_max2))
        comb22_planes = pl.exp(pl.sub(comb22_planes, row_max2))
        comb23_planes = pl.exp(pl.sub(comb23_planes, row_max2))
        row_sum2 = pl.add(pl.add(comb20_planes, comb21_planes), pl.add(comb22_planes, comb23_planes))
        comb20_planes = pl.add(pl.div(comb20_planes, row_sum2), PRE_EPS)
        comb21_planes = pl.add(pl.div(comb21_planes, row_sum2), PRE_EPS)
        comb22_planes = pl.add(pl.div(comb22_planes, row_sum2), PRE_EPS)
        comb23_planes = pl.add(pl.div(comb23_planes, row_sum2), PRE_EPS)
        comb30_planes = pl.add(pl.mul(pl.reshape(mix_planes[20:21, 0:64], [1, 64]), scale_comb), pl.read(mhc_base, [0, 20]))
        comb31_planes = pl.add(pl.mul(pl.reshape(mix_planes[21:22, 0:64], [1, 64]), scale_comb), pl.read(mhc_base, [0, 21]))
        comb32_planes = pl.add(pl.mul(pl.reshape(mix_planes[22:23, 0:64], [1, 64]), scale_comb), pl.read(mhc_base, [0, 22]))
        comb33_planes = pl.add(pl.mul(pl.reshape(mix_planes[23:24, 0:64], [1, 64]), scale_comb), pl.read(mhc_base, [0, 23]))
        row_max3 = pl.maximum(pl.maximum(comb30_planes, comb31_planes), pl.maximum(comb32_planes, comb33_planes))
        comb30_planes = pl.exp(pl.sub(comb30_planes, row_max3))
        comb31_planes = pl.exp(pl.sub(comb31_planes, row_max3))
        comb32_planes = pl.exp(pl.sub(comb32_planes, row_max3))
        comb33_planes = pl.exp(pl.sub(comb33_planes, row_max3))
        row_sum3 = pl.add(pl.add(comb30_planes, comb31_planes), pl.add(comb32_planes, comb33_planes))
        comb30_planes = pl.add(pl.div(comb30_planes, row_sum3), PRE_EPS)
        comb31_planes = pl.add(pl.div(comb31_planes, row_sum3), PRE_EPS)
        comb32_planes = pl.add(pl.div(comb32_planes, row_sum3), PRE_EPS)
        comb33_planes = pl.add(pl.div(comb33_planes, row_sum3), PRE_EPS)
        column_sum0 = pl.add(pl.add(pl.add(comb00_planes, comb10_planes), pl.add(comb20_planes, comb30_planes)), PRE_EPS)
        comb00_planes = pl.div(comb00_planes, column_sum0)
        comb10_planes = pl.div(comb10_planes, column_sum0)
        comb20_planes = pl.div(comb20_planes, column_sum0)
        comb30_planes = pl.div(comb30_planes, column_sum0)
        column_sum1 = pl.add(pl.add(pl.add(comb01_planes, comb11_planes), pl.add(comb21_planes, comb31_planes)), PRE_EPS)
        comb01_planes = pl.div(comb01_planes, column_sum1)
        comb11_planes = pl.div(comb11_planes, column_sum1)
        comb21_planes = pl.div(comb21_planes, column_sum1)
        comb31_planes = pl.div(comb31_planes, column_sum1)
        column_sum2 = pl.add(pl.add(pl.add(comb02_planes, comb12_planes), pl.add(comb22_planes, comb32_planes)), PRE_EPS)
        comb02_planes = pl.div(comb02_planes, column_sum2)
        comb12_planes = pl.div(comb12_planes, column_sum2)
        comb22_planes = pl.div(comb22_planes, column_sum2)
        comb32_planes = pl.div(comb32_planes, column_sum2)
        column_sum3 = pl.add(pl.add(pl.add(comb03_planes, comb13_planes), pl.add(comb23_planes, comb33_planes)), PRE_EPS)
        comb03_planes = pl.div(comb03_planes, column_sum3)
        comb13_planes = pl.div(comb13_planes, column_sum3)
        comb23_planes = pl.div(comb23_planes, column_sum3)
        comb33_planes = pl.div(comb33_planes, column_sum3)
        for iteration in pl.range(SINK - 1):
            row_sum0 = pl.add(pl.add(pl.add(comb00_planes, comb01_planes), pl.add(comb02_planes, comb03_planes)), PRE_EPS)
            comb00_planes = pl.div(comb00_planes, row_sum0)
            comb01_planes = pl.div(comb01_planes, row_sum0)
            comb02_planes = pl.div(comb02_planes, row_sum0)
            comb03_planes = pl.div(comb03_planes, row_sum0)
            row_sum1 = pl.add(pl.add(pl.add(comb10_planes, comb11_planes), pl.add(comb12_planes, comb13_planes)), PRE_EPS)
            comb10_planes = pl.div(comb10_planes, row_sum1)
            comb11_planes = pl.div(comb11_planes, row_sum1)
            comb12_planes = pl.div(comb12_planes, row_sum1)
            comb13_planes = pl.div(comb13_planes, row_sum1)
            row_sum2 = pl.add(pl.add(pl.add(comb20_planes, comb21_planes), pl.add(comb22_planes, comb23_planes)), PRE_EPS)
            comb20_planes = pl.div(comb20_planes, row_sum2)
            comb21_planes = pl.div(comb21_planes, row_sum2)
            comb22_planes = pl.div(comb22_planes, row_sum2)
            comb23_planes = pl.div(comb23_planes, row_sum2)
            row_sum3 = pl.add(pl.add(pl.add(comb30_planes, comb31_planes), pl.add(comb32_planes, comb33_planes)), PRE_EPS)
            comb30_planes = pl.div(comb30_planes, row_sum3)
            comb31_planes = pl.div(comb31_planes, row_sum3)
            comb32_planes = pl.div(comb32_planes, row_sum3)
            comb33_planes = pl.div(comb33_planes, row_sum3)
            column_sum0 = pl.add(pl.add(pl.add(comb00_planes, comb10_planes), pl.add(comb20_planes, comb30_planes)), PRE_EPS)
            comb00_planes = pl.div(comb00_planes, column_sum0)
            comb10_planes = pl.div(comb10_planes, column_sum0)
            comb20_planes = pl.div(comb20_planes, column_sum0)
            comb30_planes = pl.div(comb30_planes, column_sum0)
            column_sum1 = pl.add(pl.add(pl.add(comb01_planes, comb11_planes), pl.add(comb21_planes, comb31_planes)), PRE_EPS)
            comb01_planes = pl.div(comb01_planes, column_sum1)
            comb11_planes = pl.div(comb11_planes, column_sum1)
            comb21_planes = pl.div(comb21_planes, column_sum1)
            comb31_planes = pl.div(comb31_planes, column_sum1)
            column_sum2 = pl.add(pl.add(pl.add(comb02_planes, comb12_planes), pl.add(comb22_planes, comb32_planes)), PRE_EPS)
            comb02_planes = pl.div(comb02_planes, column_sum2)
            comb12_planes = pl.div(comb12_planes, column_sum2)
            comb22_planes = pl.div(comb22_planes, column_sum2)
            comb32_planes = pl.div(comb32_planes, column_sum2)
            column_sum3 = pl.add(pl.add(pl.add(comb03_planes, comb13_planes), pl.add(comb23_planes, comb33_planes)), PRE_EPS)
            comb03_planes = pl.div(comb03_planes, column_sum3)
            comb13_planes = pl.div(comb13_planes, column_sum3)
            comb23_planes = pl.div(comb23_planes, column_sum3)
            comb33_planes = pl.div(comb33_planes, column_sum3)
        for update_block in pl.range(64 // 32):
            update_offset = update_block * 32
            token_start = macro_start + update_offset
            post0 = pl.reshape(post0_planes[0:1, update_offset:update_offset + 32], [32, 1])
            post1 = pl.reshape(post1_planes[0:1, update_offset:update_offset + 32], [32, 1])
            post2 = pl.reshape(post2_planes[0:1, update_offset:update_offset + 32], [32, 1])
            post3 = pl.reshape(post3_planes[0:1, update_offset:update_offset + 32], [32, 1])
            comb00 = pl.reshape(comb00_planes[0:1, update_offset:update_offset + 32], [32, 1])
            comb01 = pl.reshape(comb01_planes[0:1, update_offset:update_offset + 32], [32, 1])
            comb02 = pl.reshape(comb02_planes[0:1, update_offset:update_offset + 32], [32, 1])
            comb03 = pl.reshape(comb03_planes[0:1, update_offset:update_offset + 32], [32, 1])
            comb10 = pl.reshape(comb10_planes[0:1, update_offset:update_offset + 32], [32, 1])
            comb11 = pl.reshape(comb11_planes[0:1, update_offset:update_offset + 32], [32, 1])
            comb12 = pl.reshape(comb12_planes[0:1, update_offset:update_offset + 32], [32, 1])
            comb13 = pl.reshape(comb13_planes[0:1, update_offset:update_offset + 32], [32, 1])
            comb20 = pl.reshape(comb20_planes[0:1, update_offset:update_offset + 32], [32, 1])
            comb21 = pl.reshape(comb21_planes[0:1, update_offset:update_offset + 32], [32, 1])
            comb22 = pl.reshape(comb22_planes[0:1, update_offset:update_offset + 32], [32, 1])
            comb23 = pl.reshape(comb23_planes[0:1, update_offset:update_offset + 32], [32, 1])
            comb30 = pl.reshape(comb30_planes[0:1, update_offset:update_offset + 32], [32, 1])
            comb31 = pl.reshape(comb31_planes[0:1, update_offset:update_offset + 32], [32, 1])
            comb32 = pl.reshape(comb32_planes[0:1, update_offset:update_offset + 32], [32, 1])
            comb33 = pl.reshape(comb33_planes[0:1, update_offset:update_offset + 32], [32, 1])
            square_acc = pl.full([1, 32], dtype=pl.FP32, value=0.0)
            for hidden_block in pl.range(D // 128):
                hidden_start = hidden_block * 128
                input_tile = pl.cast(x[token_start:token_start + 32, hidden_start:hidden_start + 128], pl.FP32)
                original0 = pl.cast(residual[token_start:token_start + 32, 0 * D + hidden_start:0 * D + hidden_start + 128], pl.FP32)
                original1 = pl.cast(residual[token_start:token_start + 32, 1 * D + hidden_start:1 * D + hidden_start + 128], pl.FP32)
                original2 = pl.cast(residual[token_start:token_start + 32, 2 * D + hidden_start:2 * D + hidden_start + 128], pl.FP32)
                original3 = pl.cast(residual[token_start:token_start + 32, 3 * D + hidden_start:3 * D + hidden_start + 128], pl.FP32)
                updated0 = pl.row_expand_mul(input_tile, post0)
                updated0 = pl.add(updated0, pl.row_expand_mul(original0, comb00))
                updated0 = pl.add(updated0, pl.row_expand_mul(original1, comb01))
                updated0 = pl.add(updated0, pl.row_expand_mul(original2, comb02))
                updated0 = pl.add(updated0, pl.row_expand_mul(original3, comb03))
                updated1 = pl.row_expand_mul(input_tile, post1)
                updated1 = pl.add(updated1, pl.row_expand_mul(original0, comb10))
                updated1 = pl.add(updated1, pl.row_expand_mul(original1, comb11))
                updated1 = pl.add(updated1, pl.row_expand_mul(original2, comb12))
                updated1 = pl.add(updated1, pl.row_expand_mul(original3, comb13))
                updated2 = pl.row_expand_mul(input_tile, post2)
                updated2 = pl.add(updated2, pl.row_expand_mul(original0, comb20))
                updated2 = pl.add(updated2, pl.row_expand_mul(original1, comb21))
                updated2 = pl.add(updated2, pl.row_expand_mul(original2, comb22))
                updated2 = pl.add(updated2, pl.row_expand_mul(original3, comb23))
                updated3 = pl.row_expand_mul(input_tile, post3)
                updated3 = pl.add(updated3, pl.row_expand_mul(original0, comb30))
                updated3 = pl.add(updated3, pl.row_expand_mul(original1, comb31))
                updated3 = pl.add(updated3, pl.row_expand_mul(original2, comb32))
                updated3 = pl.add(updated3, pl.row_expand_mul(original3, comb33))
                square_sum = pl.row_sum(pl.add(pl.add(pl.mul(updated0, updated0), pl.mul(updated1, updated1)), pl.add(pl.mul(updated2, updated2), pl.mul(updated3, updated3))))
                square_acc = pl.add(square_acc, pl.reshape(square_sum, [1, 32]))
                updated_fp32[token_start:token_start + 32, 0 * D + hidden_start:0 * D + hidden_start + 128] = updated0
                residual[token_start:token_start + 32, 0 * D + hidden_start:0 * D + hidden_start + 128] = pl.cast(updated0, pl.BF16, mode="rint")
                updated_fp32[token_start:token_start + 32, 1 * D + hidden_start:1 * D + hidden_start + 128] = updated1
                residual[token_start:token_start + 32, 1 * D + hidden_start:1 * D + hidden_start + 128] = pl.cast(updated1, pl.BF16, mode="rint")
                updated_fp32[token_start:token_start + 32, 2 * D + hidden_start:2 * D + hidden_start + 128] = updated2
                residual[token_start:token_start + 32, 2 * D + hidden_start:2 * D + hidden_start + 128] = pl.cast(updated2, pl.BF16, mode="rint")
                updated_fp32[token_start:token_start + 32, 3 * D + hidden_start:3 * D + hidden_start + 128] = updated3
                residual[token_start:token_start + 32, 3 * D + hidden_start:3 * D + hidden_start + 128] = pl.cast(updated3, pl.BF16, mode="rint")
            square_sum = pl.reshape(square_acc, [32, 1])
            mean_square = pl.add(pl.mul(square_sum, 1.0 / FD), RMS_EPS)
            inverse_rms = pl.rsqrt(mean_square)
            inverse_rms = pl.mul(inverse_rms, pl.add(pl.mul(pl.mul(mean_square, pl.mul(inverse_rms, inverse_rms)), -0.5), 1.5))
            rms_scratch[0:1, token_start:token_start + 32] = pl.reshape(inverse_rms, [1, 32])
    for token_start in pl.parallel(0, T, 128):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="bridge_cube_projection_0"):
            norm_row = pl.slice(normw, [1, 32], [0, 0], valid_shape=[1, 24], clamp=True)
            first_updated = pl.slice(updated_fp32, [128, 128], [token_start, 0], valid_shape=[128, 128])
            first_weight = pl.slice(fn, [32, 128], [0, 0], valid_shape=[24, 128], clamp=True)
            accumulator = pl.matmul(first_updated, first_weight, b_trans=True)
            for hidden_block in pl.range(1, FD // 128):
                hidden_start = hidden_block * 128
                updated_chunk = pl.slice(updated_fp32, [128, 128], [token_start, hidden_start], valid_shape=[128, 128])
                weight_chunk = pl.slice(fn, [32, 128], [0, hidden_start], valid_shape=[24, 128], clamp=True)
                accumulator = pl.matmul_acc(accumulator, updated_chunk, weight_chunk, b_trans=True)
            rms_tile = pl.reshape(pl.slice(rms_scratch, [1, 128], [0, token_start], valid_shape=[1, 128]), [128, 1])
            predicted = pl.row_expand_mul(pl.col_expand_mul(accumulator, norm_row), rms_tile)
            next_mixes[token_start:token_start + 128, 0:24] = predicted[0:128, 0:24]
            scale_pre = pl.read(mhc_scale, [0])
            predicted_planes = pl.transpose(predicted, 0, 1)
            pre_coefficient0 = pl.reshape(predicted_planes[0:1, 0:128], [128, 1])
            pre_raw0 = pl.add(pl.mul(pre_coefficient0, scale_pre), pl.read(mhc_base, [0, 0]))
            next_pre0 = pl.add(pl.recip(pl.add(pl.exp(pl.neg(pre_raw0)), 1.0)), PRE_EPS)
            pre_coefficient1 = pl.reshape(predicted_planes[1:2, 0:128], [128, 1])
            pre_raw1 = pl.add(pl.mul(pre_coefficient1, scale_pre), pl.read(mhc_base, [0, 1]))
            next_pre1 = pl.add(pl.recip(pl.add(pl.exp(pl.neg(pre_raw1)), 1.0)), PRE_EPS)
            pre_coefficient2 = pl.reshape(predicted_planes[2:3, 0:128], [128, 1])
            pre_raw2 = pl.add(pl.mul(pre_coefficient2, scale_pre), pl.read(mhc_base, [0, 2]))
            next_pre2 = pl.add(pl.recip(pl.add(pl.exp(pl.neg(pre_raw2)), 1.0)), PRE_EPS)
            pre_coefficient3 = pl.reshape(predicted_planes[3:4, 0:128], [128, 1])
            pre_raw3 = pl.add(pl.mul(pre_coefficient3, scale_pre), pl.read(mhc_base, [0, 3]))
            next_pre3 = pl.add(pl.recip(pl.add(pl.exp(pl.neg(pre_raw3)), 1.0)), PRE_EPS)
            for hidden_block in pl.range(D // 128):
                hidden_start = hidden_block * 128
            for hidden_block in pl.range(D // 128):
                hidden_start = hidden_block * 128
                updated_copy0 = updated_fp32[token_start:token_start + 128, 0 * D + hidden_start:0 * D + hidden_start + 128]
                hidden = pl.row_expand_mul(updated_copy0, next_pre0)
                updated_copy1 = updated_fp32[token_start:token_start + 128, 1 * D + hidden_start:1 * D + hidden_start + 128]
                hidden = pl.add(hidden, pl.row_expand_mul(updated_copy1, next_pre1))
                updated_copy2 = updated_fp32[token_start:token_start + 128, 2 * D + hidden_start:2 * D + hidden_start + 128]
                hidden = pl.add(hidden, pl.row_expand_mul(updated_copy2, next_pre2))
                updated_copy3 = updated_fp32[token_start:token_start + 128, 3 * D + hidden_start:3 * D + hidden_start + 128]
                hidden = pl.add(hidden, pl.row_expand_mul(updated_copy3, next_pre3))
                h_next[token_start:token_start + 128, hidden_start:hidden_start + 128] = pl.cast(hidden, pl.BF16, mode="rint")
    return h_next



@pl.jit
def mega_mhc_bridge_compact128(x: pl.Tensor[[T, D], pl.BF16], residual: pl.InOut[pl.Tensor[[T, FD], pl.BF16]], input_mixes: pl.Tensor[[T, M3], pl.FP32], fn: pl.Tensor[[M3, FD], pl.FP32], normw: pl.Tensor[[1, M3], pl.FP32], mhc_scale: pl.Tensor[[3], pl.FP32], mhc_base: pl.Tensor[[1, M3], pl.FP32], next_mixes: pl.Out[pl.Tensor[[T, M3], pl.FP32]], h_next: pl.Out[pl.Tensor[[T, D], pl.BF16]]):
    updated_fp32 = pl.create_tensor([T, FD], dtype=pl.FP32)
    rms_scratch = pl.create_tensor([1, T], dtype=pl.FP32)
    for task_index in pl.spmd(T // 128, name_hint="macro_producer_128"):
        macro_start = task_index * 128
        scale_pre = pl.read(mhc_scale, [0])
        scale_post = pl.read(mhc_scale, [1])
        scale_comb = pl.read(mhc_scale, [2])
        mix_planes = pl.transpose(input_mixes[macro_start:macro_start + 128, 0:M3], 0, 1)
        post_raw0 = pl.add(pl.mul(pl.reshape(mix_planes[4:5, 0:128], [1, 128]), scale_post), pl.read(mhc_base, [0, 4]))
        post0_planes = pl.mul(pl.recip(pl.add(pl.exp(pl.neg(post_raw0)), 1.0)), POST_MULT)
        post_raw1 = pl.add(pl.mul(pl.reshape(mix_planes[5:6, 0:128], [1, 128]), scale_post), pl.read(mhc_base, [0, 5]))
        post1_planes = pl.mul(pl.recip(pl.add(pl.exp(pl.neg(post_raw1)), 1.0)), POST_MULT)
        post_raw2 = pl.add(pl.mul(pl.reshape(mix_planes[6:7, 0:128], [1, 128]), scale_post), pl.read(mhc_base, [0, 6]))
        post2_planes = pl.mul(pl.recip(pl.add(pl.exp(pl.neg(post_raw2)), 1.0)), POST_MULT)
        post_raw3 = pl.add(pl.mul(pl.reshape(mix_planes[7:8, 0:128], [1, 128]), scale_post), pl.read(mhc_base, [0, 7]))
        post3_planes = pl.mul(pl.recip(pl.add(pl.exp(pl.neg(post_raw3)), 1.0)), POST_MULT)
        comb00_planes = pl.add(pl.mul(pl.reshape(mix_planes[8:9, 0:128], [1, 128]), scale_comb), pl.read(mhc_base, [0, 8]))
        comb01_planes = pl.add(pl.mul(pl.reshape(mix_planes[9:10, 0:128], [1, 128]), scale_comb), pl.read(mhc_base, [0, 9]))
        comb02_planes = pl.add(pl.mul(pl.reshape(mix_planes[10:11, 0:128], [1, 128]), scale_comb), pl.read(mhc_base, [0, 10]))
        comb03_planes = pl.add(pl.mul(pl.reshape(mix_planes[11:12, 0:128], [1, 128]), scale_comb), pl.read(mhc_base, [0, 11]))
        row_max0 = pl.maximum(pl.maximum(comb00_planes, comb01_planes), pl.maximum(comb02_planes, comb03_planes))
        comb00_planes = pl.exp(pl.sub(comb00_planes, row_max0))
        comb01_planes = pl.exp(pl.sub(comb01_planes, row_max0))
        comb02_planes = pl.exp(pl.sub(comb02_planes, row_max0))
        comb03_planes = pl.exp(pl.sub(comb03_planes, row_max0))
        row_sum0 = pl.add(pl.add(comb00_planes, comb01_planes), pl.add(comb02_planes, comb03_planes))
        comb00_planes = pl.add(pl.div(comb00_planes, row_sum0), PRE_EPS)
        comb01_planes = pl.add(pl.div(comb01_planes, row_sum0), PRE_EPS)
        comb02_planes = pl.add(pl.div(comb02_planes, row_sum0), PRE_EPS)
        comb03_planes = pl.add(pl.div(comb03_planes, row_sum0), PRE_EPS)
        comb10_planes = pl.add(pl.mul(pl.reshape(mix_planes[12:13, 0:128], [1, 128]), scale_comb), pl.read(mhc_base, [0, 12]))
        comb11_planes = pl.add(pl.mul(pl.reshape(mix_planes[13:14, 0:128], [1, 128]), scale_comb), pl.read(mhc_base, [0, 13]))
        comb12_planes = pl.add(pl.mul(pl.reshape(mix_planes[14:15, 0:128], [1, 128]), scale_comb), pl.read(mhc_base, [0, 14]))
        comb13_planes = pl.add(pl.mul(pl.reshape(mix_planes[15:16, 0:128], [1, 128]), scale_comb), pl.read(mhc_base, [0, 15]))
        row_max1 = pl.maximum(pl.maximum(comb10_planes, comb11_planes), pl.maximum(comb12_planes, comb13_planes))
        comb10_planes = pl.exp(pl.sub(comb10_planes, row_max1))
        comb11_planes = pl.exp(pl.sub(comb11_planes, row_max1))
        comb12_planes = pl.exp(pl.sub(comb12_planes, row_max1))
        comb13_planes = pl.exp(pl.sub(comb13_planes, row_max1))
        row_sum1 = pl.add(pl.add(comb10_planes, comb11_planes), pl.add(comb12_planes, comb13_planes))
        comb10_planes = pl.add(pl.div(comb10_planes, row_sum1), PRE_EPS)
        comb11_planes = pl.add(pl.div(comb11_planes, row_sum1), PRE_EPS)
        comb12_planes = pl.add(pl.div(comb12_planes, row_sum1), PRE_EPS)
        comb13_planes = pl.add(pl.div(comb13_planes, row_sum1), PRE_EPS)
        comb20_planes = pl.add(pl.mul(pl.reshape(mix_planes[16:17, 0:128], [1, 128]), scale_comb), pl.read(mhc_base, [0, 16]))
        comb21_planes = pl.add(pl.mul(pl.reshape(mix_planes[17:18, 0:128], [1, 128]), scale_comb), pl.read(mhc_base, [0, 17]))
        comb22_planes = pl.add(pl.mul(pl.reshape(mix_planes[18:19, 0:128], [1, 128]), scale_comb), pl.read(mhc_base, [0, 18]))
        comb23_planes = pl.add(pl.mul(pl.reshape(mix_planes[19:20, 0:128], [1, 128]), scale_comb), pl.read(mhc_base, [0, 19]))
        row_max2 = pl.maximum(pl.maximum(comb20_planes, comb21_planes), pl.maximum(comb22_planes, comb23_planes))
        comb20_planes = pl.exp(pl.sub(comb20_planes, row_max2))
        comb21_planes = pl.exp(pl.sub(comb21_planes, row_max2))
        comb22_planes = pl.exp(pl.sub(comb22_planes, row_max2))
        comb23_planes = pl.exp(pl.sub(comb23_planes, row_max2))
        row_sum2 = pl.add(pl.add(comb20_planes, comb21_planes), pl.add(comb22_planes, comb23_planes))
        comb20_planes = pl.add(pl.div(comb20_planes, row_sum2), PRE_EPS)
        comb21_planes = pl.add(pl.div(comb21_planes, row_sum2), PRE_EPS)
        comb22_planes = pl.add(pl.div(comb22_planes, row_sum2), PRE_EPS)
        comb23_planes = pl.add(pl.div(comb23_planes, row_sum2), PRE_EPS)
        comb30_planes = pl.add(pl.mul(pl.reshape(mix_planes[20:21, 0:128], [1, 128]), scale_comb), pl.read(mhc_base, [0, 20]))
        comb31_planes = pl.add(pl.mul(pl.reshape(mix_planes[21:22, 0:128], [1, 128]), scale_comb), pl.read(mhc_base, [0, 21]))
        comb32_planes = pl.add(pl.mul(pl.reshape(mix_planes[22:23, 0:128], [1, 128]), scale_comb), pl.read(mhc_base, [0, 22]))
        comb33_planes = pl.add(pl.mul(pl.reshape(mix_planes[23:24, 0:128], [1, 128]), scale_comb), pl.read(mhc_base, [0, 23]))
        row_max3 = pl.maximum(pl.maximum(comb30_planes, comb31_planes), pl.maximum(comb32_planes, comb33_planes))
        comb30_planes = pl.exp(pl.sub(comb30_planes, row_max3))
        comb31_planes = pl.exp(pl.sub(comb31_planes, row_max3))
        comb32_planes = pl.exp(pl.sub(comb32_planes, row_max3))
        comb33_planes = pl.exp(pl.sub(comb33_planes, row_max3))
        row_sum3 = pl.add(pl.add(comb30_planes, comb31_planes), pl.add(comb32_planes, comb33_planes))
        comb30_planes = pl.add(pl.div(comb30_planes, row_sum3), PRE_EPS)
        comb31_planes = pl.add(pl.div(comb31_planes, row_sum3), PRE_EPS)
        comb32_planes = pl.add(pl.div(comb32_planes, row_sum3), PRE_EPS)
        comb33_planes = pl.add(pl.div(comb33_planes, row_sum3), PRE_EPS)
        column_sum0 = pl.add(pl.add(pl.add(comb00_planes, comb10_planes), pl.add(comb20_planes, comb30_planes)), PRE_EPS)
        comb00_planes = pl.div(comb00_planes, column_sum0)
        comb10_planes = pl.div(comb10_planes, column_sum0)
        comb20_planes = pl.div(comb20_planes, column_sum0)
        comb30_planes = pl.div(comb30_planes, column_sum0)
        column_sum1 = pl.add(pl.add(pl.add(comb01_planes, comb11_planes), pl.add(comb21_planes, comb31_planes)), PRE_EPS)
        comb01_planes = pl.div(comb01_planes, column_sum1)
        comb11_planes = pl.div(comb11_planes, column_sum1)
        comb21_planes = pl.div(comb21_planes, column_sum1)
        comb31_planes = pl.div(comb31_planes, column_sum1)
        column_sum2 = pl.add(pl.add(pl.add(comb02_planes, comb12_planes), pl.add(comb22_planes, comb32_planes)), PRE_EPS)
        comb02_planes = pl.div(comb02_planes, column_sum2)
        comb12_planes = pl.div(comb12_planes, column_sum2)
        comb22_planes = pl.div(comb22_planes, column_sum2)
        comb32_planes = pl.div(comb32_planes, column_sum2)
        column_sum3 = pl.add(pl.add(pl.add(comb03_planes, comb13_planes), pl.add(comb23_planes, comb33_planes)), PRE_EPS)
        comb03_planes = pl.div(comb03_planes, column_sum3)
        comb13_planes = pl.div(comb13_planes, column_sum3)
        comb23_planes = pl.div(comb23_planes, column_sum3)
        comb33_planes = pl.div(comb33_planes, column_sum3)
        for iteration in pl.range(SINK - 1):
            row_sum0 = pl.add(pl.add(pl.add(comb00_planes, comb01_planes), pl.add(comb02_planes, comb03_planes)), PRE_EPS)
            comb00_planes = pl.div(comb00_planes, row_sum0)
            comb01_planes = pl.div(comb01_planes, row_sum0)
            comb02_planes = pl.div(comb02_planes, row_sum0)
            comb03_planes = pl.div(comb03_planes, row_sum0)
            row_sum1 = pl.add(pl.add(pl.add(comb10_planes, comb11_planes), pl.add(comb12_planes, comb13_planes)), PRE_EPS)
            comb10_planes = pl.div(comb10_planes, row_sum1)
            comb11_planes = pl.div(comb11_planes, row_sum1)
            comb12_planes = pl.div(comb12_planes, row_sum1)
            comb13_planes = pl.div(comb13_planes, row_sum1)
            row_sum2 = pl.add(pl.add(pl.add(comb20_planes, comb21_planes), pl.add(comb22_planes, comb23_planes)), PRE_EPS)
            comb20_planes = pl.div(comb20_planes, row_sum2)
            comb21_planes = pl.div(comb21_planes, row_sum2)
            comb22_planes = pl.div(comb22_planes, row_sum2)
            comb23_planes = pl.div(comb23_planes, row_sum2)
            row_sum3 = pl.add(pl.add(pl.add(comb30_planes, comb31_planes), pl.add(comb32_planes, comb33_planes)), PRE_EPS)
            comb30_planes = pl.div(comb30_planes, row_sum3)
            comb31_planes = pl.div(comb31_planes, row_sum3)
            comb32_planes = pl.div(comb32_planes, row_sum3)
            comb33_planes = pl.div(comb33_planes, row_sum3)
            column_sum0 = pl.add(pl.add(pl.add(comb00_planes, comb10_planes), pl.add(comb20_planes, comb30_planes)), PRE_EPS)
            comb00_planes = pl.div(comb00_planes, column_sum0)
            comb10_planes = pl.div(comb10_planes, column_sum0)
            comb20_planes = pl.div(comb20_planes, column_sum0)
            comb30_planes = pl.div(comb30_planes, column_sum0)
            column_sum1 = pl.add(pl.add(pl.add(comb01_planes, comb11_planes), pl.add(comb21_planes, comb31_planes)), PRE_EPS)
            comb01_planes = pl.div(comb01_planes, column_sum1)
            comb11_planes = pl.div(comb11_planes, column_sum1)
            comb21_planes = pl.div(comb21_planes, column_sum1)
            comb31_planes = pl.div(comb31_planes, column_sum1)
            column_sum2 = pl.add(pl.add(pl.add(comb02_planes, comb12_planes), pl.add(comb22_planes, comb32_planes)), PRE_EPS)
            comb02_planes = pl.div(comb02_planes, column_sum2)
            comb12_planes = pl.div(comb12_planes, column_sum2)
            comb22_planes = pl.div(comb22_planes, column_sum2)
            comb32_planes = pl.div(comb32_planes, column_sum2)
            column_sum3 = pl.add(pl.add(pl.add(comb03_planes, comb13_planes), pl.add(comb23_planes, comb33_planes)), PRE_EPS)
            comb03_planes = pl.div(comb03_planes, column_sum3)
            comb13_planes = pl.div(comb13_planes, column_sum3)
            comb23_planes = pl.div(comb23_planes, column_sum3)
            comb33_planes = pl.div(comb33_planes, column_sum3)
        for update_block in pl.range(128 // 32):
            update_offset = update_block * 32
            token_start = macro_start + update_offset
            post0 = pl.reshape(post0_planes[0:1, update_offset:update_offset + 32], [32, 1])
            post1 = pl.reshape(post1_planes[0:1, update_offset:update_offset + 32], [32, 1])
            post2 = pl.reshape(post2_planes[0:1, update_offset:update_offset + 32], [32, 1])
            post3 = pl.reshape(post3_planes[0:1, update_offset:update_offset + 32], [32, 1])
            comb00 = pl.reshape(comb00_planes[0:1, update_offset:update_offset + 32], [32, 1])
            comb01 = pl.reshape(comb01_planes[0:1, update_offset:update_offset + 32], [32, 1])
            comb02 = pl.reshape(comb02_planes[0:1, update_offset:update_offset + 32], [32, 1])
            comb03 = pl.reshape(comb03_planes[0:1, update_offset:update_offset + 32], [32, 1])
            comb10 = pl.reshape(comb10_planes[0:1, update_offset:update_offset + 32], [32, 1])
            comb11 = pl.reshape(comb11_planes[0:1, update_offset:update_offset + 32], [32, 1])
            comb12 = pl.reshape(comb12_planes[0:1, update_offset:update_offset + 32], [32, 1])
            comb13 = pl.reshape(comb13_planes[0:1, update_offset:update_offset + 32], [32, 1])
            comb20 = pl.reshape(comb20_planes[0:1, update_offset:update_offset + 32], [32, 1])
            comb21 = pl.reshape(comb21_planes[0:1, update_offset:update_offset + 32], [32, 1])
            comb22 = pl.reshape(comb22_planes[0:1, update_offset:update_offset + 32], [32, 1])
            comb23 = pl.reshape(comb23_planes[0:1, update_offset:update_offset + 32], [32, 1])
            comb30 = pl.reshape(comb30_planes[0:1, update_offset:update_offset + 32], [32, 1])
            comb31 = pl.reshape(comb31_planes[0:1, update_offset:update_offset + 32], [32, 1])
            comb32 = pl.reshape(comb32_planes[0:1, update_offset:update_offset + 32], [32, 1])
            comb33 = pl.reshape(comb33_planes[0:1, update_offset:update_offset + 32], [32, 1])
            square_acc = pl.full([1, 32], dtype=pl.FP32, value=0.0)
            for hidden_block in pl.range(D // 128):
                hidden_start = hidden_block * 128
                input_tile = pl.cast(x[token_start:token_start + 32, hidden_start:hidden_start + 128], pl.FP32)
                original0 = pl.cast(residual[token_start:token_start + 32, 0 * D + hidden_start:0 * D + hidden_start + 128], pl.FP32)
                original1 = pl.cast(residual[token_start:token_start + 32, 1 * D + hidden_start:1 * D + hidden_start + 128], pl.FP32)
                original2 = pl.cast(residual[token_start:token_start + 32, 2 * D + hidden_start:2 * D + hidden_start + 128], pl.FP32)
                original3 = pl.cast(residual[token_start:token_start + 32, 3 * D + hidden_start:3 * D + hidden_start + 128], pl.FP32)
                updated0 = pl.row_expand_mul(input_tile, post0)
                updated0 = pl.add(updated0, pl.row_expand_mul(original0, comb00))
                updated0 = pl.add(updated0, pl.row_expand_mul(original1, comb01))
                updated0 = pl.add(updated0, pl.row_expand_mul(original2, comb02))
                updated0 = pl.add(updated0, pl.row_expand_mul(original3, comb03))
                updated1 = pl.row_expand_mul(input_tile, post1)
                updated1 = pl.add(updated1, pl.row_expand_mul(original0, comb10))
                updated1 = pl.add(updated1, pl.row_expand_mul(original1, comb11))
                updated1 = pl.add(updated1, pl.row_expand_mul(original2, comb12))
                updated1 = pl.add(updated1, pl.row_expand_mul(original3, comb13))
                updated2 = pl.row_expand_mul(input_tile, post2)
                updated2 = pl.add(updated2, pl.row_expand_mul(original0, comb20))
                updated2 = pl.add(updated2, pl.row_expand_mul(original1, comb21))
                updated2 = pl.add(updated2, pl.row_expand_mul(original2, comb22))
                updated2 = pl.add(updated2, pl.row_expand_mul(original3, comb23))
                updated3 = pl.row_expand_mul(input_tile, post3)
                updated3 = pl.add(updated3, pl.row_expand_mul(original0, comb30))
                updated3 = pl.add(updated3, pl.row_expand_mul(original1, comb31))
                updated3 = pl.add(updated3, pl.row_expand_mul(original2, comb32))
                updated3 = pl.add(updated3, pl.row_expand_mul(original3, comb33))
                square_sum = pl.row_sum(pl.add(pl.add(pl.mul(updated0, updated0), pl.mul(updated1, updated1)), pl.add(pl.mul(updated2, updated2), pl.mul(updated3, updated3))))
                square_acc = pl.add(square_acc, pl.reshape(square_sum, [1, 32]))
                updated_fp32[token_start:token_start + 32, 0 * D + hidden_start:0 * D + hidden_start + 128] = updated0
                residual[token_start:token_start + 32, 0 * D + hidden_start:0 * D + hidden_start + 128] = pl.cast(updated0, pl.BF16, mode="rint")
                updated_fp32[token_start:token_start + 32, 1 * D + hidden_start:1 * D + hidden_start + 128] = updated1
                residual[token_start:token_start + 32, 1 * D + hidden_start:1 * D + hidden_start + 128] = pl.cast(updated1, pl.BF16, mode="rint")
                updated_fp32[token_start:token_start + 32, 2 * D + hidden_start:2 * D + hidden_start + 128] = updated2
                residual[token_start:token_start + 32, 2 * D + hidden_start:2 * D + hidden_start + 128] = pl.cast(updated2, pl.BF16, mode="rint")
                updated_fp32[token_start:token_start + 32, 3 * D + hidden_start:3 * D + hidden_start + 128] = updated3
                residual[token_start:token_start + 32, 3 * D + hidden_start:3 * D + hidden_start + 128] = pl.cast(updated3, pl.BF16, mode="rint")
            square_sum = pl.reshape(square_acc, [32, 1])
            mean_square = pl.add(pl.mul(square_sum, 1.0 / FD), RMS_EPS)
            inverse_rms = pl.rsqrt(mean_square)
            inverse_rms = pl.mul(inverse_rms, pl.add(pl.mul(pl.mul(mean_square, pl.mul(inverse_rms, inverse_rms)), -0.5), 1.5))
            rms_scratch[0:1, token_start:token_start + 32] = pl.reshape(inverse_rms, [1, 32])
    for token_start in pl.parallel(0, T, 128):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="bridge_cube_projection_0"):
            norm_row = pl.slice(normw, [1, 32], [0, 0], valid_shape=[1, 24], clamp=True)
            first_updated = pl.slice(updated_fp32, [128, 128], [token_start, 0], valid_shape=[128, 128])
            first_weight = pl.slice(fn, [32, 128], [0, 0], valid_shape=[24, 128], clamp=True)
            accumulator = pl.matmul(first_updated, first_weight, b_trans=True)
            for hidden_block in pl.range(1, FD // 128):
                hidden_start = hidden_block * 128
                updated_chunk = pl.slice(updated_fp32, [128, 128], [token_start, hidden_start], valid_shape=[128, 128])
                weight_chunk = pl.slice(fn, [32, 128], [0, hidden_start], valid_shape=[24, 128], clamp=True)
                accumulator = pl.matmul_acc(accumulator, updated_chunk, weight_chunk, b_trans=True)
            rms_tile = pl.reshape(pl.slice(rms_scratch, [1, 128], [0, token_start], valid_shape=[1, 128]), [128, 1])
            predicted = pl.row_expand_mul(pl.col_expand_mul(accumulator, norm_row), rms_tile)
            next_mixes[token_start:token_start + 128, 0:24] = predicted[0:128, 0:24]
            scale_pre = pl.read(mhc_scale, [0])
            predicted_planes = pl.transpose(predicted, 0, 1)
            pre_coefficient0 = pl.reshape(predicted_planes[0:1, 0:128], [128, 1])
            pre_raw0 = pl.add(pl.mul(pre_coefficient0, scale_pre), pl.read(mhc_base, [0, 0]))
            next_pre0 = pl.add(pl.recip(pl.add(pl.exp(pl.neg(pre_raw0)), 1.0)), PRE_EPS)
            pre_coefficient1 = pl.reshape(predicted_planes[1:2, 0:128], [128, 1])
            pre_raw1 = pl.add(pl.mul(pre_coefficient1, scale_pre), pl.read(mhc_base, [0, 1]))
            next_pre1 = pl.add(pl.recip(pl.add(pl.exp(pl.neg(pre_raw1)), 1.0)), PRE_EPS)
            pre_coefficient2 = pl.reshape(predicted_planes[2:3, 0:128], [128, 1])
            pre_raw2 = pl.add(pl.mul(pre_coefficient2, scale_pre), pl.read(mhc_base, [0, 2]))
            next_pre2 = pl.add(pl.recip(pl.add(pl.exp(pl.neg(pre_raw2)), 1.0)), PRE_EPS)
            pre_coefficient3 = pl.reshape(predicted_planes[3:4, 0:128], [128, 1])
            pre_raw3 = pl.add(pl.mul(pre_coefficient3, scale_pre), pl.read(mhc_base, [0, 3]))
            next_pre3 = pl.add(pl.recip(pl.add(pl.exp(pl.neg(pre_raw3)), 1.0)), PRE_EPS)
            for hidden_block in pl.range(D // 128):
                hidden_start = hidden_block * 128
            for hidden_block in pl.range(D // 128):
                hidden_start = hidden_block * 128
                updated_copy0 = updated_fp32[token_start:token_start + 128, 0 * D + hidden_start:0 * D + hidden_start + 128]
                hidden = pl.row_expand_mul(updated_copy0, next_pre0)
                updated_copy1 = updated_fp32[token_start:token_start + 128, 1 * D + hidden_start:1 * D + hidden_start + 128]
                hidden = pl.add(hidden, pl.row_expand_mul(updated_copy1, next_pre1))
                updated_copy2 = updated_fp32[token_start:token_start + 128, 2 * D + hidden_start:2 * D + hidden_start + 128]
                hidden = pl.add(hidden, pl.row_expand_mul(updated_copy2, next_pre2))
                updated_copy3 = updated_fp32[token_start:token_start + 128, 3 * D + hidden_start:3 * D + hidden_start + 128]
                hidden = pl.add(hidden, pl.row_expand_mul(updated_copy3, next_pre3))
                h_next[token_start:token_start + 128, hidden_start:hidden_start + 128] = pl.cast(hidden, pl.BF16, mode="rint")
    return h_next



@pl.jit.incore
def chunk_producer_32(
    x: pl.Tensor[[512, D], pl.BF16],
    residual: pl.InOut[pl.Tensor[[512, FD], pl.BF16]],
    input_mixes: pl.Tensor[[512, M3], pl.FP32],
    mhc_scale: pl.Tensor[[3], pl.FP32],
    mhc_base: pl.Tensor[[1, M3], pl.FP32],
    updated_fp32: pl.InOut[pl.Tensor[[512, FD], pl.FP32]],
    rms_scratch: pl.InOut[pl.Tensor[[1, 512], pl.FP32]],
):
    task_index = pl.tile.get_block_idx()
    macro_start = task_index * 32
    scale_pre = pl.read(mhc_scale, [0])
    scale_post = pl.read(mhc_scale, [1])
    scale_comb = pl.read(mhc_scale, [2])
    mix_planes = pl.transpose(input_mixes[macro_start:macro_start + 32, 0:M3], 0, 1)
    post_raw0 = pl.add(pl.mul(pl.reshape(mix_planes[4:5, 0:32], [1, 32]), scale_post), pl.read(mhc_base, [0, 4]))
    post0_planes = pl.mul(pl.recip(pl.add(pl.exp(pl.neg(post_raw0)), 1.0)), POST_MULT)
    post_raw1 = pl.add(pl.mul(pl.reshape(mix_planes[5:6, 0:32], [1, 32]), scale_post), pl.read(mhc_base, [0, 5]))
    post1_planes = pl.mul(pl.recip(pl.add(pl.exp(pl.neg(post_raw1)), 1.0)), POST_MULT)
    post_raw2 = pl.add(pl.mul(pl.reshape(mix_planes[6:7, 0:32], [1, 32]), scale_post), pl.read(mhc_base, [0, 6]))
    post2_planes = pl.mul(pl.recip(pl.add(pl.exp(pl.neg(post_raw2)), 1.0)), POST_MULT)
    post_raw3 = pl.add(pl.mul(pl.reshape(mix_planes[7:8, 0:32], [1, 32]), scale_post), pl.read(mhc_base, [0, 7]))
    post3_planes = pl.mul(pl.recip(pl.add(pl.exp(pl.neg(post_raw3)), 1.0)), POST_MULT)
    comb00_planes = pl.add(pl.mul(pl.reshape(mix_planes[8:9, 0:32], [1, 32]), scale_comb), pl.read(mhc_base, [0, 8]))
    comb01_planes = pl.add(pl.mul(pl.reshape(mix_planes[9:10, 0:32], [1, 32]), scale_comb), pl.read(mhc_base, [0, 9]))
    comb02_planes = pl.add(pl.mul(pl.reshape(mix_planes[10:11, 0:32], [1, 32]), scale_comb), pl.read(mhc_base, [0, 10]))
    comb03_planes = pl.add(pl.mul(pl.reshape(mix_planes[11:12, 0:32], [1, 32]), scale_comb), pl.read(mhc_base, [0, 11]))
    row_max0 = pl.maximum(pl.maximum(comb00_planes, comb01_planes), pl.maximum(comb02_planes, comb03_planes))
    comb00_planes = pl.exp(pl.sub(comb00_planes, row_max0))
    comb01_planes = pl.exp(pl.sub(comb01_planes, row_max0))
    comb02_planes = pl.exp(pl.sub(comb02_planes, row_max0))
    comb03_planes = pl.exp(pl.sub(comb03_planes, row_max0))
    row_sum0 = pl.add(pl.add(comb00_planes, comb01_planes), pl.add(comb02_planes, comb03_planes))
    comb00_planes = pl.add(pl.div(comb00_planes, row_sum0), PRE_EPS)
    comb01_planes = pl.add(pl.div(comb01_planes, row_sum0), PRE_EPS)
    comb02_planes = pl.add(pl.div(comb02_planes, row_sum0), PRE_EPS)
    comb03_planes = pl.add(pl.div(comb03_planes, row_sum0), PRE_EPS)
    comb10_planes = pl.add(pl.mul(pl.reshape(mix_planes[12:13, 0:32], [1, 32]), scale_comb), pl.read(mhc_base, [0, 12]))
    comb11_planes = pl.add(pl.mul(pl.reshape(mix_planes[13:14, 0:32], [1, 32]), scale_comb), pl.read(mhc_base, [0, 13]))
    comb12_planes = pl.add(pl.mul(pl.reshape(mix_planes[14:15, 0:32], [1, 32]), scale_comb), pl.read(mhc_base, [0, 14]))
    comb13_planes = pl.add(pl.mul(pl.reshape(mix_planes[15:16, 0:32], [1, 32]), scale_comb), pl.read(mhc_base, [0, 15]))
    row_max1 = pl.maximum(pl.maximum(comb10_planes, comb11_planes), pl.maximum(comb12_planes, comb13_planes))
    comb10_planes = pl.exp(pl.sub(comb10_planes, row_max1))
    comb11_planes = pl.exp(pl.sub(comb11_planes, row_max1))
    comb12_planes = pl.exp(pl.sub(comb12_planes, row_max1))
    comb13_planes = pl.exp(pl.sub(comb13_planes, row_max1))
    row_sum1 = pl.add(pl.add(comb10_planes, comb11_planes), pl.add(comb12_planes, comb13_planes))
    comb10_planes = pl.add(pl.div(comb10_planes, row_sum1), PRE_EPS)
    comb11_planes = pl.add(pl.div(comb11_planes, row_sum1), PRE_EPS)
    comb12_planes = pl.add(pl.div(comb12_planes, row_sum1), PRE_EPS)
    comb13_planes = pl.add(pl.div(comb13_planes, row_sum1), PRE_EPS)
    comb20_planes = pl.add(pl.mul(pl.reshape(mix_planes[16:17, 0:32], [1, 32]), scale_comb), pl.read(mhc_base, [0, 16]))
    comb21_planes = pl.add(pl.mul(pl.reshape(mix_planes[17:18, 0:32], [1, 32]), scale_comb), pl.read(mhc_base, [0, 17]))
    comb22_planes = pl.add(pl.mul(pl.reshape(mix_planes[18:19, 0:32], [1, 32]), scale_comb), pl.read(mhc_base, [0, 18]))
    comb23_planes = pl.add(pl.mul(pl.reshape(mix_planes[19:20, 0:32], [1, 32]), scale_comb), pl.read(mhc_base, [0, 19]))
    row_max2 = pl.maximum(pl.maximum(comb20_planes, comb21_planes), pl.maximum(comb22_planes, comb23_planes))
    comb20_planes = pl.exp(pl.sub(comb20_planes, row_max2))
    comb21_planes = pl.exp(pl.sub(comb21_planes, row_max2))
    comb22_planes = pl.exp(pl.sub(comb22_planes, row_max2))
    comb23_planes = pl.exp(pl.sub(comb23_planes, row_max2))
    row_sum2 = pl.add(pl.add(comb20_planes, comb21_planes), pl.add(comb22_planes, comb23_planes))
    comb20_planes = pl.add(pl.div(comb20_planes, row_sum2), PRE_EPS)
    comb21_planes = pl.add(pl.div(comb21_planes, row_sum2), PRE_EPS)
    comb22_planes = pl.add(pl.div(comb22_planes, row_sum2), PRE_EPS)
    comb23_planes = pl.add(pl.div(comb23_planes, row_sum2), PRE_EPS)
    comb30_planes = pl.add(pl.mul(pl.reshape(mix_planes[20:21, 0:32], [1, 32]), scale_comb), pl.read(mhc_base, [0, 20]))
    comb31_planes = pl.add(pl.mul(pl.reshape(mix_planes[21:22, 0:32], [1, 32]), scale_comb), pl.read(mhc_base, [0, 21]))
    comb32_planes = pl.add(pl.mul(pl.reshape(mix_planes[22:23, 0:32], [1, 32]), scale_comb), pl.read(mhc_base, [0, 22]))
    comb33_planes = pl.add(pl.mul(pl.reshape(mix_planes[23:24, 0:32], [1, 32]), scale_comb), pl.read(mhc_base, [0, 23]))
    row_max3 = pl.maximum(pl.maximum(comb30_planes, comb31_planes), pl.maximum(comb32_planes, comb33_planes))
    comb30_planes = pl.exp(pl.sub(comb30_planes, row_max3))
    comb31_planes = pl.exp(pl.sub(comb31_planes, row_max3))
    comb32_planes = pl.exp(pl.sub(comb32_planes, row_max3))
    comb33_planes = pl.exp(pl.sub(comb33_planes, row_max3))
    row_sum3 = pl.add(pl.add(comb30_planes, comb31_planes), pl.add(comb32_planes, comb33_planes))
    comb30_planes = pl.add(pl.div(comb30_planes, row_sum3), PRE_EPS)
    comb31_planes = pl.add(pl.div(comb31_planes, row_sum3), PRE_EPS)
    comb32_planes = pl.add(pl.div(comb32_planes, row_sum3), PRE_EPS)
    comb33_planes = pl.add(pl.div(comb33_planes, row_sum3), PRE_EPS)
    column_sum0 = pl.add(pl.add(pl.add(comb00_planes, comb10_planes), pl.add(comb20_planes, comb30_planes)), PRE_EPS)
    comb00_planes = pl.div(comb00_planes, column_sum0)
    comb10_planes = pl.div(comb10_planes, column_sum0)
    comb20_planes = pl.div(comb20_planes, column_sum0)
    comb30_planes = pl.div(comb30_planes, column_sum0)
    column_sum1 = pl.add(pl.add(pl.add(comb01_planes, comb11_planes), pl.add(comb21_planes, comb31_planes)), PRE_EPS)
    comb01_planes = pl.div(comb01_planes, column_sum1)
    comb11_planes = pl.div(comb11_planes, column_sum1)
    comb21_planes = pl.div(comb21_planes, column_sum1)
    comb31_planes = pl.div(comb31_planes, column_sum1)
    column_sum2 = pl.add(pl.add(pl.add(comb02_planes, comb12_planes), pl.add(comb22_planes, comb32_planes)), PRE_EPS)
    comb02_planes = pl.div(comb02_planes, column_sum2)
    comb12_planes = pl.div(comb12_planes, column_sum2)
    comb22_planes = pl.div(comb22_planes, column_sum2)
    comb32_planes = pl.div(comb32_planes, column_sum2)
    column_sum3 = pl.add(pl.add(pl.add(comb03_planes, comb13_planes), pl.add(comb23_planes, comb33_planes)), PRE_EPS)
    comb03_planes = pl.div(comb03_planes, column_sum3)
    comb13_planes = pl.div(comb13_planes, column_sum3)
    comb23_planes = pl.div(comb23_planes, column_sum3)
    comb33_planes = pl.div(comb33_planes, column_sum3)
    for iteration in pl.range(SINK - 1):
        row_sum0 = pl.add(pl.add(pl.add(comb00_planes, comb01_planes), pl.add(comb02_planes, comb03_planes)), PRE_EPS)
        comb00_planes = pl.div(comb00_planes, row_sum0)
        comb01_planes = pl.div(comb01_planes, row_sum0)
        comb02_planes = pl.div(comb02_planes, row_sum0)
        comb03_planes = pl.div(comb03_planes, row_sum0)
        row_sum1 = pl.add(pl.add(pl.add(comb10_planes, comb11_planes), pl.add(comb12_planes, comb13_planes)), PRE_EPS)
        comb10_planes = pl.div(comb10_planes, row_sum1)
        comb11_planes = pl.div(comb11_planes, row_sum1)
        comb12_planes = pl.div(comb12_planes, row_sum1)
        comb13_planes = pl.div(comb13_planes, row_sum1)
        row_sum2 = pl.add(pl.add(pl.add(comb20_planes, comb21_planes), pl.add(comb22_planes, comb23_planes)), PRE_EPS)
        comb20_planes = pl.div(comb20_planes, row_sum2)
        comb21_planes = pl.div(comb21_planes, row_sum2)
        comb22_planes = pl.div(comb22_planes, row_sum2)
        comb23_planes = pl.div(comb23_planes, row_sum2)
        row_sum3 = pl.add(pl.add(pl.add(comb30_planes, comb31_planes), pl.add(comb32_planes, comb33_planes)), PRE_EPS)
        comb30_planes = pl.div(comb30_planes, row_sum3)
        comb31_planes = pl.div(comb31_planes, row_sum3)
        comb32_planes = pl.div(comb32_planes, row_sum3)
        comb33_planes = pl.div(comb33_planes, row_sum3)
        column_sum0 = pl.add(pl.add(pl.add(comb00_planes, comb10_planes), pl.add(comb20_planes, comb30_planes)), PRE_EPS)
        comb00_planes = pl.div(comb00_planes, column_sum0)
        comb10_planes = pl.div(comb10_planes, column_sum0)
        comb20_planes = pl.div(comb20_planes, column_sum0)
        comb30_planes = pl.div(comb30_planes, column_sum0)
        column_sum1 = pl.add(pl.add(pl.add(comb01_planes, comb11_planes), pl.add(comb21_planes, comb31_planes)), PRE_EPS)
        comb01_planes = pl.div(comb01_planes, column_sum1)
        comb11_planes = pl.div(comb11_planes, column_sum1)
        comb21_planes = pl.div(comb21_planes, column_sum1)
        comb31_planes = pl.div(comb31_planes, column_sum1)
        column_sum2 = pl.add(pl.add(pl.add(comb02_planes, comb12_planes), pl.add(comb22_planes, comb32_planes)), PRE_EPS)
        comb02_planes = pl.div(comb02_planes, column_sum2)
        comb12_planes = pl.div(comb12_planes, column_sum2)
        comb22_planes = pl.div(comb22_planes, column_sum2)
        comb32_planes = pl.div(comb32_planes, column_sum2)
        column_sum3 = pl.add(pl.add(pl.add(comb03_planes, comb13_planes), pl.add(comb23_planes, comb33_planes)), PRE_EPS)
        comb03_planes = pl.div(comb03_planes, column_sum3)
        comb13_planes = pl.div(comb13_planes, column_sum3)
        comb23_planes = pl.div(comb23_planes, column_sum3)
        comb33_planes = pl.div(comb33_planes, column_sum3)
    for update_block in pl.range(32 // 32):
        update_offset = update_block * 32
        token_start = macro_start + update_offset
        post0 = pl.reshape(post0_planes[0:1, update_offset:update_offset + 32], [32, 1])
        post1 = pl.reshape(post1_planes[0:1, update_offset:update_offset + 32], [32, 1])
        post2 = pl.reshape(post2_planes[0:1, update_offset:update_offset + 32], [32, 1])
        post3 = pl.reshape(post3_planes[0:1, update_offset:update_offset + 32], [32, 1])
        comb00 = pl.reshape(comb00_planes[0:1, update_offset:update_offset + 32], [32, 1])
        comb01 = pl.reshape(comb01_planes[0:1, update_offset:update_offset + 32], [32, 1])
        comb02 = pl.reshape(comb02_planes[0:1, update_offset:update_offset + 32], [32, 1])
        comb03 = pl.reshape(comb03_planes[0:1, update_offset:update_offset + 32], [32, 1])
        comb10 = pl.reshape(comb10_planes[0:1, update_offset:update_offset + 32], [32, 1])
        comb11 = pl.reshape(comb11_planes[0:1, update_offset:update_offset + 32], [32, 1])
        comb12 = pl.reshape(comb12_planes[0:1, update_offset:update_offset + 32], [32, 1])
        comb13 = pl.reshape(comb13_planes[0:1, update_offset:update_offset + 32], [32, 1])
        comb20 = pl.reshape(comb20_planes[0:1, update_offset:update_offset + 32], [32, 1])
        comb21 = pl.reshape(comb21_planes[0:1, update_offset:update_offset + 32], [32, 1])
        comb22 = pl.reshape(comb22_planes[0:1, update_offset:update_offset + 32], [32, 1])
        comb23 = pl.reshape(comb23_planes[0:1, update_offset:update_offset + 32], [32, 1])
        comb30 = pl.reshape(comb30_planes[0:1, update_offset:update_offset + 32], [32, 1])
        comb31 = pl.reshape(comb31_planes[0:1, update_offset:update_offset + 32], [32, 1])
        comb32 = pl.reshape(comb32_planes[0:1, update_offset:update_offset + 32], [32, 1])
        comb33 = pl.reshape(comb33_planes[0:1, update_offset:update_offset + 32], [32, 1])
        square_acc = pl.full([1, 32], dtype=pl.FP32, value=0.0)
        for hidden_block in pl.range(D // 128):
            hidden_start = hidden_block * 128
            input_tile = pl.cast(x[token_start:token_start + 32, hidden_start:hidden_start + 128], pl.FP32)
            original0 = pl.cast(residual[token_start:token_start + 32, 0 * D + hidden_start:0 * D + hidden_start + 128], pl.FP32)
            original1 = pl.cast(residual[token_start:token_start + 32, 1 * D + hidden_start:1 * D + hidden_start + 128], pl.FP32)
            original2 = pl.cast(residual[token_start:token_start + 32, 2 * D + hidden_start:2 * D + hidden_start + 128], pl.FP32)
            original3 = pl.cast(residual[token_start:token_start + 32, 3 * D + hidden_start:3 * D + hidden_start + 128], pl.FP32)
            updated0 = pl.row_expand_mul(input_tile, post0)
            updated0 = pl.add(updated0, pl.row_expand_mul(original0, comb00))
            updated0 = pl.add(updated0, pl.row_expand_mul(original1, comb01))
            updated0 = pl.add(updated0, pl.row_expand_mul(original2, comb02))
            updated0 = pl.add(updated0, pl.row_expand_mul(original3, comb03))
            updated1 = pl.row_expand_mul(input_tile, post1)
            updated1 = pl.add(updated1, pl.row_expand_mul(original0, comb10))
            updated1 = pl.add(updated1, pl.row_expand_mul(original1, comb11))
            updated1 = pl.add(updated1, pl.row_expand_mul(original2, comb12))
            updated1 = pl.add(updated1, pl.row_expand_mul(original3, comb13))
            updated2 = pl.row_expand_mul(input_tile, post2)
            updated2 = pl.add(updated2, pl.row_expand_mul(original0, comb20))
            updated2 = pl.add(updated2, pl.row_expand_mul(original1, comb21))
            updated2 = pl.add(updated2, pl.row_expand_mul(original2, comb22))
            updated2 = pl.add(updated2, pl.row_expand_mul(original3, comb23))
            updated3 = pl.row_expand_mul(input_tile, post3)
            updated3 = pl.add(updated3, pl.row_expand_mul(original0, comb30))
            updated3 = pl.add(updated3, pl.row_expand_mul(original1, comb31))
            updated3 = pl.add(updated3, pl.row_expand_mul(original2, comb32))
            updated3 = pl.add(updated3, pl.row_expand_mul(original3, comb33))
            square_sum = pl.row_sum(pl.add(pl.add(pl.mul(updated0, updated0), pl.mul(updated1, updated1)), pl.add(pl.mul(updated2, updated2), pl.mul(updated3, updated3))))
            square_acc = pl.add(square_acc, pl.reshape(square_sum, [1, 32]))
            updated_fp32[token_start:token_start + 32, 0 * D + hidden_start:0 * D + hidden_start + 128] = updated0
            residual[token_start:token_start + 32, 0 * D + hidden_start:0 * D + hidden_start + 128] = pl.cast(updated0, pl.BF16, mode="rint")
            updated_fp32[token_start:token_start + 32, 1 * D + hidden_start:1 * D + hidden_start + 128] = updated1
            residual[token_start:token_start + 32, 1 * D + hidden_start:1 * D + hidden_start + 128] = pl.cast(updated1, pl.BF16, mode="rint")
            updated_fp32[token_start:token_start + 32, 2 * D + hidden_start:2 * D + hidden_start + 128] = updated2
            residual[token_start:token_start + 32, 2 * D + hidden_start:2 * D + hidden_start + 128] = pl.cast(updated2, pl.BF16, mode="rint")
            updated_fp32[token_start:token_start + 32, 3 * D + hidden_start:3 * D + hidden_start + 128] = updated3
            residual[token_start:token_start + 32, 3 * D + hidden_start:3 * D + hidden_start + 128] = pl.cast(updated3, pl.BF16, mode="rint")
        square_sum = pl.reshape(square_acc, [32, 1])
        mean_square = pl.add(pl.mul(square_sum, 1.0 / FD), RMS_EPS)
        inverse_rms = pl.rsqrt(mean_square)
        inverse_rms = pl.mul(inverse_rms, pl.add(pl.mul(pl.mul(mean_square, pl.mul(inverse_rms, inverse_rms)), -0.5), 1.5))
        rms_scratch[0:1, token_start:token_start + 32] = pl.reshape(inverse_rms, [1, 32])
    return residual, updated_fp32, rms_scratch


@pl.jit.incore
def chunk_consumer(
    updated_fp32: pl.Tensor[[512, FD], pl.FP32],
    rms_scratch: pl.Tensor[[1, 512], pl.FP32],
    fn: pl.Tensor[[M3, FD], pl.FP32],
    normw: pl.Tensor[[1, M3], pl.FP32],
    mhc_scale: pl.Tensor[[3], pl.FP32],
    mhc_base: pl.Tensor[[1, M3], pl.FP32],
    next_mixes: pl.Out[pl.Tensor[[512, M3], pl.FP32]],
    h_next: pl.Out[pl.Tensor[[512, D], pl.BF16]],
):
    token_start = pl.tile.get_block_idx() * 128
    norm_row = pl.slice(normw, [1, 32], [0, 0], valid_shape=[1, 24], clamp=True)
    first_updated = pl.slice(updated_fp32, [128, 128], [token_start, 0], valid_shape=[128, 128])
    first_weight = pl.slice(fn, [32, 128], [0, 0], valid_shape=[24, 128], clamp=True)
    accumulator = pl.matmul(first_updated, first_weight, b_trans=True)
    for hidden_block in pl.range(1, FD // 128):
        hidden_start = hidden_block * 128
        updated_chunk = pl.slice(updated_fp32, [128, 128], [token_start, hidden_start], valid_shape=[128, 128])
        weight_chunk = pl.slice(fn, [32, 128], [0, hidden_start], valid_shape=[24, 128], clamp=True)
        accumulator = pl.matmul_acc(accumulator, updated_chunk, weight_chunk, b_trans=True)
    rms_tile = pl.reshape(pl.slice(rms_scratch, [1, 128], [0, token_start], valid_shape=[1, 128]), [128, 1])
    predicted = pl.row_expand_mul(pl.col_expand_mul(accumulator, norm_row), rms_tile)
    next_mixes[token_start:token_start + 128, 0:24] = predicted[0:128, 0:24]
    scale_pre = pl.read(mhc_scale, [0])
    predicted_planes = pl.transpose(predicted, 0, 1)
    pre_coefficient0 = pl.reshape(predicted_planes[0:1, 0:128], [128, 1])
    pre_raw0 = pl.add(pl.mul(pre_coefficient0, scale_pre), pl.read(mhc_base, [0, 0]))
    next_pre0 = pl.add(pl.recip(pl.add(pl.exp(pl.neg(pre_raw0)), 1.0)), PRE_EPS)
    pre_coefficient1 = pl.reshape(predicted_planes[1:2, 0:128], [128, 1])
    pre_raw1 = pl.add(pl.mul(pre_coefficient1, scale_pre), pl.read(mhc_base, [0, 1]))
    next_pre1 = pl.add(pl.recip(pl.add(pl.exp(pl.neg(pre_raw1)), 1.0)), PRE_EPS)
    pre_coefficient2 = pl.reshape(predicted_planes[2:3, 0:128], [128, 1])
    pre_raw2 = pl.add(pl.mul(pre_coefficient2, scale_pre), pl.read(mhc_base, [0, 2]))
    next_pre2 = pl.add(pl.recip(pl.add(pl.exp(pl.neg(pre_raw2)), 1.0)), PRE_EPS)
    pre_coefficient3 = pl.reshape(predicted_planes[3:4, 0:128], [128, 1])
    pre_raw3 = pl.add(pl.mul(pre_coefficient3, scale_pre), pl.read(mhc_base, [0, 3]))
    next_pre3 = pl.add(pl.recip(pl.add(pl.exp(pl.neg(pre_raw3)), 1.0)), PRE_EPS)
    for hidden_block in pl.range(D // 128):
        hidden_start = hidden_block * 128
    for hidden_block in pl.range(D // 128):
        hidden_start = hidden_block * 128
        updated_copy0 = updated_fp32[token_start:token_start + 128, 0 * D + hidden_start:0 * D + hidden_start + 128]
        hidden = pl.row_expand_mul(updated_copy0, next_pre0)
        updated_copy1 = updated_fp32[token_start:token_start + 128, 1 * D + hidden_start:1 * D + hidden_start + 128]
        hidden = pl.add(hidden, pl.row_expand_mul(updated_copy1, next_pre1))
        updated_copy2 = updated_fp32[token_start:token_start + 128, 2 * D + hidden_start:2 * D + hidden_start + 128]
        hidden = pl.add(hidden, pl.row_expand_mul(updated_copy2, next_pre2))
        updated_copy3 = updated_fp32[token_start:token_start + 128, 3 * D + hidden_start:3 * D + hidden_start + 128]
        hidden = pl.add(hidden, pl.row_expand_mul(updated_copy3, next_pre3))
        h_next[token_start:token_start + 128, hidden_start:hidden_start + 128] = pl.cast(hidden, pl.BF16, mode="rint")
    return next_mixes, h_next



@pl.jit.incore
def chunk_producer_64(
    x: pl.Tensor[[512, D], pl.BF16],
    residual: pl.InOut[pl.Tensor[[512, FD], pl.BF16]],
    input_mixes: pl.Tensor[[512, M3], pl.FP32],
    mhc_scale: pl.Tensor[[3], pl.FP32],
    mhc_base: pl.Tensor[[1, M3], pl.FP32],
    updated_fp32: pl.InOut[pl.Tensor[[512, FD], pl.FP32]],
    rms_scratch: pl.InOut[pl.Tensor[[1, 512], pl.FP32]],
):
    task_index = pl.tile.get_block_idx()
    macro_start = task_index * 64
    scale_pre = pl.read(mhc_scale, [0])
    scale_post = pl.read(mhc_scale, [1])
    scale_comb = pl.read(mhc_scale, [2])
    mix_planes = pl.transpose(input_mixes[macro_start:macro_start + 64, 0:M3], 0, 1)
    post_raw0 = pl.add(pl.mul(pl.reshape(mix_planes[4:5, 0:64], [1, 64]), scale_post), pl.read(mhc_base, [0, 4]))
    post0_planes = pl.mul(pl.recip(pl.add(pl.exp(pl.neg(post_raw0)), 1.0)), POST_MULT)
    post_raw1 = pl.add(pl.mul(pl.reshape(mix_planes[5:6, 0:64], [1, 64]), scale_post), pl.read(mhc_base, [0, 5]))
    post1_planes = pl.mul(pl.recip(pl.add(pl.exp(pl.neg(post_raw1)), 1.0)), POST_MULT)
    post_raw2 = pl.add(pl.mul(pl.reshape(mix_planes[6:7, 0:64], [1, 64]), scale_post), pl.read(mhc_base, [0, 6]))
    post2_planes = pl.mul(pl.recip(pl.add(pl.exp(pl.neg(post_raw2)), 1.0)), POST_MULT)
    post_raw3 = pl.add(pl.mul(pl.reshape(mix_planes[7:8, 0:64], [1, 64]), scale_post), pl.read(mhc_base, [0, 7]))
    post3_planes = pl.mul(pl.recip(pl.add(pl.exp(pl.neg(post_raw3)), 1.0)), POST_MULT)
    comb00_planes = pl.add(pl.mul(pl.reshape(mix_planes[8:9, 0:64], [1, 64]), scale_comb), pl.read(mhc_base, [0, 8]))
    comb01_planes = pl.add(pl.mul(pl.reshape(mix_planes[9:10, 0:64], [1, 64]), scale_comb), pl.read(mhc_base, [0, 9]))
    comb02_planes = pl.add(pl.mul(pl.reshape(mix_planes[10:11, 0:64], [1, 64]), scale_comb), pl.read(mhc_base, [0, 10]))
    comb03_planes = pl.add(pl.mul(pl.reshape(mix_planes[11:12, 0:64], [1, 64]), scale_comb), pl.read(mhc_base, [0, 11]))
    row_max0 = pl.maximum(pl.maximum(comb00_planes, comb01_planes), pl.maximum(comb02_planes, comb03_planes))
    comb00_planes = pl.exp(pl.sub(comb00_planes, row_max0))
    comb01_planes = pl.exp(pl.sub(comb01_planes, row_max0))
    comb02_planes = pl.exp(pl.sub(comb02_planes, row_max0))
    comb03_planes = pl.exp(pl.sub(comb03_planes, row_max0))
    row_sum0 = pl.add(pl.add(comb00_planes, comb01_planes), pl.add(comb02_planes, comb03_planes))
    comb00_planes = pl.add(pl.div(comb00_planes, row_sum0), PRE_EPS)
    comb01_planes = pl.add(pl.div(comb01_planes, row_sum0), PRE_EPS)
    comb02_planes = pl.add(pl.div(comb02_planes, row_sum0), PRE_EPS)
    comb03_planes = pl.add(pl.div(comb03_planes, row_sum0), PRE_EPS)
    comb10_planes = pl.add(pl.mul(pl.reshape(mix_planes[12:13, 0:64], [1, 64]), scale_comb), pl.read(mhc_base, [0, 12]))
    comb11_planes = pl.add(pl.mul(pl.reshape(mix_planes[13:14, 0:64], [1, 64]), scale_comb), pl.read(mhc_base, [0, 13]))
    comb12_planes = pl.add(pl.mul(pl.reshape(mix_planes[14:15, 0:64], [1, 64]), scale_comb), pl.read(mhc_base, [0, 14]))
    comb13_planes = pl.add(pl.mul(pl.reshape(mix_planes[15:16, 0:64], [1, 64]), scale_comb), pl.read(mhc_base, [0, 15]))
    row_max1 = pl.maximum(pl.maximum(comb10_planes, comb11_planes), pl.maximum(comb12_planes, comb13_planes))
    comb10_planes = pl.exp(pl.sub(comb10_planes, row_max1))
    comb11_planes = pl.exp(pl.sub(comb11_planes, row_max1))
    comb12_planes = pl.exp(pl.sub(comb12_planes, row_max1))
    comb13_planes = pl.exp(pl.sub(comb13_planes, row_max1))
    row_sum1 = pl.add(pl.add(comb10_planes, comb11_planes), pl.add(comb12_planes, comb13_planes))
    comb10_planes = pl.add(pl.div(comb10_planes, row_sum1), PRE_EPS)
    comb11_planes = pl.add(pl.div(comb11_planes, row_sum1), PRE_EPS)
    comb12_planes = pl.add(pl.div(comb12_planes, row_sum1), PRE_EPS)
    comb13_planes = pl.add(pl.div(comb13_planes, row_sum1), PRE_EPS)
    comb20_planes = pl.add(pl.mul(pl.reshape(mix_planes[16:17, 0:64], [1, 64]), scale_comb), pl.read(mhc_base, [0, 16]))
    comb21_planes = pl.add(pl.mul(pl.reshape(mix_planes[17:18, 0:64], [1, 64]), scale_comb), pl.read(mhc_base, [0, 17]))
    comb22_planes = pl.add(pl.mul(pl.reshape(mix_planes[18:19, 0:64], [1, 64]), scale_comb), pl.read(mhc_base, [0, 18]))
    comb23_planes = pl.add(pl.mul(pl.reshape(mix_planes[19:20, 0:64], [1, 64]), scale_comb), pl.read(mhc_base, [0, 19]))
    row_max2 = pl.maximum(pl.maximum(comb20_planes, comb21_planes), pl.maximum(comb22_planes, comb23_planes))
    comb20_planes = pl.exp(pl.sub(comb20_planes, row_max2))
    comb21_planes = pl.exp(pl.sub(comb21_planes, row_max2))
    comb22_planes = pl.exp(pl.sub(comb22_planes, row_max2))
    comb23_planes = pl.exp(pl.sub(comb23_planes, row_max2))
    row_sum2 = pl.add(pl.add(comb20_planes, comb21_planes), pl.add(comb22_planes, comb23_planes))
    comb20_planes = pl.add(pl.div(comb20_planes, row_sum2), PRE_EPS)
    comb21_planes = pl.add(pl.div(comb21_planes, row_sum2), PRE_EPS)
    comb22_planes = pl.add(pl.div(comb22_planes, row_sum2), PRE_EPS)
    comb23_planes = pl.add(pl.div(comb23_planes, row_sum2), PRE_EPS)
    comb30_planes = pl.add(pl.mul(pl.reshape(mix_planes[20:21, 0:64], [1, 64]), scale_comb), pl.read(mhc_base, [0, 20]))
    comb31_planes = pl.add(pl.mul(pl.reshape(mix_planes[21:22, 0:64], [1, 64]), scale_comb), pl.read(mhc_base, [0, 21]))
    comb32_planes = pl.add(pl.mul(pl.reshape(mix_planes[22:23, 0:64], [1, 64]), scale_comb), pl.read(mhc_base, [0, 22]))
    comb33_planes = pl.add(pl.mul(pl.reshape(mix_planes[23:24, 0:64], [1, 64]), scale_comb), pl.read(mhc_base, [0, 23]))
    row_max3 = pl.maximum(pl.maximum(comb30_planes, comb31_planes), pl.maximum(comb32_planes, comb33_planes))
    comb30_planes = pl.exp(pl.sub(comb30_planes, row_max3))
    comb31_planes = pl.exp(pl.sub(comb31_planes, row_max3))
    comb32_planes = pl.exp(pl.sub(comb32_planes, row_max3))
    comb33_planes = pl.exp(pl.sub(comb33_planes, row_max3))
    row_sum3 = pl.add(pl.add(comb30_planes, comb31_planes), pl.add(comb32_planes, comb33_planes))
    comb30_planes = pl.add(pl.div(comb30_planes, row_sum3), PRE_EPS)
    comb31_planes = pl.add(pl.div(comb31_planes, row_sum3), PRE_EPS)
    comb32_planes = pl.add(pl.div(comb32_planes, row_sum3), PRE_EPS)
    comb33_planes = pl.add(pl.div(comb33_planes, row_sum3), PRE_EPS)
    column_sum0 = pl.add(pl.add(pl.add(comb00_planes, comb10_planes), pl.add(comb20_planes, comb30_planes)), PRE_EPS)
    comb00_planes = pl.div(comb00_planes, column_sum0)
    comb10_planes = pl.div(comb10_planes, column_sum0)
    comb20_planes = pl.div(comb20_planes, column_sum0)
    comb30_planes = pl.div(comb30_planes, column_sum0)
    column_sum1 = pl.add(pl.add(pl.add(comb01_planes, comb11_planes), pl.add(comb21_planes, comb31_planes)), PRE_EPS)
    comb01_planes = pl.div(comb01_planes, column_sum1)
    comb11_planes = pl.div(comb11_planes, column_sum1)
    comb21_planes = pl.div(comb21_planes, column_sum1)
    comb31_planes = pl.div(comb31_planes, column_sum1)
    column_sum2 = pl.add(pl.add(pl.add(comb02_planes, comb12_planes), pl.add(comb22_planes, comb32_planes)), PRE_EPS)
    comb02_planes = pl.div(comb02_planes, column_sum2)
    comb12_planes = pl.div(comb12_planes, column_sum2)
    comb22_planes = pl.div(comb22_planes, column_sum2)
    comb32_planes = pl.div(comb32_planes, column_sum2)
    column_sum3 = pl.add(pl.add(pl.add(comb03_planes, comb13_planes), pl.add(comb23_planes, comb33_planes)), PRE_EPS)
    comb03_planes = pl.div(comb03_planes, column_sum3)
    comb13_planes = pl.div(comb13_planes, column_sum3)
    comb23_planes = pl.div(comb23_planes, column_sum3)
    comb33_planes = pl.div(comb33_planes, column_sum3)
    for iteration in pl.range(SINK - 1):
        row_sum0 = pl.add(pl.add(pl.add(comb00_planes, comb01_planes), pl.add(comb02_planes, comb03_planes)), PRE_EPS)
        comb00_planes = pl.div(comb00_planes, row_sum0)
        comb01_planes = pl.div(comb01_planes, row_sum0)
        comb02_planes = pl.div(comb02_planes, row_sum0)
        comb03_planes = pl.div(comb03_planes, row_sum0)
        row_sum1 = pl.add(pl.add(pl.add(comb10_planes, comb11_planes), pl.add(comb12_planes, comb13_planes)), PRE_EPS)
        comb10_planes = pl.div(comb10_planes, row_sum1)
        comb11_planes = pl.div(comb11_planes, row_sum1)
        comb12_planes = pl.div(comb12_planes, row_sum1)
        comb13_planes = pl.div(comb13_planes, row_sum1)
        row_sum2 = pl.add(pl.add(pl.add(comb20_planes, comb21_planes), pl.add(comb22_planes, comb23_planes)), PRE_EPS)
        comb20_planes = pl.div(comb20_planes, row_sum2)
        comb21_planes = pl.div(comb21_planes, row_sum2)
        comb22_planes = pl.div(comb22_planes, row_sum2)
        comb23_planes = pl.div(comb23_planes, row_sum2)
        row_sum3 = pl.add(pl.add(pl.add(comb30_planes, comb31_planes), pl.add(comb32_planes, comb33_planes)), PRE_EPS)
        comb30_planes = pl.div(comb30_planes, row_sum3)
        comb31_planes = pl.div(comb31_planes, row_sum3)
        comb32_planes = pl.div(comb32_planes, row_sum3)
        comb33_planes = pl.div(comb33_planes, row_sum3)
        column_sum0 = pl.add(pl.add(pl.add(comb00_planes, comb10_planes), pl.add(comb20_planes, comb30_planes)), PRE_EPS)
        comb00_planes = pl.div(comb00_planes, column_sum0)
        comb10_planes = pl.div(comb10_planes, column_sum0)
        comb20_planes = pl.div(comb20_planes, column_sum0)
        comb30_planes = pl.div(comb30_planes, column_sum0)
        column_sum1 = pl.add(pl.add(pl.add(comb01_planes, comb11_planes), pl.add(comb21_planes, comb31_planes)), PRE_EPS)
        comb01_planes = pl.div(comb01_planes, column_sum1)
        comb11_planes = pl.div(comb11_planes, column_sum1)
        comb21_planes = pl.div(comb21_planes, column_sum1)
        comb31_planes = pl.div(comb31_planes, column_sum1)
        column_sum2 = pl.add(pl.add(pl.add(comb02_planes, comb12_planes), pl.add(comb22_planes, comb32_planes)), PRE_EPS)
        comb02_planes = pl.div(comb02_planes, column_sum2)
        comb12_planes = pl.div(comb12_planes, column_sum2)
        comb22_planes = pl.div(comb22_planes, column_sum2)
        comb32_planes = pl.div(comb32_planes, column_sum2)
        column_sum3 = pl.add(pl.add(pl.add(comb03_planes, comb13_planes), pl.add(comb23_planes, comb33_planes)), PRE_EPS)
        comb03_planes = pl.div(comb03_planes, column_sum3)
        comb13_planes = pl.div(comb13_planes, column_sum3)
        comb23_planes = pl.div(comb23_planes, column_sum3)
        comb33_planes = pl.div(comb33_planes, column_sum3)
    for update_block in pl.range(64 // 32):
        update_offset = update_block * 32
        token_start = macro_start + update_offset
        post0 = pl.reshape(post0_planes[0:1, update_offset:update_offset + 32], [32, 1])
        post1 = pl.reshape(post1_planes[0:1, update_offset:update_offset + 32], [32, 1])
        post2 = pl.reshape(post2_planes[0:1, update_offset:update_offset + 32], [32, 1])
        post3 = pl.reshape(post3_planes[0:1, update_offset:update_offset + 32], [32, 1])
        comb00 = pl.reshape(comb00_planes[0:1, update_offset:update_offset + 32], [32, 1])
        comb01 = pl.reshape(comb01_planes[0:1, update_offset:update_offset + 32], [32, 1])
        comb02 = pl.reshape(comb02_planes[0:1, update_offset:update_offset + 32], [32, 1])
        comb03 = pl.reshape(comb03_planes[0:1, update_offset:update_offset + 32], [32, 1])
        comb10 = pl.reshape(comb10_planes[0:1, update_offset:update_offset + 32], [32, 1])
        comb11 = pl.reshape(comb11_planes[0:1, update_offset:update_offset + 32], [32, 1])
        comb12 = pl.reshape(comb12_planes[0:1, update_offset:update_offset + 32], [32, 1])
        comb13 = pl.reshape(comb13_planes[0:1, update_offset:update_offset + 32], [32, 1])
        comb20 = pl.reshape(comb20_planes[0:1, update_offset:update_offset + 32], [32, 1])
        comb21 = pl.reshape(comb21_planes[0:1, update_offset:update_offset + 32], [32, 1])
        comb22 = pl.reshape(comb22_planes[0:1, update_offset:update_offset + 32], [32, 1])
        comb23 = pl.reshape(comb23_planes[0:1, update_offset:update_offset + 32], [32, 1])
        comb30 = pl.reshape(comb30_planes[0:1, update_offset:update_offset + 32], [32, 1])
        comb31 = pl.reshape(comb31_planes[0:1, update_offset:update_offset + 32], [32, 1])
        comb32 = pl.reshape(comb32_planes[0:1, update_offset:update_offset + 32], [32, 1])
        comb33 = pl.reshape(comb33_planes[0:1, update_offset:update_offset + 32], [32, 1])
        square_acc = pl.full([1, 32], dtype=pl.FP32, value=0.0)
        for hidden_block in pl.range(D // 128):
            hidden_start = hidden_block * 128
            input_tile = pl.cast(x[token_start:token_start + 32, hidden_start:hidden_start + 128], pl.FP32)
            original0 = pl.cast(residual[token_start:token_start + 32, 0 * D + hidden_start:0 * D + hidden_start + 128], pl.FP32)
            original1 = pl.cast(residual[token_start:token_start + 32, 1 * D + hidden_start:1 * D + hidden_start + 128], pl.FP32)
            original2 = pl.cast(residual[token_start:token_start + 32, 2 * D + hidden_start:2 * D + hidden_start + 128], pl.FP32)
            original3 = pl.cast(residual[token_start:token_start + 32, 3 * D + hidden_start:3 * D + hidden_start + 128], pl.FP32)
            updated0 = pl.row_expand_mul(input_tile, post0)
            updated0 = pl.add(updated0, pl.row_expand_mul(original0, comb00))
            updated0 = pl.add(updated0, pl.row_expand_mul(original1, comb01))
            updated0 = pl.add(updated0, pl.row_expand_mul(original2, comb02))
            updated0 = pl.add(updated0, pl.row_expand_mul(original3, comb03))
            updated1 = pl.row_expand_mul(input_tile, post1)
            updated1 = pl.add(updated1, pl.row_expand_mul(original0, comb10))
            updated1 = pl.add(updated1, pl.row_expand_mul(original1, comb11))
            updated1 = pl.add(updated1, pl.row_expand_mul(original2, comb12))
            updated1 = pl.add(updated1, pl.row_expand_mul(original3, comb13))
            updated2 = pl.row_expand_mul(input_tile, post2)
            updated2 = pl.add(updated2, pl.row_expand_mul(original0, comb20))
            updated2 = pl.add(updated2, pl.row_expand_mul(original1, comb21))
            updated2 = pl.add(updated2, pl.row_expand_mul(original2, comb22))
            updated2 = pl.add(updated2, pl.row_expand_mul(original3, comb23))
            updated3 = pl.row_expand_mul(input_tile, post3)
            updated3 = pl.add(updated3, pl.row_expand_mul(original0, comb30))
            updated3 = pl.add(updated3, pl.row_expand_mul(original1, comb31))
            updated3 = pl.add(updated3, pl.row_expand_mul(original2, comb32))
            updated3 = pl.add(updated3, pl.row_expand_mul(original3, comb33))
            square_sum = pl.row_sum(pl.add(pl.add(pl.mul(updated0, updated0), pl.mul(updated1, updated1)), pl.add(pl.mul(updated2, updated2), pl.mul(updated3, updated3))))
            square_acc = pl.add(square_acc, pl.reshape(square_sum, [1, 32]))
            updated_fp32[token_start:token_start + 32, 0 * D + hidden_start:0 * D + hidden_start + 128] = updated0
            residual[token_start:token_start + 32, 0 * D + hidden_start:0 * D + hidden_start + 128] = pl.cast(updated0, pl.BF16, mode="rint")
            updated_fp32[token_start:token_start + 32, 1 * D + hidden_start:1 * D + hidden_start + 128] = updated1
            residual[token_start:token_start + 32, 1 * D + hidden_start:1 * D + hidden_start + 128] = pl.cast(updated1, pl.BF16, mode="rint")
            updated_fp32[token_start:token_start + 32, 2 * D + hidden_start:2 * D + hidden_start + 128] = updated2
            residual[token_start:token_start + 32, 2 * D + hidden_start:2 * D + hidden_start + 128] = pl.cast(updated2, pl.BF16, mode="rint")
            updated_fp32[token_start:token_start + 32, 3 * D + hidden_start:3 * D + hidden_start + 128] = updated3
            residual[token_start:token_start + 32, 3 * D + hidden_start:3 * D + hidden_start + 128] = pl.cast(updated3, pl.BF16, mode="rint")
        square_sum = pl.reshape(square_acc, [32, 1])
        mean_square = pl.add(pl.mul(square_sum, 1.0 / FD), RMS_EPS)
        inverse_rms = pl.rsqrt(mean_square)
        inverse_rms = pl.mul(inverse_rms, pl.add(pl.mul(pl.mul(mean_square, pl.mul(inverse_rms, inverse_rms)), -0.5), 1.5))
        rms_scratch[0:1, token_start:token_start + 32] = pl.reshape(inverse_rms, [1, 32])
    return residual, updated_fp32, rms_scratch



@pl.jit.incore
def chunk_producer_128(
    x: pl.Tensor[[512, D], pl.BF16],
    residual: pl.InOut[pl.Tensor[[512, FD], pl.BF16]],
    input_mixes: pl.Tensor[[512, M3], pl.FP32],
    mhc_scale: pl.Tensor[[3], pl.FP32],
    mhc_base: pl.Tensor[[1, M3], pl.FP32],
    updated_fp32: pl.InOut[pl.Tensor[[512, FD], pl.FP32]],
    rms_scratch: pl.InOut[pl.Tensor[[1, 512], pl.FP32]],
):
    task_index = pl.tile.get_block_idx()
    macro_start = task_index * 128
    scale_pre = pl.read(mhc_scale, [0])
    scale_post = pl.read(mhc_scale, [1])
    scale_comb = pl.read(mhc_scale, [2])
    mix_planes = pl.transpose(input_mixes[macro_start:macro_start + 128, 0:M3], 0, 1)
    post_raw0 = pl.add(pl.mul(pl.reshape(mix_planes[4:5, 0:128], [1, 128]), scale_post), pl.read(mhc_base, [0, 4]))
    post0_planes = pl.mul(pl.recip(pl.add(pl.exp(pl.neg(post_raw0)), 1.0)), POST_MULT)
    post_raw1 = pl.add(pl.mul(pl.reshape(mix_planes[5:6, 0:128], [1, 128]), scale_post), pl.read(mhc_base, [0, 5]))
    post1_planes = pl.mul(pl.recip(pl.add(pl.exp(pl.neg(post_raw1)), 1.0)), POST_MULT)
    post_raw2 = pl.add(pl.mul(pl.reshape(mix_planes[6:7, 0:128], [1, 128]), scale_post), pl.read(mhc_base, [0, 6]))
    post2_planes = pl.mul(pl.recip(pl.add(pl.exp(pl.neg(post_raw2)), 1.0)), POST_MULT)
    post_raw3 = pl.add(pl.mul(pl.reshape(mix_planes[7:8, 0:128], [1, 128]), scale_post), pl.read(mhc_base, [0, 7]))
    post3_planes = pl.mul(pl.recip(pl.add(pl.exp(pl.neg(post_raw3)), 1.0)), POST_MULT)
    comb00_planes = pl.add(pl.mul(pl.reshape(mix_planes[8:9, 0:128], [1, 128]), scale_comb), pl.read(mhc_base, [0, 8]))
    comb01_planes = pl.add(pl.mul(pl.reshape(mix_planes[9:10, 0:128], [1, 128]), scale_comb), pl.read(mhc_base, [0, 9]))
    comb02_planes = pl.add(pl.mul(pl.reshape(mix_planes[10:11, 0:128], [1, 128]), scale_comb), pl.read(mhc_base, [0, 10]))
    comb03_planes = pl.add(pl.mul(pl.reshape(mix_planes[11:12, 0:128], [1, 128]), scale_comb), pl.read(mhc_base, [0, 11]))
    row_max0 = pl.maximum(pl.maximum(comb00_planes, comb01_planes), pl.maximum(comb02_planes, comb03_planes))
    comb00_planes = pl.exp(pl.sub(comb00_planes, row_max0))
    comb01_planes = pl.exp(pl.sub(comb01_planes, row_max0))
    comb02_planes = pl.exp(pl.sub(comb02_planes, row_max0))
    comb03_planes = pl.exp(pl.sub(comb03_planes, row_max0))
    row_sum0 = pl.add(pl.add(comb00_planes, comb01_planes), pl.add(comb02_planes, comb03_planes))
    comb00_planes = pl.add(pl.div(comb00_planes, row_sum0), PRE_EPS)
    comb01_planes = pl.add(pl.div(comb01_planes, row_sum0), PRE_EPS)
    comb02_planes = pl.add(pl.div(comb02_planes, row_sum0), PRE_EPS)
    comb03_planes = pl.add(pl.div(comb03_planes, row_sum0), PRE_EPS)
    comb10_planes = pl.add(pl.mul(pl.reshape(mix_planes[12:13, 0:128], [1, 128]), scale_comb), pl.read(mhc_base, [0, 12]))
    comb11_planes = pl.add(pl.mul(pl.reshape(mix_planes[13:14, 0:128], [1, 128]), scale_comb), pl.read(mhc_base, [0, 13]))
    comb12_planes = pl.add(pl.mul(pl.reshape(mix_planes[14:15, 0:128], [1, 128]), scale_comb), pl.read(mhc_base, [0, 14]))
    comb13_planes = pl.add(pl.mul(pl.reshape(mix_planes[15:16, 0:128], [1, 128]), scale_comb), pl.read(mhc_base, [0, 15]))
    row_max1 = pl.maximum(pl.maximum(comb10_planes, comb11_planes), pl.maximum(comb12_planes, comb13_planes))
    comb10_planes = pl.exp(pl.sub(comb10_planes, row_max1))
    comb11_planes = pl.exp(pl.sub(comb11_planes, row_max1))
    comb12_planes = pl.exp(pl.sub(comb12_planes, row_max1))
    comb13_planes = pl.exp(pl.sub(comb13_planes, row_max1))
    row_sum1 = pl.add(pl.add(comb10_planes, comb11_planes), pl.add(comb12_planes, comb13_planes))
    comb10_planes = pl.add(pl.div(comb10_planes, row_sum1), PRE_EPS)
    comb11_planes = pl.add(pl.div(comb11_planes, row_sum1), PRE_EPS)
    comb12_planes = pl.add(pl.div(comb12_planes, row_sum1), PRE_EPS)
    comb13_planes = pl.add(pl.div(comb13_planes, row_sum1), PRE_EPS)
    comb20_planes = pl.add(pl.mul(pl.reshape(mix_planes[16:17, 0:128], [1, 128]), scale_comb), pl.read(mhc_base, [0, 16]))
    comb21_planes = pl.add(pl.mul(pl.reshape(mix_planes[17:18, 0:128], [1, 128]), scale_comb), pl.read(mhc_base, [0, 17]))
    comb22_planes = pl.add(pl.mul(pl.reshape(mix_planes[18:19, 0:128], [1, 128]), scale_comb), pl.read(mhc_base, [0, 18]))
    comb23_planes = pl.add(pl.mul(pl.reshape(mix_planes[19:20, 0:128], [1, 128]), scale_comb), pl.read(mhc_base, [0, 19]))
    row_max2 = pl.maximum(pl.maximum(comb20_planes, comb21_planes), pl.maximum(comb22_planes, comb23_planes))
    comb20_planes = pl.exp(pl.sub(comb20_planes, row_max2))
    comb21_planes = pl.exp(pl.sub(comb21_planes, row_max2))
    comb22_planes = pl.exp(pl.sub(comb22_planes, row_max2))
    comb23_planes = pl.exp(pl.sub(comb23_planes, row_max2))
    row_sum2 = pl.add(pl.add(comb20_planes, comb21_planes), pl.add(comb22_planes, comb23_planes))
    comb20_planes = pl.add(pl.div(comb20_planes, row_sum2), PRE_EPS)
    comb21_planes = pl.add(pl.div(comb21_planes, row_sum2), PRE_EPS)
    comb22_planes = pl.add(pl.div(comb22_planes, row_sum2), PRE_EPS)
    comb23_planes = pl.add(pl.div(comb23_planes, row_sum2), PRE_EPS)
    comb30_planes = pl.add(pl.mul(pl.reshape(mix_planes[20:21, 0:128], [1, 128]), scale_comb), pl.read(mhc_base, [0, 20]))
    comb31_planes = pl.add(pl.mul(pl.reshape(mix_planes[21:22, 0:128], [1, 128]), scale_comb), pl.read(mhc_base, [0, 21]))
    comb32_planes = pl.add(pl.mul(pl.reshape(mix_planes[22:23, 0:128], [1, 128]), scale_comb), pl.read(mhc_base, [0, 22]))
    comb33_planes = pl.add(pl.mul(pl.reshape(mix_planes[23:24, 0:128], [1, 128]), scale_comb), pl.read(mhc_base, [0, 23]))
    row_max3 = pl.maximum(pl.maximum(comb30_planes, comb31_planes), pl.maximum(comb32_planes, comb33_planes))
    comb30_planes = pl.exp(pl.sub(comb30_planes, row_max3))
    comb31_planes = pl.exp(pl.sub(comb31_planes, row_max3))
    comb32_planes = pl.exp(pl.sub(comb32_planes, row_max3))
    comb33_planes = pl.exp(pl.sub(comb33_planes, row_max3))
    row_sum3 = pl.add(pl.add(comb30_planes, comb31_planes), pl.add(comb32_planes, comb33_planes))
    comb30_planes = pl.add(pl.div(comb30_planes, row_sum3), PRE_EPS)
    comb31_planes = pl.add(pl.div(comb31_planes, row_sum3), PRE_EPS)
    comb32_planes = pl.add(pl.div(comb32_planes, row_sum3), PRE_EPS)
    comb33_planes = pl.add(pl.div(comb33_planes, row_sum3), PRE_EPS)
    column_sum0 = pl.add(pl.add(pl.add(comb00_planes, comb10_planes), pl.add(comb20_planes, comb30_planes)), PRE_EPS)
    comb00_planes = pl.div(comb00_planes, column_sum0)
    comb10_planes = pl.div(comb10_planes, column_sum0)
    comb20_planes = pl.div(comb20_planes, column_sum0)
    comb30_planes = pl.div(comb30_planes, column_sum0)
    column_sum1 = pl.add(pl.add(pl.add(comb01_planes, comb11_planes), pl.add(comb21_planes, comb31_planes)), PRE_EPS)
    comb01_planes = pl.div(comb01_planes, column_sum1)
    comb11_planes = pl.div(comb11_planes, column_sum1)
    comb21_planes = pl.div(comb21_planes, column_sum1)
    comb31_planes = pl.div(comb31_planes, column_sum1)
    column_sum2 = pl.add(pl.add(pl.add(comb02_planes, comb12_planes), pl.add(comb22_planes, comb32_planes)), PRE_EPS)
    comb02_planes = pl.div(comb02_planes, column_sum2)
    comb12_planes = pl.div(comb12_planes, column_sum2)
    comb22_planes = pl.div(comb22_planes, column_sum2)
    comb32_planes = pl.div(comb32_planes, column_sum2)
    column_sum3 = pl.add(pl.add(pl.add(comb03_planes, comb13_planes), pl.add(comb23_planes, comb33_planes)), PRE_EPS)
    comb03_planes = pl.div(comb03_planes, column_sum3)
    comb13_planes = pl.div(comb13_planes, column_sum3)
    comb23_planes = pl.div(comb23_planes, column_sum3)
    comb33_planes = pl.div(comb33_planes, column_sum3)
    for iteration in pl.range(SINK - 1):
        row_sum0 = pl.add(pl.add(pl.add(comb00_planes, comb01_planes), pl.add(comb02_planes, comb03_planes)), PRE_EPS)
        comb00_planes = pl.div(comb00_planes, row_sum0)
        comb01_planes = pl.div(comb01_planes, row_sum0)
        comb02_planes = pl.div(comb02_planes, row_sum0)
        comb03_planes = pl.div(comb03_planes, row_sum0)
        row_sum1 = pl.add(pl.add(pl.add(comb10_planes, comb11_planes), pl.add(comb12_planes, comb13_planes)), PRE_EPS)
        comb10_planes = pl.div(comb10_planes, row_sum1)
        comb11_planes = pl.div(comb11_planes, row_sum1)
        comb12_planes = pl.div(comb12_planes, row_sum1)
        comb13_planes = pl.div(comb13_planes, row_sum1)
        row_sum2 = pl.add(pl.add(pl.add(comb20_planes, comb21_planes), pl.add(comb22_planes, comb23_planes)), PRE_EPS)
        comb20_planes = pl.div(comb20_planes, row_sum2)
        comb21_planes = pl.div(comb21_planes, row_sum2)
        comb22_planes = pl.div(comb22_planes, row_sum2)
        comb23_planes = pl.div(comb23_planes, row_sum2)
        row_sum3 = pl.add(pl.add(pl.add(comb30_planes, comb31_planes), pl.add(comb32_planes, comb33_planes)), PRE_EPS)
        comb30_planes = pl.div(comb30_planes, row_sum3)
        comb31_planes = pl.div(comb31_planes, row_sum3)
        comb32_planes = pl.div(comb32_planes, row_sum3)
        comb33_planes = pl.div(comb33_planes, row_sum3)
        column_sum0 = pl.add(pl.add(pl.add(comb00_planes, comb10_planes), pl.add(comb20_planes, comb30_planes)), PRE_EPS)
        comb00_planes = pl.div(comb00_planes, column_sum0)
        comb10_planes = pl.div(comb10_planes, column_sum0)
        comb20_planes = pl.div(comb20_planes, column_sum0)
        comb30_planes = pl.div(comb30_planes, column_sum0)
        column_sum1 = pl.add(pl.add(pl.add(comb01_planes, comb11_planes), pl.add(comb21_planes, comb31_planes)), PRE_EPS)
        comb01_planes = pl.div(comb01_planes, column_sum1)
        comb11_planes = pl.div(comb11_planes, column_sum1)
        comb21_planes = pl.div(comb21_planes, column_sum1)
        comb31_planes = pl.div(comb31_planes, column_sum1)
        column_sum2 = pl.add(pl.add(pl.add(comb02_planes, comb12_planes), pl.add(comb22_planes, comb32_planes)), PRE_EPS)
        comb02_planes = pl.div(comb02_planes, column_sum2)
        comb12_planes = pl.div(comb12_planes, column_sum2)
        comb22_planes = pl.div(comb22_planes, column_sum2)
        comb32_planes = pl.div(comb32_planes, column_sum2)
        column_sum3 = pl.add(pl.add(pl.add(comb03_planes, comb13_planes), pl.add(comb23_planes, comb33_planes)), PRE_EPS)
        comb03_planes = pl.div(comb03_planes, column_sum3)
        comb13_planes = pl.div(comb13_planes, column_sum3)
        comb23_planes = pl.div(comb23_planes, column_sum3)
        comb33_planes = pl.div(comb33_planes, column_sum3)
    for update_block in pl.range(128 // 32):
        update_offset = update_block * 32
        token_start = macro_start + update_offset
        post0 = pl.reshape(post0_planes[0:1, update_offset:update_offset + 32], [32, 1])
        post1 = pl.reshape(post1_planes[0:1, update_offset:update_offset + 32], [32, 1])
        post2 = pl.reshape(post2_planes[0:1, update_offset:update_offset + 32], [32, 1])
        post3 = pl.reshape(post3_planes[0:1, update_offset:update_offset + 32], [32, 1])
        comb00 = pl.reshape(comb00_planes[0:1, update_offset:update_offset + 32], [32, 1])
        comb01 = pl.reshape(comb01_planes[0:1, update_offset:update_offset + 32], [32, 1])
        comb02 = pl.reshape(comb02_planes[0:1, update_offset:update_offset + 32], [32, 1])
        comb03 = pl.reshape(comb03_planes[0:1, update_offset:update_offset + 32], [32, 1])
        comb10 = pl.reshape(comb10_planes[0:1, update_offset:update_offset + 32], [32, 1])
        comb11 = pl.reshape(comb11_planes[0:1, update_offset:update_offset + 32], [32, 1])
        comb12 = pl.reshape(comb12_planes[0:1, update_offset:update_offset + 32], [32, 1])
        comb13 = pl.reshape(comb13_planes[0:1, update_offset:update_offset + 32], [32, 1])
        comb20 = pl.reshape(comb20_planes[0:1, update_offset:update_offset + 32], [32, 1])
        comb21 = pl.reshape(comb21_planes[0:1, update_offset:update_offset + 32], [32, 1])
        comb22 = pl.reshape(comb22_planes[0:1, update_offset:update_offset + 32], [32, 1])
        comb23 = pl.reshape(comb23_planes[0:1, update_offset:update_offset + 32], [32, 1])
        comb30 = pl.reshape(comb30_planes[0:1, update_offset:update_offset + 32], [32, 1])
        comb31 = pl.reshape(comb31_planes[0:1, update_offset:update_offset + 32], [32, 1])
        comb32 = pl.reshape(comb32_planes[0:1, update_offset:update_offset + 32], [32, 1])
        comb33 = pl.reshape(comb33_planes[0:1, update_offset:update_offset + 32], [32, 1])
        square_acc = pl.full([1, 32], dtype=pl.FP32, value=0.0)
        for hidden_block in pl.range(D // 128):
            hidden_start = hidden_block * 128
            input_tile = pl.cast(x[token_start:token_start + 32, hidden_start:hidden_start + 128], pl.FP32)
            original0 = pl.cast(residual[token_start:token_start + 32, 0 * D + hidden_start:0 * D + hidden_start + 128], pl.FP32)
            original1 = pl.cast(residual[token_start:token_start + 32, 1 * D + hidden_start:1 * D + hidden_start + 128], pl.FP32)
            original2 = pl.cast(residual[token_start:token_start + 32, 2 * D + hidden_start:2 * D + hidden_start + 128], pl.FP32)
            original3 = pl.cast(residual[token_start:token_start + 32, 3 * D + hidden_start:3 * D + hidden_start + 128], pl.FP32)
            updated0 = pl.row_expand_mul(input_tile, post0)
            updated0 = pl.add(updated0, pl.row_expand_mul(original0, comb00))
            updated0 = pl.add(updated0, pl.row_expand_mul(original1, comb01))
            updated0 = pl.add(updated0, pl.row_expand_mul(original2, comb02))
            updated0 = pl.add(updated0, pl.row_expand_mul(original3, comb03))
            updated1 = pl.row_expand_mul(input_tile, post1)
            updated1 = pl.add(updated1, pl.row_expand_mul(original0, comb10))
            updated1 = pl.add(updated1, pl.row_expand_mul(original1, comb11))
            updated1 = pl.add(updated1, pl.row_expand_mul(original2, comb12))
            updated1 = pl.add(updated1, pl.row_expand_mul(original3, comb13))
            updated2 = pl.row_expand_mul(input_tile, post2)
            updated2 = pl.add(updated2, pl.row_expand_mul(original0, comb20))
            updated2 = pl.add(updated2, pl.row_expand_mul(original1, comb21))
            updated2 = pl.add(updated2, pl.row_expand_mul(original2, comb22))
            updated2 = pl.add(updated2, pl.row_expand_mul(original3, comb23))
            updated3 = pl.row_expand_mul(input_tile, post3)
            updated3 = pl.add(updated3, pl.row_expand_mul(original0, comb30))
            updated3 = pl.add(updated3, pl.row_expand_mul(original1, comb31))
            updated3 = pl.add(updated3, pl.row_expand_mul(original2, comb32))
            updated3 = pl.add(updated3, pl.row_expand_mul(original3, comb33))
            square_sum = pl.row_sum(pl.add(pl.add(pl.mul(updated0, updated0), pl.mul(updated1, updated1)), pl.add(pl.mul(updated2, updated2), pl.mul(updated3, updated3))))
            square_acc = pl.add(square_acc, pl.reshape(square_sum, [1, 32]))
            updated_fp32[token_start:token_start + 32, 0 * D + hidden_start:0 * D + hidden_start + 128] = updated0
            residual[token_start:token_start + 32, 0 * D + hidden_start:0 * D + hidden_start + 128] = pl.cast(updated0, pl.BF16, mode="rint")
            updated_fp32[token_start:token_start + 32, 1 * D + hidden_start:1 * D + hidden_start + 128] = updated1
            residual[token_start:token_start + 32, 1 * D + hidden_start:1 * D + hidden_start + 128] = pl.cast(updated1, pl.BF16, mode="rint")
            updated_fp32[token_start:token_start + 32, 2 * D + hidden_start:2 * D + hidden_start + 128] = updated2
            residual[token_start:token_start + 32, 2 * D + hidden_start:2 * D + hidden_start + 128] = pl.cast(updated2, pl.BF16, mode="rint")
            updated_fp32[token_start:token_start + 32, 3 * D + hidden_start:3 * D + hidden_start + 128] = updated3
            residual[token_start:token_start + 32, 3 * D + hidden_start:3 * D + hidden_start + 128] = pl.cast(updated3, pl.BF16, mode="rint")
        square_sum = pl.reshape(square_acc, [32, 1])
        mean_square = pl.add(pl.mul(square_sum, 1.0 / FD), RMS_EPS)
        inverse_rms = pl.rsqrt(mean_square)
        inverse_rms = pl.mul(inverse_rms, pl.add(pl.mul(pl.mul(mean_square, pl.mul(inverse_rms, inverse_rms)), -0.5), 1.5))
        rms_scratch[0:1, token_start:token_start + 32] = pl.reshape(inverse_rms, [1, 32])
    return residual, updated_fp32, rms_scratch



@pl.jit
def mega_mhc_group512_32(x: pl.Tensor[[T, D], pl.BF16], residual: pl.InOut[pl.Tensor[[T, FD], pl.BF16]], input_mixes: pl.Tensor[[T, M3], pl.FP32], fn: pl.Tensor[[M3, FD], pl.FP32], normw: pl.Tensor[[1, M3], pl.FP32], mhc_scale: pl.Tensor[[3], pl.FP32], mhc_base: pl.Tensor[[1, M3], pl.FP32], next_mixes: pl.Out[pl.Tensor[[T, M3], pl.FP32]], h_next: pl.Out[pl.Tensor[[T, D], pl.BF16]]):
    updated_fp32 = pl.create_tensor([T, FD], dtype=pl.FP32)
    rms_scratch = pl.create_tensor([1, T], dtype=pl.FP32)
    producer_tids = pl.array.create(32, pl.TASK_ID)
    for group_index in pl.range(T // 512):
        group_start = group_index * 512
        group_x = pl.slice(x, [512, D], [group_start, 0])
        group_residual = pl.slice(residual, [512, FD], [group_start, 0])
        group_input_mixes = pl.slice(input_mixes, [512, M3], [group_start, 0])
        group_updated = pl.slice(updated_fp32, [512, FD], [group_start, 0])
        group_rms = pl.slice(rms_scratch, [1, 512], [0, group_start])
        with pl.spmd(16, name_hint="group512_producer_32") as producer_tid:
            group_residual, group_updated, group_rms = chunk_producer_32(group_x, group_residual, group_input_mixes, mhc_scale, mhc_base, group_updated, group_rms)
        producer_tids[group_index] = producer_tid
    for tid_slot in pl.range(T // 512, 32):
        producer_tids[tid_slot] = producer_tids[T // 512 - 1]
    for group_index in pl.range(T // 512):
        group_start = group_index * 512
        group_updated = pl.slice(updated_fp32, [512, FD], [group_start, 0])
        group_rms = pl.slice(rms_scratch, [1, 512], [0, group_start])
        group_next_mixes = pl.slice(next_mixes, [512, M3], [group_start, 0])
        group_h_next = pl.slice(h_next, [512, D], [group_start, 0])
        with pl.spmd(4, name_hint="group512_consumer", deps=[producer_tids[group_index]]) as consumer_tid:
            group_next_mixes, group_h_next = chunk_consumer(group_updated, group_rms, fn, normw, mhc_scale, mhc_base, group_next_mixes, group_h_next)
    return h_next



@pl.jit
def mega_mhc_group512_64(x: pl.Tensor[[T, D], pl.BF16], residual: pl.InOut[pl.Tensor[[T, FD], pl.BF16]], input_mixes: pl.Tensor[[T, M3], pl.FP32], fn: pl.Tensor[[M3, FD], pl.FP32], normw: pl.Tensor[[1, M3], pl.FP32], mhc_scale: pl.Tensor[[3], pl.FP32], mhc_base: pl.Tensor[[1, M3], pl.FP32], next_mixes: pl.Out[pl.Tensor[[T, M3], pl.FP32]], h_next: pl.Out[pl.Tensor[[T, D], pl.BF16]]):
    updated_fp32 = pl.create_tensor([T, FD], dtype=pl.FP32)
    rms_scratch = pl.create_tensor([1, T], dtype=pl.FP32)
    producer_tids = pl.array.create(32, pl.TASK_ID)
    for group_index in pl.range(T // 512):
        group_start = group_index * 512
        group_x = pl.slice(x, [512, D], [group_start, 0])
        group_residual = pl.slice(residual, [512, FD], [group_start, 0])
        group_input_mixes = pl.slice(input_mixes, [512, M3], [group_start, 0])
        group_updated = pl.slice(updated_fp32, [512, FD], [group_start, 0])
        group_rms = pl.slice(rms_scratch, [1, 512], [0, group_start])
        with pl.spmd(8, name_hint="group512_producer_64") as producer_tid:
            group_residual, group_updated, group_rms = chunk_producer_64(group_x, group_residual, group_input_mixes, mhc_scale, mhc_base, group_updated, group_rms)
        producer_tids[group_index] = producer_tid
    for tid_slot in pl.range(T // 512, 32):
        producer_tids[tid_slot] = producer_tids[T // 512 - 1]
    for group_index in pl.range(T // 512):
        group_start = group_index * 512
        group_updated = pl.slice(updated_fp32, [512, FD], [group_start, 0])
        group_rms = pl.slice(rms_scratch, [1, 512], [0, group_start])
        group_next_mixes = pl.slice(next_mixes, [512, M3], [group_start, 0])
        group_h_next = pl.slice(h_next, [512, D], [group_start, 0])
        with pl.spmd(4, name_hint="group512_consumer", deps=[producer_tids[group_index]]) as consumer_tid:
            group_next_mixes, group_h_next = chunk_consumer(group_updated, group_rms, fn, normw, mhc_scale, mhc_base, group_next_mixes, group_h_next)
    return h_next



@pl.jit
def mega_mhc_group512_128(x: pl.Tensor[[T, D], pl.BF16], residual: pl.InOut[pl.Tensor[[T, FD], pl.BF16]], input_mixes: pl.Tensor[[T, M3], pl.FP32], fn: pl.Tensor[[M3, FD], pl.FP32], normw: pl.Tensor[[1, M3], pl.FP32], mhc_scale: pl.Tensor[[3], pl.FP32], mhc_base: pl.Tensor[[1, M3], pl.FP32], next_mixes: pl.Out[pl.Tensor[[T, M3], pl.FP32]], h_next: pl.Out[pl.Tensor[[T, D], pl.BF16]]):
    updated_fp32 = pl.create_tensor([T, FD], dtype=pl.FP32)
    rms_scratch = pl.create_tensor([1, T], dtype=pl.FP32)
    producer_tids = pl.array.create(32, pl.TASK_ID)
    for group_index in pl.range(T // 512):
        group_start = group_index * 512
        group_x = pl.slice(x, [512, D], [group_start, 0])
        group_residual = pl.slice(residual, [512, FD], [group_start, 0])
        group_input_mixes = pl.slice(input_mixes, [512, M3], [group_start, 0])
        group_updated = pl.slice(updated_fp32, [512, FD], [group_start, 0])
        group_rms = pl.slice(rms_scratch, [1, 512], [0, group_start])
        with pl.spmd(4, name_hint="group512_producer_128") as producer_tid:
            group_residual, group_updated, group_rms = chunk_producer_128(group_x, group_residual, group_input_mixes, mhc_scale, mhc_base, group_updated, group_rms)
        producer_tids[group_index] = producer_tid
    for tid_slot in pl.range(T // 512, 32):
        producer_tids[tid_slot] = producer_tids[T // 512 - 1]
    for group_index in pl.range(T // 512):
        group_start = group_index * 512
        group_updated = pl.slice(updated_fp32, [512, FD], [group_start, 0])
        group_rms = pl.slice(rms_scratch, [1, 512], [0, group_start])
        group_next_mixes = pl.slice(next_mixes, [512, M3], [group_start, 0])
        group_h_next = pl.slice(h_next, [512, D], [group_start, 0])
        with pl.spmd(4, name_hint="group512_consumer", deps=[producer_tids[group_index]]) as consumer_tid:
            group_next_mixes, group_h_next = chunk_consumer(group_updated, group_rms, fn, normw, mhc_scale, mhc_base, group_next_mixes, group_h_next)
    return h_next



@pl.jit
def mega_mhc_mixed512_64(
    x: pl.Tensor[[T, D], pl.BF16],
    residual: pl.InOut[pl.Tensor[[T, FD], pl.BF16]],
    input_mixes: pl.Tensor[[T, M3], pl.FP32],
    fn: pl.Tensor[[M3, FD], pl.FP32],
    normw: pl.Tensor[[1, M3], pl.FP32],
    mhc_scale: pl.Tensor[[3], pl.FP32],
    mhc_base: pl.Tensor[[1, M3], pl.FP32],
    next_mixes: pl.Out[pl.Tensor[[T, M3], pl.FP32]],
    h_next: pl.Out[pl.Tensor[[T, D], pl.BF16]],
):
    updated_fp32 = pl.create_tensor([T, FD], dtype=pl.FP32)
    rms_scratch = pl.create_tensor([1, T], dtype=pl.FP32)
    for group_index in pl.range(T // 512):
        group_start = group_index * 512
        group_x = pl.slice(x, [512, D], [group_start, 0])
        group_residual = pl.slice(residual, [512, FD], [group_start, 0])
        group_input_mixes = pl.slice(input_mixes, [512, M3], [group_start, 0])
        group_updated = pl.slice(updated_fp32, [512, FD], [group_start, 0])
        group_rms = pl.slice(rms_scratch, [1, 512], [0, group_start])
        with pl.spmd(8, name_hint="mixed64_producer") as producer_tid:
            group_residual, group_updated, group_rms = chunk_producer_64(
                group_x, group_residual, group_input_mixes, mhc_scale, mhc_base, group_updated, group_rms
            )
        group_next_mixes = pl.slice(next_mixes, [512, M3], [group_start, 0])
        group_h_next = pl.slice(h_next, [512, D], [group_start, 0])
        with pl.spmd(4, name_hint="mixed64_consumer", deps=[producer_tid]):
            group_next_mixes, group_h_next = chunk_consumer(
                group_updated, group_rms, fn, normw, mhc_scale, mhc_base, group_next_mixes, group_h_next
            )
    return h_next

@pl.jit
def mega_mhc_owner512(x: pl.Tensor[[T, D], pl.BF16], residual: pl.InOut[pl.Tensor[[T, FD], pl.BF16]], input_mixes: pl.Tensor[[T, M3], pl.FP32], fn: pl.Tensor[[M3, FD], pl.FP32], normw: pl.Tensor[[1, M3], pl.FP32], mhc_scale: pl.Tensor[[3], pl.FP32], mhc_base: pl.Tensor[[1, M3], pl.FP32], next_mixes: pl.Out[pl.Tensor[[T, M3], pl.FP32]], h_next: pl.Out[pl.Tensor[[T, D], pl.BF16]]):
    updated_fp32 = pl.create_tensor([T, FD], dtype=pl.FP32)
    rms_scratch = pl.create_tensor([1, T], dtype=pl.FP32)
    ffts_workspace = pl.create_tensor([256], dtype=pl.INT64)
    for group_index in pl.range(T // 512):
        group_start = group_index * 512
        group_x = pl.slice(x, [512, D], [group_start, 0])
        group_residual = pl.slice(residual, [512, FD], [group_start, 0])
        group_input_mixes = pl.slice(input_mixes, [512, M3], [group_start, 0])
        group_updated = pl.slice(updated_fp32, [512, FD], [group_start, 0])
        group_rms = pl.slice(rms_scratch, [1, 512], [0, group_start])
        group_next_mixes = pl.slice(next_mixes, [512, M3], [group_start, 0])
        group_h_next = pl.slice(h_next, [512, D], [group_start, 0])
        with pl.spmd(4, name_hint="owner_fused"):
            with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.cross_core_slot(slot_num=1)]):
                owner_start = pl.tile.get_block_idx() * 128
                pl.system.set_ffts(ffts_workspace)
                for producer_lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                    for coefficient_block in pl.range(1):
                        macro_start = owner_start + 64 * producer_lane + coefficient_block * 64
                        scale_pre = pl.read(mhc_scale, [0])
                        scale_post = pl.read(mhc_scale, [1])
                        scale_comb = pl.read(mhc_scale, [2])
                        mix_planes = pl.transpose(group_input_mixes[macro_start:macro_start + 64, 0:M3], 0, 1)
                        post_raw0 = pl.add(pl.mul(pl.reshape(mix_planes[4:5, 0:64], [1, 64]), scale_post), pl.read(mhc_base, [0, 4]))
                        post0_planes = pl.mul(pl.recip(pl.add(pl.exp(pl.neg(post_raw0)), 1.0)), POST_MULT)
                        post_raw1 = pl.add(pl.mul(pl.reshape(mix_planes[5:6, 0:64], [1, 64]), scale_post), pl.read(mhc_base, [0, 5]))
                        post1_planes = pl.mul(pl.recip(pl.add(pl.exp(pl.neg(post_raw1)), 1.0)), POST_MULT)
                        post_raw2 = pl.add(pl.mul(pl.reshape(mix_planes[6:7, 0:64], [1, 64]), scale_post), pl.read(mhc_base, [0, 6]))
                        post2_planes = pl.mul(pl.recip(pl.add(pl.exp(pl.neg(post_raw2)), 1.0)), POST_MULT)
                        post_raw3 = pl.add(pl.mul(pl.reshape(mix_planes[7:8, 0:64], [1, 64]), scale_post), pl.read(mhc_base, [0, 7]))
                        post3_planes = pl.mul(pl.recip(pl.add(pl.exp(pl.neg(post_raw3)), 1.0)), POST_MULT)
                        comb00_planes = pl.add(pl.mul(pl.reshape(mix_planes[8:9, 0:64], [1, 64]), scale_comb), pl.read(mhc_base, [0, 8]))
                        comb01_planes = pl.add(pl.mul(pl.reshape(mix_planes[9:10, 0:64], [1, 64]), scale_comb), pl.read(mhc_base, [0, 9]))
                        comb02_planes = pl.add(pl.mul(pl.reshape(mix_planes[10:11, 0:64], [1, 64]), scale_comb), pl.read(mhc_base, [0, 10]))
                        comb03_planes = pl.add(pl.mul(pl.reshape(mix_planes[11:12, 0:64], [1, 64]), scale_comb), pl.read(mhc_base, [0, 11]))
                        comb10_planes = pl.add(pl.mul(pl.reshape(mix_planes[12:13, 0:64], [1, 64]), scale_comb), pl.read(mhc_base, [0, 12]))
                        comb11_planes = pl.add(pl.mul(pl.reshape(mix_planes[13:14, 0:64], [1, 64]), scale_comb), pl.read(mhc_base, [0, 13]))
                        comb12_planes = pl.add(pl.mul(pl.reshape(mix_planes[14:15, 0:64], [1, 64]), scale_comb), pl.read(mhc_base, [0, 14]))
                        comb13_planes = pl.add(pl.mul(pl.reshape(mix_planes[15:16, 0:64], [1, 64]), scale_comb), pl.read(mhc_base, [0, 15]))
                        comb20_planes = pl.add(pl.mul(pl.reshape(mix_planes[16:17, 0:64], [1, 64]), scale_comb), pl.read(mhc_base, [0, 16]))
                        comb21_planes = pl.add(pl.mul(pl.reshape(mix_planes[17:18, 0:64], [1, 64]), scale_comb), pl.read(mhc_base, [0, 17]))
                        comb22_planes = pl.add(pl.mul(pl.reshape(mix_planes[18:19, 0:64], [1, 64]), scale_comb), pl.read(mhc_base, [0, 18]))
                        comb23_planes = pl.add(pl.mul(pl.reshape(mix_planes[19:20, 0:64], [1, 64]), scale_comb), pl.read(mhc_base, [0, 19]))
                        comb30_planes = pl.add(pl.mul(pl.reshape(mix_planes[20:21, 0:64], [1, 64]), scale_comb), pl.read(mhc_base, [0, 20]))
                        comb31_planes = pl.add(pl.mul(pl.reshape(mix_planes[21:22, 0:64], [1, 64]), scale_comb), pl.read(mhc_base, [0, 21]))
                        comb32_planes = pl.add(pl.mul(pl.reshape(mix_planes[22:23, 0:64], [1, 64]), scale_comb), pl.read(mhc_base, [0, 22]))
                        comb33_planes = pl.add(pl.mul(pl.reshape(mix_planes[23:24, 0:64], [1, 64]), scale_comb), pl.read(mhc_base, [0, 23]))
                        packed_row0 = pl.concat(pl.concat(comb00_planes, comb01_planes), pl.concat(comb02_planes, comb03_planes))
                        packed_row1 = pl.concat(pl.concat(comb10_planes, comb11_planes), pl.concat(comb12_planes, comb13_planes))
                        packed_row2 = pl.concat(pl.concat(comb20_planes, comb21_planes), pl.concat(comb22_planes, comb23_planes))
                        packed_row3 = pl.concat(pl.concat(comb30_planes, comb31_planes), pl.concat(comb32_planes, comb33_planes))
                        superrows = pl.reshape(pl.concat(pl.concat(packed_row0, packed_row1), pl.concat(packed_row2, packed_row3)), [4, 256])
                        quarter0 = superrows[0:4, 0:64]
                        quarter1 = superrows[0:4, 64:128]
                        quarter2 = superrows[0:4, 128:192]
                        quarter3 = superrows[0:4, 192:256]
                        row_maximum = pl.maximum(pl.maximum(quarter0, quarter1), pl.maximum(quarter2, quarter3))
                        replicated_half = pl.concat(row_maximum, row_maximum)
                        replicated = pl.concat(replicated_half, replicated_half)
                        superrows = pl.exp(pl.sub(superrows, replicated))
                        quarter0 = superrows[0:4, 0:64]
                        quarter1 = superrows[0:4, 64:128]
                        quarter2 = superrows[0:4, 128:192]
                        quarter3 = superrows[0:4, 192:256]
                        row_denominator = pl.add(pl.add(quarter0, quarter1), pl.add(quarter2, quarter3))
                        replicated_half = pl.concat(row_denominator, row_denominator)
                        replicated = pl.concat(replicated_half, replicated_half)
                        superrows = pl.add(pl.div(superrows, replicated), PRE_EPS)
                        column_row0 = superrows[0:1, 0:256]
                        column_row1 = superrows[1:2, 0:256]
                        column_row2 = superrows[2:3, 0:256]
                        column_row3 = superrows[3:4, 0:256]
                        column_denominator = pl.add(pl.add(pl.add(column_row0, column_row1), pl.add(column_row2, column_row3)), PRE_EPS)
                        superrows = pl.col_expand_div(superrows, column_denominator)
                        for super_iteration in pl.range(SINK - 1):
                            quarter0 = superrows[0:4, 0:64]
                            quarter1 = superrows[0:4, 64:128]
                            quarter2 = superrows[0:4, 128:192]
                            quarter3 = superrows[0:4, 192:256]
                            row_denominator = pl.add(pl.add(quarter0, quarter1), pl.add(quarter2, quarter3))
                            row_denominator = pl.add(row_denominator, PRE_EPS)
                            replicated_half = pl.concat(row_denominator, row_denominator)
                            replicated = pl.concat(replicated_half, replicated_half)
                            superrows = pl.div(superrows, replicated)
                            column_row0 = superrows[0:1, 0:256]
                            column_row1 = superrows[1:2, 0:256]
                            column_row2 = superrows[2:3, 0:256]
                            column_row3 = superrows[3:4, 0:256]
                            column_denominator = pl.add(pl.add(pl.add(column_row0, column_row1), pl.add(column_row2, column_row3)), PRE_EPS)
                            superrows = pl.col_expand_div(superrows, column_denominator)
                        superplanes = pl.reshape(superrows, [16, 64])
                        comb00_planes = superplanes[0:1, 0:64]
                        comb01_planes = superplanes[1:2, 0:64]
                        comb02_planes = superplanes[2:3, 0:64]
                        comb03_planes = superplanes[3:4, 0:64]
                        comb10_planes = superplanes[4:5, 0:64]
                        comb11_planes = superplanes[5:6, 0:64]
                        comb12_planes = superplanes[6:7, 0:64]
                        comb13_planes = superplanes[7:8, 0:64]
                        comb20_planes = superplanes[8:9, 0:64]
                        comb21_planes = superplanes[9:10, 0:64]
                        comb22_planes = superplanes[10:11, 0:64]
                        comb23_planes = superplanes[11:12, 0:64]
                        comb30_planes = superplanes[12:13, 0:64]
                        comb31_planes = superplanes[13:14, 0:64]
                        comb32_planes = superplanes[14:15, 0:64]
                        comb33_planes = superplanes[15:16, 0:64]
                        for update_block in pl.range(64 // 32):
                            update_offset = update_block * 32
                            token_start = macro_start + update_offset
                            post0 = pl.reshape(post0_planes[0:1, update_offset:update_offset + 32], [32, 1])
                            post1 = pl.reshape(post1_planes[0:1, update_offset:update_offset + 32], [32, 1])
                            post2 = pl.reshape(post2_planes[0:1, update_offset:update_offset + 32], [32, 1])
                            post3 = pl.reshape(post3_planes[0:1, update_offset:update_offset + 32], [32, 1])
                            comb00 = pl.reshape(comb00_planes[0:1, update_offset:update_offset + 32], [32, 1])
                            comb01 = pl.reshape(comb01_planes[0:1, update_offset:update_offset + 32], [32, 1])
                            comb02 = pl.reshape(comb02_planes[0:1, update_offset:update_offset + 32], [32, 1])
                            comb03 = pl.reshape(comb03_planes[0:1, update_offset:update_offset + 32], [32, 1])
                            comb10 = pl.reshape(comb10_planes[0:1, update_offset:update_offset + 32], [32, 1])
                            comb11 = pl.reshape(comb11_planes[0:1, update_offset:update_offset + 32], [32, 1])
                            comb12 = pl.reshape(comb12_planes[0:1, update_offset:update_offset + 32], [32, 1])
                            comb13 = pl.reshape(comb13_planes[0:1, update_offset:update_offset + 32], [32, 1])
                            comb20 = pl.reshape(comb20_planes[0:1, update_offset:update_offset + 32], [32, 1])
                            comb21 = pl.reshape(comb21_planes[0:1, update_offset:update_offset + 32], [32, 1])
                            comb22 = pl.reshape(comb22_planes[0:1, update_offset:update_offset + 32], [32, 1])
                            comb23 = pl.reshape(comb23_planes[0:1, update_offset:update_offset + 32], [32, 1])
                            comb30 = pl.reshape(comb30_planes[0:1, update_offset:update_offset + 32], [32, 1])
                            comb31 = pl.reshape(comb31_planes[0:1, update_offset:update_offset + 32], [32, 1])
                            comb32 = pl.reshape(comb32_planes[0:1, update_offset:update_offset + 32], [32, 1])
                            comb33 = pl.reshape(comb33_planes[0:1, update_offset:update_offset + 32], [32, 1])
                            square_acc = pl.full([1, 32], dtype=pl.FP32, value=0.0)
                            for hidden_block in pl.range(D // 128):
                                hidden_start = hidden_block * 128
                                input_tile = pl.cast(group_x[token_start:token_start + 32, hidden_start:hidden_start + 128], pl.FP32)
                                original0 = pl.cast(group_residual[token_start:token_start + 32, 0 * D + hidden_start:0 * D + hidden_start + 128], pl.FP32)
                                original1 = pl.cast(group_residual[token_start:token_start + 32, 1 * D + hidden_start:1 * D + hidden_start + 128], pl.FP32)
                                original2 = pl.cast(group_residual[token_start:token_start + 32, 2 * D + hidden_start:2 * D + hidden_start + 128], pl.FP32)
                                original3 = pl.cast(group_residual[token_start:token_start + 32, 3 * D + hidden_start:3 * D + hidden_start + 128], pl.FP32)
                                updated0 = pl.row_expand_mul(input_tile, post0)
                                updated0 = pl.add(updated0, pl.row_expand_mul(original0, comb00))
                                updated0 = pl.add(updated0, pl.row_expand_mul(original1, comb01))
                                updated0 = pl.add(updated0, pl.row_expand_mul(original2, comb02))
                                updated0 = pl.add(updated0, pl.row_expand_mul(original3, comb03))
                                updated1 = pl.row_expand_mul(input_tile, post1)
                                updated1 = pl.add(updated1, pl.row_expand_mul(original0, comb10))
                                updated1 = pl.add(updated1, pl.row_expand_mul(original1, comb11))
                                updated1 = pl.add(updated1, pl.row_expand_mul(original2, comb12))
                                updated1 = pl.add(updated1, pl.row_expand_mul(original3, comb13))
                                updated2 = pl.row_expand_mul(input_tile, post2)
                                updated2 = pl.add(updated2, pl.row_expand_mul(original0, comb20))
                                updated2 = pl.add(updated2, pl.row_expand_mul(original1, comb21))
                                updated2 = pl.add(updated2, pl.row_expand_mul(original2, comb22))
                                updated2 = pl.add(updated2, pl.row_expand_mul(original3, comb23))
                                updated3 = pl.row_expand_mul(input_tile, post3)
                                updated3 = pl.add(updated3, pl.row_expand_mul(original0, comb30))
                                updated3 = pl.add(updated3, pl.row_expand_mul(original1, comb31))
                                updated3 = pl.add(updated3, pl.row_expand_mul(original2, comb32))
                                updated3 = pl.add(updated3, pl.row_expand_mul(original3, comb33))
                                square_sum = pl.row_sum(pl.add(pl.add(pl.mul(updated0, updated0), pl.mul(updated1, updated1)), pl.add(pl.mul(updated2, updated2), pl.mul(updated3, updated3))))
                                square_acc = pl.add(square_acc, pl.reshape(square_sum, [1, 32]))
                                group_updated[token_start:token_start + 32, 0 * D + hidden_start:0 * D + hidden_start + 128] = updated0
                                group_residual[token_start:token_start + 32, 0 * D + hidden_start:0 * D + hidden_start + 128] = pl.cast(updated0, pl.BF16, mode="rint")
                                group_updated[token_start:token_start + 32, 1 * D + hidden_start:1 * D + hidden_start + 128] = updated1
                                group_residual[token_start:token_start + 32, 1 * D + hidden_start:1 * D + hidden_start + 128] = pl.cast(updated1, pl.BF16, mode="rint")
                                group_updated[token_start:token_start + 32, 2 * D + hidden_start:2 * D + hidden_start + 128] = updated2
                                group_residual[token_start:token_start + 32, 2 * D + hidden_start:2 * D + hidden_start + 128] = pl.cast(updated2, pl.BF16, mode="rint")
                                group_updated[token_start:token_start + 32, 3 * D + hidden_start:3 * D + hidden_start + 128] = updated3
                                group_residual[token_start:token_start + 32, 3 * D + hidden_start:3 * D + hidden_start + 128] = pl.cast(updated3, pl.BF16, mode="rint")
                            square_sum = pl.reshape(square_acc, [32, 1])
                            mean_square = pl.add(pl.mul(square_sum, 1.0 / FD), RMS_EPS)
                            inverse_rms = pl.rsqrt(mean_square)
                            inverse_rms = pl.mul(inverse_rms, pl.add(pl.mul(pl.mul(mean_square, pl.mul(inverse_rms, inverse_rms)), -0.5), 1.5))
                            group_rms[0:1, token_start:token_start + 32] = pl.reshape(inverse_rms, [1, 32])
                    pl.system.barrier(pipe=pl.PipeType.ALL)
                    pl.system.cacheinvalid()
                    pl.system.fence()
                    pl.system.sync_set(4, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV)
                pl.system.sync_wait(4, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                pl.system.cacheinvalid()
                pl.system.fence()
                accumulator = pl.create_tensor([128, 32], dtype=pl.FP32)
                for projection_block in pl.pipeline(0, FD // 128, stage=2):
                    projection_start = projection_block * 128
                    updated_chunk = pl.slice(group_updated, [128, 128], [owner_start, projection_start], valid_shape=[128, 128])
                    weight_chunk = pl.slice(fn, [32, 128], [0, projection_start], valid_shape=[24, 128], clamp=True)
                    accumulator = pl.matmul_acc(accumulator, updated_chunk, weight_chunk, b_trans=True, init_cond=(projection_block == 0))
                for consumer_lane in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                    lane_accumulator = pl.aiv_shard(accumulator)
                    token_start = owner_start + consumer_lane * 64
                    norm_row = pl.slice(normw, [1, 32], [0, 0], valid_shape=[1, 24], clamp=True)
                    rms_tile = pl.reshape(pl.slice(group_rms, [1, 64], [0, token_start], valid_shape=[1, 64]), [64, 1])
                    predicted = pl.row_expand_mul(pl.col_expand_mul(lane_accumulator, norm_row), rms_tile)
                    group_next_mixes[token_start:token_start + 64, 0:24] = predicted[0:64, 0:24]
                    scale_pre = pl.read(mhc_scale, [0])
                    coefficient_indices = pl.cast(pl.arange(0, [1, 32], dtype=pl.INT32), pl.FP32)
                    pre_selector0 = pl.maximum(pl.add(pl.neg(pl.abs(pl.sub(coefficient_indices, 0.0))), 1.0), 0.0)
                    pre_coefficient0 = pl.row_sum(pl.col_expand_mul(predicted, pre_selector0))
                    pre_raw0 = pl.add(pl.mul(pre_coefficient0, scale_pre), pl.read(mhc_base, [0, 0]))
                    next_pre0 = pl.add(pl.recip(pl.add(pl.exp(pl.neg(pre_raw0)), 1.0)), PRE_EPS)
                    pre_selector1 = pl.maximum(pl.add(pl.neg(pl.abs(pl.sub(coefficient_indices, 1.0))), 1.0), 0.0)
                    pre_coefficient1 = pl.row_sum(pl.col_expand_mul(predicted, pre_selector1))
                    pre_raw1 = pl.add(pl.mul(pre_coefficient1, scale_pre), pl.read(mhc_base, [0, 1]))
                    next_pre1 = pl.add(pl.recip(pl.add(pl.exp(pl.neg(pre_raw1)), 1.0)), PRE_EPS)
                    pre_selector2 = pl.maximum(pl.add(pl.neg(pl.abs(pl.sub(coefficient_indices, 2.0))), 1.0), 0.0)
                    pre_coefficient2 = pl.row_sum(pl.col_expand_mul(predicted, pre_selector2))
                    pre_raw2 = pl.add(pl.mul(pre_coefficient2, scale_pre), pl.read(mhc_base, [0, 2]))
                    next_pre2 = pl.add(pl.recip(pl.add(pl.exp(pl.neg(pre_raw2)), 1.0)), PRE_EPS)
                    pre_selector3 = pl.maximum(pl.add(pl.neg(pl.abs(pl.sub(coefficient_indices, 3.0))), 1.0), 0.0)
                    pre_coefficient3 = pl.row_sum(pl.col_expand_mul(predicted, pre_selector3))
                    pre_raw3 = pl.add(pl.mul(pre_coefficient3, scale_pre), pl.read(mhc_base, [0, 3]))
                    next_pre3 = pl.add(pl.recip(pl.add(pl.exp(pl.neg(pre_raw3)), 1.0)), PRE_EPS)
                    for hidden_block in pl.range(D // 64):
                        hidden_start = hidden_block * 64
                    for hidden_block in pl.range(D // 64):
                        hidden_start = hidden_block * 64
                        updated_copy0 = group_updated[token_start:token_start + 64, 0 * D + hidden_start:0 * D + hidden_start + 64]
                        hidden = pl.row_expand_mul(updated_copy0, next_pre0)
                        updated_copy1 = group_updated[token_start:token_start + 64, 1 * D + hidden_start:1 * D + hidden_start + 64]
                        hidden = pl.add(hidden, pl.row_expand_mul(updated_copy1, next_pre1))
                        updated_copy2 = group_updated[token_start:token_start + 64, 2 * D + hidden_start:2 * D + hidden_start + 64]
                        hidden = pl.add(hidden, pl.row_expand_mul(updated_copy2, next_pre2))
                        updated_copy3 = group_updated[token_start:token_start + 64, 3 * D + hidden_start:3 * D + hidden_start + 64]
                        hidden = pl.add(hidden, pl.row_expand_mul(updated_copy3, next_pre3))
                        group_h_next[token_start:token_start + 64, hidden_start:hidden_start + 64] = pl.cast(hidden, pl.BF16, mode="rint")
    return h_next



@pl.jit
def mega_mhc_packed_owner512(x: pl.Tensor[[T, D], pl.BF16], residual: pl.InOut[pl.Tensor[[T, FD], pl.BF16]], input_mixes: pl.Tensor[[T, M3], pl.FP32], fn: pl.Tensor[[M3, FD], pl.FP32], normw: pl.Tensor[[1, M3], pl.FP32], mhc_scale: pl.Tensor[[3], pl.FP32], mhc_base: pl.Tensor[[1, M3], pl.FP32], next_mixes: pl.Out[pl.Tensor[[T, M3], pl.FP32]], h_next: pl.Out[pl.Tensor[[T, D], pl.BF16]]):
    updated_fp32 = pl.create_tensor([T // 512 * 576, FD], dtype=pl.FP32)
    rms_scratch = pl.create_tensor([1, T // 512 * 576], dtype=pl.FP32)
    ffts_workspace = pl.create_tensor([256], dtype=pl.INT64)
    for group_index in pl.range(T // 512):
        group_start = group_index * 512
        group_x = pl.slice(x, [512, D], [group_start, 0])
        group_residual = pl.slice(residual, [512, FD], [group_start, 0])
        group_input_mixes = pl.slice(input_mixes, [512, M3], [group_start, 0])
        group_updated = pl.slice(updated_fp32, [576, FD], [group_index * 576, 0])
        group_rms = pl.slice(rms_scratch, [1, 576], [0, group_index * 576])
        group_next_mixes = pl.slice(next_mixes, [512, M3], [group_start, 0])
        group_h_next = pl.slice(h_next, [512, D], [group_start, 0])
        with pl.spmd(3, name_hint='packed_owner_fused'):
            with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.cross_core_slot(slot_num=2, local_slot_num=1)]):
                owner_index = pl.tile.get_block_idx()
                owner_start = owner_index * 192
                pl.system.set_ffts(ffts_workspace)
                for producer_lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                    if owner_index < 2 or producer_lane == 0:
                        long_segment96_macro_start = owner_start + producer_lane * 96
                        long_segment96_scale_pre = pl.read(mhc_scale, [0])
                        long_segment96_scale_post = pl.read(mhc_scale, [1])
                        long_segment96_scale_comb = pl.read(mhc_scale, [2])
                        long_segment96_mix_planes = pl.transpose(group_input_mixes[long_segment96_macro_start:long_segment96_macro_start + 96, 0:M3], 0, 1)
                        long_segment96_post_raw0 = pl.add(pl.mul(pl.reshape(long_segment96_mix_planes[4:5, 0:96], [1, 96]), long_segment96_scale_post), pl.read(mhc_base, [0, 4]))
                        long_segment96_post0_planes = pl.mul(pl.recip(pl.add(pl.exp(pl.neg(long_segment96_post_raw0)), 1.0)), POST_MULT)
                        long_segment96_post_raw1 = pl.add(pl.mul(pl.reshape(long_segment96_mix_planes[5:6, 0:96], [1, 96]), long_segment96_scale_post), pl.read(mhc_base, [0, 5]))
                        long_segment96_post1_planes = pl.mul(pl.recip(pl.add(pl.exp(pl.neg(long_segment96_post_raw1)), 1.0)), POST_MULT)
                        long_segment96_post_raw2 = pl.add(pl.mul(pl.reshape(long_segment96_mix_planes[6:7, 0:96], [1, 96]), long_segment96_scale_post), pl.read(mhc_base, [0, 6]))
                        long_segment96_post2_planes = pl.mul(pl.recip(pl.add(pl.exp(pl.neg(long_segment96_post_raw2)), 1.0)), POST_MULT)
                        long_segment96_post_raw3 = pl.add(pl.mul(pl.reshape(long_segment96_mix_planes[7:8, 0:96], [1, 96]), long_segment96_scale_post), pl.read(mhc_base, [0, 7]))
                        long_segment96_post3_planes = pl.mul(pl.recip(pl.add(pl.exp(pl.neg(long_segment96_post_raw3)), 1.0)), POST_MULT)
                        long_segment96_comb00_planes = pl.add(pl.mul(pl.reshape(long_segment96_mix_planes[8:9, 0:96], [1, 96]), long_segment96_scale_comb), pl.read(mhc_base, [0, 8]))
                        long_segment96_comb01_planes = pl.add(pl.mul(pl.reshape(long_segment96_mix_planes[9:10, 0:96], [1, 96]), long_segment96_scale_comb), pl.read(mhc_base, [0, 9]))
                        long_segment96_comb02_planes = pl.add(pl.mul(pl.reshape(long_segment96_mix_planes[10:11, 0:96], [1, 96]), long_segment96_scale_comb), pl.read(mhc_base, [0, 10]))
                        long_segment96_comb03_planes = pl.add(pl.mul(pl.reshape(long_segment96_mix_planes[11:12, 0:96], [1, 96]), long_segment96_scale_comb), pl.read(mhc_base, [0, 11]))
                        long_segment96_comb10_planes = pl.add(pl.mul(pl.reshape(long_segment96_mix_planes[12:13, 0:96], [1, 96]), long_segment96_scale_comb), pl.read(mhc_base, [0, 12]))
                        long_segment96_comb11_planes = pl.add(pl.mul(pl.reshape(long_segment96_mix_planes[13:14, 0:96], [1, 96]), long_segment96_scale_comb), pl.read(mhc_base, [0, 13]))
                        long_segment96_comb12_planes = pl.add(pl.mul(pl.reshape(long_segment96_mix_planes[14:15, 0:96], [1, 96]), long_segment96_scale_comb), pl.read(mhc_base, [0, 14]))
                        long_segment96_comb13_planes = pl.add(pl.mul(pl.reshape(long_segment96_mix_planes[15:16, 0:96], [1, 96]), long_segment96_scale_comb), pl.read(mhc_base, [0, 15]))
                        long_segment96_comb20_planes = pl.add(pl.mul(pl.reshape(long_segment96_mix_planes[16:17, 0:96], [1, 96]), long_segment96_scale_comb), pl.read(mhc_base, [0, 16]))
                        long_segment96_comb21_planes = pl.add(pl.mul(pl.reshape(long_segment96_mix_planes[17:18, 0:96], [1, 96]), long_segment96_scale_comb), pl.read(mhc_base, [0, 17]))
                        long_segment96_comb22_planes = pl.add(pl.mul(pl.reshape(long_segment96_mix_planes[18:19, 0:96], [1, 96]), long_segment96_scale_comb), pl.read(mhc_base, [0, 18]))
                        long_segment96_comb23_planes = pl.add(pl.mul(pl.reshape(long_segment96_mix_planes[19:20, 0:96], [1, 96]), long_segment96_scale_comb), pl.read(mhc_base, [0, 19]))
                        long_segment96_comb30_planes = pl.add(pl.mul(pl.reshape(long_segment96_mix_planes[20:21, 0:96], [1, 96]), long_segment96_scale_comb), pl.read(mhc_base, [0, 20]))
                        long_segment96_comb31_planes = pl.add(pl.mul(pl.reshape(long_segment96_mix_planes[21:22, 0:96], [1, 96]), long_segment96_scale_comb), pl.read(mhc_base, [0, 21]))
                        long_segment96_comb32_planes = pl.add(pl.mul(pl.reshape(long_segment96_mix_planes[22:23, 0:96], [1, 96]), long_segment96_scale_comb), pl.read(mhc_base, [0, 22]))
                        long_segment96_comb33_planes = pl.add(pl.mul(pl.reshape(long_segment96_mix_planes[23:24, 0:96], [1, 96]), long_segment96_scale_comb), pl.read(mhc_base, [0, 23]))
                        long_segment96_packed_row0 = pl.concat(pl.concat(long_segment96_comb00_planes, long_segment96_comb01_planes), pl.concat(long_segment96_comb02_planes, long_segment96_comb03_planes))
                        long_segment96_packed_row1 = pl.concat(pl.concat(long_segment96_comb10_planes, long_segment96_comb11_planes), pl.concat(long_segment96_comb12_planes, long_segment96_comb13_planes))
                        long_segment96_packed_row2 = pl.concat(pl.concat(long_segment96_comb20_planes, long_segment96_comb21_planes), pl.concat(long_segment96_comb22_planes, long_segment96_comb23_planes))
                        long_segment96_packed_row3 = pl.concat(pl.concat(long_segment96_comb30_planes, long_segment96_comb31_planes), pl.concat(long_segment96_comb32_planes, long_segment96_comb33_planes))
                        long_segment96_superrows = pl.reshape(pl.concat(pl.concat(long_segment96_packed_row0, long_segment96_packed_row1), pl.concat(long_segment96_packed_row2, long_segment96_packed_row3)), [4, 384])
                        long_segment96_quarter0 = long_segment96_superrows[0:4, 0:96]
                        long_segment96_quarter1 = long_segment96_superrows[0:4, 96:192]
                        long_segment96_quarter2 = long_segment96_superrows[0:4, 192:288]
                        long_segment96_quarter3 = long_segment96_superrows[0:4, 288:384]
                        long_segment96_row_maximum = pl.maximum(pl.maximum(long_segment96_quarter0, long_segment96_quarter1), pl.maximum(long_segment96_quarter2, long_segment96_quarter3))
                        long_segment96_replicated_half = pl.concat(long_segment96_row_maximum, long_segment96_row_maximum)
                        long_segment96_replicated = pl.concat(long_segment96_replicated_half, long_segment96_replicated_half)
                        long_segment96_superrows = pl.exp(pl.sub(long_segment96_superrows, long_segment96_replicated))
                        long_segment96_quarter0 = long_segment96_superrows[0:4, 0:96]
                        long_segment96_quarter1 = long_segment96_superrows[0:4, 96:192]
                        long_segment96_quarter2 = long_segment96_superrows[0:4, 192:288]
                        long_segment96_quarter3 = long_segment96_superrows[0:4, 288:384]
                        long_segment96_row_denominator = pl.add(pl.add(long_segment96_quarter0, long_segment96_quarter1), pl.add(long_segment96_quarter2, long_segment96_quarter3))
                        long_segment96_replicated_half = pl.concat(long_segment96_row_denominator, long_segment96_row_denominator)
                        long_segment96_replicated = pl.concat(long_segment96_replicated_half, long_segment96_replicated_half)
                        long_segment96_superrows = pl.add(pl.div(long_segment96_superrows, long_segment96_replicated), PRE_EPS)
                        long_segment96_column_row0 = long_segment96_superrows[0:1, 0:384]
                        long_segment96_column_row1 = long_segment96_superrows[1:2, 0:384]
                        long_segment96_column_row2 = long_segment96_superrows[2:3, 0:384]
                        long_segment96_column_row3 = long_segment96_superrows[3:4, 0:384]
                        long_segment96_column_denominator = pl.add(pl.add(pl.add(long_segment96_column_row0, long_segment96_column_row1), pl.add(long_segment96_column_row2, long_segment96_column_row3)), PRE_EPS)
                        long_segment96_superrows = pl.col_expand_div(long_segment96_superrows, long_segment96_column_denominator)
                        for long_segment96_super_iteration in pl.range(SINK - 1):
                            long_segment96_quarter0 = long_segment96_superrows[0:4, 0:96]
                            long_segment96_quarter1 = long_segment96_superrows[0:4, 96:192]
                            long_segment96_quarter2 = long_segment96_superrows[0:4, 192:288]
                            long_segment96_quarter3 = long_segment96_superrows[0:4, 288:384]
                            long_segment96_row_denominator = pl.add(pl.add(long_segment96_quarter0, long_segment96_quarter1), pl.add(long_segment96_quarter2, long_segment96_quarter3))
                            long_segment96_row_denominator = pl.add(long_segment96_row_denominator, PRE_EPS)
                            long_segment96_replicated_half = pl.concat(long_segment96_row_denominator, long_segment96_row_denominator)
                            long_segment96_replicated = pl.concat(long_segment96_replicated_half, long_segment96_replicated_half)
                            long_segment96_superrows = pl.div(long_segment96_superrows, long_segment96_replicated)
                            long_segment96_column_row0 = long_segment96_superrows[0:1, 0:384]
                            long_segment96_column_row1 = long_segment96_superrows[1:2, 0:384]
                            long_segment96_column_row2 = long_segment96_superrows[2:3, 0:384]
                            long_segment96_column_row3 = long_segment96_superrows[3:4, 0:384]
                            long_segment96_column_denominator = pl.add(pl.add(pl.add(long_segment96_column_row0, long_segment96_column_row1), pl.add(long_segment96_column_row2, long_segment96_column_row3)), PRE_EPS)
                            long_segment96_superrows = pl.col_expand_div(long_segment96_superrows, long_segment96_column_denominator)
                        long_segment96_superplanes = pl.reshape(long_segment96_superrows, [16, 96])
                        long_segment96_comb00_planes = long_segment96_superplanes[0:1, 0:96]
                        long_segment96_comb01_planes = long_segment96_superplanes[1:2, 0:96]
                        long_segment96_comb02_planes = long_segment96_superplanes[2:3, 0:96]
                        long_segment96_comb03_planes = long_segment96_superplanes[3:4, 0:96]
                        long_segment96_comb10_planes = long_segment96_superplanes[4:5, 0:96]
                        long_segment96_comb11_planes = long_segment96_superplanes[5:6, 0:96]
                        long_segment96_comb12_planes = long_segment96_superplanes[6:7, 0:96]
                        long_segment96_comb13_planes = long_segment96_superplanes[7:8, 0:96]
                        long_segment96_comb20_planes = long_segment96_superplanes[8:9, 0:96]
                        long_segment96_comb21_planes = long_segment96_superplanes[9:10, 0:96]
                        long_segment96_comb22_planes = long_segment96_superplanes[10:11, 0:96]
                        long_segment96_comb23_planes = long_segment96_superplanes[11:12, 0:96]
                        long_segment96_comb30_planes = long_segment96_superplanes[12:13, 0:96]
                        long_segment96_comb31_planes = long_segment96_superplanes[13:14, 0:96]
                        long_segment96_comb32_planes = long_segment96_superplanes[14:15, 0:96]
                        long_segment96_comb33_planes = long_segment96_superplanes[15:16, 0:96]
                        if owner_index < 2 and producer_lane == 0:
                            pl.system.sync_set(4, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV)
                        for long_segment96_update_block in pl.range(4):
                            if producer_lane == 1:
                                long_segment96_update_offset = (3 - long_segment96_update_block) * 24
                            else:
                                long_segment96_update_offset = long_segment96_update_block * 24
                            long_segment96_token_start = long_segment96_macro_start + long_segment96_update_offset
                            long_segment96_post0 = pl.reshape(long_segment96_post0_planes[0:1, long_segment96_update_offset:long_segment96_update_offset + 24], [24, 1])
                            long_segment96_post1 = pl.reshape(long_segment96_post1_planes[0:1, long_segment96_update_offset:long_segment96_update_offset + 24], [24, 1])
                            long_segment96_post2 = pl.reshape(long_segment96_post2_planes[0:1, long_segment96_update_offset:long_segment96_update_offset + 24], [24, 1])
                            long_segment96_post3 = pl.reshape(long_segment96_post3_planes[0:1, long_segment96_update_offset:long_segment96_update_offset + 24], [24, 1])
                            long_segment96_comb00 = pl.reshape(long_segment96_comb00_planes[0:1, long_segment96_update_offset:long_segment96_update_offset + 24], [24, 1])
                            long_segment96_comb01 = pl.reshape(long_segment96_comb01_planes[0:1, long_segment96_update_offset:long_segment96_update_offset + 24], [24, 1])
                            long_segment96_comb02 = pl.reshape(long_segment96_comb02_planes[0:1, long_segment96_update_offset:long_segment96_update_offset + 24], [24, 1])
                            long_segment96_comb03 = pl.reshape(long_segment96_comb03_planes[0:1, long_segment96_update_offset:long_segment96_update_offset + 24], [24, 1])
                            long_segment96_comb10 = pl.reshape(long_segment96_comb10_planes[0:1, long_segment96_update_offset:long_segment96_update_offset + 24], [24, 1])
                            long_segment96_comb11 = pl.reshape(long_segment96_comb11_planes[0:1, long_segment96_update_offset:long_segment96_update_offset + 24], [24, 1])
                            long_segment96_comb12 = pl.reshape(long_segment96_comb12_planes[0:1, long_segment96_update_offset:long_segment96_update_offset + 24], [24, 1])
                            long_segment96_comb13 = pl.reshape(long_segment96_comb13_planes[0:1, long_segment96_update_offset:long_segment96_update_offset + 24], [24, 1])
                            long_segment96_comb20 = pl.reshape(long_segment96_comb20_planes[0:1, long_segment96_update_offset:long_segment96_update_offset + 24], [24, 1])
                            long_segment96_comb21 = pl.reshape(long_segment96_comb21_planes[0:1, long_segment96_update_offset:long_segment96_update_offset + 24], [24, 1])
                            long_segment96_comb22 = pl.reshape(long_segment96_comb22_planes[0:1, long_segment96_update_offset:long_segment96_update_offset + 24], [24, 1])
                            long_segment96_comb23 = pl.reshape(long_segment96_comb23_planes[0:1, long_segment96_update_offset:long_segment96_update_offset + 24], [24, 1])
                            long_segment96_comb30 = pl.reshape(long_segment96_comb30_planes[0:1, long_segment96_update_offset:long_segment96_update_offset + 24], [24, 1])
                            long_segment96_comb31 = pl.reshape(long_segment96_comb31_planes[0:1, long_segment96_update_offset:long_segment96_update_offset + 24], [24, 1])
                            long_segment96_comb32 = pl.reshape(long_segment96_comb32_planes[0:1, long_segment96_update_offset:long_segment96_update_offset + 24], [24, 1])
                            long_segment96_comb33 = pl.reshape(long_segment96_comb33_planes[0:1, long_segment96_update_offset:long_segment96_update_offset + 24], [24, 1])
                            long_segment96_square_acc = pl.full([1, 24], dtype=pl.FP32, value=0.0)
                            for long_segment96_hidden_block in pl.range(D // 128):
                                long_segment96_hidden_start = long_segment96_hidden_block * 128
                                long_segment96_input_tile = pl.cast(group_x[long_segment96_token_start:long_segment96_token_start + 24, long_segment96_hidden_start:long_segment96_hidden_start + 128], pl.FP32)
                                long_segment96_original0 = pl.cast(group_residual[long_segment96_token_start:long_segment96_token_start + 24, 0 * D + long_segment96_hidden_start:0 * D + long_segment96_hidden_start + 128], pl.FP32)
                                long_segment96_original1 = pl.cast(group_residual[long_segment96_token_start:long_segment96_token_start + 24, 1 * D + long_segment96_hidden_start:1 * D + long_segment96_hidden_start + 128], pl.FP32)
                                long_segment96_original2 = pl.cast(group_residual[long_segment96_token_start:long_segment96_token_start + 24, 2 * D + long_segment96_hidden_start:2 * D + long_segment96_hidden_start + 128], pl.FP32)
                                long_segment96_original3 = pl.cast(group_residual[long_segment96_token_start:long_segment96_token_start + 24, 3 * D + long_segment96_hidden_start:3 * D + long_segment96_hidden_start + 128], pl.FP32)
                                long_segment96_updated0 = pl.row_expand_mul(long_segment96_input_tile, long_segment96_post0)
                                long_segment96_updated0 = pl.add(long_segment96_updated0, pl.row_expand_mul(long_segment96_original0, long_segment96_comb00))
                                long_segment96_updated0 = pl.add(long_segment96_updated0, pl.row_expand_mul(long_segment96_original1, long_segment96_comb01))
                                long_segment96_updated0 = pl.add(long_segment96_updated0, pl.row_expand_mul(long_segment96_original2, long_segment96_comb02))
                                long_segment96_updated0 = pl.add(long_segment96_updated0, pl.row_expand_mul(long_segment96_original3, long_segment96_comb03))
                                long_segment96_updated1 = pl.row_expand_mul(long_segment96_input_tile, long_segment96_post1)
                                long_segment96_updated1 = pl.add(long_segment96_updated1, pl.row_expand_mul(long_segment96_original0, long_segment96_comb10))
                                long_segment96_updated1 = pl.add(long_segment96_updated1, pl.row_expand_mul(long_segment96_original1, long_segment96_comb11))
                                long_segment96_updated1 = pl.add(long_segment96_updated1, pl.row_expand_mul(long_segment96_original2, long_segment96_comb12))
                                long_segment96_updated1 = pl.add(long_segment96_updated1, pl.row_expand_mul(long_segment96_original3, long_segment96_comb13))
                                long_segment96_updated2 = pl.row_expand_mul(long_segment96_input_tile, long_segment96_post2)
                                long_segment96_updated2 = pl.add(long_segment96_updated2, pl.row_expand_mul(long_segment96_original0, long_segment96_comb20))
                                long_segment96_updated2 = pl.add(long_segment96_updated2, pl.row_expand_mul(long_segment96_original1, long_segment96_comb21))
                                long_segment96_updated2 = pl.add(long_segment96_updated2, pl.row_expand_mul(long_segment96_original2, long_segment96_comb22))
                                long_segment96_updated2 = pl.add(long_segment96_updated2, pl.row_expand_mul(long_segment96_original3, long_segment96_comb23))
                                long_segment96_updated3 = pl.row_expand_mul(long_segment96_input_tile, long_segment96_post3)
                                long_segment96_updated3 = pl.add(long_segment96_updated3, pl.row_expand_mul(long_segment96_original0, long_segment96_comb30))
                                long_segment96_updated3 = pl.add(long_segment96_updated3, pl.row_expand_mul(long_segment96_original1, long_segment96_comb31))
                                long_segment96_updated3 = pl.add(long_segment96_updated3, pl.row_expand_mul(long_segment96_original2, long_segment96_comb32))
                                long_segment96_updated3 = pl.add(long_segment96_updated3, pl.row_expand_mul(long_segment96_original3, long_segment96_comb33))
                                long_segment96_square_sum = pl.row_sum(pl.add(pl.add(pl.mul(long_segment96_updated0, long_segment96_updated0), pl.mul(long_segment96_updated1, long_segment96_updated1)), pl.add(pl.mul(long_segment96_updated2, long_segment96_updated2), pl.mul(long_segment96_updated3, long_segment96_updated3))))
                                long_segment96_square_acc = pl.add(long_segment96_square_acc, pl.reshape(long_segment96_square_sum, [1, 24]))
                                group_updated[long_segment96_token_start - owner_start + owner_index * 192:long_segment96_token_start + 24 - owner_start + owner_index * 192, 0 * D + long_segment96_hidden_start:0 * D + long_segment96_hidden_start + 128] = long_segment96_updated0
                                group_updated[long_segment96_token_start - owner_start + owner_index * 192:long_segment96_token_start + 24 - owner_start + owner_index * 192, 1 * D + long_segment96_hidden_start:1 * D + long_segment96_hidden_start + 128] = long_segment96_updated1
                                group_updated[long_segment96_token_start - owner_start + owner_index * 192:long_segment96_token_start + 24 - owner_start + owner_index * 192, 2 * D + long_segment96_hidden_start:2 * D + long_segment96_hidden_start + 128] = long_segment96_updated2
                                group_updated[long_segment96_token_start - owner_start + owner_index * 192:long_segment96_token_start + 24 - owner_start + owner_index * 192, 3 * D + long_segment96_hidden_start:3 * D + long_segment96_hidden_start + 128] = long_segment96_updated3
                            long_segment96_square_sum = pl.reshape(long_segment96_square_acc, [24, 1])
                            long_segment96_mean_square = pl.add(pl.mul(long_segment96_square_sum, 1.0 / FD), RMS_EPS)
                            long_segment96_inverse_rms = pl.rsqrt(long_segment96_mean_square)
                            long_segment96_inverse_rms = pl.mul(long_segment96_inverse_rms, pl.add(pl.mul(pl.mul(long_segment96_mean_square, pl.mul(long_segment96_inverse_rms, long_segment96_inverse_rms)), -0.5), 1.5))
                            group_rms[0:1, long_segment96_token_start - owner_start + owner_index * 192:long_segment96_token_start + 24 - owner_start + owner_index * 192] = pl.reshape(long_segment96_inverse_rms, [1, 24])
                            if producer_lane == 1 and long_segment96_update_block == 2:
                                pl.system.barrier(pipe=pl.PipeType.ALL)
                                pl.system.cacheinvalid()
                                pl.system.fence()
                                pl.system.sync_set(4, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV)
                    else:
                        short_segment32_macro_start = owner_start + 96
                        short_segment32_scale_pre = pl.read(mhc_scale, [0])
                        short_segment32_scale_post = pl.read(mhc_scale, [1])
                        short_segment32_scale_comb = pl.read(mhc_scale, [2])
                        short_segment32_mix_planes = pl.transpose(group_input_mixes[short_segment32_macro_start:short_segment32_macro_start + 32, 0:M3], 0, 1)
                        short_segment32_post_raw0 = pl.add(pl.mul(pl.reshape(short_segment32_mix_planes[4:5, 0:32], [1, 32]), short_segment32_scale_post), pl.read(mhc_base, [0, 4]))
                        short_segment32_post0_planes = pl.mul(pl.recip(pl.add(pl.exp(pl.neg(short_segment32_post_raw0)), 1.0)), POST_MULT)
                        short_segment32_post_raw1 = pl.add(pl.mul(pl.reshape(short_segment32_mix_planes[5:6, 0:32], [1, 32]), short_segment32_scale_post), pl.read(mhc_base, [0, 5]))
                        short_segment32_post1_planes = pl.mul(pl.recip(pl.add(pl.exp(pl.neg(short_segment32_post_raw1)), 1.0)), POST_MULT)
                        short_segment32_post_raw2 = pl.add(pl.mul(pl.reshape(short_segment32_mix_planes[6:7, 0:32], [1, 32]), short_segment32_scale_post), pl.read(mhc_base, [0, 6]))
                        short_segment32_post2_planes = pl.mul(pl.recip(pl.add(pl.exp(pl.neg(short_segment32_post_raw2)), 1.0)), POST_MULT)
                        short_segment32_post_raw3 = pl.add(pl.mul(pl.reshape(short_segment32_mix_planes[7:8, 0:32], [1, 32]), short_segment32_scale_post), pl.read(mhc_base, [0, 7]))
                        short_segment32_post3_planes = pl.mul(pl.recip(pl.add(pl.exp(pl.neg(short_segment32_post_raw3)), 1.0)), POST_MULT)
                        short_segment32_comb00_planes = pl.add(pl.mul(pl.reshape(short_segment32_mix_planes[8:9, 0:32], [1, 32]), short_segment32_scale_comb), pl.read(mhc_base, [0, 8]))
                        short_segment32_comb01_planes = pl.add(pl.mul(pl.reshape(short_segment32_mix_planes[9:10, 0:32], [1, 32]), short_segment32_scale_comb), pl.read(mhc_base, [0, 9]))
                        short_segment32_comb02_planes = pl.add(pl.mul(pl.reshape(short_segment32_mix_planes[10:11, 0:32], [1, 32]), short_segment32_scale_comb), pl.read(mhc_base, [0, 10]))
                        short_segment32_comb03_planes = pl.add(pl.mul(pl.reshape(short_segment32_mix_planes[11:12, 0:32], [1, 32]), short_segment32_scale_comb), pl.read(mhc_base, [0, 11]))
                        short_segment32_comb10_planes = pl.add(pl.mul(pl.reshape(short_segment32_mix_planes[12:13, 0:32], [1, 32]), short_segment32_scale_comb), pl.read(mhc_base, [0, 12]))
                        short_segment32_comb11_planes = pl.add(pl.mul(pl.reshape(short_segment32_mix_planes[13:14, 0:32], [1, 32]), short_segment32_scale_comb), pl.read(mhc_base, [0, 13]))
                        short_segment32_comb12_planes = pl.add(pl.mul(pl.reshape(short_segment32_mix_planes[14:15, 0:32], [1, 32]), short_segment32_scale_comb), pl.read(mhc_base, [0, 14]))
                        short_segment32_comb13_planes = pl.add(pl.mul(pl.reshape(short_segment32_mix_planes[15:16, 0:32], [1, 32]), short_segment32_scale_comb), pl.read(mhc_base, [0, 15]))
                        short_segment32_comb20_planes = pl.add(pl.mul(pl.reshape(short_segment32_mix_planes[16:17, 0:32], [1, 32]), short_segment32_scale_comb), pl.read(mhc_base, [0, 16]))
                        short_segment32_comb21_planes = pl.add(pl.mul(pl.reshape(short_segment32_mix_planes[17:18, 0:32], [1, 32]), short_segment32_scale_comb), pl.read(mhc_base, [0, 17]))
                        short_segment32_comb22_planes = pl.add(pl.mul(pl.reshape(short_segment32_mix_planes[18:19, 0:32], [1, 32]), short_segment32_scale_comb), pl.read(mhc_base, [0, 18]))
                        short_segment32_comb23_planes = pl.add(pl.mul(pl.reshape(short_segment32_mix_planes[19:20, 0:32], [1, 32]), short_segment32_scale_comb), pl.read(mhc_base, [0, 19]))
                        short_segment32_comb30_planes = pl.add(pl.mul(pl.reshape(short_segment32_mix_planes[20:21, 0:32], [1, 32]), short_segment32_scale_comb), pl.read(mhc_base, [0, 20]))
                        short_segment32_comb31_planes = pl.add(pl.mul(pl.reshape(short_segment32_mix_planes[21:22, 0:32], [1, 32]), short_segment32_scale_comb), pl.read(mhc_base, [0, 21]))
                        short_segment32_comb32_planes = pl.add(pl.mul(pl.reshape(short_segment32_mix_planes[22:23, 0:32], [1, 32]), short_segment32_scale_comb), pl.read(mhc_base, [0, 22]))
                        short_segment32_comb33_planes = pl.add(pl.mul(pl.reshape(short_segment32_mix_planes[23:24, 0:32], [1, 32]), short_segment32_scale_comb), pl.read(mhc_base, [0, 23]))
                        short_segment32_packed_row0 = pl.concat(pl.concat(short_segment32_comb00_planes, short_segment32_comb01_planes), pl.concat(short_segment32_comb02_planes, short_segment32_comb03_planes))
                        short_segment32_packed_row1 = pl.concat(pl.concat(short_segment32_comb10_planes, short_segment32_comb11_planes), pl.concat(short_segment32_comb12_planes, short_segment32_comb13_planes))
                        short_segment32_packed_row2 = pl.concat(pl.concat(short_segment32_comb20_planes, short_segment32_comb21_planes), pl.concat(short_segment32_comb22_planes, short_segment32_comb23_planes))
                        short_segment32_packed_row3 = pl.concat(pl.concat(short_segment32_comb30_planes, short_segment32_comb31_planes), pl.concat(short_segment32_comb32_planes, short_segment32_comb33_planes))
                        short_segment32_superrows = pl.reshape(pl.concat(pl.concat(short_segment32_packed_row0, short_segment32_packed_row1), pl.concat(short_segment32_packed_row2, short_segment32_packed_row3)), [4, 128])
                        short_segment32_quarter0 = short_segment32_superrows[0:4, 0:32]
                        short_segment32_quarter1 = short_segment32_superrows[0:4, 32:64]
                        short_segment32_quarter2 = short_segment32_superrows[0:4, 64:96]
                        short_segment32_quarter3 = short_segment32_superrows[0:4, 96:128]
                        short_segment32_row_maximum = pl.maximum(pl.maximum(short_segment32_quarter0, short_segment32_quarter1), pl.maximum(short_segment32_quarter2, short_segment32_quarter3))
                        short_segment32_replicated_half = pl.concat(short_segment32_row_maximum, short_segment32_row_maximum)
                        short_segment32_replicated = pl.concat(short_segment32_replicated_half, short_segment32_replicated_half)
                        short_segment32_superrows = pl.exp(pl.sub(short_segment32_superrows, short_segment32_replicated))
                        short_segment32_quarter0 = short_segment32_superrows[0:4, 0:32]
                        short_segment32_quarter1 = short_segment32_superrows[0:4, 32:64]
                        short_segment32_quarter2 = short_segment32_superrows[0:4, 64:96]
                        short_segment32_quarter3 = short_segment32_superrows[0:4, 96:128]
                        short_segment32_row_denominator = pl.add(pl.add(short_segment32_quarter0, short_segment32_quarter1), pl.add(short_segment32_quarter2, short_segment32_quarter3))
                        short_segment32_replicated_half = pl.concat(short_segment32_row_denominator, short_segment32_row_denominator)
                        short_segment32_replicated = pl.concat(short_segment32_replicated_half, short_segment32_replicated_half)
                        short_segment32_superrows = pl.add(pl.div(short_segment32_superrows, short_segment32_replicated), PRE_EPS)
                        short_segment32_column_row0 = short_segment32_superrows[0:1, 0:128]
                        short_segment32_column_row1 = short_segment32_superrows[1:2, 0:128]
                        short_segment32_column_row2 = short_segment32_superrows[2:3, 0:128]
                        short_segment32_column_row3 = short_segment32_superrows[3:4, 0:128]
                        short_segment32_column_denominator = pl.add(pl.add(pl.add(short_segment32_column_row0, short_segment32_column_row1), pl.add(short_segment32_column_row2, short_segment32_column_row3)), PRE_EPS)
                        short_segment32_superrows = pl.col_expand_div(short_segment32_superrows, short_segment32_column_denominator)
                        for short_segment32_super_iteration in pl.range(SINK - 1):
                            short_segment32_quarter0 = short_segment32_superrows[0:4, 0:32]
                            short_segment32_quarter1 = short_segment32_superrows[0:4, 32:64]
                            short_segment32_quarter2 = short_segment32_superrows[0:4, 64:96]
                            short_segment32_quarter3 = short_segment32_superrows[0:4, 96:128]
                            short_segment32_row_denominator = pl.add(pl.add(short_segment32_quarter0, short_segment32_quarter1), pl.add(short_segment32_quarter2, short_segment32_quarter3))
                            short_segment32_row_denominator = pl.add(short_segment32_row_denominator, PRE_EPS)
                            short_segment32_replicated_half = pl.concat(short_segment32_row_denominator, short_segment32_row_denominator)
                            short_segment32_replicated = pl.concat(short_segment32_replicated_half, short_segment32_replicated_half)
                            short_segment32_superrows = pl.div(short_segment32_superrows, short_segment32_replicated)
                            short_segment32_column_row0 = short_segment32_superrows[0:1, 0:128]
                            short_segment32_column_row1 = short_segment32_superrows[1:2, 0:128]
                            short_segment32_column_row2 = short_segment32_superrows[2:3, 0:128]
                            short_segment32_column_row3 = short_segment32_superrows[3:4, 0:128]
                            short_segment32_column_denominator = pl.add(pl.add(pl.add(short_segment32_column_row0, short_segment32_column_row1), pl.add(short_segment32_column_row2, short_segment32_column_row3)), PRE_EPS)
                            short_segment32_superrows = pl.col_expand_div(short_segment32_superrows, short_segment32_column_denominator)
                        short_segment32_superplanes = pl.reshape(short_segment32_superrows, [16, 32])
                        short_segment32_comb00_planes = short_segment32_superplanes[0:1, 0:32]
                        short_segment32_comb01_planes = short_segment32_superplanes[1:2, 0:32]
                        short_segment32_comb02_planes = short_segment32_superplanes[2:3, 0:32]
                        short_segment32_comb03_planes = short_segment32_superplanes[3:4, 0:32]
                        short_segment32_comb10_planes = short_segment32_superplanes[4:5, 0:32]
                        short_segment32_comb11_planes = short_segment32_superplanes[5:6, 0:32]
                        short_segment32_comb12_planes = short_segment32_superplanes[6:7, 0:32]
                        short_segment32_comb13_planes = short_segment32_superplanes[7:8, 0:32]
                        short_segment32_comb20_planes = short_segment32_superplanes[8:9, 0:32]
                        short_segment32_comb21_planes = short_segment32_superplanes[9:10, 0:32]
                        short_segment32_comb22_planes = short_segment32_superplanes[10:11, 0:32]
                        short_segment32_comb23_planes = short_segment32_superplanes[11:12, 0:32]
                        short_segment32_comb30_planes = short_segment32_superplanes[12:13, 0:32]
                        short_segment32_comb31_planes = short_segment32_superplanes[13:14, 0:32]
                        short_segment32_comb32_planes = short_segment32_superplanes[14:15, 0:32]
                        short_segment32_comb33_planes = short_segment32_superplanes[15:16, 0:32]
                        for short_segment32_update_block in pl.range(1):
                            short_segment32_update_offset = short_segment32_update_block * 24
                            short_segment32_token_start = short_segment32_macro_start + short_segment32_update_offset
                            short_segment32_post0 = pl.reshape(short_segment32_post0_planes[0:1, short_segment32_update_offset:short_segment32_update_offset + 24], [24, 1])
                            short_segment32_post1 = pl.reshape(short_segment32_post1_planes[0:1, short_segment32_update_offset:short_segment32_update_offset + 24], [24, 1])
                            short_segment32_post2 = pl.reshape(short_segment32_post2_planes[0:1, short_segment32_update_offset:short_segment32_update_offset + 24], [24, 1])
                            short_segment32_post3 = pl.reshape(short_segment32_post3_planes[0:1, short_segment32_update_offset:short_segment32_update_offset + 24], [24, 1])
                            short_segment32_comb00 = pl.reshape(short_segment32_comb00_planes[0:1, short_segment32_update_offset:short_segment32_update_offset + 24], [24, 1])
                            short_segment32_comb01 = pl.reshape(short_segment32_comb01_planes[0:1, short_segment32_update_offset:short_segment32_update_offset + 24], [24, 1])
                            short_segment32_comb02 = pl.reshape(short_segment32_comb02_planes[0:1, short_segment32_update_offset:short_segment32_update_offset + 24], [24, 1])
                            short_segment32_comb03 = pl.reshape(short_segment32_comb03_planes[0:1, short_segment32_update_offset:short_segment32_update_offset + 24], [24, 1])
                            short_segment32_comb10 = pl.reshape(short_segment32_comb10_planes[0:1, short_segment32_update_offset:short_segment32_update_offset + 24], [24, 1])
                            short_segment32_comb11 = pl.reshape(short_segment32_comb11_planes[0:1, short_segment32_update_offset:short_segment32_update_offset + 24], [24, 1])
                            short_segment32_comb12 = pl.reshape(short_segment32_comb12_planes[0:1, short_segment32_update_offset:short_segment32_update_offset + 24], [24, 1])
                            short_segment32_comb13 = pl.reshape(short_segment32_comb13_planes[0:1, short_segment32_update_offset:short_segment32_update_offset + 24], [24, 1])
                            short_segment32_comb20 = pl.reshape(short_segment32_comb20_planes[0:1, short_segment32_update_offset:short_segment32_update_offset + 24], [24, 1])
                            short_segment32_comb21 = pl.reshape(short_segment32_comb21_planes[0:1, short_segment32_update_offset:short_segment32_update_offset + 24], [24, 1])
                            short_segment32_comb22 = pl.reshape(short_segment32_comb22_planes[0:1, short_segment32_update_offset:short_segment32_update_offset + 24], [24, 1])
                            short_segment32_comb23 = pl.reshape(short_segment32_comb23_planes[0:1, short_segment32_update_offset:short_segment32_update_offset + 24], [24, 1])
                            short_segment32_comb30 = pl.reshape(short_segment32_comb30_planes[0:1, short_segment32_update_offset:short_segment32_update_offset + 24], [24, 1])
                            short_segment32_comb31 = pl.reshape(short_segment32_comb31_planes[0:1, short_segment32_update_offset:short_segment32_update_offset + 24], [24, 1])
                            short_segment32_comb32 = pl.reshape(short_segment32_comb32_planes[0:1, short_segment32_update_offset:short_segment32_update_offset + 24], [24, 1])
                            short_segment32_comb33 = pl.reshape(short_segment32_comb33_planes[0:1, short_segment32_update_offset:short_segment32_update_offset + 24], [24, 1])
                            short_segment32_square_acc = pl.full([1, 24], dtype=pl.FP32, value=0.0)
                            for short_segment32_hidden_block in pl.range(D // 128):
                                short_segment32_hidden_start = short_segment32_hidden_block * 128
                                short_segment32_input_tile = pl.cast(group_x[short_segment32_token_start:short_segment32_token_start + 24, short_segment32_hidden_start:short_segment32_hidden_start + 128], pl.FP32)
                                short_segment32_original0 = pl.cast(group_residual[short_segment32_token_start:short_segment32_token_start + 24, 0 * D + short_segment32_hidden_start:0 * D + short_segment32_hidden_start + 128], pl.FP32)
                                short_segment32_original1 = pl.cast(group_residual[short_segment32_token_start:short_segment32_token_start + 24, 1 * D + short_segment32_hidden_start:1 * D + short_segment32_hidden_start + 128], pl.FP32)
                                short_segment32_original2 = pl.cast(group_residual[short_segment32_token_start:short_segment32_token_start + 24, 2 * D + short_segment32_hidden_start:2 * D + short_segment32_hidden_start + 128], pl.FP32)
                                short_segment32_original3 = pl.cast(group_residual[short_segment32_token_start:short_segment32_token_start + 24, 3 * D + short_segment32_hidden_start:3 * D + short_segment32_hidden_start + 128], pl.FP32)
                                short_segment32_updated0 = pl.row_expand_mul(short_segment32_input_tile, short_segment32_post0)
                                short_segment32_updated0 = pl.add(short_segment32_updated0, pl.row_expand_mul(short_segment32_original0, short_segment32_comb00))
                                short_segment32_updated0 = pl.add(short_segment32_updated0, pl.row_expand_mul(short_segment32_original1, short_segment32_comb01))
                                short_segment32_updated0 = pl.add(short_segment32_updated0, pl.row_expand_mul(short_segment32_original2, short_segment32_comb02))
                                short_segment32_updated0 = pl.add(short_segment32_updated0, pl.row_expand_mul(short_segment32_original3, short_segment32_comb03))
                                short_segment32_updated1 = pl.row_expand_mul(short_segment32_input_tile, short_segment32_post1)
                                short_segment32_updated1 = pl.add(short_segment32_updated1, pl.row_expand_mul(short_segment32_original0, short_segment32_comb10))
                                short_segment32_updated1 = pl.add(short_segment32_updated1, pl.row_expand_mul(short_segment32_original1, short_segment32_comb11))
                                short_segment32_updated1 = pl.add(short_segment32_updated1, pl.row_expand_mul(short_segment32_original2, short_segment32_comb12))
                                short_segment32_updated1 = pl.add(short_segment32_updated1, pl.row_expand_mul(short_segment32_original3, short_segment32_comb13))
                                short_segment32_updated2 = pl.row_expand_mul(short_segment32_input_tile, short_segment32_post2)
                                short_segment32_updated2 = pl.add(short_segment32_updated2, pl.row_expand_mul(short_segment32_original0, short_segment32_comb20))
                                short_segment32_updated2 = pl.add(short_segment32_updated2, pl.row_expand_mul(short_segment32_original1, short_segment32_comb21))
                                short_segment32_updated2 = pl.add(short_segment32_updated2, pl.row_expand_mul(short_segment32_original2, short_segment32_comb22))
                                short_segment32_updated2 = pl.add(short_segment32_updated2, pl.row_expand_mul(short_segment32_original3, short_segment32_comb23))
                                short_segment32_updated3 = pl.row_expand_mul(short_segment32_input_tile, short_segment32_post3)
                                short_segment32_updated3 = pl.add(short_segment32_updated3, pl.row_expand_mul(short_segment32_original0, short_segment32_comb30))
                                short_segment32_updated3 = pl.add(short_segment32_updated3, pl.row_expand_mul(short_segment32_original1, short_segment32_comb31))
                                short_segment32_updated3 = pl.add(short_segment32_updated3, pl.row_expand_mul(short_segment32_original2, short_segment32_comb32))
                                short_segment32_updated3 = pl.add(short_segment32_updated3, pl.row_expand_mul(short_segment32_original3, short_segment32_comb33))
                                short_segment32_square_sum = pl.row_sum(pl.add(pl.add(pl.mul(short_segment32_updated0, short_segment32_updated0), pl.mul(short_segment32_updated1, short_segment32_updated1)), pl.add(pl.mul(short_segment32_updated2, short_segment32_updated2), pl.mul(short_segment32_updated3, short_segment32_updated3))))
                                short_segment32_square_acc = pl.add(short_segment32_square_acc, pl.reshape(short_segment32_square_sum, [1, 24]))
                                group_updated[short_segment32_token_start - owner_start + owner_index * 192:short_segment32_token_start + 24 - owner_start + owner_index * 192, 0 * D + short_segment32_hidden_start:0 * D + short_segment32_hidden_start + 128] = short_segment32_updated0
                                group_updated[short_segment32_token_start - owner_start + owner_index * 192:short_segment32_token_start + 24 - owner_start + owner_index * 192, 1 * D + short_segment32_hidden_start:1 * D + short_segment32_hidden_start + 128] = short_segment32_updated1
                                group_updated[short_segment32_token_start - owner_start + owner_index * 192:short_segment32_token_start + 24 - owner_start + owner_index * 192, 2 * D + short_segment32_hidden_start:2 * D + short_segment32_hidden_start + 128] = short_segment32_updated2
                                group_updated[short_segment32_token_start - owner_start + owner_index * 192:short_segment32_token_start + 24 - owner_start + owner_index * 192, 3 * D + short_segment32_hidden_start:3 * D + short_segment32_hidden_start + 128] = short_segment32_updated3
                            short_segment32_square_sum = pl.reshape(short_segment32_square_acc, [24, 1])
                            short_segment32_mean_square = pl.add(pl.mul(short_segment32_square_sum, 1.0 / FD), RMS_EPS)
                            short_segment32_inverse_rms = pl.rsqrt(short_segment32_mean_square)
                            short_segment32_inverse_rms = pl.mul(short_segment32_inverse_rms, pl.add(pl.mul(pl.mul(short_segment32_mean_square, pl.mul(short_segment32_inverse_rms, short_segment32_inverse_rms)), -0.5), 1.5))
                            group_rms[0:1, short_segment32_token_start - owner_start + owner_index * 192:short_segment32_token_start + 24 - owner_start + owner_index * 192] = pl.reshape(short_segment32_inverse_rms, [1, 24])
                        short_segment32_tail_update_offset = 24
                        short_segment32_tail_token_start = short_segment32_macro_start + short_segment32_tail_update_offset
                        short_segment32_tail_post0 = pl.reshape(short_segment32_post0_planes[0:1, short_segment32_tail_update_offset:short_segment32_tail_update_offset + 8], [8, 1])
                        short_segment32_tail_post1 = pl.reshape(short_segment32_post1_planes[0:1, short_segment32_tail_update_offset:short_segment32_tail_update_offset + 8], [8, 1])
                        short_segment32_tail_post2 = pl.reshape(short_segment32_post2_planes[0:1, short_segment32_tail_update_offset:short_segment32_tail_update_offset + 8], [8, 1])
                        short_segment32_tail_post3 = pl.reshape(short_segment32_post3_planes[0:1, short_segment32_tail_update_offset:short_segment32_tail_update_offset + 8], [8, 1])
                        short_segment32_tail_comb00 = pl.reshape(short_segment32_comb00_planes[0:1, short_segment32_tail_update_offset:short_segment32_tail_update_offset + 8], [8, 1])
                        short_segment32_tail_comb01 = pl.reshape(short_segment32_comb01_planes[0:1, short_segment32_tail_update_offset:short_segment32_tail_update_offset + 8], [8, 1])
                        short_segment32_tail_comb02 = pl.reshape(short_segment32_comb02_planes[0:1, short_segment32_tail_update_offset:short_segment32_tail_update_offset + 8], [8, 1])
                        short_segment32_tail_comb03 = pl.reshape(short_segment32_comb03_planes[0:1, short_segment32_tail_update_offset:short_segment32_tail_update_offset + 8], [8, 1])
                        short_segment32_tail_comb10 = pl.reshape(short_segment32_comb10_planes[0:1, short_segment32_tail_update_offset:short_segment32_tail_update_offset + 8], [8, 1])
                        short_segment32_tail_comb11 = pl.reshape(short_segment32_comb11_planes[0:1, short_segment32_tail_update_offset:short_segment32_tail_update_offset + 8], [8, 1])
                        short_segment32_tail_comb12 = pl.reshape(short_segment32_comb12_planes[0:1, short_segment32_tail_update_offset:short_segment32_tail_update_offset + 8], [8, 1])
                        short_segment32_tail_comb13 = pl.reshape(short_segment32_comb13_planes[0:1, short_segment32_tail_update_offset:short_segment32_tail_update_offset + 8], [8, 1])
                        short_segment32_tail_comb20 = pl.reshape(short_segment32_comb20_planes[0:1, short_segment32_tail_update_offset:short_segment32_tail_update_offset + 8], [8, 1])
                        short_segment32_tail_comb21 = pl.reshape(short_segment32_comb21_planes[0:1, short_segment32_tail_update_offset:short_segment32_tail_update_offset + 8], [8, 1])
                        short_segment32_tail_comb22 = pl.reshape(short_segment32_comb22_planes[0:1, short_segment32_tail_update_offset:short_segment32_tail_update_offset + 8], [8, 1])
                        short_segment32_tail_comb23 = pl.reshape(short_segment32_comb23_planes[0:1, short_segment32_tail_update_offset:short_segment32_tail_update_offset + 8], [8, 1])
                        short_segment32_tail_comb30 = pl.reshape(short_segment32_comb30_planes[0:1, short_segment32_tail_update_offset:short_segment32_tail_update_offset + 8], [8, 1])
                        short_segment32_tail_comb31 = pl.reshape(short_segment32_comb31_planes[0:1, short_segment32_tail_update_offset:short_segment32_tail_update_offset + 8], [8, 1])
                        short_segment32_tail_comb32 = pl.reshape(short_segment32_comb32_planes[0:1, short_segment32_tail_update_offset:short_segment32_tail_update_offset + 8], [8, 1])
                        short_segment32_tail_comb33 = pl.reshape(short_segment32_comb33_planes[0:1, short_segment32_tail_update_offset:short_segment32_tail_update_offset + 8], [8, 1])
                        short_segment32_tail_square_acc = pl.full([1, 8], dtype=pl.FP32, value=0.0)
                        for short_segment32_tail_hidden_block in pl.range(D // 128):
                            short_segment32_tail_hidden_start = short_segment32_tail_hidden_block * 128
                            short_segment32_tail_input_tile = pl.cast(group_x[short_segment32_tail_token_start:short_segment32_tail_token_start + 8, short_segment32_tail_hidden_start:short_segment32_tail_hidden_start + 128], pl.FP32)
                            short_segment32_tail_original0 = pl.cast(group_residual[short_segment32_tail_token_start:short_segment32_tail_token_start + 8, 0 * D + short_segment32_tail_hidden_start:0 * D + short_segment32_tail_hidden_start + 128], pl.FP32)
                            short_segment32_tail_original1 = pl.cast(group_residual[short_segment32_tail_token_start:short_segment32_tail_token_start + 8, 1 * D + short_segment32_tail_hidden_start:1 * D + short_segment32_tail_hidden_start + 128], pl.FP32)
                            short_segment32_tail_original2 = pl.cast(group_residual[short_segment32_tail_token_start:short_segment32_tail_token_start + 8, 2 * D + short_segment32_tail_hidden_start:2 * D + short_segment32_tail_hidden_start + 128], pl.FP32)
                            short_segment32_tail_original3 = pl.cast(group_residual[short_segment32_tail_token_start:short_segment32_tail_token_start + 8, 3 * D + short_segment32_tail_hidden_start:3 * D + short_segment32_tail_hidden_start + 128], pl.FP32)
                            short_segment32_tail_updated0 = pl.row_expand_mul(short_segment32_tail_input_tile, short_segment32_tail_post0)
                            short_segment32_tail_updated0 = pl.add(short_segment32_tail_updated0, pl.row_expand_mul(short_segment32_tail_original0, short_segment32_tail_comb00))
                            short_segment32_tail_updated0 = pl.add(short_segment32_tail_updated0, pl.row_expand_mul(short_segment32_tail_original1, short_segment32_tail_comb01))
                            short_segment32_tail_updated0 = pl.add(short_segment32_tail_updated0, pl.row_expand_mul(short_segment32_tail_original2, short_segment32_tail_comb02))
                            short_segment32_tail_updated0 = pl.add(short_segment32_tail_updated0, pl.row_expand_mul(short_segment32_tail_original3, short_segment32_tail_comb03))
                            short_segment32_tail_updated1 = pl.row_expand_mul(short_segment32_tail_input_tile, short_segment32_tail_post1)
                            short_segment32_tail_updated1 = pl.add(short_segment32_tail_updated1, pl.row_expand_mul(short_segment32_tail_original0, short_segment32_tail_comb10))
                            short_segment32_tail_updated1 = pl.add(short_segment32_tail_updated1, pl.row_expand_mul(short_segment32_tail_original1, short_segment32_tail_comb11))
                            short_segment32_tail_updated1 = pl.add(short_segment32_tail_updated1, pl.row_expand_mul(short_segment32_tail_original2, short_segment32_tail_comb12))
                            short_segment32_tail_updated1 = pl.add(short_segment32_tail_updated1, pl.row_expand_mul(short_segment32_tail_original3, short_segment32_tail_comb13))
                            short_segment32_tail_updated2 = pl.row_expand_mul(short_segment32_tail_input_tile, short_segment32_tail_post2)
                            short_segment32_tail_updated2 = pl.add(short_segment32_tail_updated2, pl.row_expand_mul(short_segment32_tail_original0, short_segment32_tail_comb20))
                            short_segment32_tail_updated2 = pl.add(short_segment32_tail_updated2, pl.row_expand_mul(short_segment32_tail_original1, short_segment32_tail_comb21))
                            short_segment32_tail_updated2 = pl.add(short_segment32_tail_updated2, pl.row_expand_mul(short_segment32_tail_original2, short_segment32_tail_comb22))
                            short_segment32_tail_updated2 = pl.add(short_segment32_tail_updated2, pl.row_expand_mul(short_segment32_tail_original3, short_segment32_tail_comb23))
                            short_segment32_tail_updated3 = pl.row_expand_mul(short_segment32_tail_input_tile, short_segment32_tail_post3)
                            short_segment32_tail_updated3 = pl.add(short_segment32_tail_updated3, pl.row_expand_mul(short_segment32_tail_original0, short_segment32_tail_comb30))
                            short_segment32_tail_updated3 = pl.add(short_segment32_tail_updated3, pl.row_expand_mul(short_segment32_tail_original1, short_segment32_tail_comb31))
                            short_segment32_tail_updated3 = pl.add(short_segment32_tail_updated3, pl.row_expand_mul(short_segment32_tail_original2, short_segment32_tail_comb32))
                            short_segment32_tail_updated3 = pl.add(short_segment32_tail_updated3, pl.row_expand_mul(short_segment32_tail_original3, short_segment32_tail_comb33))
                            short_segment32_tail_square_sum = pl.row_sum(pl.add(pl.add(pl.mul(short_segment32_tail_updated0, short_segment32_tail_updated0), pl.mul(short_segment32_tail_updated1, short_segment32_tail_updated1)), pl.add(pl.mul(short_segment32_tail_updated2, short_segment32_tail_updated2), pl.mul(short_segment32_tail_updated3, short_segment32_tail_updated3))))
                            short_segment32_tail_square_acc = pl.add(short_segment32_tail_square_acc, pl.reshape(short_segment32_tail_square_sum, [1, 8]))
                            group_updated[short_segment32_tail_token_start - owner_start + owner_index * 192:short_segment32_tail_token_start + 8 - owner_start + owner_index * 192, 0 * D + short_segment32_tail_hidden_start:0 * D + short_segment32_tail_hidden_start + 128] = short_segment32_tail_updated0
                            group_updated[short_segment32_tail_token_start - owner_start + owner_index * 192:short_segment32_tail_token_start + 8 - owner_start + owner_index * 192, 1 * D + short_segment32_tail_hidden_start:1 * D + short_segment32_tail_hidden_start + 128] = short_segment32_tail_updated1
                            group_updated[short_segment32_tail_token_start - owner_start + owner_index * 192:short_segment32_tail_token_start + 8 - owner_start + owner_index * 192, 2 * D + short_segment32_tail_hidden_start:2 * D + short_segment32_tail_hidden_start + 128] = short_segment32_tail_updated2
                            group_updated[short_segment32_tail_token_start - owner_start + owner_index * 192:short_segment32_tail_token_start + 8 - owner_start + owner_index * 192, 3 * D + short_segment32_tail_hidden_start:3 * D + short_segment32_tail_hidden_start + 128] = short_segment32_tail_updated3
                        short_segment32_tail_square_sum = pl.reshape(short_segment32_tail_square_acc, [8, 1])
                        short_segment32_tail_mean_square = pl.add(pl.mul(short_segment32_tail_square_sum, 1.0 / FD), RMS_EPS)
                        short_segment32_tail_inverse_rms = pl.rsqrt(short_segment32_tail_mean_square)
                        short_segment32_tail_inverse_rms = pl.mul(short_segment32_tail_inverse_rms, pl.add(pl.mul(pl.mul(short_segment32_tail_mean_square, pl.mul(short_segment32_tail_inverse_rms, short_segment32_tail_inverse_rms)), -0.5), 1.5))
                        group_rms[0:1, short_segment32_tail_token_start - owner_start + owner_index * 192:short_segment32_tail_token_start + 8 - owner_start + owner_index * 192] = pl.reshape(short_segment32_tail_inverse_rms, [1, 8])
                    pl.system.barrier(pipe=pl.PipeType.ALL)
                    pl.system.cacheinvalid()
                    pl.system.fence()
                    pl.system.sync_set(5, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV)
                if owner_index < 2:
                    pl.system.sync_wait(4, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                    pl.system.cacheinvalid()
                    pl.system.fence()
                    partial_accumulator = pl.create_tensor([64, 32], dtype=pl.FP32)
                    for partial_projection_block in pl.pipeline(0, FD // 128, stage=2):
                        partial_projection_start = partial_projection_block * 128
                        partial_updated_chunk = pl.slice(group_updated, [64, 128], [owner_index * 192 + 128, partial_projection_start], valid_shape=[64, 128])
                        partial_weight_chunk = pl.slice(fn, [32, 128], [0, partial_projection_start], valid_shape=[24, 128], clamp=True)
                        partial_accumulator = pl.matmul_acc(partial_accumulator, partial_updated_chunk, partial_weight_chunk, b_trans=True, init_cond=(partial_projection_block == 0))
                    for partial_consumer_lane in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                        partial_lane_accumulator = pl.aiv_shard(partial_accumulator)
                        partial_token_start = owner_start + 128 + partial_consumer_lane * 32
                        partial_scratch_start = owner_index * 192 + 128 + partial_consumer_lane * 32
                        partial_norm_row = pl.slice(normw, [1, 32], [0, 0], valid_shape=[1, 24], clamp=True)
                        partial_rms_tile = pl.reshape(pl.slice(group_rms, [1, 32], [0, partial_scratch_start], valid_shape=[1, 32]), [32, 1])
                        partial_predicted = pl.row_expand_mul(pl.col_expand_mul(partial_lane_accumulator, partial_norm_row), partial_rms_tile)
                        group_next_mixes[partial_token_start:partial_token_start + 32, 0:24] = partial_predicted[0:32, 0:24]
                        partial_scale_pre = pl.read(mhc_scale, [0])
                        partial_coefficient_indices = pl.cast(pl.arange(0, [1, 32], dtype=pl.INT32), pl.FP32)
                        partial_pre_selector0 = pl.maximum(pl.add(pl.neg(pl.abs(pl.sub(partial_coefficient_indices, 0.0))), 1.0), 0.0)
                        partial_pre_coefficient0 = pl.row_sum(pl.col_expand_mul(partial_predicted, partial_pre_selector0))
                        partial_pre_raw0 = pl.add(pl.mul(partial_pre_coefficient0, partial_scale_pre), pl.read(mhc_base, [0, 0]))
                        partial_next_pre0 = pl.add(pl.recip(pl.add(pl.exp(pl.neg(partial_pre_raw0)), 1.0)), PRE_EPS)
                        partial_pre_selector1 = pl.maximum(pl.add(pl.neg(pl.abs(pl.sub(partial_coefficient_indices, 1.0))), 1.0), 0.0)
                        partial_pre_coefficient1 = pl.row_sum(pl.col_expand_mul(partial_predicted, partial_pre_selector1))
                        partial_pre_raw1 = pl.add(pl.mul(partial_pre_coefficient1, partial_scale_pre), pl.read(mhc_base, [0, 1]))
                        partial_next_pre1 = pl.add(pl.recip(pl.add(pl.exp(pl.neg(partial_pre_raw1)), 1.0)), PRE_EPS)
                        partial_pre_selector2 = pl.maximum(pl.add(pl.neg(pl.abs(pl.sub(partial_coefficient_indices, 2.0))), 1.0), 0.0)
                        partial_pre_coefficient2 = pl.row_sum(pl.col_expand_mul(partial_predicted, partial_pre_selector2))
                        partial_pre_raw2 = pl.add(pl.mul(partial_pre_coefficient2, partial_scale_pre), pl.read(mhc_base, [0, 2]))
                        partial_next_pre2 = pl.add(pl.recip(pl.add(pl.exp(pl.neg(partial_pre_raw2)), 1.0)), PRE_EPS)
                        partial_pre_selector3 = pl.maximum(pl.add(pl.neg(pl.abs(pl.sub(partial_coefficient_indices, 3.0))), 1.0), 0.0)
                        partial_pre_coefficient3 = pl.row_sum(pl.col_expand_mul(partial_predicted, partial_pre_selector3))
                        partial_pre_raw3 = pl.add(pl.mul(partial_pre_coefficient3, partial_scale_pre), pl.read(mhc_base, [0, 3]))
                        partial_next_pre3 = pl.add(pl.recip(pl.add(pl.exp(pl.neg(partial_pre_raw3)), 1.0)), PRE_EPS)
                        for partial_hidden_block in pl.range(D // 64):
                            partial_hidden_start = partial_hidden_block * 64
                        for partial_strip_index in pl.range(4):
                            partial_strip_start = partial_strip_index * 8
                            partial_updated_rows = group_updated[partial_scratch_start + partial_strip_start:partial_scratch_start + partial_strip_start + 8, 0:FD]
                            partial_strip_pre0 = partial_next_pre0[partial_strip_start:partial_strip_start + 8, 0:1]
                            partial_strip_pre1 = partial_next_pre1[partial_strip_start:partial_strip_start + 8, 0:1]
                            partial_strip_pre2 = partial_next_pre2[partial_strip_start:partial_strip_start + 8, 0:1]
                            partial_strip_pre3 = partial_next_pre3[partial_strip_start:partial_strip_start + 8, 0:1]
                            partial_updated_copy0 = partial_updated_rows[0:8, 0 * D:(0 + 1) * D]
                            partial_hidden = pl.row_expand_mul(partial_updated_copy0, partial_strip_pre0)
                            partial_updated_copy1 = partial_updated_rows[0:8, 1 * D:(1 + 1) * D]
                            partial_hidden = pl.add(partial_hidden, pl.row_expand_mul(partial_updated_copy1, partial_strip_pre1))
                            partial_updated_copy2 = partial_updated_rows[0:8, 2 * D:(2 + 1) * D]
                            partial_hidden = pl.add(partial_hidden, pl.row_expand_mul(partial_updated_copy2, partial_strip_pre2))
                            partial_updated_copy3 = partial_updated_rows[0:8, 3 * D:(3 + 1) * D]
                            partial_hidden = pl.add(partial_hidden, pl.row_expand_mul(partial_updated_copy3, partial_strip_pre3))
                            group_residual[partial_token_start + partial_strip_start:partial_token_start + partial_strip_start + 8, 0:FD] = pl.cast(partial_updated_rows, pl.BF16, mode='rint')
                            group_h_next[partial_token_start + partial_strip_start:partial_token_start + partial_strip_start + 8, 0:D] = pl.cast(partial_hidden, pl.BF16, mode='rint')
                pl.system.sync_wait(5, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                pl.system.cacheinvalid()
                pl.system.fence()
                full_accumulator = pl.create_tensor([128, 32], dtype=pl.FP32)
                for full_projection_block in pl.pipeline(0, FD // 128, stage=2):
                    full_projection_start = full_projection_block * 128
                    full_updated_chunk = pl.slice(group_updated, [128, 128], [owner_index * 192 + 0, full_projection_start], valid_shape=[128, 128])
                    full_weight_chunk = pl.slice(fn, [32, 128], [0, full_projection_start], valid_shape=[24, 128], clamp=True)
                    full_accumulator = pl.matmul_acc(full_accumulator, full_updated_chunk, full_weight_chunk, b_trans=True, init_cond=(full_projection_block == 0))
                for full_consumer_lane in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                    full_lane_accumulator = pl.aiv_shard(full_accumulator)
                    full_token_start = owner_start + 0 + full_consumer_lane * 64
                    full_scratch_start = owner_index * 192 + 0 + full_consumer_lane * 64
                    full_norm_row = pl.slice(normw, [1, 32], [0, 0], valid_shape=[1, 24], clamp=True)
                    full_rms_tile = pl.reshape(pl.slice(group_rms, [1, 64], [0, full_scratch_start], valid_shape=[1, 64]), [64, 1])
                    full_predicted = pl.row_expand_mul(pl.col_expand_mul(full_lane_accumulator, full_norm_row), full_rms_tile)
                    group_next_mixes[full_token_start:full_token_start + 64, 0:24] = full_predicted[0:64, 0:24]
                    full_scale_pre = pl.read(mhc_scale, [0])
                    full_coefficient_indices = pl.cast(pl.arange(0, [1, 32], dtype=pl.INT32), pl.FP32)
                    full_pre_selector0 = pl.maximum(pl.add(pl.neg(pl.abs(pl.sub(full_coefficient_indices, 0.0))), 1.0), 0.0)
                    full_pre_coefficient0 = pl.row_sum(pl.col_expand_mul(full_predicted, full_pre_selector0))
                    full_pre_raw0 = pl.add(pl.mul(full_pre_coefficient0, full_scale_pre), pl.read(mhc_base, [0, 0]))
                    full_next_pre0 = pl.add(pl.recip(pl.add(pl.exp(pl.neg(full_pre_raw0)), 1.0)), PRE_EPS)
                    full_pre_selector1 = pl.maximum(pl.add(pl.neg(pl.abs(pl.sub(full_coefficient_indices, 1.0))), 1.0), 0.0)
                    full_pre_coefficient1 = pl.row_sum(pl.col_expand_mul(full_predicted, full_pre_selector1))
                    full_pre_raw1 = pl.add(pl.mul(full_pre_coefficient1, full_scale_pre), pl.read(mhc_base, [0, 1]))
                    full_next_pre1 = pl.add(pl.recip(pl.add(pl.exp(pl.neg(full_pre_raw1)), 1.0)), PRE_EPS)
                    full_pre_selector2 = pl.maximum(pl.add(pl.neg(pl.abs(pl.sub(full_coefficient_indices, 2.0))), 1.0), 0.0)
                    full_pre_coefficient2 = pl.row_sum(pl.col_expand_mul(full_predicted, full_pre_selector2))
                    full_pre_raw2 = pl.add(pl.mul(full_pre_coefficient2, full_scale_pre), pl.read(mhc_base, [0, 2]))
                    full_next_pre2 = pl.add(pl.recip(pl.add(pl.exp(pl.neg(full_pre_raw2)), 1.0)), PRE_EPS)
                    full_pre_selector3 = pl.maximum(pl.add(pl.neg(pl.abs(pl.sub(full_coefficient_indices, 3.0))), 1.0), 0.0)
                    full_pre_coefficient3 = pl.row_sum(pl.col_expand_mul(full_predicted, full_pre_selector3))
                    full_pre_raw3 = pl.add(pl.mul(full_pre_coefficient3, full_scale_pre), pl.read(mhc_base, [0, 3]))
                    full_next_pre3 = pl.add(pl.recip(pl.add(pl.exp(pl.neg(full_pre_raw3)), 1.0)), PRE_EPS)
                    for full_hidden_block in pl.range(D // 64):
                        full_hidden_start = full_hidden_block * 64
                    for full_strip_index in pl.range(8):
                        full_strip_start = full_strip_index * 8
                        full_updated_rows = group_updated[full_scratch_start + full_strip_start:full_scratch_start + full_strip_start + 8, 0:FD]
                        full_strip_pre0 = full_next_pre0[full_strip_start:full_strip_start + 8, 0:1]
                        full_strip_pre1 = full_next_pre1[full_strip_start:full_strip_start + 8, 0:1]
                        full_strip_pre2 = full_next_pre2[full_strip_start:full_strip_start + 8, 0:1]
                        full_strip_pre3 = full_next_pre3[full_strip_start:full_strip_start + 8, 0:1]
                        full_updated_copy0 = full_updated_rows[0:8, 0 * D:(0 + 1) * D]
                        full_hidden = pl.row_expand_mul(full_updated_copy0, full_strip_pre0)
                        full_updated_copy1 = full_updated_rows[0:8, 1 * D:(1 + 1) * D]
                        full_hidden = pl.add(full_hidden, pl.row_expand_mul(full_updated_copy1, full_strip_pre1))
                        full_updated_copy2 = full_updated_rows[0:8, 2 * D:(2 + 1) * D]
                        full_hidden = pl.add(full_hidden, pl.row_expand_mul(full_updated_copy2, full_strip_pre2))
                        full_updated_copy3 = full_updated_rows[0:8, 3 * D:(3 + 1) * D]
                        full_hidden = pl.add(full_hidden, pl.row_expand_mul(full_updated_copy3, full_strip_pre3))
                        group_residual[full_token_start + full_strip_start:full_token_start + full_strip_start + 8, 0:FD] = pl.cast(full_updated_rows, pl.BF16, mode='rint')
                        group_h_next[full_token_start + full_strip_start:full_token_start + full_strip_start + 8, 0:D] = pl.cast(full_hidden, pl.BF16, mode='rint')
    return h_next


def build(shape):
    import torch

    T = shape['T']
    dtype = shape.get('dtype', 'fp32')
    if dtype not in ('fp32', 'bf16') or T <= 0 or T % 8:
        raise ValueError('unsupported shape')
    activation_dtype = torch.bfloat16 if dtype == 'bf16' else torch.float32
    FD = M * D

    def specs():
        from golden import TensorSpec

        def rnd(*sh, s=0.02, storage=torch.float32):
            return lambda: (torch.randn(*sh) * s).to(storage)
        return [TensorSpec('x', [T, D], activation_dtype, init_value=rnd(T, D, storage=activation_dtype)), TensorSpec('residual', [T, FD], activation_dtype, init_value=rnd(T, FD, s=0.5, storage=activation_dtype)), TensorSpec('input_mixes', [T, M3], torch.float32, init_value=rnd(T, M3, s=0.5)), TensorSpec('fn', [M3, FD], torch.float32, init_value=rnd(M3, FD)), TensorSpec('normw', [1, M3], torch.float32, init_value=lambda: torch.rand(1, M3) + 0.5), TensorSpec('mhc_scale', [3], torch.float32, init_value=lambda: torch.rand(3) + 0.5), TensorSpec('mhc_base', [1, M3], torch.float32, init_value=rnd(1, M3, s=0.1)), TensorSpec('next_mixes', [T, M3], torch.float32), TensorSpec('h_next', [T, D], activation_dtype)]

    def golden(t):
        x = t['x'].float()
        res = t['residual'].float().view(T, M, D).clone()
        mixes = t['input_mixes'].float()
        s = t['mhc_scale']
        base = t['mhc_base'].float().view(-1)
        nw = t['normw'].float().view(-1)
        pre = torch.sigmoid(mixes[:, 0:M] * s[0] + base[0:M]) + PRE_EPS
        post = torch.sigmoid(mixes[:, M:2 * M] * s[1] + base[M:2 * M]) * POST_MULT
        comb = (mixes[:, 2 * M:] * s[2] + base[2 * M:]).view(T, M, M)
        cs = torch.softmax(comb, dim=-1) + PRE_EPS
        cs = cs / (cs.sum(-2, keepdim=True) + PRE_EPS)
        for _ in range(SINK - 1):
            cs = cs / (cs.sum(-1, keepdim=True) + PRE_EPS)
            cs = cs / (cs.sum(-2, keepdim=True) + PRE_EPS)
        res2 = x.unsqueeze(1) * post.unsqueeze(-1) + torch.einsum('tmn,tnd->tmd', cs, res)
        flat = res2.reshape(T, FD)
        rn = torch.rsqrt(flat.pow(2).mean(-1, keepdim=True) + RMS_EPS)
        nxt = flat @ (t['fn'].float() * nw.unsqueeze(1)).t() * rn
        npre = torch.sigmoid(nxt[:, 0:M] * s[0] + base[0:M]) + PRE_EPS
        h = (res2 * npre.unsqueeze(-1)).sum(1)
        t['residual'][:] = res2.reshape(T, FD)
        t['next_mixes'][:] = nxt
        t['h_next'][:] = h
    if dtype == 'bf16':
        if T % 128:
            entry = mega_mhc_bridge_tail_bf16_8
        elif T <= 1536:
            entry = mega_mhc_bridge_compact32
        elif T <= 3072:
            entry = mega_mhc_bridge_compact64
        else:
            entry = mega_mhc_bridge_compact128
        if T >= 1024 and T % 512 == 0 and T // 512 <= 32:
            entry = mega_mhc_packed_owner512
            if (T + 127) // 128 <= FUSED_OWNER_LIMIT:
                entry = mega_mhc_owner512
        return entry, specs(), golden, 0.02, 0.02
    return mega_mhc_plane_8, specs(), golden, 0.0001, 0.0001
