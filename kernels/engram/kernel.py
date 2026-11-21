import pypto.language as pl
HEADS = 24
HEAD_DIM = 32
EMBED = 768
HIDDEN = 1024
VOCAB = 4096
ROW_TILE = 128
COL_TILE = 128
RED_TILE = 256
GATE_ROWS = 16
GATE_TILE = 256
SCALAR_GATE_TILE = 128
EPS = 1e-06

def make_fused_engram_fp32(tokens):
    if tokens < GATE_ROWS:
        @pl.program
        class FusedEngramFp32Region0Program:
            @pl.function(type=pl.FunctionType.Opaque)
            def fused_engram_fp32(self, hash_ids: pl.Tensor[[tokens, HEADS], pl.INT32], head_off: pl.Tensor[[1, HEADS], pl.INT32], table: pl.Tensor[[HEADS * VOCAB, HEAD_DIM], pl.FP32], wk: pl.Tensor[[EMBED, HIDDEN], pl.FP32], wv: pl.Tensor[[EMBED, HIDDEN], pl.FP32], hidden: pl.Tensor[[tokens, HIDDEN], pl.FP32], out: pl.Out[pl.Tensor[[tokens, HIDDEN], pl.FP32]]) -> pl.Tensor[[tokens, HIDDEN], pl.FP32]:
                token_count = tokens
                padded_tokens = (token_count + 16 - 1) // 16 * 16
                workspace = pl.create_tensor([padded_tokens, EMBED], dtype=pl.FP32)
                for block in pl.spmd(padded_tokens // 16, name_hint='gather_embedding', allow_early_resolve=True):
                    base_token = block * 16
                    if base_token + 16 <= token_count:
                        local_ids = pl.load(hash_ids, [base_token, 0], [16, HEADS])
                        offsets = pl.load(head_off, [0, 0], [1, HEADS])
                        row_ids = pl.tile.col_expand_add(local_ids, offsets)
                        flat_rows = pl.reshape(row_ids, [1, 16 * HEADS])
                        bulk_rows = pl.tile.mgather(table, flat_rows, coalesce='row', gather_oob='undefined')
                        bulk_embedding = pl.reshape(bulk_rows, [16, EMBED])
                        workspace = pl.store(bulk_embedding, [base_token, 0], workspace)
                    else:
                        if base_token < token_count:
                            gather_valid_rows = pl.min(16, token_count - base_token)
                            partial_ids = pl.load(hash_ids, [base_token, 0], [16, HEADS], valid_shape=[gather_valid_rows, HEADS])
                            padded_ids = pl.tile.fillpad(partial_ids, pad_value=0)
                            partial_defined_ids = pl.tile.set_validshape(padded_ids, 16, HEADS)
                            partial_offsets = pl.load(head_off, [0, 0], [1, HEADS])
                            partial_row_ids = pl.tile.col_expand_add(partial_defined_ids, partial_offsets)
                            partial_flat_rows = pl.reshape(partial_row_ids, [1, 16 * HEADS])
                            partial_bulk_rows = pl.tile.mgather(table, partial_flat_rows, coalesce='row', gather_oob='undefined')
                            partial_embedding = pl.reshape(partial_bulk_rows, [16, EMBED])
                            workspace = pl.store(partial_embedding, [base_token, 0], workspace)
                        else:
                            zero_embedding = pl.tile.full([16, EMBED], dtype=pl.FP32, value=0.0)
                            workspace = pl.store(zero_embedding, [base_token, 0], workspace)
                key_workspace = pl.create_tensor([padded_tokens, HIDDEN], dtype=pl.FP32)
                value_workspace = pl.create_tensor([padded_tokens, HIDDEN], dtype=pl.FP32)
                for block in pl.spmd((padded_tokens // 16 * (HIDDEN // 64) - 0) // 1, name_hint='paired_projection', allow_early_resolve=True):
                    owner = 0 + block * 1
                    row = owner // (HIDDEN // 64) * 16
                    column = owner % (HIDDEN // 64) * 64
                    key_acc = pl.create_tensor([16, 64], dtype=pl.FP32)
                    value_acc = pl.create_tensor([16, 64], dtype=pl.FP32)
                    for reduction in pl.pipeline(EMBED // 64, stage=2):
                        embedding_tile = workspace[row:row + 16, reduction * 64:(reduction + 1) * 64]
                        key_weights = wk[reduction * 64:(reduction + 1) * 64, column:column + 64]
                        value_weights = wv[reduction * 64:(reduction + 1) * 64, column:column + 64]
                        key_acc = pl.matmul_acc(key_acc, embedding_tile, key_weights, init_cond=reduction == 0)
                        value_acc = pl.matmul_acc(value_acc, embedding_tile, value_weights, init_cond=reduction == 0)
                    key_workspace[row:row + 16, column:column + 64] = key_acc
                    value_workspace[row:row + 16, column:column + 64] = value_acc
                for block in pl.spmd((token_count % GATE_ROWS + GATE_ROWS - 1) // GATE_ROWS, name_hint='vector_tail_gate'):
                    tail_row = token_count // GATE_ROWS * GATE_ROWS
                    tail_valid_rows = token_count - tail_row
                    tail_vector_key_zero = pl.tile.full([16, GATE_TILE], dtype=pl.FP32, value=0.0)
                    tail_vector_query_zero = pl.tile.full([16, GATE_TILE], dtype=pl.FP32, value=0.0)
                    tail_vector_dot_zero = pl.tile.full([16, GATE_TILE], dtype=pl.FP32, value=0.0)
                    tail_vector_key_squares = pl.tile.set_validshape(tail_vector_key_zero, tail_valid_rows, GATE_TILE)
                    tail_vector_query_squares = pl.tile.set_validshape(tail_vector_query_zero, tail_valid_rows, GATE_TILE)
                    tail_vector_key_dot = pl.tile.set_validshape(tail_vector_dot_zero, tail_valid_rows, GATE_TILE)
                    for tail_vector_column in pl.range(HIDDEN // GATE_TILE):
                        tail_vector_key = pl.load(key_workspace, [tail_row, tail_vector_column * GATE_TILE], [16, GATE_TILE], valid_shape=[tail_valid_rows, GATE_TILE])
                        tail_vector_query = pl.load(hidden, [tail_row, tail_vector_column * GATE_TILE], [16, GATE_TILE], valid_shape=[tail_valid_rows, GATE_TILE])
                        tail_vector_key_squares = pl.tile.add(tail_vector_key_squares, pl.tile.mul(tail_vector_key, tail_vector_key))
                        tail_vector_query_squares = pl.tile.add(tail_vector_query_squares, pl.tile.mul(tail_vector_query, tail_vector_query))
                        tail_vector_key_dot = pl.tile.add(tail_vector_key_dot, pl.tile.mul(tail_vector_key, tail_vector_query))
                    tail_vector_reduce_scratch = pl.create_tile([16, GATE_TILE], dtype=pl.FP32)
                    tail_vector_key_sum = pl.tile.row_sum(tail_vector_key_squares, tail_vector_reduce_scratch)
                    tail_vector_query_sum = pl.tile.row_sum(tail_vector_query_squares, tail_vector_reduce_scratch)
                    tail_vector_dot_sum = pl.tile.row_sum(tail_vector_key_dot, tail_vector_reduce_scratch)
                    tail_vector_ones = pl.tile.full([16, GATE_TILE], dtype=pl.FP32, value=1.0)
                    tail_vector_carrier = pl.tile.set_validshape(tail_vector_ones, tail_valid_rows, GATE_TILE)
                    tail_vector_key_expanded = pl.tile.row_expand_mul(tail_vector_carrier, tail_vector_key_sum)
                    tail_vector_query_expanded = pl.tile.row_expand_mul(tail_vector_carrier, tail_vector_query_sum)
                    tail_vector_dot_expanded = pl.tile.row_expand_mul(tail_vector_carrier, tail_vector_dot_sum)
                    tail_vector_rsqrt_scratch = pl.create_tile([16, GATE_TILE], dtype=pl.FP32)
                    tail_vector_key_scale = pl.tile.rsqrt(pl.tile.add(pl.tile.mul(tail_vector_key_expanded, 1.0 / HIDDEN), EPS), tail_vector_rsqrt_scratch)
                    tail_vector_query_scale = pl.tile.rsqrt(pl.tile.add(pl.tile.mul(tail_vector_query_expanded, 1.0 / HIDDEN), EPS), tail_vector_rsqrt_scratch)
                    tail_vector_argument = pl.tile.mul(pl.tile.mul(tail_vector_dot_expanded, pl.tile.mul(tail_vector_key_scale, tail_vector_query_scale)), HIDDEN ** (-0.5))
                    tail_vector_magnitude = pl.tile.sqrt(pl.tile.add(pl.tile.abs(tail_vector_argument), EPS))
                    tail_vector_negative_or_zero = pl.tile.select(pl.tile.cmps(tail_vector_argument, 0.0, cmp_type=2), pl.tile.neg(tail_vector_magnitude), 0.0)
                    tail_vector_signed_root = pl.tile.select(pl.tile.cmps(tail_vector_argument, 0.0, cmp_type=4), tail_vector_magnitude, tail_vector_negative_or_zero)
                    tail_vector_denominator = pl.tile.add(pl.tile.exp(pl.tile.neg(tail_vector_signed_root)), 1.0)
                    tail_vector_gate_estimate = pl.tile.recip(tail_vector_denominator)
                    tail_vector_twos = pl.tile.full([16, GATE_TILE], dtype=pl.FP32, value=2.0)
                    tail_vector_correction = pl.tile.sub(pl.tile.set_validshape(tail_vector_twos, tail_valid_rows, GATE_TILE), pl.tile.mul(tail_vector_denominator, tail_vector_gate_estimate))
                    tail_vector_gate = pl.tile.mul(tail_vector_gate_estimate, tail_vector_correction)
                    for tail_vector_column in pl.range(HIDDEN // GATE_TILE):
                        tail_vector_value = pl.load(value_workspace, [tail_row, tail_vector_column * GATE_TILE], [16, GATE_TILE], valid_shape=[tail_valid_rows, GATE_TILE])
                        tail_vector_result = pl.tile.mul(tail_vector_value, tail_vector_gate)
                        out = pl.store(tail_vector_result, [tail_row, tail_vector_column * GATE_TILE], out)
                return out

        return FusedEngramFp32Region0Program
    elif tokens % GATE_ROWS == 0:
        @pl.program
        class FusedEngramFp32Region1Program:
            @pl.function(type=pl.FunctionType.Opaque)
            def fused_engram_fp32(self, hash_ids: pl.Tensor[[tokens, HEADS], pl.INT32], head_off: pl.Tensor[[1, HEADS], pl.INT32], table: pl.Tensor[[HEADS * VOCAB, HEAD_DIM], pl.FP32], wk: pl.Tensor[[EMBED, HIDDEN], pl.FP32], wv: pl.Tensor[[EMBED, HIDDEN], pl.FP32], hidden: pl.Tensor[[tokens, HIDDEN], pl.FP32], out: pl.Out[pl.Tensor[[tokens, HIDDEN], pl.FP32]]) -> pl.Tensor[[tokens, HIDDEN], pl.FP32]:
                token_count = tokens
                padded_tokens = (token_count + 16 - 1) // 16 * 16
                workspace = pl.create_tensor([padded_tokens, EMBED], dtype=pl.FP32)
                for block in pl.spmd(padded_tokens // 16, name_hint='gather_embedding', allow_early_resolve=True):
                    base_token = block * 16
                    if base_token + 16 <= token_count:
                        local_ids = pl.load(hash_ids, [base_token, 0], [16, HEADS])
                        offsets = pl.load(head_off, [0, 0], [1, HEADS])
                        row_ids = pl.tile.col_expand_add(local_ids, offsets)
                        flat_rows = pl.reshape(row_ids, [1, 16 * HEADS])
                        bulk_rows = pl.tile.mgather(table, flat_rows, coalesce='row', gather_oob='undefined')
                        bulk_embedding = pl.reshape(bulk_rows, [16, EMBED])
                        workspace = pl.store(bulk_embedding, [base_token, 0], workspace)
                    else:
                        if base_token < token_count:
                            gather_valid_rows = pl.min(16, token_count - base_token)
                            partial_ids = pl.load(hash_ids, [base_token, 0], [16, HEADS], valid_shape=[gather_valid_rows, HEADS])
                            padded_ids = pl.tile.fillpad(partial_ids, pad_value=0)
                            partial_defined_ids = pl.tile.set_validshape(padded_ids, 16, HEADS)
                            partial_offsets = pl.load(head_off, [0, 0], [1, HEADS])
                            partial_row_ids = pl.tile.col_expand_add(partial_defined_ids, partial_offsets)
                            partial_flat_rows = pl.reshape(partial_row_ids, [1, 16 * HEADS])
                            partial_bulk_rows = pl.tile.mgather(table, partial_flat_rows, coalesce='row', gather_oob='undefined')
                            partial_embedding = pl.reshape(partial_bulk_rows, [16, EMBED])
                            workspace = pl.store(partial_embedding, [base_token, 0], workspace)
                        else:
                            zero_embedding = pl.tile.full([16, EMBED], dtype=pl.FP32, value=0.0)
                            workspace = pl.store(zero_embedding, [base_token, 0], workspace)
                key_workspace = pl.create_tensor([padded_tokens, HIDDEN], dtype=pl.FP32)
                value_workspace = pl.create_tensor([padded_tokens, HIDDEN], dtype=pl.FP32)
                for block in pl.spmd((padded_tokens // 16 * (HIDDEN // 64) - 0) // 1, name_hint='paired_projection', allow_early_resolve=True):
                    owner = 0 + block * 1
                    row = owner // (HIDDEN // 64) * 16
                    column = owner % (HIDDEN // 64) * 64
                    key_acc = pl.create_tensor([16, 64], dtype=pl.FP32)
                    value_acc = pl.create_tensor([16, 64], dtype=pl.FP32)
                    for reduction in pl.pipeline(EMBED // 64, stage=2):
                        embedding_tile = workspace[row:row + 16, reduction * 64:(reduction + 1) * 64]
                        key_weights = wk[reduction * 64:(reduction + 1) * 64, column:column + 64]
                        value_weights = wv[reduction * 64:(reduction + 1) * 64, column:column + 64]
                        key_acc = pl.matmul_acc(key_acc, embedding_tile, key_weights, init_cond=reduction == 0)
                        value_acc = pl.matmul_acc(value_acc, embedding_tile, value_weights, init_cond=reduction == 0)
                    key_workspace[row:row + 16, column:column + 64] = key_acc
                    value_workspace[row:row + 16, column:column + 64] = value_acc
                for block in pl.spmd((token_count // GATE_ROWS * GATE_ROWS - 0) // GATE_ROWS, name_hint='vector_gate'):
                    row = 0 + block * GATE_ROWS
                    vector_key_zero = pl.tile.full([16, GATE_TILE], dtype=pl.FP32, value=0.0)
                    vector_query_zero = pl.tile.full([16, GATE_TILE], dtype=pl.FP32, value=0.0)
                    vector_dot_zero = pl.tile.full([16, GATE_TILE], dtype=pl.FP32, value=0.0)
                    vector_key_squares = pl.tile.set_validshape(vector_key_zero, GATE_ROWS, GATE_TILE)
                    vector_query_squares = pl.tile.set_validshape(vector_query_zero, GATE_ROWS, GATE_TILE)
                    vector_key_dot = pl.tile.set_validshape(vector_dot_zero, GATE_ROWS, GATE_TILE)
                    for vector_column in pl.range(HIDDEN // GATE_TILE):
                        vector_key = pl.load(key_workspace, [row, vector_column * GATE_TILE], [16, GATE_TILE], valid_shape=[GATE_ROWS, GATE_TILE])
                        vector_query = pl.load(hidden, [row, vector_column * GATE_TILE], [16, GATE_TILE], valid_shape=[GATE_ROWS, GATE_TILE])
                        vector_key_squares = pl.tile.add(vector_key_squares, pl.tile.mul(vector_key, vector_key))
                        vector_query_squares = pl.tile.add(vector_query_squares, pl.tile.mul(vector_query, vector_query))
                        vector_key_dot = pl.tile.add(vector_key_dot, pl.tile.mul(vector_key, vector_query))
                    vector_reduce_scratch = pl.create_tile([16, GATE_TILE], dtype=pl.FP32)
                    vector_key_sum = pl.tile.row_sum(vector_key_squares, vector_reduce_scratch)
                    vector_query_sum = pl.tile.row_sum(vector_query_squares, vector_reduce_scratch)
                    vector_dot_sum = pl.tile.row_sum(vector_key_dot, vector_reduce_scratch)
                    vector_ones = pl.tile.full([16, GATE_TILE], dtype=pl.FP32, value=1.0)
                    vector_carrier = pl.tile.set_validshape(vector_ones, GATE_ROWS, GATE_TILE)
                    vector_key_expanded = pl.tile.row_expand_mul(vector_carrier, vector_key_sum)
                    vector_query_expanded = pl.tile.row_expand_mul(vector_carrier, vector_query_sum)
                    vector_dot_expanded = pl.tile.row_expand_mul(vector_carrier, vector_dot_sum)
                    vector_rsqrt_scratch = pl.create_tile([16, GATE_TILE], dtype=pl.FP32)
                    vector_key_scale = pl.tile.rsqrt(pl.tile.add(pl.tile.mul(vector_key_expanded, 1.0 / HIDDEN), EPS), vector_rsqrt_scratch)
                    vector_query_scale = pl.tile.rsqrt(pl.tile.add(pl.tile.mul(vector_query_expanded, 1.0 / HIDDEN), EPS), vector_rsqrt_scratch)
                    vector_argument = pl.tile.mul(pl.tile.mul(vector_dot_expanded, pl.tile.mul(vector_key_scale, vector_query_scale)), HIDDEN ** (-0.5))
                    vector_magnitude = pl.tile.sqrt(pl.tile.add(pl.tile.abs(vector_argument), EPS))
                    vector_negative_or_zero = pl.tile.select(pl.tile.cmps(vector_argument, 0.0, cmp_type=2), pl.tile.neg(vector_magnitude), 0.0)
                    vector_signed_root = pl.tile.select(pl.tile.cmps(vector_argument, 0.0, cmp_type=4), vector_magnitude, vector_negative_or_zero)
                    vector_denominator = pl.tile.add(pl.tile.exp(pl.tile.neg(vector_signed_root)), 1.0)
                    vector_gate_estimate = pl.tile.recip(vector_denominator)
                    vector_twos = pl.tile.full([16, GATE_TILE], dtype=pl.FP32, value=2.0)
                    vector_correction = pl.tile.sub(pl.tile.set_validshape(vector_twos, GATE_ROWS, GATE_TILE), pl.tile.mul(vector_denominator, vector_gate_estimate))
                    vector_gate = pl.tile.mul(vector_gate_estimate, vector_correction)
                    for vector_column in pl.range(HIDDEN // GATE_TILE):
                        vector_value = pl.load(value_workspace, [row, vector_column * GATE_TILE], [16, GATE_TILE], valid_shape=[GATE_ROWS, GATE_TILE])
                        vector_result = pl.tile.mul(vector_value, vector_gate)
                        out = pl.store(vector_result, [row, vector_column * GATE_TILE], out)
                return out

        return FusedEngramFp32Region1Program
    else:
        @pl.program
        class FusedEngramFp32Region2Program:
            @pl.function(type=pl.FunctionType.Opaque)
            def fused_engram_fp32(self, hash_ids: pl.Tensor[[tokens, HEADS], pl.INT32], head_off: pl.Tensor[[1, HEADS], pl.INT32], table: pl.Tensor[[HEADS * VOCAB, HEAD_DIM], pl.FP32], wk: pl.Tensor[[EMBED, HIDDEN], pl.FP32], wv: pl.Tensor[[EMBED, HIDDEN], pl.FP32], hidden: pl.Tensor[[tokens, HIDDEN], pl.FP32], out: pl.Out[pl.Tensor[[tokens, HIDDEN], pl.FP32]]) -> pl.Tensor[[tokens, HIDDEN], pl.FP32]:
                token_count = tokens
                padded_tokens = (token_count + 16 - 1) // 16 * 16
                workspace = pl.create_tensor([padded_tokens, EMBED], dtype=pl.FP32)
                for block in pl.spmd(padded_tokens // 16, name_hint='gather_embedding', allow_early_resolve=True):
                    base_token = block * 16
                    if base_token + 16 <= token_count:
                        local_ids = pl.load(hash_ids, [base_token, 0], [16, HEADS])
                        offsets = pl.load(head_off, [0, 0], [1, HEADS])
                        row_ids = pl.tile.col_expand_add(local_ids, offsets)
                        flat_rows = pl.reshape(row_ids, [1, 16 * HEADS])
                        bulk_rows = pl.tile.mgather(table, flat_rows, coalesce='row', gather_oob='undefined')
                        bulk_embedding = pl.reshape(bulk_rows, [16, EMBED])
                        workspace = pl.store(bulk_embedding, [base_token, 0], workspace)
                    else:
                        if base_token < token_count:
                            gather_valid_rows = pl.min(16, token_count - base_token)
                            partial_ids = pl.load(hash_ids, [base_token, 0], [16, HEADS], valid_shape=[gather_valid_rows, HEADS])
                            padded_ids = pl.tile.fillpad(partial_ids, pad_value=0)
                            partial_defined_ids = pl.tile.set_validshape(padded_ids, 16, HEADS)
                            partial_offsets = pl.load(head_off, [0, 0], [1, HEADS])
                            partial_row_ids = pl.tile.col_expand_add(partial_defined_ids, partial_offsets)
                            partial_flat_rows = pl.reshape(partial_row_ids, [1, 16 * HEADS])
                            partial_bulk_rows = pl.tile.mgather(table, partial_flat_rows, coalesce='row', gather_oob='undefined')
                            partial_embedding = pl.reshape(partial_bulk_rows, [16, EMBED])
                            workspace = pl.store(partial_embedding, [base_token, 0], workspace)
                        else:
                            zero_embedding = pl.tile.full([16, EMBED], dtype=pl.FP32, value=0.0)
                            workspace = pl.store(zero_embedding, [base_token, 0], workspace)
                key_workspace = pl.create_tensor([padded_tokens, HIDDEN], dtype=pl.FP32)
                value_workspace = pl.create_tensor([padded_tokens, HIDDEN], dtype=pl.FP32)
                for block in pl.spmd((padded_tokens // 16 * (HIDDEN // 64) - 0) // 1, name_hint='paired_projection', allow_early_resolve=True):
                    owner = 0 + block * 1
                    row = owner // (HIDDEN // 64) * 16
                    column = owner % (HIDDEN // 64) * 64
                    key_acc = pl.create_tensor([16, 64], dtype=pl.FP32)
                    value_acc = pl.create_tensor([16, 64], dtype=pl.FP32)
                    for reduction in pl.pipeline(EMBED // 64, stage=2):
                        embedding_tile = workspace[row:row + 16, reduction * 64:(reduction + 1) * 64]
                        key_weights = wk[reduction * 64:(reduction + 1) * 64, column:column + 64]
                        value_weights = wv[reduction * 64:(reduction + 1) * 64, column:column + 64]
                        key_acc = pl.matmul_acc(key_acc, embedding_tile, key_weights, init_cond=reduction == 0)
                        value_acc = pl.matmul_acc(value_acc, embedding_tile, value_weights, init_cond=reduction == 0)
                    key_workspace[row:row + 16, column:column + 64] = key_acc
                    value_workspace[row:row + 16, column:column + 64] = value_acc
                for block in pl.spmd((token_count // GATE_ROWS * GATE_ROWS - 0) // GATE_ROWS, name_hint='vector_gate'):
                    row = 0 + block * GATE_ROWS
                    vector_key_zero = pl.tile.full([16, GATE_TILE], dtype=pl.FP32, value=0.0)
                    vector_query_zero = pl.tile.full([16, GATE_TILE], dtype=pl.FP32, value=0.0)
                    vector_dot_zero = pl.tile.full([16, GATE_TILE], dtype=pl.FP32, value=0.0)
                    vector_key_squares = pl.tile.set_validshape(vector_key_zero, GATE_ROWS, GATE_TILE)
                    vector_query_squares = pl.tile.set_validshape(vector_query_zero, GATE_ROWS, GATE_TILE)
                    vector_key_dot = pl.tile.set_validshape(vector_dot_zero, GATE_ROWS, GATE_TILE)
                    for vector_column in pl.range(HIDDEN // GATE_TILE):
                        vector_key = pl.load(key_workspace, [row, vector_column * GATE_TILE], [16, GATE_TILE], valid_shape=[GATE_ROWS, GATE_TILE])
                        vector_query = pl.load(hidden, [row, vector_column * GATE_TILE], [16, GATE_TILE], valid_shape=[GATE_ROWS, GATE_TILE])
                        vector_key_squares = pl.tile.add(vector_key_squares, pl.tile.mul(vector_key, vector_key))
                        vector_query_squares = pl.tile.add(vector_query_squares, pl.tile.mul(vector_query, vector_query))
                        vector_key_dot = pl.tile.add(vector_key_dot, pl.tile.mul(vector_key, vector_query))
                    vector_reduce_scratch = pl.create_tile([16, GATE_TILE], dtype=pl.FP32)
                    vector_key_sum = pl.tile.row_sum(vector_key_squares, vector_reduce_scratch)
                    vector_query_sum = pl.tile.row_sum(vector_query_squares, vector_reduce_scratch)
                    vector_dot_sum = pl.tile.row_sum(vector_key_dot, vector_reduce_scratch)
                    vector_ones = pl.tile.full([16, GATE_TILE], dtype=pl.FP32, value=1.0)
                    vector_carrier = pl.tile.set_validshape(vector_ones, GATE_ROWS, GATE_TILE)
                    vector_key_expanded = pl.tile.row_expand_mul(vector_carrier, vector_key_sum)
                    vector_query_expanded = pl.tile.row_expand_mul(vector_carrier, vector_query_sum)
                    vector_dot_expanded = pl.tile.row_expand_mul(vector_carrier, vector_dot_sum)
                    vector_rsqrt_scratch = pl.create_tile([16, GATE_TILE], dtype=pl.FP32)
                    vector_key_scale = pl.tile.rsqrt(pl.tile.add(pl.tile.mul(vector_key_expanded, 1.0 / HIDDEN), EPS), vector_rsqrt_scratch)
                    vector_query_scale = pl.tile.rsqrt(pl.tile.add(pl.tile.mul(vector_query_expanded, 1.0 / HIDDEN), EPS), vector_rsqrt_scratch)
                    vector_argument = pl.tile.mul(pl.tile.mul(vector_dot_expanded, pl.tile.mul(vector_key_scale, vector_query_scale)), HIDDEN ** (-0.5))
                    vector_magnitude = pl.tile.sqrt(pl.tile.add(pl.tile.abs(vector_argument), EPS))
                    vector_negative_or_zero = pl.tile.select(pl.tile.cmps(vector_argument, 0.0, cmp_type=2), pl.tile.neg(vector_magnitude), 0.0)
                    vector_signed_root = pl.tile.select(pl.tile.cmps(vector_argument, 0.0, cmp_type=4), vector_magnitude, vector_negative_or_zero)
                    vector_denominator = pl.tile.add(pl.tile.exp(pl.tile.neg(vector_signed_root)), 1.0)
                    vector_gate_estimate = pl.tile.recip(vector_denominator)
                    vector_twos = pl.tile.full([16, GATE_TILE], dtype=pl.FP32, value=2.0)
                    vector_correction = pl.tile.sub(pl.tile.set_validshape(vector_twos, GATE_ROWS, GATE_TILE), pl.tile.mul(vector_denominator, vector_gate_estimate))
                    vector_gate = pl.tile.mul(vector_gate_estimate, vector_correction)
                    for vector_column in pl.range(HIDDEN // GATE_TILE):
                        vector_value = pl.load(value_workspace, [row, vector_column * GATE_TILE], [16, GATE_TILE], valid_shape=[GATE_ROWS, GATE_TILE])
                        vector_result = pl.tile.mul(vector_value, vector_gate)
                        out = pl.store(vector_result, [row, vector_column * GATE_TILE], out)
                for block in pl.spmd((token_count % GATE_ROWS + GATE_ROWS - 1) // GATE_ROWS, name_hint='vector_tail_gate'):
                    tail_row = token_count // GATE_ROWS * GATE_ROWS
                    tail_valid_rows = token_count - tail_row
                    tail_vector_key_zero = pl.tile.full([16, GATE_TILE], dtype=pl.FP32, value=0.0)
                    tail_vector_query_zero = pl.tile.full([16, GATE_TILE], dtype=pl.FP32, value=0.0)
                    tail_vector_dot_zero = pl.tile.full([16, GATE_TILE], dtype=pl.FP32, value=0.0)
                    tail_vector_key_squares = pl.tile.set_validshape(tail_vector_key_zero, tail_valid_rows, GATE_TILE)
                    tail_vector_query_squares = pl.tile.set_validshape(tail_vector_query_zero, tail_valid_rows, GATE_TILE)
                    tail_vector_key_dot = pl.tile.set_validshape(tail_vector_dot_zero, tail_valid_rows, GATE_TILE)
                    for tail_vector_column in pl.range(HIDDEN // GATE_TILE):
                        tail_vector_key = pl.load(key_workspace, [tail_row, tail_vector_column * GATE_TILE], [16, GATE_TILE], valid_shape=[tail_valid_rows, GATE_TILE])
                        tail_vector_query = pl.load(hidden, [tail_row, tail_vector_column * GATE_TILE], [16, GATE_TILE], valid_shape=[tail_valid_rows, GATE_TILE])
                        tail_vector_key_squares = pl.tile.add(tail_vector_key_squares, pl.tile.mul(tail_vector_key, tail_vector_key))
                        tail_vector_query_squares = pl.tile.add(tail_vector_query_squares, pl.tile.mul(tail_vector_query, tail_vector_query))
                        tail_vector_key_dot = pl.tile.add(tail_vector_key_dot, pl.tile.mul(tail_vector_key, tail_vector_query))
                    tail_vector_reduce_scratch = pl.create_tile([16, GATE_TILE], dtype=pl.FP32)
                    tail_vector_key_sum = pl.tile.row_sum(tail_vector_key_squares, tail_vector_reduce_scratch)
                    tail_vector_query_sum = pl.tile.row_sum(tail_vector_query_squares, tail_vector_reduce_scratch)
                    tail_vector_dot_sum = pl.tile.row_sum(tail_vector_key_dot, tail_vector_reduce_scratch)
                    tail_vector_ones = pl.tile.full([16, GATE_TILE], dtype=pl.FP32, value=1.0)
                    tail_vector_carrier = pl.tile.set_validshape(tail_vector_ones, tail_valid_rows, GATE_TILE)
                    tail_vector_key_expanded = pl.tile.row_expand_mul(tail_vector_carrier, tail_vector_key_sum)
                    tail_vector_query_expanded = pl.tile.row_expand_mul(tail_vector_carrier, tail_vector_query_sum)
                    tail_vector_dot_expanded = pl.tile.row_expand_mul(tail_vector_carrier, tail_vector_dot_sum)
                    tail_vector_rsqrt_scratch = pl.create_tile([16, GATE_TILE], dtype=pl.FP32)
                    tail_vector_key_scale = pl.tile.rsqrt(pl.tile.add(pl.tile.mul(tail_vector_key_expanded, 1.0 / HIDDEN), EPS), tail_vector_rsqrt_scratch)
                    tail_vector_query_scale = pl.tile.rsqrt(pl.tile.add(pl.tile.mul(tail_vector_query_expanded, 1.0 / HIDDEN), EPS), tail_vector_rsqrt_scratch)
                    tail_vector_argument = pl.tile.mul(pl.tile.mul(tail_vector_dot_expanded, pl.tile.mul(tail_vector_key_scale, tail_vector_query_scale)), HIDDEN ** (-0.5))
                    tail_vector_magnitude = pl.tile.sqrt(pl.tile.add(pl.tile.abs(tail_vector_argument), EPS))
                    tail_vector_negative_or_zero = pl.tile.select(pl.tile.cmps(tail_vector_argument, 0.0, cmp_type=2), pl.tile.neg(tail_vector_magnitude), 0.0)
                    tail_vector_signed_root = pl.tile.select(pl.tile.cmps(tail_vector_argument, 0.0, cmp_type=4), tail_vector_magnitude, tail_vector_negative_or_zero)
                    tail_vector_denominator = pl.tile.add(pl.tile.exp(pl.tile.neg(tail_vector_signed_root)), 1.0)
                    tail_vector_gate_estimate = pl.tile.recip(tail_vector_denominator)
                    tail_vector_twos = pl.tile.full([16, GATE_TILE], dtype=pl.FP32, value=2.0)
                    tail_vector_correction = pl.tile.sub(pl.tile.set_validshape(tail_vector_twos, tail_valid_rows, GATE_TILE), pl.tile.mul(tail_vector_denominator, tail_vector_gate_estimate))
                    tail_vector_gate = pl.tile.mul(tail_vector_gate_estimate, tail_vector_correction)
                    for tail_vector_column in pl.range(HIDDEN // GATE_TILE):
                        tail_vector_value = pl.load(value_workspace, [tail_row, tail_vector_column * GATE_TILE], [16, GATE_TILE], valid_shape=[tail_valid_rows, GATE_TILE])
                        tail_vector_result = pl.tile.mul(tail_vector_value, tail_vector_gate)
                        out = pl.store(tail_vector_result, [tail_row, tail_vector_column * GATE_TILE], out)
                return out

        return FusedEngramFp32Region2Program

def make_fused_engram_bf16(tokens):
    if tokens < GATE_ROWS:
        @pl.program
        class FusedEngramBf16Region0Program:
            @pl.function(type=pl.FunctionType.Opaque)
            def fused_engram_bf16(self, hash_ids: pl.Tensor[[tokens, HEADS], pl.INT32], head_off: pl.Tensor[[1, HEADS], pl.INT32], table: pl.Tensor[[HEADS * VOCAB, HEAD_DIM], pl.BF16], wk: pl.Tensor[[EMBED, HIDDEN], pl.BF16], wv: pl.Tensor[[EMBED, HIDDEN], pl.BF16], hidden: pl.Tensor[[tokens, HIDDEN], pl.BF16], out: pl.Out[pl.Tensor[[tokens, HIDDEN], pl.BF16]]) -> pl.Tensor[[tokens, HIDDEN], pl.BF16]:
                token_count = tokens
                padded_tokens = (token_count + ROW_TILE - 1) // ROW_TILE * ROW_TILE
                workspace = pl.create_tensor([padded_tokens, EMBED], dtype=pl.BF16)
                for block in pl.spmd(padded_tokens // 16, name_hint='gather_embedding', allow_early_resolve=True):
                    base_token = block * 16
                    if base_token + 16 <= token_count:
                        local_ids = pl.load(hash_ids, [base_token, 0], [16, HEADS])
                        offsets = pl.load(head_off, [0, 0], [1, HEADS])
                        row_ids = pl.tile.col_expand_add(local_ids, offsets)
                        flat_rows = pl.reshape(row_ids, [1, 16 * HEADS])
                        bulk_rows = pl.tile.mgather(table, flat_rows, coalesce='row', gather_oob='undefined')
                        bulk_embedding = pl.reshape(bulk_rows, [16, EMBED])
                        workspace = pl.store(bulk_embedding, [base_token, 0], workspace)
                    else:
                        if base_token < token_count:
                            gather_valid_rows = pl.min(16, token_count - base_token)
                            partial_ids = pl.load(hash_ids, [base_token, 0], [16, HEADS], valid_shape=[gather_valid_rows, HEADS])
                            padded_ids = pl.tile.fillpad(partial_ids, pad_value=0)
                            partial_defined_ids = pl.tile.set_validshape(padded_ids, 16, HEADS)
                            partial_offsets = pl.load(head_off, [0, 0], [1, HEADS])
                            partial_row_ids = pl.tile.col_expand_add(partial_defined_ids, partial_offsets)
                            partial_flat_rows = pl.reshape(partial_row_ids, [1, 16 * HEADS])
                            partial_bulk_rows = pl.tile.mgather(table, partial_flat_rows, coalesce='row', gather_oob='undefined')
                            partial_embedding = pl.reshape(partial_bulk_rows, [16, EMBED])
                            workspace = pl.store(partial_embedding, [base_token, 0], workspace)
                        else:
                            zero_embedding = pl.tile.full([16, EMBED], dtype=pl.BF16, value=0.0)
                            workspace = pl.store(zero_embedding, [base_token, 0], workspace)
                key_workspace = pl.create_tensor([padded_tokens, HIDDEN], dtype=pl.FP32)
                value_workspace = pl.create_tensor([padded_tokens, HIDDEN], dtype=pl.FP32)
                for block in pl.spmd((padded_tokens // ROW_TILE * (HIDDEN // COL_TILE) - 0) // 1, name_hint='paired_projection', allow_early_resolve=True):
                    owner = 0 + block * 1
                    row = owner // (HIDDEN // COL_TILE) * ROW_TILE
                    column = owner % (HIDDEN // COL_TILE) * COL_TILE
                    key_acc = pl.create_tensor([ROW_TILE, COL_TILE], dtype=pl.FP32)
                    value_acc = pl.create_tensor([ROW_TILE, COL_TILE], dtype=pl.FP32)
                    for reduction in pl.pipeline(EMBED // RED_TILE, stage=2):
                        embedding_tile = workspace[row:row + ROW_TILE, reduction * RED_TILE:(reduction + 1) * RED_TILE]
                        key_weights = wk[reduction * RED_TILE:(reduction + 1) * RED_TILE, column:column + COL_TILE]
                        value_weights = wv[reduction * RED_TILE:(reduction + 1) * RED_TILE, column:column + COL_TILE]
                        key_acc = pl.matmul_acc(key_acc, embedding_tile, key_weights, init_cond=reduction == 0)
                        value_acc = pl.matmul_acc(value_acc, embedding_tile, value_weights, init_cond=reduction == 0)
                    key_workspace[row:row + ROW_TILE, column:column + COL_TILE] = key_acc
                    value_workspace[row:row + ROW_TILE, column:column + COL_TILE] = value_acc
                for block in pl.spmd((token_count % GATE_ROWS + GATE_ROWS - 1) // GATE_ROWS, name_hint='vector_tail_gate'):
                    tail_row = token_count // GATE_ROWS * GATE_ROWS
                    tail_valid_rows = token_count - tail_row
                    tail_vector_key_zero = pl.tile.full([16, GATE_TILE], dtype=pl.FP32, value=0.0)
                    tail_vector_query_zero = pl.tile.full([16, GATE_TILE], dtype=pl.FP32, value=0.0)
                    tail_vector_dot_zero = pl.tile.full([16, GATE_TILE], dtype=pl.FP32, value=0.0)
                    tail_vector_key_squares = pl.tile.set_validshape(tail_vector_key_zero, tail_valid_rows, GATE_TILE)
                    tail_vector_query_squares = pl.tile.set_validshape(tail_vector_query_zero, tail_valid_rows, GATE_TILE)
                    tail_vector_key_dot = pl.tile.set_validshape(tail_vector_dot_zero, tail_valid_rows, GATE_TILE)
                    for tail_vector_column in pl.range(HIDDEN // GATE_TILE):
                        tail_vector_key = pl.load(key_workspace, [tail_row, tail_vector_column * GATE_TILE], [16, GATE_TILE], valid_shape=[tail_valid_rows, GATE_TILE])
                        tail_vector_query = pl.tile.cast(pl.load(hidden, [tail_row, tail_vector_column * GATE_TILE], [16, GATE_TILE], valid_shape=[tail_valid_rows, GATE_TILE]), pl.FP32)
                        tail_vector_key_squares = pl.tile.add(tail_vector_key_squares, pl.tile.mul(tail_vector_key, tail_vector_key))
                        tail_vector_query_squares = pl.tile.add(tail_vector_query_squares, pl.tile.mul(tail_vector_query, tail_vector_query))
                        tail_vector_key_dot = pl.tile.add(tail_vector_key_dot, pl.tile.mul(tail_vector_key, tail_vector_query))
                    tail_vector_reduce_scratch = pl.create_tile([16, GATE_TILE], dtype=pl.FP32)
                    tail_vector_key_sum = pl.tile.row_sum(tail_vector_key_squares, tail_vector_reduce_scratch)
                    tail_vector_query_sum = pl.tile.row_sum(tail_vector_query_squares, tail_vector_reduce_scratch)
                    tail_vector_dot_sum = pl.tile.row_sum(tail_vector_key_dot, tail_vector_reduce_scratch)
                    tail_vector_ones = pl.tile.full([16, GATE_TILE], dtype=pl.FP32, value=1.0)
                    tail_vector_carrier = pl.tile.set_validshape(tail_vector_ones, tail_valid_rows, GATE_TILE)
                    tail_vector_key_expanded = pl.tile.row_expand_mul(tail_vector_carrier, tail_vector_key_sum)
                    tail_vector_query_expanded = pl.tile.row_expand_mul(tail_vector_carrier, tail_vector_query_sum)
                    tail_vector_dot_expanded = pl.tile.row_expand_mul(tail_vector_carrier, tail_vector_dot_sum)
                    tail_vector_rsqrt_scratch = pl.create_tile([16, GATE_TILE], dtype=pl.FP32)
                    tail_vector_key_scale = pl.tile.rsqrt(pl.tile.add(pl.tile.mul(tail_vector_key_expanded, 1.0 / HIDDEN), EPS), tail_vector_rsqrt_scratch)
                    tail_vector_query_scale = pl.tile.rsqrt(pl.tile.add(pl.tile.mul(tail_vector_query_expanded, 1.0 / HIDDEN), EPS), tail_vector_rsqrt_scratch)
                    tail_vector_argument = pl.tile.mul(pl.tile.mul(tail_vector_dot_expanded, pl.tile.mul(tail_vector_key_scale, tail_vector_query_scale)), HIDDEN ** (-0.5))
                    tail_vector_magnitude = pl.tile.sqrt(pl.tile.add(pl.tile.abs(tail_vector_argument), EPS))
                    tail_vector_negative_or_zero = pl.tile.select(pl.tile.cmps(tail_vector_argument, 0.0, cmp_type=2), pl.tile.neg(tail_vector_magnitude), 0.0)
                    tail_vector_signed_root = pl.tile.select(pl.tile.cmps(tail_vector_argument, 0.0, cmp_type=4), tail_vector_magnitude, tail_vector_negative_or_zero)
                    tail_vector_denominator = pl.tile.add(pl.tile.exp(pl.tile.neg(tail_vector_signed_root)), 1.0)
                    tail_vector_gate_estimate = pl.tile.recip(tail_vector_denominator)
                    tail_vector_twos = pl.tile.full([16, GATE_TILE], dtype=pl.FP32, value=2.0)
                    tail_vector_correction = pl.tile.sub(pl.tile.set_validshape(tail_vector_twos, tail_valid_rows, GATE_TILE), pl.tile.mul(tail_vector_denominator, tail_vector_gate_estimate))
                    tail_vector_gate = pl.tile.mul(tail_vector_gate_estimate, tail_vector_correction)
                    for tail_vector_column in pl.range(HIDDEN // GATE_TILE):
                        tail_vector_value = pl.load(value_workspace, [tail_row, tail_vector_column * GATE_TILE], [16, GATE_TILE], valid_shape=[tail_valid_rows, GATE_TILE])
                        tail_vector_result = pl.tile.mul(tail_vector_value, tail_vector_gate)
                        tail_vector_output = pl.tile.cast(tail_vector_result, pl.BF16)
                        out = pl.store(tail_vector_output, [tail_row, tail_vector_column * GATE_TILE], out)
                return out

        return FusedEngramBf16Region0Program
    elif tokens % GATE_ROWS == 0:
        @pl.program
        class FusedEngramBf16Region1Program:
            @pl.function(type=pl.FunctionType.Opaque)
            def fused_engram_bf16(self, hash_ids: pl.Tensor[[tokens, HEADS], pl.INT32], head_off: pl.Tensor[[1, HEADS], pl.INT32], table: pl.Tensor[[HEADS * VOCAB, HEAD_DIM], pl.BF16], wk: pl.Tensor[[EMBED, HIDDEN], pl.BF16], wv: pl.Tensor[[EMBED, HIDDEN], pl.BF16], hidden: pl.Tensor[[tokens, HIDDEN], pl.BF16], out: pl.Out[pl.Tensor[[tokens, HIDDEN], pl.BF16]]) -> pl.Tensor[[tokens, HIDDEN], pl.BF16]:
                token_count = tokens
                padded_tokens = (token_count + ROW_TILE - 1) // ROW_TILE * ROW_TILE
                workspace = pl.create_tensor([padded_tokens, EMBED], dtype=pl.BF16)
                for block in pl.spmd(padded_tokens // 16, name_hint='gather_embedding', allow_early_resolve=True):
                    base_token = block * 16
                    if base_token + 16 <= token_count:
                        local_ids = pl.load(hash_ids, [base_token, 0], [16, HEADS])
                        offsets = pl.load(head_off, [0, 0], [1, HEADS])
                        row_ids = pl.tile.col_expand_add(local_ids, offsets)
                        flat_rows = pl.reshape(row_ids, [1, 16 * HEADS])
                        bulk_rows = pl.tile.mgather(table, flat_rows, coalesce='row', gather_oob='undefined')
                        bulk_embedding = pl.reshape(bulk_rows, [16, EMBED])
                        workspace = pl.store(bulk_embedding, [base_token, 0], workspace)
                    else:
                        if base_token < token_count:
                            gather_valid_rows = pl.min(16, token_count - base_token)
                            partial_ids = pl.load(hash_ids, [base_token, 0], [16, HEADS], valid_shape=[gather_valid_rows, HEADS])
                            padded_ids = pl.tile.fillpad(partial_ids, pad_value=0)
                            partial_defined_ids = pl.tile.set_validshape(padded_ids, 16, HEADS)
                            partial_offsets = pl.load(head_off, [0, 0], [1, HEADS])
                            partial_row_ids = pl.tile.col_expand_add(partial_defined_ids, partial_offsets)
                            partial_flat_rows = pl.reshape(partial_row_ids, [1, 16 * HEADS])
                            partial_bulk_rows = pl.tile.mgather(table, partial_flat_rows, coalesce='row', gather_oob='undefined')
                            partial_embedding = pl.reshape(partial_bulk_rows, [16, EMBED])
                            workspace = pl.store(partial_embedding, [base_token, 0], workspace)
                        else:
                            zero_embedding = pl.tile.full([16, EMBED], dtype=pl.BF16, value=0.0)
                            workspace = pl.store(zero_embedding, [base_token, 0], workspace)
                key_workspace = pl.create_tensor([padded_tokens, HIDDEN], dtype=pl.FP32)
                value_workspace = pl.create_tensor([padded_tokens, HIDDEN], dtype=pl.FP32)
                for block in pl.spmd((padded_tokens // ROW_TILE * (HIDDEN // COL_TILE) - 0) // 1, name_hint='paired_projection', allow_early_resolve=True):
                    owner = 0 + block * 1
                    row = owner // (HIDDEN // COL_TILE) * ROW_TILE
                    column = owner % (HIDDEN // COL_TILE) * COL_TILE
                    key_acc = pl.create_tensor([ROW_TILE, COL_TILE], dtype=pl.FP32)
                    value_acc = pl.create_tensor([ROW_TILE, COL_TILE], dtype=pl.FP32)
                    for reduction in pl.pipeline(EMBED // RED_TILE, stage=2):
                        embedding_tile = workspace[row:row + ROW_TILE, reduction * RED_TILE:(reduction + 1) * RED_TILE]
                        key_weights = wk[reduction * RED_TILE:(reduction + 1) * RED_TILE, column:column + COL_TILE]
                        value_weights = wv[reduction * RED_TILE:(reduction + 1) * RED_TILE, column:column + COL_TILE]
                        key_acc = pl.matmul_acc(key_acc, embedding_tile, key_weights, init_cond=reduction == 0)
                        value_acc = pl.matmul_acc(value_acc, embedding_tile, value_weights, init_cond=reduction == 0)
                    key_workspace[row:row + ROW_TILE, column:column + COL_TILE] = key_acc
                    value_workspace[row:row + ROW_TILE, column:column + COL_TILE] = value_acc
                for block in pl.spmd((token_count // GATE_ROWS * GATE_ROWS - 0) // GATE_ROWS, name_hint='vector_gate'):
                    row = 0 + block * GATE_ROWS
                    vector_key_zero = pl.tile.full([16, GATE_TILE], dtype=pl.FP32, value=0.0)
                    vector_query_zero = pl.tile.full([16, GATE_TILE], dtype=pl.FP32, value=0.0)
                    vector_dot_zero = pl.tile.full([16, GATE_TILE], dtype=pl.FP32, value=0.0)
                    vector_key_squares = pl.tile.set_validshape(vector_key_zero, GATE_ROWS, GATE_TILE)
                    vector_query_squares = pl.tile.set_validshape(vector_query_zero, GATE_ROWS, GATE_TILE)
                    vector_key_dot = pl.tile.set_validshape(vector_dot_zero, GATE_ROWS, GATE_TILE)
                    for vector_column in pl.range(HIDDEN // GATE_TILE):
                        vector_key = pl.load(key_workspace, [row, vector_column * GATE_TILE], [16, GATE_TILE], valid_shape=[GATE_ROWS, GATE_TILE])
                        vector_query = pl.tile.cast(pl.load(hidden, [row, vector_column * GATE_TILE], [16, GATE_TILE], valid_shape=[GATE_ROWS, GATE_TILE]), pl.FP32)
                        vector_key_squares = pl.tile.add(vector_key_squares, pl.tile.mul(vector_key, vector_key))
                        vector_query_squares = pl.tile.add(vector_query_squares, pl.tile.mul(vector_query, vector_query))
                        vector_key_dot = pl.tile.add(vector_key_dot, pl.tile.mul(vector_key, vector_query))
                    vector_reduce_scratch = pl.create_tile([16, GATE_TILE], dtype=pl.FP32)
                    vector_key_sum = pl.tile.row_sum(vector_key_squares, vector_reduce_scratch)
                    vector_query_sum = pl.tile.row_sum(vector_query_squares, vector_reduce_scratch)
                    vector_dot_sum = pl.tile.row_sum(vector_key_dot, vector_reduce_scratch)
                    compact_key_sum = pl.reshape(vector_key_sum, [1, GATE_ROWS])
                    compact_query_sum = pl.reshape(vector_query_sum, [1, GATE_ROWS])
                    compact_dot_sum = pl.reshape(vector_dot_sum, [1, GATE_ROWS])
                    compact_scratch = pl.create_tile([1, GATE_ROWS], dtype=pl.FP32)
                    compact_key_scale = pl.tile.rsqrt(pl.tile.add(pl.tile.mul(compact_key_sum, 1.0 / HIDDEN), EPS), compact_scratch)
                    compact_query_scale = pl.tile.rsqrt(pl.tile.add(pl.tile.mul(compact_query_sum, 1.0 / HIDDEN), EPS), compact_scratch)
                    compact_argument = pl.tile.mul(pl.tile.mul(compact_dot_sum, pl.tile.mul(compact_key_scale, compact_query_scale)), HIDDEN ** (-0.5))
                    compact_magnitude = pl.tile.sqrt(pl.tile.add(pl.tile.abs(compact_argument), EPS))
                    compact_negative_or_zero = pl.tile.select(pl.tile.cmps(compact_argument, 0.0, cmp_type=2), pl.tile.neg(compact_magnitude), 0.0)
                    compact_signed_root = pl.tile.select(pl.tile.cmps(compact_argument, 0.0, cmp_type=4), compact_magnitude, compact_negative_or_zero)
                    compact_denominator = pl.tile.add(pl.tile.exp(pl.tile.neg(compact_signed_root)), 1.0)
                    compact_estimate = pl.tile.recip(compact_denominator)
                    compact_correction = pl.tile.sub(pl.tile.full([1, GATE_ROWS], dtype=pl.FP32, value=2.0), pl.tile.mul(compact_denominator, compact_estimate))
                    compact_gate = pl.tile.mul(compact_estimate, compact_correction)
                    gate_column = pl.reshape(compact_gate, [GATE_ROWS, 1])
                    for vector_column in pl.range(HIDDEN // GATE_TILE):
                        vector_value = pl.load(value_workspace, [row, vector_column * GATE_TILE], [GATE_ROWS, GATE_TILE])
                        vector_result = pl.tile.row_expand_mul(vector_value, gate_column)
                        vector_output = pl.tile.cast(vector_result, pl.BF16)
                        out = pl.store(vector_output, [row, vector_column * GATE_TILE], out)
                return out

        return FusedEngramBf16Region1Program
    else:
        @pl.program
        class FusedEngramBf16Region2Program:
            @pl.function(type=pl.FunctionType.Opaque)
            def fused_engram_bf16(self, hash_ids: pl.Tensor[[tokens, HEADS], pl.INT32], head_off: pl.Tensor[[1, HEADS], pl.INT32], table: pl.Tensor[[HEADS * VOCAB, HEAD_DIM], pl.BF16], wk: pl.Tensor[[EMBED, HIDDEN], pl.BF16], wv: pl.Tensor[[EMBED, HIDDEN], pl.BF16], hidden: pl.Tensor[[tokens, HIDDEN], pl.BF16], out: pl.Out[pl.Tensor[[tokens, HIDDEN], pl.BF16]]) -> pl.Tensor[[tokens, HIDDEN], pl.BF16]:
                token_count = tokens
                padded_tokens = (token_count + ROW_TILE - 1) // ROW_TILE * ROW_TILE
                workspace = pl.create_tensor([padded_tokens, EMBED], dtype=pl.BF16)
                for block in pl.spmd(padded_tokens // 16, name_hint='gather_embedding', allow_early_resolve=True):
                    base_token = block * 16
                    if base_token + 16 <= token_count:
                        local_ids = pl.load(hash_ids, [base_token, 0], [16, HEADS])
                        offsets = pl.load(head_off, [0, 0], [1, HEADS])
                        row_ids = pl.tile.col_expand_add(local_ids, offsets)
                        flat_rows = pl.reshape(row_ids, [1, 16 * HEADS])
                        bulk_rows = pl.tile.mgather(table, flat_rows, coalesce='row', gather_oob='undefined')
                        bulk_embedding = pl.reshape(bulk_rows, [16, EMBED])
                        workspace = pl.store(bulk_embedding, [base_token, 0], workspace)
                    else:
                        if base_token < token_count:
                            gather_valid_rows = pl.min(16, token_count - base_token)
                            partial_ids = pl.load(hash_ids, [base_token, 0], [16, HEADS], valid_shape=[gather_valid_rows, HEADS])
                            padded_ids = pl.tile.fillpad(partial_ids, pad_value=0)
                            partial_defined_ids = pl.tile.set_validshape(padded_ids, 16, HEADS)
                            partial_offsets = pl.load(head_off, [0, 0], [1, HEADS])
                            partial_row_ids = pl.tile.col_expand_add(partial_defined_ids, partial_offsets)
                            partial_flat_rows = pl.reshape(partial_row_ids, [1, 16 * HEADS])
                            partial_bulk_rows = pl.tile.mgather(table, partial_flat_rows, coalesce='row', gather_oob='undefined')
                            partial_embedding = pl.reshape(partial_bulk_rows, [16, EMBED])
                            workspace = pl.store(partial_embedding, [base_token, 0], workspace)
                        else:
                            zero_embedding = pl.tile.full([16, EMBED], dtype=pl.BF16, value=0.0)
                            workspace = pl.store(zero_embedding, [base_token, 0], workspace)
                key_workspace = pl.create_tensor([padded_tokens, HIDDEN], dtype=pl.FP32)
                value_workspace = pl.create_tensor([padded_tokens, HIDDEN], dtype=pl.FP32)
                for block in pl.spmd((padded_tokens // ROW_TILE * (HIDDEN // COL_TILE) - 0) // 1, name_hint='paired_projection', allow_early_resolve=True):
                    owner = 0 + block * 1
                    row = owner // (HIDDEN // COL_TILE) * ROW_TILE
                    column = owner % (HIDDEN // COL_TILE) * COL_TILE
                    key_acc = pl.create_tensor([ROW_TILE, COL_TILE], dtype=pl.FP32)
                    value_acc = pl.create_tensor([ROW_TILE, COL_TILE], dtype=pl.FP32)
                    for reduction in pl.pipeline(EMBED // RED_TILE, stage=2):
                        embedding_tile = workspace[row:row + ROW_TILE, reduction * RED_TILE:(reduction + 1) * RED_TILE]
                        key_weights = wk[reduction * RED_TILE:(reduction + 1) * RED_TILE, column:column + COL_TILE]
                        value_weights = wv[reduction * RED_TILE:(reduction + 1) * RED_TILE, column:column + COL_TILE]
                        key_acc = pl.matmul_acc(key_acc, embedding_tile, key_weights, init_cond=reduction == 0)
                        value_acc = pl.matmul_acc(value_acc, embedding_tile, value_weights, init_cond=reduction == 0)
                    key_workspace[row:row + ROW_TILE, column:column + COL_TILE] = key_acc
                    value_workspace[row:row + ROW_TILE, column:column + COL_TILE] = value_acc
                for block in pl.spmd((token_count // GATE_ROWS * GATE_ROWS - 0) // GATE_ROWS, name_hint='vector_gate'):
                    row = 0 + block * GATE_ROWS
                    vector_key_zero = pl.tile.full([16, GATE_TILE], dtype=pl.FP32, value=0.0)
                    vector_query_zero = pl.tile.full([16, GATE_TILE], dtype=pl.FP32, value=0.0)
                    vector_dot_zero = pl.tile.full([16, GATE_TILE], dtype=pl.FP32, value=0.0)
                    vector_key_squares = pl.tile.set_validshape(vector_key_zero, GATE_ROWS, GATE_TILE)
                    vector_query_squares = pl.tile.set_validshape(vector_query_zero, GATE_ROWS, GATE_TILE)
                    vector_key_dot = pl.tile.set_validshape(vector_dot_zero, GATE_ROWS, GATE_TILE)
                    for vector_column in pl.range(HIDDEN // GATE_TILE):
                        vector_key = pl.load(key_workspace, [row, vector_column * GATE_TILE], [16, GATE_TILE], valid_shape=[GATE_ROWS, GATE_TILE])
                        vector_query = pl.tile.cast(pl.load(hidden, [row, vector_column * GATE_TILE], [16, GATE_TILE], valid_shape=[GATE_ROWS, GATE_TILE]), pl.FP32)
                        vector_key_squares = pl.tile.add(vector_key_squares, pl.tile.mul(vector_key, vector_key))
                        vector_query_squares = pl.tile.add(vector_query_squares, pl.tile.mul(vector_query, vector_query))
                        vector_key_dot = pl.tile.add(vector_key_dot, pl.tile.mul(vector_key, vector_query))
                    vector_reduce_scratch = pl.create_tile([16, GATE_TILE], dtype=pl.FP32)
                    vector_key_sum = pl.tile.row_sum(vector_key_squares, vector_reduce_scratch)
                    vector_query_sum = pl.tile.row_sum(vector_query_squares, vector_reduce_scratch)
                    vector_dot_sum = pl.tile.row_sum(vector_key_dot, vector_reduce_scratch)
                    compact_key_sum = pl.reshape(vector_key_sum, [1, GATE_ROWS])
                    compact_query_sum = pl.reshape(vector_query_sum, [1, GATE_ROWS])
                    compact_dot_sum = pl.reshape(vector_dot_sum, [1, GATE_ROWS])
                    compact_scratch = pl.create_tile([1, GATE_ROWS], dtype=pl.FP32)
                    compact_key_scale = pl.tile.rsqrt(pl.tile.add(pl.tile.mul(compact_key_sum, 1.0 / HIDDEN), EPS), compact_scratch)
                    compact_query_scale = pl.tile.rsqrt(pl.tile.add(pl.tile.mul(compact_query_sum, 1.0 / HIDDEN), EPS), compact_scratch)
                    compact_argument = pl.tile.mul(pl.tile.mul(compact_dot_sum, pl.tile.mul(compact_key_scale, compact_query_scale)), HIDDEN ** (-0.5))
                    compact_magnitude = pl.tile.sqrt(pl.tile.add(pl.tile.abs(compact_argument), EPS))
                    compact_negative_or_zero = pl.tile.select(pl.tile.cmps(compact_argument, 0.0, cmp_type=2), pl.tile.neg(compact_magnitude), 0.0)
                    compact_signed_root = pl.tile.select(pl.tile.cmps(compact_argument, 0.0, cmp_type=4), compact_magnitude, compact_negative_or_zero)
                    compact_denominator = pl.tile.add(pl.tile.exp(pl.tile.neg(compact_signed_root)), 1.0)
                    compact_estimate = pl.tile.recip(compact_denominator)
                    compact_correction = pl.tile.sub(pl.tile.full([1, GATE_ROWS], dtype=pl.FP32, value=2.0), pl.tile.mul(compact_denominator, compact_estimate))
                    compact_gate = pl.tile.mul(compact_estimate, compact_correction)
                    gate_column = pl.reshape(compact_gate, [GATE_ROWS, 1])
                    for vector_column in pl.range(HIDDEN // GATE_TILE):
                        vector_value = pl.load(value_workspace, [row, vector_column * GATE_TILE], [GATE_ROWS, GATE_TILE])
                        vector_result = pl.tile.row_expand_mul(vector_value, gate_column)
                        vector_output = pl.tile.cast(vector_result, pl.BF16)
                        out = pl.store(vector_output, [row, vector_column * GATE_TILE], out)
                for block in pl.spmd((token_count % GATE_ROWS + GATE_ROWS - 1) // GATE_ROWS, name_hint='vector_tail_gate'):
                    tail_row = token_count // GATE_ROWS * GATE_ROWS
                    tail_valid_rows = token_count - tail_row
                    tail_vector_key_zero = pl.tile.full([16, GATE_TILE], dtype=pl.FP32, value=0.0)
                    tail_vector_query_zero = pl.tile.full([16, GATE_TILE], dtype=pl.FP32, value=0.0)
                    tail_vector_dot_zero = pl.tile.full([16, GATE_TILE], dtype=pl.FP32, value=0.0)
                    tail_vector_key_squares = pl.tile.set_validshape(tail_vector_key_zero, tail_valid_rows, GATE_TILE)
                    tail_vector_query_squares = pl.tile.set_validshape(tail_vector_query_zero, tail_valid_rows, GATE_TILE)
                    tail_vector_key_dot = pl.tile.set_validshape(tail_vector_dot_zero, tail_valid_rows, GATE_TILE)
                    for tail_vector_column in pl.range(HIDDEN // GATE_TILE):
                        tail_vector_key = pl.load(key_workspace, [tail_row, tail_vector_column * GATE_TILE], [16, GATE_TILE], valid_shape=[tail_valid_rows, GATE_TILE])
                        tail_vector_query = pl.tile.cast(pl.load(hidden, [tail_row, tail_vector_column * GATE_TILE], [16, GATE_TILE], valid_shape=[tail_valid_rows, GATE_TILE]), pl.FP32)
                        tail_vector_key_squares = pl.tile.add(tail_vector_key_squares, pl.tile.mul(tail_vector_key, tail_vector_key))
                        tail_vector_query_squares = pl.tile.add(tail_vector_query_squares, pl.tile.mul(tail_vector_query, tail_vector_query))
                        tail_vector_key_dot = pl.tile.add(tail_vector_key_dot, pl.tile.mul(tail_vector_key, tail_vector_query))
                    tail_vector_reduce_scratch = pl.create_tile([16, GATE_TILE], dtype=pl.FP32)
                    tail_vector_key_sum = pl.tile.row_sum(tail_vector_key_squares, tail_vector_reduce_scratch)
                    tail_vector_query_sum = pl.tile.row_sum(tail_vector_query_squares, tail_vector_reduce_scratch)
                    tail_vector_dot_sum = pl.tile.row_sum(tail_vector_key_dot, tail_vector_reduce_scratch)
                    tail_vector_ones = pl.tile.full([16, GATE_TILE], dtype=pl.FP32, value=1.0)
                    tail_vector_carrier = pl.tile.set_validshape(tail_vector_ones, tail_valid_rows, GATE_TILE)
                    tail_vector_key_expanded = pl.tile.row_expand_mul(tail_vector_carrier, tail_vector_key_sum)
                    tail_vector_query_expanded = pl.tile.row_expand_mul(tail_vector_carrier, tail_vector_query_sum)
                    tail_vector_dot_expanded = pl.tile.row_expand_mul(tail_vector_carrier, tail_vector_dot_sum)
                    tail_vector_rsqrt_scratch = pl.create_tile([16, GATE_TILE], dtype=pl.FP32)
                    tail_vector_key_scale = pl.tile.rsqrt(pl.tile.add(pl.tile.mul(tail_vector_key_expanded, 1.0 / HIDDEN), EPS), tail_vector_rsqrt_scratch)
                    tail_vector_query_scale = pl.tile.rsqrt(pl.tile.add(pl.tile.mul(tail_vector_query_expanded, 1.0 / HIDDEN), EPS), tail_vector_rsqrt_scratch)
                    tail_vector_argument = pl.tile.mul(pl.tile.mul(tail_vector_dot_expanded, pl.tile.mul(tail_vector_key_scale, tail_vector_query_scale)), HIDDEN ** (-0.5))
                    tail_vector_magnitude = pl.tile.sqrt(pl.tile.add(pl.tile.abs(tail_vector_argument), EPS))
                    tail_vector_negative_or_zero = pl.tile.select(pl.tile.cmps(tail_vector_argument, 0.0, cmp_type=2), pl.tile.neg(tail_vector_magnitude), 0.0)
                    tail_vector_signed_root = pl.tile.select(pl.tile.cmps(tail_vector_argument, 0.0, cmp_type=4), tail_vector_magnitude, tail_vector_negative_or_zero)
                    tail_vector_denominator = pl.tile.add(pl.tile.exp(pl.tile.neg(tail_vector_signed_root)), 1.0)
                    tail_vector_gate_estimate = pl.tile.recip(tail_vector_denominator)
                    tail_vector_twos = pl.tile.full([16, GATE_TILE], dtype=pl.FP32, value=2.0)
                    tail_vector_correction = pl.tile.sub(pl.tile.set_validshape(tail_vector_twos, tail_valid_rows, GATE_TILE), pl.tile.mul(tail_vector_denominator, tail_vector_gate_estimate))
                    tail_vector_gate = pl.tile.mul(tail_vector_gate_estimate, tail_vector_correction)
                    for tail_vector_column in pl.range(HIDDEN // GATE_TILE):
                        tail_vector_value = pl.load(value_workspace, [tail_row, tail_vector_column * GATE_TILE], [16, GATE_TILE], valid_shape=[tail_valid_rows, GATE_TILE])
                        tail_vector_result = pl.tile.mul(tail_vector_value, tail_vector_gate)
                        tail_vector_output = pl.tile.cast(tail_vector_result, pl.BF16)
                        out = pl.store(tail_vector_output, [tail_row, tail_vector_column * GATE_TILE], out)
                return out

        return FusedEngramBf16Region2Program

def make_chunked_engram_bf16(tokens):
    @pl.program
    class ChunkedEngramBf16Region0Program:
        @pl.function(type=pl.FunctionType.Opaque)
        def chunked_engram_bf16(self, hash_ids: pl.Tensor[[tokens, HEADS], pl.INT32], head_off: pl.Tensor[[1, HEADS], pl.INT32], table: pl.Tensor[[HEADS * VOCAB, HEAD_DIM], pl.BF16], wk: pl.Tensor[[EMBED, HIDDEN], pl.BF16], wv: pl.Tensor[[EMBED, HIDDEN], pl.BF16], hidden: pl.Tensor[[tokens, HIDDEN], pl.BF16], out: pl.Out[pl.Tensor[[tokens, HIDDEN], pl.BF16]]) -> pl.Tensor[[tokens, HIDDEN], pl.BF16]:
            token_count = tokens
            full_chunks = (token_count + 127) // 128
            for full_chunk_index in pl.range(full_chunks):
                full_start = full_chunk_index * 128
                full_valid_rows = pl.min(128, token_count - full_start)
                full_ids = hash_ids[full_start:full_start + full_valid_rows, 0:HEADS]
                full_hidden = hidden[full_start:full_start + full_valid_rows, 0:HIDDEN]
                full_out = out[full_start:full_start + full_valid_rows, 0:HIDDEN]
                full_embedding = pl.create_tensor([128, EMBED], dtype=pl.BF16)
                for block in pl.spmd(128 // 8, name_hint='gather_embedding', allow_early_resolve=True):
                    task_start = block * 8
                    for gather_batch in pl.unroll(1):
                        base_token = task_start + gather_batch * 8
                        if base_token < full_valid_rows:
                            gather_valid_rows = pl.min(8, full_valid_rows - base_token)
                            local_ids = pl.load(full_ids, [base_token, 0], [8, HEADS], valid_shape=[gather_valid_rows, HEADS])
                            padded_ids = pl.tile.fillpad(local_ids, pad_value=0)
                            defined_ids = pl.tile.set_validshape(padded_ids, 8, HEADS)
                            offsets = pl.load(head_off, [0, 0], [1, HEADS])
                            row_ids = pl.tile.col_expand_add(defined_ids, offsets)
                            flat_rows = pl.reshape(row_ids, [1, 8 * HEADS])
                            bulk_rows = pl.tile.mgather(table, flat_rows, coalesce='row', gather_oob='undefined')
                            bulk_embedding = pl.reshape(bulk_rows, [8, EMBED])
                            full_embedding = pl.store(bulk_embedding, [base_token, 0], full_embedding)
                        else:
                            zero_embedding = pl.tile.full([8, EMBED], dtype=pl.BF16, value=0.0)
                            full_embedding = pl.store(zero_embedding, [base_token, 0], full_embedding)
                full_key = pl.create_tensor([128, HIDDEN], dtype=pl.FP32)
                full_value = pl.create_tensor([128, HIDDEN], dtype=pl.FP32)
                for block in pl.spmd((128 // ROW_TILE * (HIDDEN // COL_TILE) - 0) // 1, name_hint='paired_projection', allow_early_resolve=True):
                    owner = 0 + block * 1
                    row = owner // (HIDDEN // COL_TILE) * ROW_TILE
                    column = owner % (HIDDEN // COL_TILE) * COL_TILE
                    key_acc = pl.create_tensor([ROW_TILE, COL_TILE], dtype=pl.FP32)
                    value_acc = pl.create_tensor([ROW_TILE, COL_TILE], dtype=pl.FP32)
                    for reduction in pl.pipeline(EMBED // RED_TILE, stage=2):
                        embedding_tile = full_embedding[row:row + ROW_TILE, reduction * RED_TILE:(reduction + 1) * RED_TILE]
                        key_weights = wk[reduction * RED_TILE:(reduction + 1) * RED_TILE, column:column + COL_TILE]
                        value_weights = wv[reduction * RED_TILE:(reduction + 1) * RED_TILE, column:column + COL_TILE]
                        key_acc = pl.matmul_acc(key_acc, embedding_tile, key_weights, init_cond=reduction == 0)
                        value_acc = pl.matmul_acc(value_acc, embedding_tile, value_weights, init_cond=reduction == 0)
                    full_key[row:row + ROW_TILE, column:column + COL_TILE] = key_acc
                    full_value[row:row + ROW_TILE, column:column + COL_TILE] = value_acc
                for block in pl.spmd((full_valid_rows + GATE_ROWS - 1) // GATE_ROWS, name_hint='vector_gate'):
                    row = block * GATE_ROWS
                    gate_valid_rows = pl.min(GATE_ROWS, full_valid_rows - row)
                    if gate_valid_rows == GATE_ROWS:
                        vector_key_zero = pl.tile.full([16, GATE_TILE], dtype=pl.FP32, value=0.0)
                        vector_query_zero = pl.tile.full([16, GATE_TILE], dtype=pl.FP32, value=0.0)
                        vector_dot_zero = pl.tile.full([16, GATE_TILE], dtype=pl.FP32, value=0.0)
                        vector_key_squares = pl.tile.set_validshape(vector_key_zero, GATE_ROWS, GATE_TILE)
                        vector_query_squares = pl.tile.set_validshape(vector_query_zero, GATE_ROWS, GATE_TILE)
                        vector_key_dot = pl.tile.set_validshape(vector_dot_zero, GATE_ROWS, GATE_TILE)
                        for vector_column in pl.range(HIDDEN // GATE_TILE):
                            vector_key = pl.load(full_key, [row, vector_column * GATE_TILE], [16, GATE_TILE], valid_shape=[GATE_ROWS, GATE_TILE])
                            vector_query = pl.tile.cast(pl.load(full_hidden, [row, vector_column * GATE_TILE], [16, GATE_TILE], valid_shape=[GATE_ROWS, GATE_TILE]), pl.FP32)
                            vector_key_squares = pl.tile.add(vector_key_squares, pl.tile.mul(vector_key, vector_key))
                            vector_query_squares = pl.tile.add(vector_query_squares, pl.tile.mul(vector_query, vector_query))
                            vector_key_dot = pl.tile.add(vector_key_dot, pl.tile.mul(vector_key, vector_query))
                        vector_reduce_scratch = pl.create_tile([16, GATE_TILE], dtype=pl.FP32)
                        vector_key_sum = pl.tile.row_sum(vector_key_squares, vector_reduce_scratch)
                        vector_query_sum = pl.tile.row_sum(vector_query_squares, vector_reduce_scratch)
                        vector_dot_sum = pl.tile.row_sum(vector_key_dot, vector_reduce_scratch)
                        compact_key_sum = pl.reshape(vector_key_sum, [1, GATE_ROWS])
                        compact_query_sum = pl.reshape(vector_query_sum, [1, GATE_ROWS])
                        compact_dot_sum = pl.reshape(vector_dot_sum, [1, GATE_ROWS])
                        compact_scratch = pl.create_tile([1, GATE_ROWS], dtype=pl.FP32)
                        compact_key_scale = pl.tile.rsqrt(pl.tile.add(pl.tile.mul(compact_key_sum, 1.0 / HIDDEN), EPS), compact_scratch)
                        compact_query_scale = pl.tile.rsqrt(pl.tile.add(pl.tile.mul(compact_query_sum, 1.0 / HIDDEN), EPS), compact_scratch)
                        compact_argument = pl.tile.mul(pl.tile.mul(compact_dot_sum, pl.tile.mul(compact_key_scale, compact_query_scale)), HIDDEN ** (-0.5))
                        compact_magnitude = pl.tile.sqrt(pl.tile.add(pl.tile.abs(compact_argument), EPS))
                        compact_negative_or_zero = pl.tile.select(pl.tile.cmps(compact_argument, 0.0, cmp_type=2), pl.tile.neg(compact_magnitude), 0.0)
                        compact_signed_root = pl.tile.select(pl.tile.cmps(compact_argument, 0.0, cmp_type=4), compact_magnitude, compact_negative_or_zero)
                        compact_denominator = pl.tile.add(pl.tile.exp(pl.tile.neg(compact_signed_root)), 1.0)
                        compact_estimate = pl.tile.recip(compact_denominator)
                        compact_correction = pl.tile.sub(pl.tile.full([1, GATE_ROWS], dtype=pl.FP32, value=2.0), pl.tile.mul(compact_denominator, compact_estimate))
                        compact_gate = pl.tile.mul(compact_estimate, compact_correction)
                        gate_column = pl.reshape(compact_gate, [GATE_ROWS, 1])
                        for vector_column in pl.range(HIDDEN // GATE_TILE):
                            vector_value = pl.load(full_value, [row, vector_column * GATE_TILE], [GATE_ROWS, GATE_TILE])
                            vector_result = pl.tile.row_expand_mul(vector_value, gate_column)
                            vector_output = pl.tile.cast(vector_result, pl.BF16)
                            full_out = pl.store(vector_output, [row, vector_column * GATE_TILE], full_out)
                    else:
                        tail_vector_key_zero = pl.tile.full([16, GATE_TILE], dtype=pl.FP32, value=0.0)
                        tail_vector_query_zero = pl.tile.full([16, GATE_TILE], dtype=pl.FP32, value=0.0)
                        tail_vector_dot_zero = pl.tile.full([16, GATE_TILE], dtype=pl.FP32, value=0.0)
                        tail_vector_key_squares = pl.tile.set_validshape(tail_vector_key_zero, gate_valid_rows, GATE_TILE)
                        tail_vector_query_squares = pl.tile.set_validshape(tail_vector_query_zero, gate_valid_rows, GATE_TILE)
                        tail_vector_key_dot = pl.tile.set_validshape(tail_vector_dot_zero, gate_valid_rows, GATE_TILE)
                        for tail_vector_column in pl.range(HIDDEN // GATE_TILE):
                            tail_vector_key = pl.load(full_key, [row, tail_vector_column * GATE_TILE], [16, GATE_TILE], valid_shape=[gate_valid_rows, GATE_TILE])
                            tail_vector_query = pl.tile.cast(pl.load(full_hidden, [row, tail_vector_column * GATE_TILE], [16, GATE_TILE], valid_shape=[gate_valid_rows, GATE_TILE]), pl.FP32)
                            tail_vector_key_squares = pl.tile.add(tail_vector_key_squares, pl.tile.mul(tail_vector_key, tail_vector_key))
                            tail_vector_query_squares = pl.tile.add(tail_vector_query_squares, pl.tile.mul(tail_vector_query, tail_vector_query))
                            tail_vector_key_dot = pl.tile.add(tail_vector_key_dot, pl.tile.mul(tail_vector_key, tail_vector_query))
                        tail_vector_reduce_scratch = pl.create_tile([16, GATE_TILE], dtype=pl.FP32)
                        tail_vector_key_sum = pl.tile.row_sum(tail_vector_key_squares, tail_vector_reduce_scratch)
                        tail_vector_query_sum = pl.tile.row_sum(tail_vector_query_squares, tail_vector_reduce_scratch)
                        tail_vector_dot_sum = pl.tile.row_sum(tail_vector_key_dot, tail_vector_reduce_scratch)
                        tail_vector_ones = pl.tile.full([16, GATE_TILE], dtype=pl.FP32, value=1.0)
                        tail_vector_carrier = pl.tile.set_validshape(tail_vector_ones, gate_valid_rows, GATE_TILE)
                        tail_vector_key_expanded = pl.tile.row_expand_mul(tail_vector_carrier, tail_vector_key_sum)
                        tail_vector_query_expanded = pl.tile.row_expand_mul(tail_vector_carrier, tail_vector_query_sum)
                        tail_vector_dot_expanded = pl.tile.row_expand_mul(tail_vector_carrier, tail_vector_dot_sum)
                        tail_vector_rsqrt_scratch = pl.create_tile([16, GATE_TILE], dtype=pl.FP32)
                        tail_vector_key_scale = pl.tile.rsqrt(pl.tile.add(pl.tile.mul(tail_vector_key_expanded, 1.0 / HIDDEN), EPS), tail_vector_rsqrt_scratch)
                        tail_vector_query_scale = pl.tile.rsqrt(pl.tile.add(pl.tile.mul(tail_vector_query_expanded, 1.0 / HIDDEN), EPS), tail_vector_rsqrt_scratch)
                        tail_vector_argument = pl.tile.mul(pl.tile.mul(tail_vector_dot_expanded, pl.tile.mul(tail_vector_key_scale, tail_vector_query_scale)), HIDDEN ** (-0.5))
                        tail_vector_magnitude = pl.tile.sqrt(pl.tile.add(pl.tile.abs(tail_vector_argument), EPS))
                        tail_vector_negative_or_zero = pl.tile.select(pl.tile.cmps(tail_vector_argument, 0.0, cmp_type=2), pl.tile.neg(tail_vector_magnitude), 0.0)
                        tail_vector_signed_root = pl.tile.select(pl.tile.cmps(tail_vector_argument, 0.0, cmp_type=4), tail_vector_magnitude, tail_vector_negative_or_zero)
                        tail_vector_denominator = pl.tile.add(pl.tile.exp(pl.tile.neg(tail_vector_signed_root)), 1.0)
                        tail_vector_gate_estimate = pl.tile.recip(tail_vector_denominator)
                        tail_vector_twos = pl.tile.full([16, GATE_TILE], dtype=pl.FP32, value=2.0)
                        tail_vector_correction = pl.tile.sub(pl.tile.set_validshape(tail_vector_twos, gate_valid_rows, GATE_TILE), pl.tile.mul(tail_vector_denominator, tail_vector_gate_estimate))
                        tail_vector_gate = pl.tile.mul(tail_vector_gate_estimate, tail_vector_correction)
                        for tail_vector_column in pl.range(HIDDEN // GATE_TILE):
                            tail_vector_value = pl.load(full_value, [row, tail_vector_column * GATE_TILE], [16, GATE_TILE], valid_shape=[gate_valid_rows, GATE_TILE])
                            tail_vector_result = pl.tile.mul(tail_vector_value, tail_vector_gate)
                            tail_vector_output = pl.tile.cast(tail_vector_result, pl.BF16)
                            full_out = pl.store(tail_vector_output, [row, tail_vector_column * GATE_TILE], full_out)
            return out

    return ChunkedEngramBf16Region0Program

def build(shape):
    import torch
    from golden import TensorSpec
    tokens = shape['T']
    dtype = shape.get('dtype', 'fp32')
    if dtype not in ('fp32', 'bf16'):
        raise ValueError(f'unsupported dtype: {dtype}')
    storage = torch.bfloat16 if dtype == 'bf16' else torch.float32

    def random_scaled(*dimensions, scale=0.02):
        return lambda: (torch.randn(*dimensions) * scale).to(storage)
    specs = [TensorSpec('hash_ids', [tokens, HEADS], torch.int32, init_value=lambda: torch.randint(0, VOCAB, (tokens, HEADS), dtype=torch.int32)), TensorSpec('head_off', [1, HEADS], torch.int32, init_value=lambda: (torch.arange(HEADS, dtype=torch.int32) * VOCAB).view(1, HEADS)), TensorSpec('table', [HEADS * VOCAB, HEAD_DIM], storage, init_value=random_scaled(HEADS * VOCAB, HEAD_DIM, scale=0.5)), TensorSpec('wk', [EMBED, HIDDEN], storage, init_value=random_scaled(EMBED, HIDDEN)), TensorSpec('wv', [EMBED, HIDDEN], storage, init_value=random_scaled(EMBED, HIDDEN)), TensorSpec('hidden', [tokens, HIDDEN], storage, init_value=random_scaled(tokens, HIDDEN, scale=0.5)), TensorSpec('out', [tokens, HIDDEN], storage)]


    def golden(tensors):
        rows = tensors['hash_ids'].long() + tensors['head_off'].long()
        embedding = tensors['table'].float()[rows].reshape(tokens, EMBED)
        key = embedding @ tensors['wk'].float()
        value = embedding @ tensors['wv'].float()
        query = tensors['hidden'].float()
        normalized_key = key * torch.rsqrt(key.square().mean(-1, keepdim=True) + EPS)
        normalized_query = query * torch.rsqrt(query.square().mean(-1, keepdim=True) + EPS)
        gate_argument = (normalized_key * normalized_query).sum(-1, keepdim=True) / HIDDEN ** 0.5
        gate = torch.sigmoid(torch.sign(gate_argument) * torch.sqrt(torch.abs(gate_argument) + EPS))
        tensors['out'][:] = (gate * value).to(tensors['out'].dtype)
    if dtype == 'bf16':
        return (make_fused_engram_bf16(tokens) if tokens <= ROW_TILE else make_chunked_engram_bf16(tokens), specs, golden, 0.02, 0.02)
    return (make_fused_engram_fp32(tokens), specs, golden, 0.0001, 0.0001)
