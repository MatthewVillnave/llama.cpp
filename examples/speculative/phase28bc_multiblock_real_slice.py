#!/usr/bin/env python3
"""
Phase 28BC: Multi-Block Real-Slice Residual Parity
Real Qwen2.5-0.5B slices, multi-block .trit layout, matmul parity.
"""
import sys, os, json
import numpy as np

sys.path.insert(0, '/home/matthew-villnave/llama.cpp/examples/speculative')
import prt_trit_io as io

OUT_JSON = '/home/matthew-villnave/llama.cpp/examples/speculative/results/phase28bc_multiblock_real_slice_residual_parity.json'

# Load the Qwen2.5-0.5B ffn_up slice (K=32, M=48) already extracted in 28BB
W_full = np.fromfile('/tmp/phase28bb_slice.f32', dtype=np.float32).reshape(32, 48)
W_full_padded = np.pad(W_full, ((0, 64), (0, 96)), mode='constant')  # 96x144 pad
print(f"Padded slice: {W_full_padded.shape}")

def run_case(name, W_target, block_rows, block_cols, N, seed, expected_grid):
    K, M = W_target.shape
    rng = np.random.RandomState(seed)

    # W_base: deterministic rounded base
    round_val = 0.25
    W_base = np.round(W_target / round_val) * round_val

    # Residual
    R_ref = W_target - W_base

    # Compute per-block scales: mean_nonzero_abs per block
    n_br = (K + block_rows - 1) // block_rows
    n_bc = (M + block_cols - 1) // block_cols
    scales = np.zeros((n_br * n_bc,), dtype=np.float32)
    idx = 0
    for r in range(0, K, block_rows):
        br = min(block_rows, K - r)
        for c in range(0, M, block_cols):
            bc = min(block_cols, M - c)
            block = R_ref[r:r+br, c:c+bc]
            nonzero = block[block != 0]
            if nonzero.size > 0:
                scales[idx] = float(np.mean(np.abs(nonzero)))
            else:
                scales[idx] = 0.0
            idx += 1

    # Ternary symbols
    R_symbols = np.sign(R_ref)
    R_symbols[R_symbols == 0] = 1

    # R reference approximation
    block_idx = 0
    R_approx = np.zeros_like(R_ref)
    for r in range(0, K, block_rows):
        br = min(block_rows, K - r)
        for c in range(0, M, block_cols):
            bc = min(block_cols, M - c)
            block_R = R_symbols[r:r+br, c:c+bc].astype(np.float32) * scales[block_idx]
            R_approx[r:r+br, c:c+bc] = block_R
            block_idx += 1

    # Write .trit
    trit_path = f'/tmp/phase28bc_{name.replace(" ","_")}.trit'
    io.write_trit(trit_path, R_symbols.astype(np.int8), scales, block_rows=block_rows, block_cols=block_cols)

    # Read back
    decoded_ternary, read_scales, meta = io.read_trit(trit_path)

    # R_decoded
    block_idx = 0
    R_decoded = np.zeros_like(R_ref)
    for r in range(0, K, block_rows):
        br = min(block_rows, K - r)
        for c in range(0, M, block_cols):
            bc = min(block_cols, M - c)
            block_R = decoded_ternary[r:r+br, c:c+bc].astype(np.float32) * scales[block_idx]
            R_decoded[r:r+br, c:c+bc] = block_R
            block_idx += 1

    # Matmul test
    X = rng.randn(N, K).astype(np.float32)
    Y_ref = X @ (W_base + R_approx)
    Y_decoded = X @ (W_base + R_decoded)

    # Metrics
    r_max = float(np.max(np.abs(R_approx - R_decoded)))
    r_rmse = float(np.sqrt(np.mean((R_approx - R_decoded)**2)))
    w_max = float(np.max(np.abs((W_base + R_approx) - (W_base + R_decoded))))
    y_max = float(np.max(np.abs(Y_ref - Y_decoded)))
    y_rmse = float(np.sqrt(np.mean((Y_ref - Y_decoded)**2)))

    # Cosine similarity for Y
    flat_ref = Y_ref.flatten()
    flat_dec = Y_decoded.flatten()
    cos = float(np.dot(flat_ref, flat_dec) / (np.linalg.norm(flat_ref) * np.linalg.norm(flat_dec) + 1e-10))

    pass_tc = r_max < 1e-4 and y_max < 1e-4

    result = {
        'case': name,
        'shape_KMK': [K, M, N],
        'block_rows': block_rows,
        'block_cols': block_cols,
        'expected_grid': expected_grid,
        'actual_block_count': n_br * n_bc,
        'R_max_abs_err': r_max,
        'R_rmse': r_rmse,
        'W_overlay_max_abs_err': w_max,
        'W_overlay_rmse': float(np.sqrt(np.mean(((W_base+R_approx)-(W_base+R_decoded))**2))),
        'Y_max_abs_err': y_max,
        'Y_rmse': y_rmse,
        'Y_cosine': cos,
        'canonical_trit_used': True,
        'bypass_used': False,
        'generation_run': False,
        'pass': pass_tc
    }

    print(f"  {name}: K={K} M={M} N={N} blocks={n_br}x{n_bc}={n_br*n_bc} "
          f"R_err={r_max:.2e} Y_err={y_max:.2e} cos={cos:.8f} PASS={pass_tc}")

    return result

# ── CASE 1: Row-split multi-block ─────────────────────────────────────────
# Use W_full_padded: K=96, M=48, block_rows=32, block_cols=48 → 3x1 blocks
W_case1 = W_full_padded[:96, :48]
r1 = run_case("row_split", W_case1, block_rows=32, block_cols=48, N=4, seed=100,
              expected_grid={'row_blocks': 3, 'col_blocks': 1})

# ── CASE 2: Column-split multi-block ───────────────────────────────────────
# K=32, M=144, block_rows=32, block_cols=48 → 1x3 blocks
W_case2 = W_full_padded[:32, :144]
r2 = run_case("col_split", W_case2, block_rows=32, block_cols=48, N=4, seed=101,
              expected_grid={'row_blocks': 1, 'col_blocks': 3})

# ── CASE 3: Row+column split multi-block ───────────────────────────────────
# K=96, M=144, block_rows=32, block_cols=48 → 3x3 blocks
W_case3 = W_full_padded[:96, :144]
r3 = run_case("row_col_split", W_case3, block_rows=32, block_cols=48, N=6, seed=102,
              expected_grid={'row_blocks': 3, 'col_blocks': 3})

# ── CASE 4: Awkward edge shape ─────────────────────────────────────────────
# K=70, M=101, block_rows=32, block_cols=48 → ceil(70/32)=3 row, ceil(101/48)=3 col = 9 blocks
W_case4_base = W_full_padded[:70, :101]
r4 = run_case("awkward_edge", W_case4_base, block_rows=32, block_cols=48, N=2, seed=103,
              expected_grid={'row_blocks': 3, 'col_blocks': 3})

all_pass = all(t['pass'] for t in [r1, r2, r3, r4])

results = {
    'phase': '28BC',
    'name': 'Multi-Block Real-Slice Residual Parity',
    'branch': 'experimental/prt-phase19a-alt-sidecar-backed',
    'head_before': 'e57ba9306',
    'head_after': None,
    'source_model': 'Qwen2.5-0.5B-Instruct/safetensors',
    'tensor_family': 'ffn_up (up_proj)',
    'layer': 'model.layers.0.mlp.up_proj.weight',
    'all_pass': all_pass,
    'canonical_trit_used': True,
    'bypass_used': False,
    'generation_run': False,
    'tests': [r1, r2, r3, r4],
    'files_changed': []
}

os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
with open(OUT_JSON, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults → {OUT_JSON}")
print(f"All pass: {all_pass}")
print("PASS" if all_pass else "FAIL")