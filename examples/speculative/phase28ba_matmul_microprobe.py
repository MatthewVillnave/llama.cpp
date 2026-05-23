#!/usr/bin/env python3
"""
Phase 28BA: Decoded Residual Matmul Microprobe
"""
import sys, os
sys.path.insert(0, '/home/matthew-villnave/llama.cpp/examples/speculative')
import numpy as np
import json

# Import after path setup
import prt_trit_io as io

RESULTS = {
    'phase': '28BA',
    'name': 'Decoded Residual Matmul Microprobe',
    'branch': 'experimental/prt-phase19a-alt-sidecar-backed',
    'head_before': '8addc0eb3',
    'head_after': None,
    'tests': []
}

test_cases = [
    {'K': 8, 'M': 16, 'N': 1, 'seed': 42},
    {'K': 17, 'M': 19, 'N': 4, 'seed': 43},
    {'K': 31, 'M': 33, 'N': 7, 'seed': 44},
]

for tc in test_cases:
    K, M, N, seed = tc['K'], tc['M'], tc['N'], tc['seed']
    rng = np.random.RandomState(seed)
    
    X = rng.randn(N, K).astype(np.float32)
    W_base = rng.randn(K, M).astype(np.float32)
    
    # Ternary residual: alternating pattern row-wise
    R_symbols = np.zeros((K, M), dtype=np.int8)
    for r in range(K):
        R_symbols[r, :] = 1 if (r % 2 == 0) else -1
    
    # Scale per row: block_rows=K means 1 block, 1 scale
    scale_val = 0.5
    scales = np.array([scale_val], dtype=np.float32)
    
    # Reference residual (float)
    R_ref = R_symbols.astype(np.float32) * scale_val
    
    # Write to .trit, read back using canonical prt_trit_io
    trit_path = f'/tmp/phase28ba_K{K}_M{M}_N{N}.trit'
    io.write_trit(trit_path, R_symbols, scales, block_rows=K, block_cols=M)
    decoded_ternary, read_scales, meta = io.read_trit(trit_path)
    
    # R_decoded: apply scale
    R_decoded = decoded_ternary.astype(np.float32) * scale_val
    
    # Matmul
    Y_ref = X @ (W_base + R_ref)
    Y_decoded = X @ (W_base + R_decoded)
    
    max_err = float(np.max(np.abs(Y_ref - Y_decoded)))
    mean_err = float(np.mean(np.abs(Y_ref - Y_decoded)))
    rmse = float(np.sqrt(np.mean((Y_ref - Y_decoded)**2)))
    
    pass_tc = max_err < 1e-4
    
    RESULTS['tests'].append({
        'case': f'K={K}_M={M}_N={N}_seed={seed}',
        'K': K, 'M': M, 'N': N, 'seed': seed,
        'max_abs_err': max_err,
        'mean_abs_err': mean_err,
        'rmse': rmse,
        'pass': pass_tc,
        'canonical_trit_used': True,
        'bypass_used': False
    })
    print(f"  Case {K}x{M}x{N}: max_err={max_err:.2e}, pass={pass_tc}")

all_pass = all(t['pass'] for t in RESULTS['tests'])
RESULTS['all_pass'] = all_pass

# Write JSON
out_json = '/home/matthew-villnave/llama.cpp/examples/speculative/results/phase28ba_decoded_residual_matmul_microprobe.json'
os.makedirs(os.path.dirname(out_json), exist_ok=True)
with open(out_json, 'w') as f:
    json.dump(RESULTS, f, indent=2)
print(f"Results written to {out_json}")
print(f"All pass: {all_pass}")