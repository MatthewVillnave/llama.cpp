#!/usr/bin/env python3
"""PRT Phase 13Y-FIX: Inline reference checker - no args needed."""
import struct, numpy as np

M, N = 896, 4864

# Load sidecar
with open('/tmp/prt_sidecars/ffn_up_layer0_prt.bin', 'rb') as f:
    sidecar = np.frombuffer(f.read(), dtype=np.float32).copy()

# Verify size
assert len(sidecar) == M * N, f"Size mismatch: {len(sidecar)} vs {M*N}"

# Reshape as [ffn,hidden]=[N,M] row-major
sc2d = sidecar.reshape((N, M))

# Test X vectors
X_ones = np.ones(M, dtype=np.float32)
X_rand = np.random.randn(M).astype(np.float32)
X_sparse = np.zeros(M, dtype=np.float32); X_sparse[:32] = 1.0

for name, X in [('ones', X_ones), ('random', X_rand), ('sparse', X_sparse)]:
    Y_ref = sc2d @ X
    
    # WRONG: treat as [hidden,ffn]=[M,N]
    sc_wrong = sidecar.reshape((M, N))
    Y_wrong = sc_wrong @ X
    
    diff = Y_ref - Y_wrong
    norm_ref = np.linalg.norm(Y_ref)
    norm_wrong = np.linalg.norm(Y_wrong)
    cos = np.dot(Y_ref, Y_wrong)/(norm_ref*norm_wrong+1e-10) if norm_ref > 0 and norm_wrong > 0 else 0.0
    
    print(f'{name}: ref_norm={norm_ref:.4f} wrong_norm={norm_wrong:.4f} cos_sim={cos:.6f} max_abs_err={np.abs(diff).max():.4f}')
    print(f'  ref[:8]  = {[round(y,6) for y in Y_ref[:8]]}')
    print(f'  wrong[:8]= {[round(y,6) for y in Y_wrong[:8]]}')
    print()