#!/usr/bin/env python3
"""PRT_3P offline accuracy test for blk.0.ffn_up.
Correct understanding: PRT_3P decomposes INPUT activations X, not weight matrix W.
Weights W stay in float32. X is split into 3 planes by magnitude thresholds."""
import struct
import statistics
import json
import numpy as np

# PRT_3P thresholds (from Phase 8)
T_HIGH_0 = 2.0   # Plane 0: |x| > 2.0
T_HIGH_1 = 0.5   # Plane 1: 0.5 < |x| <= 2.0
T_HIGH_2 = 0.1   # Plane 2: 0.1 < |x| <= 0.5
# Below 0.1: pruned (zeroed)

def prt_3plane_X(X, M, batch):
    """Split input activations X into 3 ternary planes.
    X: flat array of shape (batch * M,)
    Returns: X_q0, X_q1, X_q2, S0, S1, S2 (all flat batch*M arrays)
    """
    n = batch * M
    X_q0 = np.zeros(n, dtype=np.int8)
    X_q1 = np.zeros(n, dtype=np.int8)
    X_q2 = np.zeros(n, dtype=np.int8)
    S0 = np.zeros(n, dtype=np.float32)
    S1 = np.zeros(n, dtype=np.float32)
    S2 = np.zeros(n, dtype=np.float32)
    
    for i in range(n):
        x = X[i]
        absx = abs(x)
        
        if absx > T_HIGH_0:
            X_q0[i] = 1 if x > 0 else -1
            S0[i] = absx
        elif absx > T_HIGH_1:
            X_q1[i] = 1 if x > 0 else -1
            S1[i] = absx
        elif absx > T_HIGH_2:
            X_q2[i] = 1 if x > 0 else -1
            S2[i] = absx
        # below T_HIGH_2: all zeros (pruned)
    
    return X_q0, X_q1, X_q2, S0, S1, S2

def matmul_prt_3plane(X_q0, X_q1, X_q2, S0, S1, S2, W, batch, M, N):
    """Y = X @ W using PRT_3P computation.
    X_q*: int8 ternary {-1,0,+1} planes
    S*: float32 scales (magnitudes)
    W: float32 weights (M x N)
    Y: output (batch x N)
    """
    Y = np.zeros((batch, N), dtype=np.float32)
    W = W.reshape(M, N)
    
    for b in range(batch):
        for k in range(M):
            contrib = 0.0
            xq0 = X_q0[b*M + k]
            xq1 = X_q1[b*M + k]
            xq2 = X_q2[b*M + k]
            
            if xq0 != 0:
                contrib += S0[b*M + k] * xq0
            if xq1 != 0:
                contrib += S1[b*M + k] * xq1
            if xq2 != 0:
                contrib += S2[b*M + k] * xq2
            
            if contrib != 0.0:
                Y[b] += contrib * W[k]
    
    return Y

def cosine_sim(a, b):
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return float('nan')
    return dot / (norm_a * norm_b)

def main():
    print("=== PRT Phase 10A: Offline Accuracy Test (Correct PRT_3P) ===\n")
    
    M, N = 2048, 11008
    
    # Load weight matrix W (column-major flat array)
    print(f"Loading W: {M} x {N} = {M*N}")
    with open("/tmp/ffn_up_layer0_float.bin", "rb") as f:
        W_flat = np.array(struct.unpack(f'{M*N}f', f.read()), dtype=np.float32)
    W = W_flat.reshape(M, N)
    print(f"Loaded W: shape={W.shape}, min={W.min():.4f}, max={W.max():.4f}")
    
    for batch in [16, 17]:
        print(f"\n{'='*50}")
        print(f"Batch {batch}")
        print(f"{'='*50}")
        
        # Deterministic random X (uniform -1 to 1, like Phase 8)
        rng = np.random.default_rng(42)
        X = rng.uniform(-1, 1, size=(batch, M)).astype(np.float32)
        
        print(f"X: {X.shape}")
        print(f"X stats: min={X.min():.4f} max={X.max():.4f} mean={X.mean():.4f}")
        print(f"X |x|>2.0: {(np.abs(X) > 2.0).sum()} / {batch*M}")
        print(f"X |x|>0.5: {(np.abs(X) > 0.5).sum()} / {batch*M}")
        print(f"X |x|>0.1: {(np.abs(X) > 0.1).sum()} / {batch*M}")
        
        # Float reference: Y = X @ W
        print("\nComputing Y_float = X @ W...")
        Y_float = X @ W
        
        # PRT_3P: split X into planes, compute Y_prt = X_prt @ W
        print("Splitting X into PRT_3P planes...")
        X_q0, X_q1, X_q2, S0, S1, S2 = prt_3plane_X(X.flatten(), M, batch)
        
        nonzero_per_plane = [
            np.count_nonzero(X_q0),
            np.count_nonzero(X_q1),
            np.count_nonzero(X_q2)
        ]
        print(f"Plane 0 (|x|>2.0):  {nonzero_per_plane[0]} non-zero")
        print(f"Plane 1 (0.5<|x|<=2.0): {nonzero_per_plane[1]} non-zero")
        print(f"Plane 2 (0.1<|x|<=0.5): {nonzero_per_plane[2]} non-zero")
        print(f"Pruned (|x|<=0.1): {batch*M - sum(nonzero_per_plane)}")
        
        print("Computing Y_prt = X_prt @ W...")
        Y_prt = matmul_prt_3plane(X_q0, X_q1, X_q2, S0, S1, S2, W_flat, batch, M, N)
        
        # Compare
        Y_float_flat = Y_float.flatten()
        Y_prt_flat = Y_prt.flatten()
        
        cos = cosine_sim(Y_float_flat, Y_prt_flat)
        errors = Y_float_flat - Y_prt_flat
        abs_errors = np.abs(errors)
        
        print(f"\n--- Results ---")
        print(f"Cosine similarity: {cos:.6f}")
        print(f"Max abs error:     {abs_errors.max():.6f}")
        print(f"Mean abs error:    {abs_errors.mean():.6f}")
        print(f"Std abs error:     {abs_errors.std():.6f}")
        
        # Per-row cosine
        print(f"\nPer-row cosine (first 5):")
        for m in range(min(5, batch)):
            rc = cosine_sim(Y_float[m], Y_prt[m])
            print(f"  row[{m}]: {rc:.6f}")
        
        result = {
            "batch": batch,
            "cosine_sim": float(cos),
            "max_abs_error": float(abs_errors.max()),
            "mean_abs_error": float(abs_errors.mean()),
            "std_abs_error": float(abs_errors.std()),
            "pass_cosine_095": bool(cos >= 0.95),
            "plane0_nonzero": nonzero_per_plane[0],
            "plane1_nonzero": nonzero_per_plane[1],
            "plane2_nonzero": nonzero_per_plane[2],
            "pruned": batch*M - sum(nonzero_per_plane)
        }
        with open(f"/tmp/prt_phase10a_batch{batch}_accuracy.json", "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved: /tmp/prt_phase10a_batch{batch}_accuracy.json")
    
    print("\n=== DONE ===")
    return 0

if __name__ == '__main__':
    exit(main())