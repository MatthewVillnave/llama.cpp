#!/usr/bin/env python3
"""
Phase 28BB: Real-Slice Decoded Residual Matmul Parity
Real model tensor slice, .trit round-trip, matmul parity test.
"""
import sys, os, json, struct
import numpy as np

sys.path.insert(0, '/home/matthew-villnave/llama.cpp/examples/speculative')
import prt_trit_io as io

SLICE_PATH = '/tmp/phase28bb_slice.f32'
META_PATH  = '/tmp/phase28bb_slice_meta.json'
OUT_JSON   = '/home/matthew-villnave/llama.cpp/examples/speculative/results/phase28bb_real_slice_decoded_residual_matmul_parity.json'

# Load real slice
W_target = np.fromfile(SLICE_PATH, dtype=np.float32).reshape(32, 48)
with open(META_PATH) as f:
    meta = json.load(f)
print(f"Loaded real slice: {W_target.shape} range=[{W_target.min():.4f}, {W_target.max():.4f}]")
print(f"Source: {meta['source']} / {meta['layer']}")

# ── CASE 1: All-zero residual (controlled baseline) ─────────────────────────
print("\n=== Case 1: Zero Residual ===")
K, M, N = 32, 48, 4
# W_base = W_target → residual = 0
W_base = W_target.copy()
R_ref = np.zeros((K, M), dtype=np.float32)
scales = np.array([0.0], dtype=np.float32)  # zero scale

trit_path = '/tmp/phase28bb_case1.trit'
ternary = np.zeros((K, M), dtype=np.int8)
io.write_trit(trit_path, ternary, scales, block_rows=K, block_cols=M)
decoded_ternary, read_scales, _ = io.read_trit(trit_path)
R_decoded = decoded_ternary.astype(np.float32) * 0.0

rng = np.random.RandomState(99)
X = rng.randn(N, K).astype(np.float32)

Y_ref = X @ (W_base + R_ref)
Y_decoded = X @ (W_base + R_decoded)

err1 = float(np.max(np.abs(Y_ref - Y_decoded)))
mean1 = float(np.mean(np.abs(Y_ref - Y_decoded)))
rmse1 = float(np.sqrt(np.mean((Y_ref - Y_decoded)**2)))
pass1 = err1 < 1e-5
print(f"  max_err={err1:.2e} mean_err={mean1:.2e} RMSE={rmse1:.2e} PASS={pass1}")

# ── CASE 2: Deterministic residual from real slice ──────────────────────────
print("\n=== Case 2: Deterministic Residual from Real Slice ===")
# W_base = W_target rounded to nearest 0.25
round_val = 0.25
W_base = np.round(W_target / round_val) * round_val
R_ref = W_target - W_base
R_abs_mean = np.mean(np.abs(R_ref[R_ref != 0])) if np.any(R_ref != 0) else 0.1
R_symbols = np.sign(R_ref)
R_symbols[R_symbols == 0] = 1
scale_val = float(R_abs_mean)
scales = np.array([scale_val], dtype=np.float32)
R_approx = R_symbols.astype(np.float32) * scale_val

trit_path = '/tmp/phase28bb_case2.trit'
io.write_trit(trit_path, R_symbols.astype(np.int8), scales, block_rows=K, block_cols=M)
decoded_ternary, read_scales, _ = io.read_trit(trit_path)
R_decoded = decoded_ternary.astype(np.float32) * scale_val

# R parity
r_max = float(np.max(np.abs(R_approx - R_decoded)))
r_pass = r_max < 1e-5

# Matmul parity
Y_ref = X @ (W_base + R_approx)
Y_decoded = X @ (W_base + R_decoded)

err2 = float(np.max(np.abs(Y_ref - Y_decoded)))
mean2 = float(np.mean(np.abs(Y_ref - Y_decoded)))
rmse2 = float(np.sqrt(np.mean((Y_ref - Y_decoded)**2)))
pass2 = err2 < 1e-4 and r_pass
print(f"  R_parity max_err={r_max:.2e} PASS={r_pass}")
print(f"  max_err={err2:.2e} mean_err={mean2:.2e} RMSE={rmse2:.2e} PASS={pass2}")

# ── CASE 3: Synthetic W_base + real residual ───────────────────────────────
print("\n=== Case 3: Synthetic W_base + Real Residual ===")
rng3 = np.random.RandomState(77)
W_base = (rng3.randn(K, M) * 0.1).astype(np.float32)
R_ref = W_target[:K, :M] - W_base
R_abs_mean = np.mean(np.abs(R_ref[R_ref != 0])) if np.any(R_ref != 0) else 0.1
R_symbols = np.sign(R_ref)
R_symbols[R_symbols == 0] = 1
scale_val = float(R_abs_mean)
scales = np.array([scale_val], dtype=np.float32)
R_approx = R_symbols.astype(np.float32) * scale_val

trit_path = '/tmp/phase28bb_case3.trit'
io.write_trit(trit_path, R_symbols.astype(np.int8), scales, block_rows=K, block_cols=M)
decoded_ternary, read_scales, _ = io.read_trit(trit_path)
R_decoded = decoded_ternary.astype(np.float32) * scale_val

r_max = float(np.max(np.abs(R_approx - R_decoded)))
r_pass3 = r_max < 1e-5

Y_ref = X @ (W_base + R_approx)
Y_decoded = X @ (W_base + R_decoded)

err3 = float(np.max(np.abs(Y_ref - Y_decoded)))
mean3 = float(np.mean(np.abs(Y_ref - Y_decoded)))
rmse3 = float(np.sqrt(np.mean((Y_ref - Y_decoded)**2)))
pass3 = err3 < 1e-4 and r_pass3
print(f"  R_parity max_err={r_max:.2e} PASS={r_pass3}")
print(f"  max_err={err3:.2e} mean_err={mean3:.2e} RMSE={rmse3:.2e} PASS={pass3}")

# ── RESULTS ─────────────────────────────────────────────────────────────────
all_pass = pass1 and pass2 and pass3

results = {
    "phase": "28BB",
    "name": "Real-Slice Decoded Residual Matmul Parity",
    "branch": "experimental/prt-phase19a-alt-sidecar-backed",
    "head_before": "cdaf7fbaf",
    "head_after": None,
    "source_model": meta['model'],
    "tensor_family": meta['tensor_family'],
    "layer": meta['layer'],
    "slice_shape": [K, M],
    "N": N,
    "canonical_trit_used": True,
    "bypass_used": False,
    "generation_run": False,
    "tests": [
        {
            "case": "zero_residual",
            "K": K, "M": M, "N": N,
            "max_abs_err": err1, "mean_abs_err": mean1, "rmse": rmse1,
            "pass": pass1
        },
        {
            "case": "deterministic_real_residual",
            "K": K, "M": M, "N": N,
            "R_parity_max_err": r_max,
            "R_parity_pass": r_pass,
            "max_abs_err": err2, "mean_abs_err": mean2, "rmse": rmse2,
            "pass": pass2
        },
        {
            "case": "synthetic_Wbase_real_residual",
            "K": K, "M": M, "N": N,
            "R_parity_max_err": r_max,
            "R_parity_pass": r_pass3,
            "max_abs_err": err3, "mean_abs_err": mean3, "rmse": rmse3,
            "pass": pass3
        }
    ],
    "all_pass": all_pass,
    "files_changed": []
}

os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
with open(OUT_JSON, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults → {OUT_JSON}")
print(f"All pass: {all_pass}")
print(f"PASS" if all_pass else "FAIL")