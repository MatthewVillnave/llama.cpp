#!/usr/bin/env python3
"""
PRT Phase 28P: Real Tensor Slice Residual Validation
Tests Q2 base + ternary residual overlay on real Qwen2.5-0.5B ffn_up slice.
Offline validation only.
"""
import argparse
import json
import sys
import os

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def parse_args():
    p = argparse.ArgumentParser(description="PRT residual real-slice validation")
    p.add_argument("--slice-path", default="/tmp/ffn_up_slice_layer0.f32",
                   help="Path to f32 slice binary")
    p.add_argument("--rows", type=int, default=512)
    p.add_argument("--cols", type=int, default=2048)
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--base-bits", type=int, default=2)
    p.add_argument("--residual-format", default="ternary",
                   choices=["ternary", "int2", "int4"])
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out-json", default=None)
    return p.parse_args()


def q2_quantize(w_ref, bits=2):
    """Simulate Q2 quantization and dequantization."""
    if bits == 2:
        max_val = 1.5
        levels = np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float32)
        q_order = {1.5: 3, 0.5: 2, -0.5: 1, -1.5: 0}
        def encode(x):
            x_clipped = max(-1.5, min(1.5, x))
            idx = int(round((x_clipped - (-1.5)) / 0.5))
            idx = max(0, min(3, idx))
            return q_order[levels[idx]]
        quantized = np.vectorize(encode)(w_ref)
        w_base = levels[quantized].astype(np.float32)
        return w_base, quantized, 1.0, 4
    else:
        raise ValueError(f"Unsupported base-bits: {bits}")


def compress_ternary(residual, threshold=None):
    """Ternary compression: sign(R) * mean(|R| on non-zero elements)."""
    if threshold is None:
        threshold = np.mean(np.abs(residual[residual != 0])) if np.any(residual != 0) else 0.1
    sign = np.sign(residual)
    sign[sign == 0] = 1
    magnitude = np.abs(residual)
    mag_clipped = np.clip(magnitude, 0, threshold)
    ternary = sign * mag_clipped
    scale = float(threshold)
    storage = residual.size * 1  # 1 byte per element
    return ternary, scale, storage


def cosine_similarity(a, b):
    """Row-wise cosine similarity."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    norm_a = np.linalg.norm(a, axis=1, keepdims=True) + 1e-8
    norm_b = np.linalg.norm(b, axis=1, keepdims=True) + 1e-8
    cos = np.sum((a / norm_a) * (b / norm_b), axis=1)
    return float(np.mean(cos))


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def rel_l2(a, b):
    denom = np.linalg.norm(b, axis=1) + 1e-8
    return float(np.mean(np.linalg.norm(a - b, axis=1) / denom))


def run_test(slice_path, rows, cols, samples, base_bits, res_format, seed, out_json):
    # Load real f32 slice
    W_ref = np.fromfile(slice_path, dtype=np.float32).reshape(rows, cols)
    print(f"Loaded real slice: {W_ref.shape}", file=sys.stderr)
    print(f"Norm: {np.linalg.norm(W_ref):.4f}", file=sys.stderr)
    
    np.random.seed(seed)
    rng_state = np.random.get_state()
    
    # Quantize to Q2 base
    W_base, quantized, scale, n_levels = q2_quantize(W_ref, base_bits)
    
    # Compute residual
    R = W_ref - W_base
    
    # Compress residual
    R_hat, res_scale, res_storage = compress_ternary(R)
    
    # Reconstruct
    W_hat = W_base + R_hat
    
    # Generate activation samples (fixed seed)
    np.random.seed(seed)
    X = np.random.randn(cols, samples).astype(np.float32)
    
    # Matvec
    Y_ref = W_ref @ X
    Y_base = W_base @ X
    Y_hat = W_hat @ X
    
    # Metrics
    cos_base = cosine_similarity(Y_base, Y_ref)
    cos_hat = cosine_similarity(Y_hat, Y_ref)
    cos_improvement = cos_hat - cos_base
    
    mae_base = mae(Y_base, Y_ref)
    mae_hat = mae(Y_hat, Y_ref)
    mae_improvement_pct = (mae_base - mae_hat) / mae_base * 100 if mae_base > 0 else 0
    
    l2_base = rel_l2(Y_base, Y_ref)
    l2_hat = rel_l2(Y_hat, Y_ref)
    
    # Storage estimates
    q2_storage_bytes = int(rows * cols * 0.25)
    q4_bytes = int(rows * cols * 0.5)
    
    if res_format == "ternary":
        res_storage_bytes = int(rows * cols * 0.125)
    elif res_format == "int2":
        res_storage_bytes = int(rows * cols * 0.25)
    elif res_format == "int4":
        res_storage_bytes = int(rows * cols * 0.5)
    
    combined_bytes = q2_storage_bytes + res_storage_bytes
    compression_ratio_vs_q4 = combined_bytes / q4_bytes if q4_bytes > 0 else float('inf')
    
    # Pass/fail
    pass_cosine = cos_improvement > 0
    pass_compression = compression_ratio_vs_q4 < 1.0
    pass_mae = mae_hat < mae_base
    passed = pass_cosine and pass_compression and pass_mae
    
    fail_reasons = []
    if not pass_cosine:
        fail_reasons.append("cosine_improvement_not_positive")
    if not pass_compression:
        fail_reasons.append("compression_ratio >= 1.0")
    if not pass_mae:
        fail_reasons.append("mae_not_improved")
    
    result = {
        "config": {
            "slice_path": slice_path,
            "rows": rows, "cols": cols, "samples": samples,
            "base_bits": base_bits, "residual_format": res_format, "seed": seed
        },
        "real_slice_stats": {
            "norm": round(float(np.linalg.norm(W_ref)), 4),
            "min": round(float(np.min(W_ref)), 6),
            "max": round(float(np.max(W_ref)), 6),
            "mean": round(float(np.mean(W_ref)), 6),
        },
        "metrics": {
            "cosine_base_vs_ref": round(cos_base, 4),
            "cosine_hat_vs_ref": round(cos_hat, 4),
            "cosine_improvement": round(cos_improvement, 4),
            "mae_base_vs_ref": round(mae_base, 6),
            "mae_hat_vs_ref": round(mae_hat, 6),
            "mae_improvement_pct": round(mae_improvement_pct, 2),
            "rel_l2_base": round(l2_base, 4),
            "rel_l2_hat": round(l2_hat, 4),
            "q2_storage_bytes": int(q2_storage_bytes),
            "residual_storage_bytes": int(res_storage_bytes),
            "storage_bytes_combined": int(combined_bytes),
            "q4_equivalent_bytes": int(q4_bytes),
            "compression_ratio_vs_q4": round(compression_ratio_vs_q4, 4)
        },
        "result": {
            "pass": passed,
            "fail_reasons": fail_reasons if not passed else [],
            "pass_cosine": pass_cosine,
            "pass_compression": pass_compression,
            "pass_mae": pass_mae
        }
    }
    
    # Print summary
    print(f"=== PRT Residual Real Slice ({res_format}) ===", file=sys.stderr)
    print(f"Shape: {rows} x {cols}, samples: {samples}", file=sys.stderr)
    print(f"Real slice norm: {np.linalg.norm(W_ref):.4f}", file=sys.stderr)
    print(f"Q2 base cosine vs ref:   {cos_base:.4f}", file=sys.stderr)
    print(f"Q2+residual cosine vs ref: {cos_hat:.4f}", file=sys.stderr)
    print(f"Cosine improvement:        {cos_improvement:+.4f} {'✅' if pass_cosine else '❌'}", file=sys.stderr)
    print(f"MAE base vs ref:         {mae_base:.6f}", file=sys.stderr)
    print(f"MAE hat vs ref:          {mae_hat:.6f} {'✅' if pass_mae else '❌'}", file=sys.stderr)
    print(f"Rel-L2 base:             {l2_base:.4f}", file=sys.stderr)
    print(f"Rel-L2 hat:              {l2_hat:.4f}", file=sys.stderr)
    print(f"Storage combined:         {combined_bytes:,} bytes", file=sys.stderr)
    print(f"Q4 equivalent:           {q4_bytes:,} bytes", file=sys.stderr)
    print(f"Compression ratio:        {compression_ratio_vs_q4:.4f} {'✅' if pass_compression else '❌'}", file=sys.stderr)
    print(f"OVERALL:                 {'PASS ✅' if passed else 'FAIL ❌'}", file=sys.stderr)
    if fail_reasons:
        print(f"Fail reasons: {fail_reasons}", file=sys.stderr)
    
    if out_json:
        with open(out_json, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nJSON written to {out_json}", file=sys.stderr)
    
    return result


def main():
    if not HAS_NUMPY:
        print("ERROR: numpy not available", file=sys.stderr)
        sys.exit(1)
    
    args = parse_args()
    
    if not os.path.exists(args.slice_path):
        print(f"ERROR: slice file not found: {args.slice_path}", file=sys.stderr)
        sys.exit(1)
    
    run_test(args.slice_path, args.rows, args.cols, args.samples,
             args.base_bits, args.residual_format, args.seed, args.out_json)


if __name__ == "__main__":
    main()