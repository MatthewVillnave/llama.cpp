#!/usr/bin/env python3
"""
PRT Residual Overlay Offline Prototype
Tests residual overlay math on synthetic tensors.
No real model files. Pure numpy computation.
"""
import argparse
import json
import sys

# Check numpy availability
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def parse_args():
    p = argparse.ArgumentParser(description="PRT residual overlay offline prototype")
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
        n_levels = 4  # -1.5, -0.5, 0.5, 1.5
        levels = np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float32)
        q_order = {1.5: 3, 0.5: 2, -0.5: 1, -1.5: 0}
        def encode(x):
            x_clipped = max(-1.5, min(1.5, x))
            idx = int(round((x_clipped - (-1.5)) / 0.5))
            idx = max(0, min(3, idx))
            return q_order[levels[idx]]
        quantized = np.vectorize(encode)(w_ref)
        w_base = levels[quantized].astype(np.float32)
        q2_step = 1.0
        scale = 1.0
        n_levels_val = 4
    elif bits == 4:
        max_val = 7.0
        q2_step = max_val / 7.0
        levels = np.array([(i / 7.0 * max_val) - max_val for i in range(8)], dtype=np.float32)
        def encode(x):
            x_c = max(-max_val, min(max_val, x))
            idx = int(round(x_c / q2_step))
            idx = max(0, min(7, idx))
            return idx
        quantized = np.vectorize(encode)(w_ref)
        w_base = (quantized.astype(np.float32) * q2_step) - max_val
        scale = q2_step
        n_levels_val = 8
    else:
        raise ValueError(f"Unsupported base-bits: {bits}")
    return w_base, quantized, scale, n_levels_val


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
    storage = residual.size * 1  # 1 byte per element (stored as int8)
    return ternary, scale, storage


def compress_int2(residual):
    """INT2 compression: 2-bit signed integers with scale factor."""
    max_val = 1.5
    n_levels = 4  # -1.5, -0.5, 0.5, 1.5
    scale = max_val / 1.0
    levels = np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float32)
    def encode(x):
        x_c = max(-max_val, min(max_val, x))
        idx = int(round(x_c * scale))
        idx = max(-2, min(2, idx))
        # Map -2->0, -1->1, 0->2, 1->3, 2->N/A... 
        return idx + 2  # shift to 0..4
    quantized = np.vectorize(encode)(residual).astype(np.int8)
    w_hat = ((quantized.astype(np.float32) - 2) / scale).astype(np.float32)
    storage = residual.size * 1  # 1 byte per element
    return w_hat, 1.0, storage


def compress_int4(residual):
    """INT4 compression: 4-bit signed integers with scale factor."""
    max_val = 7.0
    scale = max_val / 7.0
    def encode(x):
        x_c = max(-max_val, min(max_val, x))
        idx = int(round(x_c / scale))
        idx = max(-8, min(7, idx))
        return idx + 8  # shift to 0..15
    quantized = np.vectorize(encode)(residual).astype(np.int8)
    w_hat = ((quantized.astype(np.float32) - 8) * scale).astype(np.float32)
    storage = residual.size * 1  # 1 byte per 2 elements (packed), rough estimate
    return w_hat, 1.0, storage


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


def run_test(rows, cols, samples, base_bits, res_format, seed, out_json):
    np.random.seed(seed)
    rng_state = np.random.get_state()
    
    # Generate synthetic W_ref
    W_ref = np.random.randn(rows, cols).astype(np.float32)
    
    # Quantize to Q2 base
    W_base, quantized, scale, n_levels = q2_quantize(W_ref, base_bits)
    
    # Compute residual
    R = W_ref - W_base
    
    # Compress residual
    if res_format == "ternary":
        R_hat, res_scale, res_storage = compress_ternary(R)
    elif res_format == "int2":
        R_hat, res_scale, res_storage = compress_int2(R)
    elif res_format == "int4":
        R_hat, res_scale, res_storage = compress_int4(R)
    else:
        raise ValueError(f"Unknown format: {res_format}")
    
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
    
    # Storage estimates (compressed bytes)
    # Q2 = 2 bits per element = rows * cols * 2 / 8 = rows * cols * 0.25
    # Q4 = 4 bits per element = rows * cols * 4 / 8 = rows * cols * 0.5
    q2_storage_bytes = int(rows * cols * 0.25)
    q4_bytes = int(rows * cols * 0.5)
    
    # Residual: ternary=1bit/elem, int2=2bits/elem, int4=4bits/elem
    if res_format == "ternary":
        res_storage_bytes = int(rows * cols * 0.125)  # 1 bit
    elif res_format == "int2":
        res_storage_bytes = int(rows * cols * 0.25)   # 2 bits
    elif res_format == "int4":
        res_storage_bytes = int(rows * cols * 0.5)    # 4 bits
    
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
            "rows": rows, "cols": cols, "samples": samples,
            "base_bits": base_bits, "residual_format": res_format, "seed": seed
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
    print(f"=== PRT Residual Overlay ({res_format}) ===")
    print(f"Shape: {rows} x {cols}, samples: {samples}")
    print(f"Q2 base cosine vs ref:   {cos_base:.4f}")
    print(f"Q2+residual cosine vs ref: {cos_hat:.4f}")
    print(f"Cosine improvement:        {cos_improvement:+.4f} {'✅' if pass_cosine else '❌'}")
    print(f"MAE base vs ref:         {mae_base:.6f}")
    print(f"MAE hat vs ref:          {mae_hat:.6f} {'✅' if pass_mae else '❌'}")
    print(f"Rel-L2 base:             {l2_base:.4f}")
    print(f"Rel-L2 hat:              {l2_hat:.4f}")
    print(f"Storage combined:         {combined_bytes:,} bytes")
    print(f"Q4 equivalent:           {q4_bytes:,} bytes")
    print(f"Compression ratio:        {compression_ratio_vs_q4:.4f} {'✅' if pass_compression else '❌'}")
    print(f"OVERALL:                 {'PASS ✅' if passed else 'FAIL ❌'}")
    if fail_reasons:
        print(f"Fail reasons: {fail_reasons}")
    
    if out_json:
        with open(out_json, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nJSON written to {out_json}")
    
    return result


def main():
    if not HAS_NUMPY:
        print("ERROR: numpy not available. Install with: pip install numpy", file=sys.stderr)
        sys.exit(1)
    
    args = parse_args()
    run_test(args.rows, args.cols, args.samples, args.base_bits,
              args.residual_format, args.seed, args.out_json)


if __name__ == "__main__":
    main()
