#!/usr/bin/env python3
"""
PRT Phase 28Q: Selected-Layer Ternary Residual Sensitivity Test
Tests Q2 base + ternary residual overlay across multiple layers of Qwen2.5-0.5B.
Offline validation only.
"""
import argparse
import json
import os
import sys

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import gguf
    HAS_GGUF = True
except ImportError:
    HAS_GGUF = False


def parse_args():
    p = argparse.ArgumentParser(description="PRT residual multi-layer sensitivity")
    p.add_argument("--model-path",
                   default="/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf")
    p.add_argument("--layers", default="0,11,23",
                   help="Comma-separated layer indices to test")
    p.add_argument("--tensor-type", default="ffn_up",
                   choices=["ffn_up", "ffn_down", "attn_q", "attn_k", "attn_v", "attn_output"])
    p.add_argument("--rows", type=int, default=512)
    p.add_argument("--cols", type=int, default=2048)
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--base-bits", type=int, default=2)
    p.add_argument("--residual-format", default="ternary",
                   choices=["ternary", "int2", "int4"])
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out-json", default="/tmp/phase28q_results.json")
    p.add_argument("--out-md", default="/tmp/phase28q_results.md")
    return p.parse_args()


def q2_quantize(w_ref, bits=2):
    if bits == 2:
        levels = np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float32)
        q_order = {1.5: 3, 0.5: 2, -0.5: 1, -1.5: 0}
        def encode(x):
            x_clipped = max(-1.5, min(1.5, x))
            idx = int(round((x_clipped - (-1.5)) / 0.5))
            idx = max(0, min(3, idx))
            return q_order[levels[idx]]
        quantized = np.vectorize(encode)(w_ref)
        w_base = levels[quantized].astype(np.float32)
        return w_base, quantized
    else:
        raise ValueError(f"Unsupported base-bits: {bits}")


def compress_ternary(residual):
    threshold = np.mean(np.abs(residual[residual != 0])) if np.any(residual != 0) else 0.1
    sign = np.sign(residual)
    sign[sign == 0] = 1
    magnitude = np.abs(residual)
    mag_clipped = np.clip(magnitude, 0, threshold)
    ternary = sign * mag_clipped
    scale = float(threshold)
    return ternary, scale


def cosine_similarity(a, b):
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


def extract_slice(model_path, layer_idx, tensor_type, rows, cols):
    """Extract a slice from a specific layer and tensor type."""
    reader = gguf.GGUFReader(model_path)
    tensor_name = f"blk.{layer_idx}.{tensor_type}.weight"
    
    # Find tensor
    tensor = None
    for t in reader.tensors:
        if t.name == tensor_name:
            tensor = t
            break
    
    if tensor is None:
        raise ValueError(f"Tensor not found: {tensor_name}")
    
    # Dequantize to f32
    data = tensor.data
    if hasattr(data, 'dtype') and data.dtype.itemsize > 1:
        pass
    w_full = gguf.dequantize(data, tensor.tensor_type)
    
    # Handle shape - ffn_up: [896, 4864] -> transpose to [512, 2048] slice
    # ffn_down: [4864, 896] -> [512, 2048] slice
    if w_full.ndim == 2:
        # ffn_up: [896, 4864] -> slice [512, 2048] from original
        # ffn_down: [4864, 896] -> slice [512, 2048] from original
        # ffn_qkv: [896, 2304] -> slice [512, 1536] from original
        # For ffn_up: take rows[:512] and cols[:2048] -> [512, 2048]
        # For ffn_down: take rows[:2048] and cols[:512] -> transpose to [512, 2048]
        r_orig, c_orig = w_full.shape
        if c_orig == 4864:  # ffn_up [896, 4864] -> [512, 2048] directly from cols
            slice_mat = w_full[:rows, :cols]
        elif r_orig == 4864:  # ffn_down [4864, 896] -> need [2048, 512] then transpose
            slice_mat = w_full[:cols, :rows].T  # [2048, 512] -> [512, 2048]
        else:
            # Generic: take rows x cols from original
            slice_mat = w_full[:rows, :cols]
    else:
        slice_mat = w_full[:rows]
    
    return slice_mat, tensor.tensor_type.name, list(tensor.shape)


def run_layer_test(W_ref, rows, cols, samples, base_bits, res_format, seed):
    """Run residual test on a single slice."""
    np.random.seed(seed)
    
    # Q2 base
    W_base, _ = q2_quantize(W_ref, base_bits)
    
    # Residual
    R = W_ref - W_base
    
    # Ternary compress
    R_hat, res_scale = compress_ternary(R)
    
    # Reconstruct
    W_hat = W_base + R_hat
    
    # Activation samples
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
    
    l2_base = rel_l2(Y_base, Y_ref)
    l2_hat = rel_l2(Y_hat, Y_ref)
    
    # Storage
    q2_bytes = int(rows * cols * 0.25)
    q4_bytes = int(rows * cols * 0.5)
    res_bytes = int(rows * cols * 0.125)
    combined_bytes = q2_bytes + res_bytes
    compression_ratio = combined_bytes / q4_bytes
    
    # Verdict
    if cos_improvement > 0.5 and compression_ratio < 1.0:
        verdict = "STRONG_RECOVERY"
    elif cos_improvement > 0.1 and compression_ratio < 1.0:
        verdict = "MODERATE_RECOVERY"
    elif cos_improvement > 0:
        verdict = "WEAK_RECOVERY"
    elif cos_improvement <= 0:
        verdict = "NO_RECOVERY"
    
    return {
        "cosine_base": round(cos_base, 4),
        "cosine_hat": round(cos_hat, 4),
        "cosine_improvement": round(cos_improvement, 4),
        "mae_base": round(mae_base, 6),
        "mae_hat": round(mae_hat, 6),
        "rel_l2_base": round(l2_base, 4),
        "rel_l2_hat": round(l2_hat, 4),
        "compression_ratio": round(compression_ratio, 4),
        "verdict": verdict,
        "residual_scale": round(res_scale, 6),
        "norm_ref": round(float(np.linalg.norm(W_ref)), 4),
    }


def main():
    if not HAS_NUMPY:
        print("ERROR: numpy not available", file=sys.stderr)
        sys.exit(1)
    if not HAS_GGUF:
        print("ERROR: gguf not available", file=sys.stderr)
        sys.exit(1)
    
    args = parse_args()
    layer_indices = [int(l) for l in args.layers.split(",")]
    
    results = []
    
    for layer_idx in layer_indices:
        print(f"\n=== Testing layer {layer_idx} ({args.tensor_type}) ===", file=sys.stderr)
        
        # Extract slice
        slice_path = f"/tmp/ffn_{args.tensor_type}_layer{layer_idx}.f32"
        
        # Try loading from cache first
        if os.path.exists(slice_path):
            print(f"Loading cached slice: {slice_path}", file=sys.stderr)
            W_ref = np.fromfile(slice_path, dtype=np.float32).reshape(args.rows, args.cols)
        else:
            print(f"Extracting layer {layer_idx} from model...", file=sys.stderr)
            W_ref, qtype, full_shape = extract_slice(args.model_path, layer_idx,
                                                     args.tensor_type, args.rows, args.cols)
            W_ref.tofile(slice_path)
            print(f"Saved to {slice_path}", file=sys.stderr)
        
        print(f"Slice shape: {W_ref.shape}, norm: {np.linalg.norm(W_ref):.4f}", file=sys.stderr)
        
        # Run test
        r = run_layer_test(W_ref, args.rows, args.cols, args.samples,
                           args.base_bits, args.residual_format, args.seed)
        r["layer"] = layer_idx
        r["tensor_type"] = args.tensor_type
        r["full_qtype"] = qtype if 'qtype' in dir() else "unknown"
        r["full_shape"] = full_shape if 'full_shape' in dir() else [0, 0]
        
        print(f"Q2 base cosine:   {r['cosine_base']:.4f}", file=sys.stderr)
        print(f"Q2+ternary cosine: {r['cosine_hat']:.4f}", file=sys.stderr)
        print(f"Cosine improvement: {r['cosine_improvement']:+.4f}", file=sys.stderr)
        print(f"Compression vs Q4: {r['compression_ratio']:.4f}", file=sys.stderr)
        print(f"Verdict: {r['verdict']}", file=sys.stderr)
        
        results.append(r)
    
    # Determine overall consistency
    improvements = [r['cosine_improvement'] for r in results]
    all_positive = all(i > 0 for i in improvements)
    all_strong = all(r['verdict'] in ("STRONG_RECOVERY", "MODERATE_RECOVERY") for r in results)
    
    if all_positive and all_strong:
        consistency = "CONSISTENT_STRONG"
    elif all_positive:
        consistency = "CONSISTENT_WEAK"
    elif sum(1 for i in improvements if i > 0) >= len(improvements) / 2:
        consistency = "PARTIAL"
    else:
        consistency = "FAIL"
    
    # Best layers
    best_layer = max(results, key=lambda r: r['cosine_improvement'])
    
    output = {
        "config": {
            "model_path": args.model_path,
            "layers": layer_indices,
            "tensor_type": args.tensor_type,
            "rows": args.rows,
            "cols": args.cols,
            "samples": args.samples,
            "base_bits": args.base_bits,
            "residual_format": args.residual_format,
            "seed": args.seed,
        },
        "layers_results": results,
        "summary": {
            "consistency": consistency,
            "all_positive": all_positive,
            "best_layer": best_layer['layer'],
            "best_improvement": best_layer['cosine_improvement'],
        }
    }
    
    # Write JSON
    with open(args.out_json, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nJSON written to {args.out_json}", file=sys.stderr)
    
    # Print summary table
    print(f"\n{'Layer':<8} {'Tensor':<12} {'Q2 cos':<10} {'Q2+T cos':<10} {'Δ cos':<10} {'MAE base':<12} {'MAE hat':<12} {'Compress':<10} {'Verdict':<20}", file=sys.stderr)
    print("-" * 120, file=sys.stderr)
    for r in results:
        print(f"{r['layer']:<8} {r['tensor_type']:<12} {r['cosine_base']:<10.4f} {r['cosine_hat']:<10.4f} {r['cosine_improvement']:<+10.4f} {r['mae_base']:<12.6f} {r['mae_hat']:<12.6f} {r['compression_ratio']:<10.4f} {r['verdict']:<20}", file=sys.stderr)
    
    print(f"\nConsistency: {consistency}", file=sys.stderr)
    print(f"Best layer: {best_layer['layer']} (Δ cos: {best_layer['cosine_improvement']:+.4f})", file=sys.stderr)
    
    return output


if __name__ == "__main__":
    main()