#!/usr/bin/env python3
"""
PRT Phase 28T: Qwen2.5-3B FFN_UP Ternary Residual Transfer Check
Extracts and tests FFN_UP slices from 3B model for comparison with 0.5B results.
Outputs to /tmp only.
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
    p = argparse.ArgumentParser(description="PRT 3B FFN_UP transfer check")
    p.add_argument("--model-path",
                   default="/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-3B-Instruct-Q4_K_M.gguf")
    p.add_argument("--layers", default="0,8,17,26,35",
                   help="Comma-separated layer indices")
    p.add_argument("--rows", type=int, default=512)
    p.add_argument("--cols", type=int, default=2048)
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out-json", default="/tmp/phase28t_3b_transfer.json")
    return p.parse_args()


def q2_quantize(w_ref, bits=2):
    levels = np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float32)
    q_order = {1.5: 3, 0.5: 2, -0.5: 1, -1.5: 0}
    def encode(x):
        x_c = max(-1.5, min(1.5, x))
        idx = int(round((x_c - (-1.5)) / 0.5))
        idx = max(0, min(3, idx))
        return q_order[levels[idx]]
    quantized = np.vectorize(encode)(w_ref)
    w_base = levels[quantized].astype(np.float32)
    return w_base


def compress_ternary(residual):
    threshold = np.mean(np.abs(residual[residual != 0])) if np.any(residual != 0) else 0.1
    sign = np.sign(residual)
    sign[sign == 0] = 1
    magnitude = np.abs(residual)
    mag_clipped = np.clip(magnitude, 0, threshold)
    return sign * mag_clipped, float(threshold)


def cosine_sim(a, b):
    a = a.astype(np.float32); b = b.astype(np.float32)
    norm_a = np.linalg.norm(a, axis=1, keepdims=True) + 1e-8
    norm_b = np.linalg.norm(b, axis=1, keepdims=True) + 1e-8
    return float(np.mean(np.sum((a / norm_a) * (b / norm_b), axis=1)))


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def rel_l2(a, b):
    denom = np.linalg.norm(b, axis=1) + 1e-8
    return float(np.mean(np.linalg.norm(a - b, axis=1) / denom))


def extract_slice(reader, tensor, rows, cols):
    """Extract f32 slice: ffn_up [2048, 11008] dequant (11008, 2048) -> (512, 2048)"""
    w_full = gguf.dequantize(tensor.data, tensor.tensor_type)
    # w_full: (11008, 2048), take [:cols, :rows] = [:2048, :512] then transpose -> (512, 2048)
    slice_mat = w_full[:cols, :rows].T
    return slice_mat


def run_test(W_ref, rows, cols, samples, seed):
    np.random.seed(seed)
    W_base = q2_quantize(W_ref)
    R = W_ref - W_base
    R_hat, res_scale = compress_ternary(R)
    W_hat = W_base + R_hat

    np.random.seed(seed)
    X = np.random.randn(cols, samples).astype(np.float32)
    Y_ref = W_ref @ X
    Y_base = W_base @ X
    Y_hat = W_hat @ X

    cos_base = cosine_sim(Y_base, Y_ref)
    cos_hat = cosine_sim(Y_hat, Y_ref)
    cos_imp = cos_hat - cos_base
    mae_b = mae(Y_base, Y_ref)
    mae_h = mae(Y_hat, Y_ref)
    l2_b = rel_l2(Y_base, Y_ref)
    l2_h = rel_l2(Y_hat, Y_ref)

    q2_bytes = int(rows * cols * 0.25)
    q4_bytes = int(rows * cols * 0.5)
    res_bytes = int(rows * cols * 0.125)
    combined = q2_bytes + res_bytes
    comp_ratio = combined / q4_bytes

    residual_norm = float(np.linalg.norm(R))
    tensor_norm = float(np.linalg.norm(W_ref))

    if cos_imp > 0.5 and comp_ratio < 1.0:
        verdict = "STRONG_RECOVERY"
    elif cos_imp > 0.1 and comp_ratio < 1.0:
        verdict = "MODERATE_RECOVERY"
    elif cos_imp > 0:
        verdict = "WEAK_RECOVERY"
    else:
        verdict = "NO_RECOVERY"

    return {
        "q2_cosine": round(cos_base, 4),
        "q2_ternary_cosine": round(cos_hat, 4),
        "cosine_improvement": round(cos_imp, 4),
        "mae_base": round(mae_b, 4),
        "mae_hat": round(mae_h, 4),
        "rel_l2_base": round(l2_b, 4),
        "rel_l2_hat": round(l2_h, 4),
        "compression_ratio_vs_q4": round(comp_ratio, 4),
        "residual_norm": round(residual_norm, 4),
        "tensor_norm": round(tensor_norm, 4),
        "residual_scale": round(res_scale, 6),
        "verdict": verdict,
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

    print(f"Reading model: {args.model_path}", file=sys.stderr)
    reader = gguf.GGUFReader(args.model_path)

    results = []

    for layer_idx in layer_indices:
        tensor_name = f"blk.{layer_idx}.ffn_up.weight"
        tensor = None
        for t in reader.tensors:
            if t.name == tensor_name:
                tensor = t
                break

        if tensor is None:
            print(f"ERROR: tensor not found: {tensor_name}", file=sys.stderr)
            continue

        cache_path = f"/tmp/ffn3b_ffn_up_layer{layer_idx}.f32"

        if os.path.exists(cache_path):
            W_ref = np.fromfile(cache_path, dtype=np.float32).reshape(args.rows, args.cols)
            print(f"L{layer_idx}: loaded cached, norm={np.linalg.norm(W_ref):.4f}", file=sys.stderr)
        else:
            print(f"L{layer_idx}: extracting...", file=sys.stderr)
            W_ref = extract_slice(reader, tensor, args.rows, args.cols)
            W_ref.tofile(cache_path)
            print(f"L{layer_idx}: saved to {cache_path}", file=sys.stderr)

        r = run_test(W_ref, args.rows, args.cols, args.samples, args.seed)
        r["layer"] = layer_idx
        r["tensor_name"] = tensor_name
        r["shape"] = [args.rows, args.cols]
        r["qtype"] = tensor.tensor_type.name

        print(f"  Q2 cos={r['q2_cosine']:.4f} Q2+T={r['q2_ternary_cosine']:.4f} "
              f"Δ={r['cosine_improvement']:+.4f} MAE={r['mae_hat']:.4f} "
              f"comp={r['compression_ratio_vs_q4']:.4f} {r['verdict']}", file=sys.stderr)

        results.append(r)

    # Compute stats
    improvements = [rr['cosine_improvement'] for rr in results]
    mean_imp = sum(improvements) / len(improvements)
    std_imp = (sum((x - mean_imp) ** 2 for x in improvements) / len(improvements)) ** 0.5
    strong_count = sum(1 for rr in results if rr['verdict'] == 'STRONG_RECOVERY')

    # Deterministic repeat for layer 0
    print(f"\nRunning deterministic repeat for layer 0...", file=sys.stderr)
    W_ref = np.fromfile(cache_path, dtype=np.float32).reshape(args.rows, args.cols)
    r2 = run_test(W_ref, args.rows, args.cols, args.samples, args.seed)
    deterministic_match = (
        abs(r2['q2_cosine'] - results[0]['q2_cosine']) < 1e-6 and
        abs(r2['q2_ternary_cosine'] - results[0]['q2_ternary_cosine']) < 1e-6
    )
    print(f"Deterministic repeat match: {deterministic_match}", file=sys.stderr)

    # 0.5B reference stats from Phase 28S
    ref_05b = {
        "mean_improvement": 0.7079,
        "std_improvement": 0.0084,
        "min_improvement": 0.6862,
        "max_improvement": 0.7247,
        "strong_count": 24,
        "total_layers": 24,
    }

    output = {
        "phase": "28T",
        "config": {
            "model": "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
            "layers": layer_indices,
            "slice_shape": [args.rows, args.cols],
            "samples": args.samples,
            "seed": args.seed,
        },
        "3b_results": results,
        "3b_statistics": {
            "total_layers": len(results),
            "mean_cosine_improvement": round(mean_imp, 4),
            "std_cosine_improvement": round(std_imp, 4),
            "min_cosine_improvement": round(min(improvements), 4),
            "max_cosine_improvement": round(max(improvements), 4),
            "strong_recovery_count": strong_count,
        },
        "reference_05b": ref_05b,
        "comparison": {
            "mean_delta_05b_vs_3b": round(ref_05b["mean_improvement"] - mean_imp, 4),
            "std_delta_05b_vs_3b": round(ref_05b["std_improvement"] - std_imp, 4),
            "03b_looks_like_05b": abs(ref_05b["mean_improvement"] - mean_imp) < 0.05,
        },
        "deterministic_repeat": {
            "layer": layer_indices[0],
            "match": deterministic_match,
        }
    }

    with open(args.out_json, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nJSON written to {args.out_json}", file=sys.stderr)

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"3B TRANSFER CHECK COMPLETE", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"3B mean Δ cos: {mean_imp:.4f} ± {std_imp:.4f}", file=sys.stderr)
    print(f"0.5B mean Δ cos: {ref_05b['mean_improvement']:.4f} ± {ref_05b['std_improvement']:.4f}", file=sys.stderr)
    print(f"Delta (0.5B - 3B): {ref_05b['mean_improvement'] - mean_imp:+.4f}", file=sys.stderr)
    print(f"3B strong recovery: {strong_count}/{len(results)}", file=sys.stderr)

    return output


if __name__ == "__main__":
    main()