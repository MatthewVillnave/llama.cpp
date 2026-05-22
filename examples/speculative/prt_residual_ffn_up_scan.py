#!/usr/bin/env python3
"""
PRT Phase 28S: All-FFN_UP Slice Scan + Budgeted Layer Selection
Offline validation of all 24 FFN_UP layers on Qwen2.5-0.5B.
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
    p = argparse.ArgumentParser(description="PRT all-FFN_UP layer scan")
    p.add_argument("--model-path",
                   default="/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf")
    p.add_argument("--rows", type=int, default=512)
    p.add_argument("--cols", type=int, default=2048)
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--base-bits", type=int, default=2)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out-json", default="/tmp/prt_ffn_up_24layer_scan.json")
    p.add_argument("--out-md", default="/tmp/prt_ffn_up_24layer_scan.md")
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


def get_ffn_up_layers(reader):
    """Get all 24 ffn_up tensor indices in layer order 0-23."""
    layers = {}
    for t in reader.tensors:
        if 'ffn_up' in t.name:
            layer_idx = int(t.name.split('.')[1])
            layers[layer_idx] = t
    return layers


def extract_slice(reader, tensor, rows, cols):
    """Extract and return f32 slice from tensor."""
    data = tensor.data
    w_full = gguf.dequantize(data, tensor.tensor_type)
    # ffn_up meta [R, C]=[896, 4864], dequant [R, C]=(4864, 896)
    # Target slice: [rows, cols] = [512, 2048]
    # Strategy: take [rows, cols] from dequant -> need (512, 2048)
    # dequant is (4864, 896), so [rows, cols] = [2048, 512] then transpose
    # to get (512, 2048)
    slice_mat = w_full[:cols, :rows].T  # (2048, 512).T = (512, 2048)
    return slice_mat


def run_layer_test(W_ref, rows, cols, samples, seed):
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
    cos_improvement = cos_hat - cos_base
    
    mae_base = mae(Y_base, Y_ref)
    mae_hat = mae(Y_hat, Y_ref)
    
    l2_base = rel_l2(Y_base, Y_ref)
    l2_hat = rel_l2(Y_hat, Y_ref)
    
    q2_bytes = int(rows * cols * 0.25)
    q4_bytes = int(rows * cols * 0.5)
    res_bytes = int(rows * cols * 0.125)
    combined = q2_bytes + res_bytes
    comp_ratio = combined / q4_bytes
    
    residual_norm = float(np.linalg.norm(R))
    tensor_norm = float(np.linalg.norm(W_ref))
    
    if cos_improvement > 0.5 and comp_ratio < 1.0:
        verdict = "STRONG_RECOVERY"
    elif cos_improvement > 0.1 and comp_ratio < 1.0:
        verdict = "MODERATE_RECOVERY"
    elif cos_improvement > 0:
        verdict = "WEAK_RECOVERY"
    else:
        verdict = "NO_RECOVERY"
    
    return {
        "q2_cosine": round(cos_base, 4),
        "q2_ternary_cosine": round(cos_hat, 4),
        "cosine_improvement": round(cos_improvement, 4),
        "mae_base": round(mae_base, 4),
        "mae_hat": round(mae_hat, 4),
        "rel_l2_base": round(l2_base, 4),
        "rel_l2_hat": round(l2_hat, 4),
        "compression_ratio_vs_q4": round(comp_ratio, 4),
        "residual_norm": round(residual_norm, 4),
        "tensor_norm": round(tensor_norm, 4),
        "residual_scale": round(res_scale, 6),
        "verdict": verdict,
        "q2_bytes": q2_bytes,
        "residual_bytes": res_bytes,
        "combined_bytes": combined,
        "q4_bytes": q4_bytes,
    }


def main():
    if not HAS_NUMPY:
        print("ERROR: numpy not available", file=sys.stderr)
        sys.exit(1)
    if not HAS_GGUF:
        print("ERROR: gguf not available", file=sys.stderr)
        sys.exit(1)
    
    args = parse_args()
    
    print(f"Reading model: {args.model_path}", file=sys.stderr)
    reader = gguf.GGUFReader(args.model_path)
    
    ffn_up_layers = get_ffn_up_layers(reader)
    layer_indices = sorted(ffn_up_layers.keys())
    
    print(f"Found {len(layer_indices)} ffn_up layers: {layer_indices}", file=sys.stderr)
    
    results = []
    
    for layer_idx in layer_indices:
        tensor = ffn_up_layers[layer_idx]
        tensor_name = tensor.name
        
        # Check cache first
        cache_path = f"/tmp/ffn_ffn_up_layer{layer_idx}.f32"
        
        if os.path.exists(cache_path):
            W_ref = np.fromfile(cache_path, dtype=np.float32).reshape(args.rows, args.cols)
            print(f"L{layer_idx}: loaded cached slice, norm={np.linalg.norm(W_ref):.4f}", file=sys.stderr)
        else:
            print(f"L{layer_idx}: extracting from model...", file=sys.stderr)
            W_ref = extract_slice(reader, tensor, args.rows, args.cols)
            W_ref.tofile(cache_path)
            print(f"L{layer_idx}: saved to {cache_path}", file=sys.stderr)
        
        r = run_layer_test(W_ref, args.rows, args.cols, args.samples, args.seed)
        r["layer"] = layer_idx
        r["tensor_name"] = tensor_name
        r["shape"] = [args.rows, args.cols]
        r["qtype"] = tensor.tensor_type.name
        
        print(f"  Q2 cos={r['q2_cosine']:.4f} Q2+T={r['q2_ternary_cosine']:.4f} "
              f"Δ={r['cosine_improvement']:+.4f} MAE={r['mae_hat']:.4f} "
              f"comp={r['compression_ratio_vs_q4']:.4f} {r['verdict']}", file=sys.stderr)
        
        results.append(r)
    
    # Compute statistics
    improvements = [rr['cosine_improvement'] for rr in results]
    mean_imp = sum(improvements) / len(improvements)
    std_imp = (sum((x - mean_imp) ** 2 for x in improvements) / len(improvements)) ** 0.5
    residuals = [rr['residual_norm'] for rr in results]
    mean_res = sum(residuals) / len(residuals)
    
    strong_count = sum(1 for rr in results if rr['verdict'] == 'STRONG_RECOVERY')
    
    # Budget scenarios
    all_combined = sum(rr['combined_bytes'] for rr in results)
    all_q4 = sum(rr['q4_bytes'] for rr in results)
    
    # Rank by score (residual_norm × cos_improvement)
    ranked = sorted(results, key=lambda x: x['residual_norm'] * x['cosine_improvement'], reverse=True)
    
    scenarios = []
    
    # All 24
    scenarios.append({
        "name": "all_24_ffn_up",
        "description": "All 24 FFN_UP layers with ternary residual",
        "layers": list(range(24)),
        "total_combined_bytes": all_combined,
        "total_q4_bytes": all_q4,
        "compression_ratio": round(all_combined / all_q4, 4),
        "savings_mb": round((all_q4 - all_combined) / 1024 / 1024, 2),
        "expected_recovery": sum(rr['cosine_improvement'] for rr in results),
        "strong_count": strong_count,
    })
    
    # Top 12 by residual_norm × cos
    top12 = ranked[:12]
    scenarios.append({
        "name": "top_12_by_score",
        "description": "Top 12 layers by residual_norm × cos_improvement",
        "layers": [rr['layer'] for rr in top12],
        "total_combined_bytes": sum(rr['combined_bytes'] for rr in top12),
        "total_q4_bytes": sum(rr['q4_bytes'] for rr in top12),
        "compression_ratio": round(sum(rr['combined_bytes'] for rr in top12) / sum(rr['q4_bytes'] for rr in top12), 4),
        "savings_mb": round((sum(rr['q4_bytes'] for rr in top12) - sum(rr['combined_bytes'] for rr in top12)) / 1024 / 1024, 2),
        "expected_recovery": sum(rr['cosine_improvement'] for rr in top12),
        "strong_count": sum(1 for rr in top12 if rr['verdict'] == 'STRONG_RECOVERY'),
    })
    
    # Top 6
    top6 = ranked[:6]
    scenarios.append({
        "name": "top_6_by_score",
        "description": "Top 6 layers by residual_norm × cos_improvement",
        "layers": [rr['layer'] for rr in top6],
        "total_combined_bytes": sum(rr['combined_bytes'] for rr in top6),
        "total_q4_bytes": sum(rr['q4_bytes'] for rr in top6),
        "compression_ratio": round(sum(rr['combined_bytes'] for rr in top6) / sum(rr['q4_bytes'] for rr in top6), 4),
        "savings_mb": round((sum(rr['q4_bytes'] for rr in top6) - sum(rr['combined_bytes'] for rr in top6)) / 1024 / 1024, 2),
        "expected_recovery": sum(rr['cosine_improvement'] for rr in top6),
        "strong_count": sum(1 for rr in top6 if rr['verdict'] == 'STRONG_RECOVERY'),
    })
    
    # Top 4
    top4 = ranked[:4]
    scenarios.append({
        "name": "top_4_by_score",
        "description": "Top 4 layers by residual_norm × cos_improvement",
        "layers": [rr['layer'] for rr in top4],
        "total_combined_bytes": sum(rr['combined_bytes'] for rr in top4),
        "total_q4_bytes": sum(rr['q4_bytes'] for rr in top4),
        "compression_ratio": round(sum(rr['combined_bytes'] for rr in top4) / sum(rr['q4_bytes'] for rr in top4), 4),
        "savings_mb": round((sum(rr['q4_bytes'] for rr in top4) - sum(rr['combined_bytes'] for rr in top4)) / 1024 / 1024, 2),
        "expected_recovery": sum(rr['cosine_improvement'] for rr in top4),
        "strong_count": sum(1 for rr in top4 if rr['verdict'] == 'STRONG_RECOVERY'),
    })
    
    # Outlier detection
    outliers = [rr for rr in results if abs(rr['cosine_improvement'] - mean_imp) > 2 * std_imp]
    if not outliers:
        outliers = []
    
    output = {
        "config": {
            "model_path": args.model_path,
            "rows": args.rows,
            "cols": args.cols,
            "samples": args.samples,
            "seed": args.seed,
        },
        "layers_results": results,
        "statistics": {
            "total_layers": len(results),
            "mean_cosine_improvement": round(mean_imp, 4),
            "std_cosine_improvement": round(std_imp, 4),
            "mean_residual_norm": round(mean_res, 4),
            "strong_recovery_count": strong_count,
            "weak_or_no_recovery_count": len(results) - strong_count,
            "outliers": [rr['layer'] for rr in outliers] if outliers else [],
        },
        "budget_scenarios": scenarios,
        "ranked_by_score": [rr['layer'] for rr in ranked],
    }
    
    # Write JSON
    with open(args.out_json, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nJSON written to {args.out_json}", file=sys.stderr)
    
    # Print summary
    print(f"\n{'='*80}", file=sys.stderr)
    print(f"ALL-24 FFN_UP SCAN COMPLETE", file=sys.stderr)
    print(f"{'='*80}", file=sys.stderr)
    print(f"Mean Δ cosine: {mean_imp:.4f} ± {std_imp:.4f}", file=sys.stderr)
    print(f"Strong recovery: {strong_count}/24 layers", file=sys.stderr)
    print(f"All combined bytes: {all_combined:,} ({all_combined/1024/1024:.2f} MB)", file=sys.stderr)
    print(f"All Q4 equivalent:   {all_q4:,} ({all_q4/1024/1024:.2f} MB)", file=sys.stderr)
    print(f"Compression ratio: {all_combined/all_q4:.4f} vs Q4", file=sys.stderr)
    print(f"Outliers: {len(outliers)} - {outliers}", file=sys.stderr)
    
    print(f"\nBudget Scenarios:", file=sys.stderr)
    for s in scenarios:
        print(f"  {s['name']}: {s['total_combined_bytes']/1024/1024:.2f}MB "
              f"vs {s['total_q4_bytes']/1024/1024:.2f}MB Q4, "
              f"comp={s['compression_ratio']:.4f}, "
              f"savings={s['savings_mb']:.2f}MB, "
              f"strong={s['strong_count']}/{'top12' if '12' in s['name'] else 'top6' if '6' in s['name'] else 'top4' if '4' in s['name'] else '24'}",
              file=sys.stderr)
    
    print(f"\nTop 6 by score (residual_norm × cos): {ranked[:6]}", file=sys.stderr)
    
    return output


if __name__ == "__main__":
    main()