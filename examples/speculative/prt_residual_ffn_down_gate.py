#!/usr/bin/env python3
"""
PRT Phase 28V: FFN_DOWN + FFN_GATE Multi-Layer Ternary Residual Validation
Validates Q2+ternary recovery on FFN_DOWN and FFN_GATE across 0.5B and 3B models.
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
    p = argparse.ArgumentParser(description="PRT FFN_DOWN/GATE validation")
    p.add_argument("--out-json", default="/tmp/phase28v_results.json")
    return p.parse_args()


def q2_quantize(w_ref):
    levels = np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float32)
    q_order = {1.5: 3, 0.5: 2, -0.5: 1, -1.5: 0}
    def encode(x):
        x_c = max(-1.5, min(1.5, x))
        idx = int(round((x_c - (-1.5)) / 0.5))
        return q_order[levels[max(0, min(3, idx))]]
    quantized = np.vectorize(encode)(w_ref)
    return levels[quantized].astype(np.float32)


def compress_ternary(residual):
    threshold = np.mean(np.abs(residual[residual != 0])) if np.any(residual != 0) else 0.1
    sign = np.sign(residual)
    sign[sign == 0] = 1
    return sign * np.clip(np.abs(residual), 0, threshold), float(threshold)


def cosine_sim(a, b):
    a, b = a.astype(np.float32), b.astype(np.float32)
    na = np.linalg.norm(a, axis=1, keepdims=True) + 1e-8
    nb = np.linalg.norm(b, axis=1, keepdims=True) + 1e-8
    return float(np.mean(np.sum((a/na) * (b/nb), axis=1)))


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def rel_l2(a, b):
    denom = np.linalg.norm(b, axis=1) + 1e-8
    return float(np.mean(np.linalg.norm(a - b, axis=1) / denom))


def extract_slice(reader, tensor_name, rows, cols):
    """Extract f32 slice. Orientation: dequant (C, R), slice [:cols,:rows].T -> (rows, cols)."""
    tensor = None
    for t in reader.tensors:
        if t.name == tensor_name:
            tensor = t
            break
    if tensor is None:
        raise ValueError(f"Tensor not found: {tensor_name}")
    
    w = gguf.dequantize(tensor.data, tensor.tensor_type)
    # dequant (C, R) where C=intermediate, R=hidden for ffn_up/ffn_gate
    # and (C, R) where C=hidden, R=intermediate for ffn_down
    # For all: slice [:cols, :rows] then .T -> (rows, cols)
    slice_mat = w[:cols, :rows].T
    return slice_mat, tensor.tensor_type.name, list(tensor.shape)


def run_test(W_ref, rows, cols, seed):
    np.random.seed(seed)
    W_base = q2_quantize(W_ref)
    R = W_ref - W_base
    R_hat, _ = compress_ternary(R)
    W_hat = W_base + R_hat

    np.random.seed(seed)
    X = np.random.randn(cols, 32).astype(np.float32)
    Y_ref = W_ref @ X
    Y_base = W_base @ X
    Y_hat = W_hat @ X

    cos_b = cosine_sim(Y_base, Y_ref)
    cos_h = cosine_sim(Y_hat, Y_ref)
    cos_imp = cos_h - cos_b
    mae_b = mae(Y_base, Y_ref)
    mae_h = mae(Y_hat, Y_ref)
    l2_b = rel_l2(Y_base, Y_ref)
    l2_h = rel_l2(Y_hat, Y_ref)

    q2_bytes = int(rows * cols * 0.25)
    q4_bytes = int(rows * cols * 0.5)
    res_bytes = int(rows * cols * 0.125)
    combined = q2_bytes + res_bytes
    comp = combined / q4_bytes

    verdict = "STRONG_RECOVERY" if cos_imp > 0.5 and comp < 1.0 else \
              "MODERATE_RECOVERY" if cos_imp > 0.1 else \
              "WEAK_RECOVERY" if cos_imp > 0 else "NO_RECOVERY"

    return {
        "q2_cosine": round(cos_b, 4),
        "q2_ternary_cosine": round(cos_h, 4),
        "cosine_improvement": round(cos_imp, 4),
        "mae_base": round(mae_b, 4),
        "mae_hat": round(mae_h, 4),
        "rel_l2_base": round(l2_b, 4),
        "rel_l2_hat": round(l2_h, 4),
        "compression_ratio_vs_q4": round(comp, 4),
        "residual_norm": round(float(np.linalg.norm(R)), 4),
        "tensor_norm": round(float(np.linalg.norm(W_ref)), 4),
        "verdict": verdict,
    }


def test_model(model_path, model_name, tensor_configs, out_json):
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Testing {model_name}: {model_path.split('/')[-1]}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    reader = gguf.GGUFReader(model_path)
    results = []

    for config in tensor_configs:
        tensor_name = config["tensor_name"]
        layer = config["layer"]
        rows = config["rows"]
        cols = config["cols"]
        cache_path = f"/tmp/ffn_{model_name.replace('-', '')}_L{layer}_{config['family']}.f32"

        if os.path.exists(cache_path):
            W_ref = np.fromfile(cache_path, dtype=np.float32).reshape(rows, cols)
            print(f"  {tensor_name}: loaded cached {W_ref.shape}", file=sys.stderr)
        else:
            print(f"  {tensor_name}: extracting...", file=sys.stderr)
            try:
                W_ref, qtype, full_shape = extract_slice(reader, tensor_name, rows, cols)
                W_ref.tofile(cache_path)
            except Exception as e:
                print(f"  {tensor_name}: EXTRACTION FAILED: {e}", file=sys.stderr)
                results.append({
                    "layer": layer, "tensor_name": tensor_name, "family": config["family"],
                    "error": str(e), "verdict": "BLOCKED_EXTRACTION"
                })
                continue
            print(f"  {tensor_name}: saved {W_ref.shape} to {cache_path}", file=sys.stderr)

        r = run_test(W_ref, rows, cols, seed=123)
        r["layer"] = layer
        r["tensor_name"] = tensor_name
        r["family"] = config["family"]
        r["shape"] = [rows, cols]
        r["qtype"] = qtype
        r["full_shape"] = full_shape

        print(f"    Q2 cos={r['q2_cosine']:.4f} Q2+T={r['q2_ternary_cosine']:.4f} "
              f"Δ={r['cosine_improvement']:+.4f} MAE={r['mae_hat']:.4f} "
              f"comp={r['compression_ratio_vs_q4']:.4f} {r['verdict']}", file=sys.stderr)

        results.append(r)

    return results


def main():
    if not HAS_NUMPY or not HAS_GGUF:
        print("ERROR: missing numpy or gguf", file=sys.stderr)
        sys.exit(1)

    args = parse_args()
    all_results = {}

    # === 0.5B: FFN_DOWN (512x896), FFN_GATE (512x896) ===
    # FFN_DOWN 0.5B: meta [4864, 896] -> dequant (896, 4864)
    # ffn_gate 0.5B: meta [896, 4864] -> dequant (4864, 896)
    # For 0.5B, rows=512, cols=896 for both (896 is the limiting dim)
    
    configs_05b = []
    for layer in [0, 5, 11, 23]:
        # ffn_down
        configs_05b.append({
            "tensor_name": f"blk.{layer}.ffn_down.weight",
            "layer": layer, "family": "ffn_down",
            "rows": 512, "cols": 896
        })
        # ffn_gate
        configs_05b.append({
            "tensor_name": f"blk.{layer}.ffn_gate.weight",
            "layer": layer, "family": "ffn_gate",
            "rows": 512, "cols": 896  # gate [intermediate=896, hidden=4864] -> dequant (4864, 896) -> [:896, :512].T -> (512, 896)
        })

    results_05b = test_model(
        "/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf",
        "0.5B", configs_05b, args.out_json
    )
    all_results["0.5B"] = results_05b

    # === 3B: FFN_DOWN (512x2048), FFN_GATE (512x2048) ===
    # FFN_DOWN 3B: meta [11008, 2048] -> dequant (2048, 11008) -> [:2048, :512].T -> (512, 2048) ✓
    # FFN_GATE 3B: meta [2048, 11008] -> dequant (11008, 2048) -> [:2048, :512].T -> (512, 2048) ✓

    configs_3b = []
    for layer in [0, 8, 17, 26, 35]:
        configs_3b.append({
            "tensor_name": f"blk.{layer}.ffn_down.weight",
            "layer": layer, "family": "ffn_down",
            "rows": 512, "cols": 2048
        })
        configs_3b.append({
            "tensor_name": f"blk.{layer}.ffn_gate.weight",
            "layer": layer, "family": "ffn_gate",
            "rows": 512, "cols": 2048
        })

    results_3b = test_model(
        "/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-3B-Instruct-Q4_K_M.gguf",
        "3B", configs_3b, args.out_json
    )
    all_results["3B"] = results_3b

    # === Deterministic repeat ===
    print(f"\nDeterministic repeat check (3B FFN_DOWN L0)...", file=sys.stderr)
    for r in results_3b:
        if r.get("layer") == 0 and r.get("family") == "ffn_down" and "error" not in r:
            cache = f"/tmp/ffn_3B_L0_ffn_down.f32"
            if os.path.exists(cache):
                W = np.fromfile(cache, dtype=np.float32).reshape(512, 2048)
                r2 = run_test(W, 512, 2048, seed=123)
                det_match = abs(r2['q2_cosine'] - r['q2_cosine']) < 1e-6 and \
                            abs(r2['q2_ternary_cosine'] - r['q2_ternary_cosine']) < 1e-6
                print(f"  Deterministic match: {det_match}", file=sys.stderr)
                all_results["deterministic_repeat"] = {"layer": 0, "family": "ffn_down", "match": det_match}
            break

    # === Summary by family ===
    print(f"\n{'='*60}", file=sys.stderr)
    print("SUMMARY BY TENSOR FAMILY", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    for model_key in ["0.5B", "3B"]:
        for family in ["ffn_down", "ffn_gate"]:
            family_results = [r for r in all_results[model_key] if r.get("family") == family and "error" not in r]
            if not family_results:
                print(f"\n{model_key} {family}: BLOCKED (no valid results)", file=sys.stderr)
                continue
            imps = [r['cosine_improvement'] for r in family_results]
            comps = [r['compression_ratio_vs_q4'] for r in family_results]
            strong = sum(1 for r in family_results if r['verdict'] == 'STRONG_RECOVERY')
            mean_imp = sum(imps) / len(imps)
            std_imp = (sum((x - mean_imp)**2 for x in imps) / len(imps)) ** 0.5
            print(f"\n{model_key} {family}:", file=sys.stderr)
            print(f"  Layers: {[r['layer'] for r in family_results]}", file=sys.stderr)
            print(f"  Shapes: {set(tuple(r['shape']) for r in family_results)}", file=sys.stderr)
            print(f"  Mean Δ cos: {mean_imp:+.4f} ± {std_imp:.4f}", file=sys.stderr)
            print(f"  Min/Max: {min(imps):+.4f} / {max(imps):+.4f}", file=sys.stderr)
            print(f"  Compressions: {set(comps)}", file=sys.stderr)
            print(f"  Strong: {strong}/{len(family_results)}", file=sys.stderr)
            print(f"  Verdict: {family_results[0]['verdict']}", file=sys.stderr)

    with open(args.out_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nJSON written to {args.out_json}", file=sys.stderr)


if __name__ == "__main__":
    main()