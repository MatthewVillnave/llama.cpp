#!/usr/bin/env python3
"""
PRT Phase 28W: Attention Projection Ternary Residual Validation
Validates Q2+ternary recovery on attention projections across 0.5B and 3B.
"""
import argparse
import json
import os
import sys

try:
    import numpy as np
    import gguf
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


def q2_quantize(w_ref):
    levels = np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float32)
    q_order = {1.5: 3, 0.5: 2, -0.5: 1, -1.5: 0}
    def encode(x):
        x_c = max(-1.5, min(1.5, x))
        idx = int(round((x_c - (-1.5)) / 0.5))
        return q_order[levels[max(0, min(3, idx))]]
    return levels[np.vectorize(encode)(w_ref)].astype(np.float32)


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


def run_test(W_ref, rows, cols, seed=123):
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

    comp = (int(rows * cols * 0.25) + int(rows * cols * 0.125)) / int(rows * cols * 0.5)
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


def extract_attn_output(reader, layer, rows, cols):
    """Extract attn_output slice. 2D weight matrix: [hidden, hidden]."""
    name = f"blk.{layer}.attn_output.weight"
    for t in reader.tensors:
        if t.name == name:
            w = gguf.dequantize(t.data, t.tensor_type)
            # w: (hidden, hidden) — take [:rows, :cols] directly
            slice_mat = w[:rows, :cols]
            return slice_mat, t.tensor_type.name, list(t.shape)
    raise ValueError(f"Tensor not found: {name}")


def extract_attn_qkv(reader, layer, rows, cols, family):
    """Extract q/k/v projection slice. These may be 1D or 2D depending on storage."""
    name = f"blk.{layer}.{family}.weight"
    for t in reader.tensors:
        if t.name == name:
            w = gguf.dequantize(t.data, t.tensor_type)
            print(f"    {family} dequant shape: {w.shape}, ndim: {w.ndim}", file=sys.stderr)
            if w.ndim == 1:
                # 1D — can't do 2D matvec. Try reshape if factors work.
                n = w.shape[0]
                # Try (heads, head_dim) reshaping
                factors = [(h, n//h) for h in range(1, int(n**0.5)+1) if n % h == 0]
                for h, hd in factors:
                    if hd > 10:  # reasonable head_dim
                        print(f"    {family} trying reshape ({h} heads x {hd} = {n})", file=sys.stderr)
                        w2d = w.reshape(h, hd)
                        if w2d.shape[0] >= rows and w2d.shape[1] >= cols:
                            slice_mat = w2d[:rows, :cols]
                            print(f"    {family} using slice {slice_mat.shape}", file=sys.stderr)
                            return slice_mat, t.tensor_type.name, list(t.shape)
                print(f"    {family}: 1D tensor, cannot reshape to 2D for matvec — BLOCKED", file=sys.stderr)
                return None, t.tensor_type.name, list(t.shape)
            elif w.ndim == 2:
                # 2D: take [:rows, :cols]
                slice_mat = w[:rows, :cols]
                return slice_mat, t.tensor_type.name, list(t.shape)
            else:
                print(f"    {family}: unsupported ndim {w.ndim} — BLOCKED", file=sys.stderr)
                return None, t.tensor_type.name, list(t.shape)
    raise ValueError(f"Tensor not found: {name}")


def test_model(model_path, model_name, configs, out_json):
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Testing {model_name}: {model_path.split('/')[-1]}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    reader = gguf.GGUFReader(model_path)
    results = []

    for cfg in configs:
        layer = cfg["layer"]
        family = cfg["family"]
        rows = cfg["rows"]
        cols = cfg["cols"]
        cache_path = f"/tmp/attn_{model_name}_L{layer}_{family}.f32"

        if os.path.exists(cache_path):
            W_ref = np.fromfile(cache_path, dtype=np.float32).reshape(rows, cols)
            print(f"  L{layer} {family}: loaded cached {W_ref.shape}", file=sys.stderr)
        else:
            print(f"  L{layer} {family}: extracting...", file=sys.stderr)
            try:
                if family == "attn_output":
                    W_ref, qtype, full_shape = extract_attn_output(reader, layer, rows, cols)
                else:
                    W_ref, qtype, full_shape = extract_attn_qkv(reader, layer, rows, cols, family)
                
                if W_ref is None:
                    results.append({"layer": layer, "family": family, "verdict": "BLOCKED_EXTRACTION", "error": "1D tensor cannot be reshaped"})
                    continue
                    
                W_ref.tofile(cache_path)
                print(f"  L{layer} {family}: saved {W_ref.shape} to {cache_path}", file=sys.stderr)
            except Exception as e:
                print(f"  L{layer} {family}: EXTRACTION FAILED: {e}", file=sys.stderr)
                results.append({"layer": layer, "family": family, "verdict": "BLOCKED_EXTRACTION", "error": str(e)})
                continue

        r = run_test(W_ref, rows, cols, seed=123)
        r["layer"] = layer
        r["family"] = family
        r["shape"] = [rows, cols]
        r["qtype"] = qtype

        print(f"    Q2 cos={r['q2_cosine']:.4f} Q2+T={r['q2_ternary_cosine']:.4f} "
              f"Δ={r['cosine_improvement']:+.4f} MAE={r['mae_hat']:.4f} "
              f"comp={r['compression_ratio_vs_q4']:.4f} {r['verdict']}", file=sys.stderr)

        results.append(r)

    return results


def main():
    if not HAS_DEPS:
        print("ERROR: missing numpy or gguf", file=sys.stderr)
        sys.exit(1)

    out_json = "/tmp/phase28w_results.json"

    # === 0.5B: attn_output (512x512), q_proj attempt ===
    # attn_output 0.5B: meta [896, 896], dequant (896, 896), slice [:512, :512] -> (512, 512)
    # q_proj 0.5B: meta [896, 896] Q5_0, dequant (896, 896) -> (512, 512) if we take [:512,:512]
    
    configs_05b = []
    for layer in [0, 5, 11, 23]:
        # attn_output: 512x512 slice
        configs_05b.append({"layer": layer, "family": "attn_output", "rows": 512, "cols": 512})
        # q_proj: try 512x512
        configs_05b.append({"layer": layer, "family": "attn_q", "rows": 512, "cols": 512})

    results_05b = test_model(
        "/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf",
        "05B", configs_05b, out_json
    )

    # === 3B: attn_output (512x512), q_proj attempt ===
    # attn_output 3B: meta [2048, 2048], slice [:512, :512] -> (512, 512)
    # q_proj 3B: meta [2048, 2048], slice [:512, :512] -> (512, 512)
    # k_proj 3B: meta [256] maybe 2D reshape possible
    
    configs_3b = []
    for layer in [0, 8, 17, 26, 35]:
        configs_3b.append({"layer": layer, "family": "attn_output", "rows": 512, "cols": 512})
        configs_3b.append({"layer": layer, "family": "attn_q", "rows": 512, "cols": 512})

    results_3b = test_model(
        "/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-3B-Instruct-Q4_K_M.gguf",
        "3B", configs_3b, out_json
    )

    # === Deterministic repeat ===
    print(f"\nDeterministic repeat check (3B attn_output L0)...", file=sys.stderr)
    for r in results_3b:
        if r.get("layer") == 0 and r.get("family") == "attn_output" and r.get("verdict") != "BLOCKED_EXTRACTION":
            cache = "/tmp/attn_3B_L0_attn_output.f32"
            if os.path.exists(cache):
                W = np.fromfile(cache, dtype=np.float32).reshape(512, 512)
                r2 = run_test(W, 512, 512, seed=123)
                det_match = abs(r2['q2_cosine'] - r['q2_cosine']) < 1e-6 and \
                            abs(r2['q2_ternary_cosine'] - r['q2_ternary_cosine']) < 1e-6
                print(f"  Deterministic match: {det_match}", file=sys.stderr)
                break

    # === Summary ===
    print(f"\n{'='*60}", file=sys.stderr)
    print("SUMMARY BY TENSOR FAMILY", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    all_results = {"0.5B": results_05b, "3B": results_3b}

    for model_key in ["0.5B", "3B"]:
        families = set(r.get("family") for r in all_results[model_key])
        for family in sorted(families):
            fam_results = [r for r in all_results[model_key] if r.get("family") == family and r.get("verdict") != "BLOCKED_EXTRACTION"]
            blocked = [r for r in all_results[model_key] if r.get("family") == family and r.get("verdict") == "BLOCKED_EXTRACTION"]
            
            if fam_results:
                imps = [r['cosine_improvement'] for r in fam_results]
                mean_imp = sum(imps) / len(imps)
                std_imp = (sum((x - mean_imp)**2 for x in imps) / len(imps)) ** 0.5
                strong = sum(1 for r in fam_results if r['verdict'] == 'STRONG_RECOVERY')
                print(f"\n{model_key} {family}:", file=sys.stderr)
                print(f"  Layers: {[r['layer'] for r in fam_results]}", file=sys.stderr)
                print(f"  Mean Δ cos: {mean_imp:+.4f} ± {std_imp:.4f}", file=sys.stderr)
                print(f"  Min/Max: {min(imps):+.4f} / {max(imps):+.4f}", file=sys.stderr)
                print(f"  Strong: {strong}/{len(fam_results)}", file=sys.stderr)
            if blocked:
                print(f"\n{model_key} {family}: BLOCKED ({len(blocked)}) — {[r.get('error', 'unknown') for r in blocked]}", file=sys.stderr)

    all_results["deterministic"] = {"family": "attn_output", "layer": 0, "match": det_match}

    with open(out_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nJSON written to {out_json}", file=sys.stderr)


if __name__ == "__main__":
    main()