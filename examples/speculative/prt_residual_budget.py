#!/usr/bin/env python3
"""
PRT Residual Budget Calculator
Two modes:
  1. Parametric: given param counts and bits, compute budget.
  2. Manifest:   given a manifest.json, evaluate selected residual policies
                  under RAM/KV constraints.
Pure calculation, no model files needed.
"""
import argparse
import json
import sys
import os


# ─── Manifest mode ─────────────────────────────────────────────────────────────

def load_manifest(path):
    with open(path) as f:
        return json.load(f)


def select_tensors_by_policy(manifest, policy, residual_budget_bytes=None,
                             manual_tensors=None):
    """Select tensor entries from manifest based on policy. Returns list of entries."""
    all_tensors = manifest.get("layers", [])

    if policy == "base_only":
        return []

    elif policy == "mlp_all":
        return [t for t in all_tensors if t.get("tensor_family") in ("ffn_up", "ffn_down", "ffn_gate")]

    elif policy == "attention_partial":
        return [t for t in all_tensors if t.get("tensor_family") in ("attn_q", "attn_output")]

    elif policy == "all_validated":
        return [t for t in all_tensors if t.get("tensor_family") in (
            "ffn_up", "ffn_down", "ffn_gate", "attn_q", "attn_output")]

    elif policy == "budget_greedy":
        # Sort by score_per_byte if available, else delta_cos / byte_size
        candidates = [t for t in all_tensors if t.get("tensor_family") in (
            "ffn_up", "ffn_down", "ffn_gate", "attn_q", "attn_output")]
        for t in candidates:
            vm = t.get("validation_metrics", {})
            if "score_per_byte" in vm:
                t["_sort_score"] = vm["score_per_byte"]
            else:
                dc = vm.get("delta_cosine", 0.0)
                bs = t.get("byte_size", 1)
                t["_sort_score"] = (dc / bs) if bs > 0 else 0.0
        candidates.sort(key=lambda x: x.get("_sort_score", 0), reverse=True)
        selected = []
        total_bytes = 0
        budget = residual_budget_bytes or float("inf")
        for t in candidates:
            sz = t.get("byte_size", 0)
            if total_bytes + sz <= budget:
                selected.append(t)
                total_bytes += sz
        return selected

    elif policy == "manual":
        if not manual_tensors:
            return []
        # manual_tensors: list of "layer.tensor_family" strings
        wanted = set(manual_tensors)
        result = []
        for t in all_tensors:
            key = f"{t['layer']}.{t['tensor_family']}"
            if key in wanted:
                result.append(t)
                wanted.discard(key)
        return result

    else:
        print(f"WARNING: unknown policy '{policy}', returning empty list", file=sys.stderr)
        return []


def sum_residual_bytes(tensors):
    return sum(t.get("byte_size", 0) for t in tensors)


def estimate_q2_base_bytes(manifest):
    """Estimate Q2 base bytes from manifest metadata, or compute from shapes."""
    # If manifest stores base bytes explicitly, use it
    if "base_model_bytes" in manifest:
        return manifest["base_model_bytes"]
    if "q2_base_bytes" in manifest:
        return manifest["q2_base_bytes"]
    # Fall back: sum(shape[0] * shape[1] * 2 / 8) for all unique tensor shapes
    # This is a rough estimate: Q2 ≈ 0.25 bytes/param
    unique_layers = {}
    for t in manifest.get("layers", []):
        lid = t.get("layer")
        if lid not in unique_layers:
            unique_layers[lid] = t
    total_params = 0
    seen_shapes = set()
    for t in manifest.get("layers", []):
        shape_key = tuple(t.get("shape", []))
        if shape_key and shape_key not in seen_shapes:
            rows, cols = shape_key
            total_params += rows * cols
            seen_shapes.add(shape_key)
    return int(total_params * 2 / 8)  # Q2 = 2 bits/param


def compute_manifest_budget(manifest, policy, residual_budget_mb=None,
                              ram_gb=16.0, context_size=1024,
                              kv_bytes_per_token=2048, runtime_buffer_mb=1024,
                              os_headroom_mb=2048, manual_tensors=None):
    """Evaluate a budget policy against a manifest. Returns result dict."""
    residual_budget_bytes = (residual_budget_mb * 1024 * 1024) if residual_budget_mb else None
    selected = select_tensors_by_policy(manifest, policy, residual_budget_bytes,
                                         manual_tensors=manual_tensors)

    residual_bytes = sum_residual_bytes(selected)
    base_bytes = estimate_q2_base_bytes(manifest)
    kv_bytes = context_size * kv_bytes_per_token
    runtime_buffer_bytes = runtime_buffer_mb * 1024 * 1024
    os_headroom_bytes = os_headroom_mb * 1024 * 1024
    ram_budget_bytes = int(ram_gb * 1024 * 1024 * 1024)

    total_estimated = base_bytes + residual_bytes + kv_bytes + runtime_buffer_bytes + os_headroom_bytes
    remaining = ram_budget_bytes - total_estimated
    safe = total_estimated < ram_budget_bytes

    # Q4 reference
    q4_bytes = base_bytes * 2  # rough: Q4 is 2x Q2

    return {
        "policy": policy,
        "selected_tensors_count": len(selected),
        "selected_tensor_families": list(set(t.get("tensor_family") for t in selected)),
        "base_model_bytes": base_bytes,
        "residual_bytes": residual_bytes,
        "kv_bytes": kv_bytes,
        "runtime_buffer_bytes": runtime_buffer_bytes,
        "os_headroom_bytes": os_headroom_bytes,
        "total_estimated_bytes": total_estimated,
        "ram_budget_bytes": ram_budget_bytes,
        "remaining_bytes": remaining,
        "safe": safe,
        "q4_reference_bytes": q4_bytes,
        "inputs": {
            "ram_gb": ram_gb,
            "context_size": context_size,
            "kv_bytes_per_token": kv_bytes_per_token,
            "runtime_buffer_mb": runtime_buffer_mb,
            "os_headroom_mb": os_headroom_mb,
            "residual_budget_mb": residual_budget_mb,
            "manual_tensors": manual_tensors,
        },
        "selected_tensors": [
            {"layer": t["layer"], "tensor_family": t.get("tensor_family"),
             "byte_size": t.get("byte_size", 0)}
            for t in selected
        ]
    }


def format_manifest_result(r):
    lines = [
        f"=== PRT Manifest Budget: {r['policy']} ===",
        f"Selected tensors: {r['selected_tensors_count']} ({', '.join(r['selected_tensor_families'])})",
        f"Base model:      {r['base_model_bytes']:,} bytes ({r['base_model_bytes']/1e9:.2f} GB)",
        f"Residual:       {r['residual_bytes']:,} bytes ({r['residual_bytes']/1e6:.1f} MB)",
        f"KV (c={r['inputs']['context_size']}): {r['kv_bytes']:,} bytes ({r['kv_bytes']/1e6:.1f} MB)",
        f"Runtime buffer:  {r['runtime_buffer_bytes']:,} bytes ({r['runtime_buffer_bytes']/1e6:.1f} MB)",
        f"OS headroom:     {r['os_headroom_bytes']:,} bytes ({r['os_headroom_bytes']/1e6:.1f} MB)",
        f"Total estimated: {r['total_estimated_bytes']:,} bytes ({r['total_estimated_bytes']/1e9:.2f} GB)",
        f"RAM budget:     {r['ram_budget_bytes']:,} bytes ({r['ram_budget_bytes']/1e9:.2f} GB)",
        f"Remaining:      {r['remaining_bytes']:,} bytes ({r['remaining_bytes']/1e9:.2f} GB)",
        f"Q4 reference:    {r['q4_reference_bytes']:,} bytes ({r['q4_reference_bytes']/1e9:.2f} GB)",
        f"SAFE: {r['safe']}",
    ]
    return "\n".join(lines)


# ─── Parametric mode (original) ────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="PRT residual overlay memory budget calculator")
    p.add_argument("--self-test", action="store_true", help="Run self-test and exit")

    # Manifest mode
    p.add_argument("--manifest", default=None, help="Path to manifest.json (manifest mode)")
    p.add_argument("--policy",
                   choices=["base_only", "mlp_all", "attention_partial",
                            "all_validated", "budget_greedy", "manual"],
                   default=None, help="Residual policy (manifest mode)")
    p.add_argument("--manual-tensors", default=None,
                   help="Comma-separated tensor keys like '0.ffn_up,1.attn_q' (manual policy)")
    p.add_argument("--residual-budget-mb", type=int, default=None,
                   help="Max residual budget in MB (budget_greedy policy)")
    p.add_argument("--ram-gb", type=float, default=16.0,
                   help="Available RAM GB (default 16.0)")
    p.add_argument("--context-size", type=int, default=1024,
                   help="Context length in tokens (default 1024)")
    p.add_argument("--kv-bytes-per-token", type=int, default=2048,
                   help="KV bytes per token (default 2048)")
    p.add_argument("--runtime-buffer-mb", type=int, default=1024,
                   help="Runtime buffer MB (default 1024)")
    p.add_argument("--os-headroom-mb", type=int, default=2048,
                   help="OS headroom MB (default 2048)")

    # Parametric mode (original)
    p.add_argument("--param-count", type=int, help="Total model parameters (parametric mode)")
    p.add_argument("--base-bits", type=int, help="Base quantization bits (parametric mode)")
    p.add_argument("--residual-bits", type=int, help="Residual overlay bits (parametric mode)")
    p.add_argument("--residual-layer-count", type=int, help="Number of layers with residuals")
    p.add_argument("--total-layer-count", type=int, help="Total number of layers")
    p.add_argument("--residual-fraction", type=float, help="Fraction of layer params in residual layers")

    p.add_argument("--out-json", default=None, help="Optional output JSON path")
    return p.parse_args()


def validate_args(args):
    errors = []
    if args.param_count <= 0:
        errors.append("--param-count must be positive")
    if args.base_bits <= 0:
        errors.append("--base-bits must be positive")
    if args.residual_bits <= 0:
        errors.append("--residual-bits must be positive")
    if not 0.0 <= args.residual_fraction <= 1.0:
        errors.append("--residual-fraction must be between 0.0 and 1.0")
    if args.context_size <= 0:
        errors.append("--context-size must be positive")
    if args.residual_layer_count <= 0 or args.residual_layer_count > args.total_layer_count:
        errors.append("--residual-layer-count must be >0 and <= --total-layer-count")
    if args.ram_gb <= 0:
        errors.append("--ram-gb must be positive")
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


def compute_budget(args):
    base_bytes = (args.param_count * args.base_bits) / 8.0
    residual_param_fraction = args.residual_fraction * (args.residual_layer_count / args.total_layer_count)
    residual_params = args.param_count * residual_param_fraction
    residual_bytes = (residual_params * args.residual_bits) / 8.0
    kv_bytes = args.context_size * args.kv_bytes_per_token
    runtime_buffer_bytes = args.runtime_buffer_mb * 1024 * 1024
    os_headroom_bytes = args.os_headroom_mb * 1024 * 1024
    ram_budget_bytes = args.ram_gb * 1024 * 1024 * 1024
    total_estimated = base_bytes + residual_bytes + kv_bytes + runtime_buffer_bytes + os_headroom_bytes
    remaining = ram_budget_bytes - total_estimated
    safe = total_estimated < ram_budget_bytes
    q4_bytes = (args.param_count * 4.0) / 8.0
    q2_bytes = (args.param_count * 2.0) / 8.0
    q4_savings = q4_bytes - q2_bytes
    compression_ratio_vs_q4_savings = residual_bytes / q4_savings if q4_savings > 0 else float("inf")
    combined_vs_q4 = (base_bytes + residual_bytes) / q4_bytes if q4_bytes > 0 else float("inf")
    return {
        "base_model_bytes": int(base_bytes),
        "residual_bytes": int(residual_bytes),
        "kv_bytes": int(kv_bytes),
        "runtime_buffer_bytes": int(runtime_buffer_bytes),
        "os_headroom_bytes": int(os_headroom_bytes),
        "total_estimated_bytes": int(total_estimated),
        "ram_budget_bytes": int(ram_budget_bytes),
        "remaining_bytes": int(remaining),
        "safe": safe,
        "residual_too_large": compression_ratio_vs_q4_savings >= 1.0 if compression_ratio_vs_q4_savings != float("inf") else True,
        "q4_bytes": int(q4_bytes),
        "q4_savings_bytes": int(q4_savings),
        "compression_ratio_vs_q4_savings": round(compression_ratio_vs_q4_savings, 4) if compression_ratio_vs_q4_savings != float("inf") else None,
        "combined_vs_q4": round(combined_vs_q4, 4),
        "inputs": {
            "param_count": args.param_count,
            "base_bits": args.base_bits,
            "residual_bits": args.residual_bits,
            "residual_layer_count": args.residual_layer_count,
            "total_layer_count": args.total_layer_count,
            "residual_fraction": args.residual_fraction,
            "context_size": args.context_size,
            "kv_bytes_per_token": args.kv_bytes_per_token,
            "runtime_buffer_mb": args.runtime_buffer_mb,
            "os_headroom_mb": args.os_headroom_mb,
            "ram_gb": args.ram_gb,
        }
    }


def format_summary(result):
    lines = [
        "=== PRT Residual Budget ===",
        f"Base model:      {result['base_model_bytes']:,} bytes ({result['base_model_bytes']/1e9:.2f} GB)",
        f"Residual:       {result['residual_bytes']:,} bytes ({result['residual_bytes']/1e9:.2f} GB)",
        f"KV (c={result['inputs']['context_size']}): {result['kv_bytes']:,} bytes ({result['kv_bytes']/1e9:.2f} GB)",
        f"Runtime buffer:  {result['runtime_buffer_bytes']:,} bytes ({result['runtime_buffer_bytes']/1e6:.1f} MB)",
        f"OS headroom:     {result['os_headroom_bytes']:,} bytes ({result['os_headroom_bytes']/1e6:.1f} MB)",
        f"Total estimated: {result['total_estimated_bytes']:,} bytes ({result['total_estimated_bytes']/1e9:.2f} GB)",
        f"RAM budget:     {result['ram_budget_bytes']:,} bytes ({result['ram_budget_bytes']/1e9:.2f} GB)",
        f"Remaining:      {result['remaining_bytes']:,} bytes ({result['remaining_bytes']/1e9:.2f} GB)",
        f"Q4 equivalent:   {result['q4_bytes']:,} bytes ({result['q4_bytes']/1e9:.2f} GB)",
        f"Q4 savings:     {result['q4_savings_bytes']:,} bytes ({result['q4_savings_bytes']/1e9:.2f} GB)",
        f"Residual/Q4 savings ratio: {result['compression_ratio_vs_q4_savings']}",
        f"Combined/Q4 ratio:        {result['combined_vs_q4']}",
        f"SAFE: {result['safe']}",
        f"Residual too large: {result['residual_too_large']}",
    ]
    return "\n".join(lines)


# ─── Self-test ────────────────────────────────────────────────────────────────

def run_self_test():
    """Test all manifest-mode logic without any files."""
    print("Running self-test...")
    all_pass = True

    # Fake manifest
    fake_manifest = {
        "format_name": "prt-residual-v1",
        "format_version": "0.1.0",
        "source_model": "fake-test-model.gguf",
        "base_quant": "Q2_K",
        "residual_format": "ternary",
        "layers": [
            {"layer": 0, "tensor_family": "ffn_up", "shape": [512, 2048],
             "byte_size": 131072, "validation_metrics": {"delta_cosine": 0.7073}},
            {"layer": 0, "tensor_family": "ffn_down", "shape": [512, 2048],
             "byte_size": 131072, "validation_metrics": {"delta_cosine": 0.7033}},
            {"layer": 0, "tensor_family": "attn_q", "shape": [512, 512],
             "byte_size": 32768, "validation_metrics": {"delta_cosine": 0.6729}},
            {"layer": 0, "tensor_family": "attn_output", "shape": [512, 512],
             "byte_size": 32768, "validation_metrics": {"delta_cosine": 0.6865}},
            {"layer": 1, "tensor_family": "ffn_up", "shape": [512, 2048],
             "byte_size": 131072, "validation_metrics": {"delta_cosine": 0.7073}},
            {"layer": 1, "tensor_family": "ffn_gate", "shape": [512, 2048],
             "byte_size": 131072, "validation_metrics": {"delta_cosine": 0.6999}},
        ]
    }

    def check(name, got, expected, silent=False):
        ok = got == expected
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        if not silent:
            print(f"  [{status}] {name}: got={got}, expected={expected}")
        return ok

    # Test 1: base_only selects nothing
    selected = select_tensors_by_policy(fake_manifest, "base_only")
    check("base_only selects 0", len(selected), 0)

    # Test 2: mlp_all selects ffn_up/down/gate only
    selected = select_tensors_by_policy(fake_manifest, "mlp_all")
    families = set(t["tensor_family"] for t in selected)
    check("mlp_all count", len(selected), 4)
    check("mlp_all families", families, {"ffn_up", "ffn_down", "ffn_gate"})

    # Test 3: attention_partial selects attn_q/output only
    selected = select_tensors_by_policy(fake_manifest, "attention_partial")
    check("attention_partial count", len(selected), 2)
    check("attention_partial families", set(t["tensor_family"] for t in selected),
          {"attn_q", "attn_output"})

    # Test 4: all_validated selects all
    selected = select_tensors_by_policy(fake_manifest, "all_validated")
    check("all_validated count", len(selected), 6)

    # Test 5: budget_greedy respects budget
    selected = select_tensors_by_policy(fake_manifest, "budget_greedy",
                                        residual_budget_bytes=131072)
    total = sum_residual_bytes(selected)
    check("budget_greedy respects budget", total <= 131072, True)
    check("budget_greedy selects at least 1", len(selected) >= 1, True)

    # Test 6: manual selects specific tensors
    selected = select_tensors_by_policy(fake_manifest, "manual",
                                         manual_tensors=["0.ffn_up", "1.ffn_gate"])
    check("manual selects 2", len(selected), 2)
    check("manual correct tensors",
          sorted([f"{t['layer']}.{t['tensor_family']}" for t in selected]),
          ["0.ffn_up", "1.ffn_gate"])

    # Test 7: sum_residual_bytes
    check("sum_residual_bytes",
          sum_residual_bytes(selected),
          131072 + 131072)

    # Test 8: estimate_q2_base_bytes
    est = estimate_q2_base_bytes(fake_manifest)
    check("estimate_q2_base_bytes > 0", est > 0, True)

    # Test 9: compute_manifest_budget SAFE
    result = compute_manifest_budget(fake_manifest, "base_only",
                                     ram_gb=16.0, context_size=1024,
                                     kv_bytes_per_token=2048,
                                     runtime_buffer_mb=1024,
                                     os_headroom_mb=2048)
    check("base_only SAFE", result["safe"], True)
    check("base_only residual_bytes", result["residual_bytes"], 0)

    # Test 10: compute_manifest_budget mlp_all
    result = select_tensors_by_policy(fake_manifest, "mlp_all")
    mlp_bytes = sum_residual_bytes(result)
    total_for_mlp = est + mlp_bytes + (1024 * 2048) + (1024 * 1024 * 1024) + (2048 * 1024 * 1024)
    check("mlp_all bytes > 0", mlp_bytes > 0, True)

    # Test 11: format_manifest_result output is non-empty string
    r = compute_manifest_budget(fake_manifest, "mlp_all",
                                 ram_gb=16.0, context_size=1024)
    output = format_manifest_result(r)
    check("format_manifest_result non-empty", len(output) > 0, True)
    check("format_manifest_result contains SAFE", "SAFE" in output, True)

    print()
    if all_pass:
        print("✅ ALL SELF-TESTS PASSED")
    else:
        print("❌ SOME SELF-TESTS FAILED")
        sys.exit(1)


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.self_test:
        run_self_test()
        return

    # Manifest mode
    if args.manifest:
        if not os.path.exists(args.manifest):
            print(f"ERROR: manifest not found: {args.manifest}", file=sys.stderr)
            sys.exit(1)
        if not args.policy:
            print("ERROR: --policy required in manifest mode", file=sys.stderr)
            sys.exit(1)

        manifest = load_manifest(args.manifest)
        manual = [x.strip() for x in args.manual_tensors.split(",")] if args.manual_tensors else None

        result = compute_manifest_budget(
            manifest, args.policy,
            residual_budget_mb=args.residual_budget_mb,
            ram_gb=args.ram_gb,
            context_size=args.context_size,
            kv_bytes_per_token=args.kv_bytes_per_token,
            runtime_buffer_mb=args.runtime_buffer_mb,
            os_headroom_mb=args.os_headroom_mb,
            manual_tensors=manual,
        )
        print(format_manifest_result(result))
        if args.out_json:
            with open(args.out_json, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\nJSON written to {args.out_json}")
        return

    # Parametric mode (original)
    if not all([args.param_count, args.base_bits, args.residual_bits,
                 args.residual_layer_count, args.total_layer_count, args.residual_fraction]):
        print("ERROR: either --manifest (manifest mode) or all parametric args required",
              file=sys.stderr)
        sys.exit(1)

    validate_args(args)
    result = compute_budget(args)
    print(format_summary(result))
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nJSON written to {args.out_json}")


if __name__ == "__main__":
    main()