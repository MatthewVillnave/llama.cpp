#!/usr/bin/env python3
"""
PRT Residual Budget Calculator
Computes memory budget for Q2 base + residual overlay on constrained hardware.
Pure calculation, no model files needed.
"""
import argparse
import json
import sys


def parse_args():
    p = argparse.ArgumentParser(description="PRT residual overlay memory budget calculator")
    p.add_argument("--param-count", type=int, required=True, help="Total model parameters")
    p.add_argument("--base-bits", type=int, required=True, help="Base quantization bits (e.g. 2 for Q2)")
    p.add_argument("--residual-bits", type=int, required=True, help="Residual overlay bits (1=ternary, 2=int2, 4=int4)")
    p.add_argument("--residual-layer-count", type=int, required=True, help="Number of layers with residuals")
    p.add_argument("--total-layer-count", type=int, required=True, help="Total number of layers")
    p.add_argument("--residual-fraction", type=float, required=True, help="Fraction of layer params in residual layers (0.0-1.0)")
    p.add_argument("--context-size", type=int, required=True, help="Context length in tokens")
    p.add_argument("--kv-bytes-per-token", type=int, default=2048, help="KV bytes per token (default 2048)")
    p.add_argument("--runtime-buffer-mb", type=int, default=1024, help="Runtime buffer MB (default 1024)")
    p.add_argument("--os-headroom-mb", type=int, default=2048, help="OS headroom MB (default 2048)")
    p.add_argument("--ram-gb", type=float, default=16.0, help="Available RAM GB (default 16.0)")
    p.add_argument("--out-json", type=str, default=None, help="Optional output JSON path")
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
    # Base model bytes = param_count * base_bits / 8
    base_bytes = (args.param_count * args.base_bits) / 8.0

    # Residual bytes: only for residual-selected layers
    # residual params = param_count * residual_fraction
    # residual layer params = residual_params * (residual_layer_count / total_layer_count)
    residual_param_fraction = args.residual_fraction * (args.residual_layer_count / args.total_layer_count)
    residual_params = args.param_count * residual_param_fraction
    residual_bytes = (residual_params * args.residual_bits) / 8.0

    # KV: context_size * kv_bytes_per_token
    kv_bytes = args.context_size * args.kv_bytes_per_token

    # Buffers and OS headroom
    runtime_buffer_bytes = args.runtime_buffer_mb * 1024 * 1024
    os_headroom_bytes = args.os_headroom_mb * 1024 * 1024
    ram_budget_bytes = args.ram_gb * 1024 * 1024 * 1024

    # Total
    total_estimated = base_bytes + residual_bytes + kv_bytes + runtime_buffer_bytes + os_headroom_bytes
    remaining = ram_budget_bytes - total_estimated
    safe = total_estimated < ram_budget_bytes

    # Memory saved by Q2 vs Q4: Q4 needs 4 bits, Q2 needs 2 bits
    # saving per base param = (4 - 2) / 8 = 0.25 bytes per param
    q4_bytes = (args.param_count * 4.0) / 8.0
    q2_bytes = (args.param_count * 2.0) / 8.0
    q4_savings = q4_bytes - q2_bytes  # bytes saved using Q2 vs Q4

    # Compression ratio: residual vs the savings from using Q2 instead of Q4
    # If residual > q4_savings, we're not saving memory vs just using Q4
    compression_ratio_vs_q4_savings = residual_bytes / q4_savings if q4_savings > 0 else float('inf')

    # Combined base+residual vs Q4
    combined_vs_q4 = (base_bytes + residual_bytes) / q4_bytes if q4_bytes > 0 else float('inf')

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
        "residual_too_large": compression_ratio_vs_q4_savings >= 1.0 if compression_ratio_vs_q4_savings != float('inf') else True,
        "q4_bytes": int(q4_bytes),
        "q4_savings_bytes": int(q4_savings),
        "compression_ratio_vs_q4_savings": round(compression_ratio_vs_q4_savings, 4) if compression_ratio_vs_q4_savings != float('inf') else None,
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


def main():
    args = parse_args()
    validate_args(args)
    result = compute_budget(args)
    print(format_summary(result))
    if args.out_json:
        with open(args.out_json, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nJSON written to {args.out_json}")


if __name__ == "__main__":
    main()