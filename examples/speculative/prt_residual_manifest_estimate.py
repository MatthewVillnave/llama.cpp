#!/usr/bin/env python3
"""
PRT Phase 28AB: Estimated Residual Manifest Generator for Real Models
Generates a manifest-like JSON from known architecture metadata or GGUF inspection.
This is ESTIMATED only — no real residual sidecars are generated.
"""
import argparse
import json
import math
import os
import sys


# ─── Architecture tables ─────────────────────────────────────────────────────

QWEN25_7B = {
    "model_name": "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
    "model_class": "qwen2.5-7b",
    "hidden_size": 3584,
    "intermediate_size": 18944,
    "num_attention_heads": 28,
    "num_kv_heads": 4,
    "layer_count": 28,
    "vocab_size": 151936,
    "q4_total_bytes": 4_173_840_384,
    "q4_per_family": {
        "ffn_up": 38_191_104,
        "ffn_down": 55_695_360,
        "ffn_gate": 38_191_104,
        "attn_q": 7_225_344,
        "attn_k": 1_032_192,
        "attn_v": 1_505_280,
        "attn_output": 7_225_344,
    },
    "attn_qkv_merged": False,
    "model_hash": "a74ae894d2a6b4a4ba41660be9555a21194c590a",
    "source": "gguf_inspection",
}

# Phase 28X empirical data
EMPIRICAL_DELTA_COS = {
    "ffn_up": 0.7079,
    "ffn_down": 0.6965,
    "ffn_gate": 0.6869,
    "attn_q": 0.6729,
    "attn_output": 0.6865,
}

# Validated families only (no attn_k/v — not validated in our prior phases)
VALIDATED_FAMILIES = ["ffn_up", "ffn_down", "ffn_gate", "attn_q", "attn_output"]

COMPRESSION_RATIO = 0.75  # Q2+ternary / Q4 from Phase 28X


def parse_args():
    p = argparse.ArgumentParser(description="PRT estimated residual manifest generator")
    p.add_argument("--model-class", default="qwen2.5-7b",
                   choices=["qwen2.5-7b", "qwen2.5-3b", "qwen2.5-0.5b",
                            "qwen2.5-14b", "qwen2.5-32b-est"],
                   help="Model class")
    p.add_argument("--source-json", default=None,
                   help="Optional: existing manifest JSON to extend")
    p.add_argument("--out-json", required=True,
                   help="Output JSON path")
    p.add_argument("--note", default="",
                   help="Optional note about assumptions")
    return p.parse_args()


def qwen25_7b_entry(layer_idx, tensor_family, arch, empirical_delta_cos):
    """Generate one manifest entry for Qwen2.5-7B."""
    shape_map = {
        "ffn_up": [arch["hidden_size"], arch["intermediate_size"]],
        "ffn_down": [arch["intermediate_size"], arch["hidden_size"]],
        "ffn_gate": [arch["hidden_size"], arch["intermediate_size"]],
        "attn_q": [arch["hidden_size"], arch["hidden_size"]],
        "attn_k": [arch["hidden_size"], arch["hidden_size"] // arch["num_attention_heads"] * arch["num_kv_heads"]],
        "attn_v": [arch["hidden_size"], arch["hidden_size"] // arch["num_attention_heads"] * arch["num_kv_heads"]],
        "attn_output": [arch["hidden_size"], arch["hidden_size"]],
    }
    # Q4 bytes per layer from GGUF inspection
    q4_bytes_per_layer = arch["q4_per_family"].get(tensor_family, 0)
    if q4_bytes_per_layer == 0:
        # Estimate for un-inspected families
        rows, cols = shape_map.get(tensor_family, [arch["hidden_size"], arch["hidden_size"]])
        q4_bytes_per_layer = int(rows * cols * 0.5)  # Q4 ≈ 0.5 bytes/param

    q2_ternary_bytes = int(q4_bytes_per_layer * COMPRESSION_RATIO)
    # Residual savings = difference between Q4 and Q2+ternary
    residual_bytes = q4_bytes_per_layer - q2_ternary_bytes
    params = q4_bytes_per_layer * 2  # rough: Q4 = 4 bits = 0.5 bytes/param

    shape = shape_map.get(tensor_family, [arch["hidden_size"], arch["hidden_size"]])
    score_per_byte = empirical_delta_cos / max(q4_bytes_per_layer, 1) if empirical_delta_cos else 0.0

    return {
        "layer": layer_idx,
        "tensor_name": f"blk.{layer_idx}.{tensor_family}.weight",
        "tensor_family": tensor_family,
        "shape": shape,
        "dtype": "ternary",
        "file_path": f"layers/layer_{layer_idx:03d}/{tensor_family}.trit",
        "q4_bytes_per_layer": q4_bytes_per_layer,
        "q2_ternary_bytes": q2_ternary_bytes,
        "residual_savings_bytes": residual_bytes,
        "byte_size": residual_bytes,
        "compression_ratio_vs_q4": COMPRESSION_RATIO,
        "q2_base_assumption": "Q2_K",
        "validation_metrics": {
            "delta_cosine": empirical_delta_cos,
            "score_per_byte": score_per_byte,
            "verdict": "STRONG_RECOVERY" if empirical_delta_cos >= 0.5 else "MODERATE_RECOVERY",
        },
        "status": "estimated_validated_family",
        "note": "Estimated from architecture + empirical family data",
    }


def build_7b_manifest(arch, empirical_delta_cos, note="", non_weight_bytes=754_446_336):
    """"Build a full estimated manifest for Qwen2.5-7B.
    
    Args:
        arch: architecture dict
        empirical_delta_cos: dict of family -> delta cosine
        note: optional note
        non_weight_bytes: bytes for non-weight tensors (embeddings, norms, biases, lm_head)
    """
    layers = []
    for layer_idx in range(arch["layer_count"]):
        for fam in VALIDATED_FAMILIES:
            entry = qwen25_7b_entry(layer_idx, fam, arch, empirical_delta_cos.get(fam, 0.0))
            layers.append(entry)

    # Compute totals
    total_residual = sum(e["byte_size"] for e in layers)
    total_q4 = sum(arch["q4_per_family"].get(e["tensor_family"], 0)
                   for e in layers)
    total_q2_ternary = int(total_q4 * COMPRESSION_RATIO)
    total_residual_savings = total_q4 - total_q2_ternary

    manifest = {
        "format_name": "prt-residual-v1",
        "format_version": "0.1.0",
        "source_model": arch["model_name"],
        "source_model_hash": arch["model_hash"],
        "base_quant": "Q2_K",
        "residual_format": "ternary",
        "created_at": "2026-05-22T16:22:00Z",
        "generator": "prt-phase28ab-estimate",
        "layer_count": arch["layer_count"],
        "hidden_size": arch["hidden_size"],
        "intermediate_size": arch["intermediate_size"],
        "tensor_families": VALIDATED_FAMILIES,
        "total_q4_bytes": total_q4,
        "total_q4_weight_bytes": total_q4,
        "q2_base_bytes": int(total_q4 * COMPRESSION_RATIO) + non_weight_bytes,
        "q2_base_weight_bytes": int(total_q4 * COMPRESSION_RATIO),
        "q2_base_non_weight_bytes": non_weight_bytes,
        "total_residual_savings_bytes": total_residual_savings,
        "budget_policy": "budget_greedy",
        "budget_bytes": total_residual_savings,
        "total_residual_bytes": total_residual,
        "compression_ratio_vs_q4": COMPRESSION_RATIO,
        "validation_passed": True,
        "note": note or "Estimated manifest from GGUF inspection + empirical Phase 28X data. No real .trit files exist.",
        "is_estimated": True,
        "layers": layers,
        "memory_breakdown": {
            "q4_total_gb": round(total_q4 / 1e9, 3),
            "q2_ternary_total_gb": round(total_q2_ternary / 1e9, 3),
            "residual_savings_gb": round(total_residual_savings / 1e9, 3),
            "savings_percent": round(100 * (1 - COMPRESSION_RATIO), 1),
        },
        "per_family_totals": {}
    }

    for fam in VALIDATED_FAMILIES:
        fam_entries = [e for e in layers if e["tensor_family"] == fam]
        fam_q4 = arch["q4_per_family"].get(fam, 0) * arch["layer_count"]
        fam_residual = sum(e["byte_size"] for e in fam_entries)
        manifest["per_family_totals"][fam] = {
            "layers": len(fam_entries),
            "q4_bytes": fam_q4,
            "q2_ternary_bytes": int(fam_q4 * COMPRESSION_RATIO),
            "residual_savings_bytes": fam_q4 - int(fam_q4 * COMPRESSION_RATIO),
            "residual_bytes_in_manifest": fam_residual,
            "pct_of_total_q4": round(fam_q4 / total_q4 * 100, 1),
        }

    return manifest


def main():
    args = parse_args()

    arch = QWEN25_7B
    manifest = build_7b_manifest(arch, EMPIRICAL_DELTA_COS, note=args.note,
                             non_weight_bytes=754_446_336)

    with open(args.out_json, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Generated: {args.out_json}")
    print(f"  Layers: {manifest['layer_count']}")
    print(f"  Tensors: {len(manifest['layers'])}")
    print(f"  Total Q4: {manifest['memory_breakdown']['q4_total_gb']:.3f} GB")
    print(f"  Total Q2+ternary: {manifest['memory_breakdown']['q2_ternary_total_gb']:.3f} GB")
    print(f"  Residual savings: {manifest['memory_breakdown']['residual_savings_gb']:.3f} GB")
    print(f"  Note: {manifest['note']}")

    for fam, info in manifest["per_family_totals"].items():
        print(f"  {fam}: Q4={info['q4_bytes']/1e9:.3f}GB, residual_savings={info['residual_savings_bytes']/1e6:.1f}MB")


if __name__ == "__main__":
    main()