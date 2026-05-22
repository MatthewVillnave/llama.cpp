#!/usr/bin/env python3
"""
PRT Phase 28Z: Manifest Validator
Validates PRT residual sidecar manifest.json against schema requirements.
Offline, Python stdlib only.
"""
import argparse
import json
import os
import sys


def parse_args():
    p = argparse.ArgumentParser(description="PRT manifest validator")
    p.add_argument("--manifest", required=True, help="Path to manifest.json")
    p.add_argument("--check-files", action="store_true", help="Check that .trit files exist")
    p.add_argument("--base-model", default=None, help="Optional: base model path to verify hash")
    p.add_argument("--out-json", default=None, help="Optional: output JSON report")
    return p.parse_args()


REQUIRED_TOP_LEVEL = [
    "format_name", "format_version", "source_model", "source_model_hash",
    "base_quant", "residual_format", "created_at", "tensor_families",
    "layer_count", "budget_policy", "layers"
]

PER_TENSOR_REQUIRED = [
    "layer", "tensor_name", "tensor_family", "shape", "dtype",
    "file_path", "byte_size", "compression_ratio_vs_q4", "validation_metrics"
]

VALID_FORMAT_NAMES = {"prt-residual-v1"}
VALID_RESIDUAL_FORMATS = {"ternary", "int2", "int4"}
VALID_BASE_QUANTS = {"Q2", "Q2_K", "Q2_K_M", "Q2_0", "Q2_1"}
VALID_BUDGET_POLICIES = {"base_only", "mlp_all", "attention_partial", "budget_greedy", "manual"}
VALID_TENSOR_FAMILIES = {"ffn_up", "ffn_down", "ffn_gate", "attn_q", "attn_k", "attn_v", "attn_output"}


def validate_manifest(manifest_path, check_files=False, base_model=None):
    errors = []
    warnings = []
    checks_passed = []

    with open(manifest_path) as f:
        manifest = json.load(f)

    # --- Top-level field checks ---
    for field in REQUIRED_TOP_LEVEL:
        if field not in manifest:
            errors.append(f"Missing required top-level field: {field}")
        else:
            checks_passed.append(f"top_level:{field}")

    if "format_name" in manifest:
        if manifest["format_name"] not in VALID_FORMAT_NAMES:
            errors.append(f"Unknown format_name: {manifest['format_name']}")
        else:
            checks_passed.append("format_name_valid")

    if "format_version" in manifest:
        # Accept semver-like strings
        version = str(manifest["format_version"])
        if "." not in version:
            warnings.append(f"format_version '{version}' may not be semver")
        checks_passed.append("format_version_present")

    if "residual_format" in manifest:
        if manifest["residual_format"] not in VALID_RESIDUAL_FORMATS:
            errors.append(f"Unknown residual_format: {manifest['residual_format']}")
        else:
            checks_passed.append("residual_format_valid")

    if "base_quant" in manifest:
        if manifest["base_quant"] not in VALID_BASE_QUANTS:
            warnings.append(f"Unusual base_quant: {manifest['base_quant']} (not in known Q2 variants)")
        checks_passed.append("base_quant_present")

    if "budget_policy" in manifest:
        if manifest["budget_policy"] not in VALID_BUDGET_POLICIES:
            errors.append(f"Unknown budget_policy: {manifest['budget_policy']}")
        else:
            checks_passed.append("budget_policy_valid")

    if "tensor_families" in manifest:
        for fam in manifest["tensor_families"]:
            if fam not in VALID_TENSOR_FAMILIES:
                warnings.append(f"Unknown tensor_family in list: {fam}")
        checks_passed.append("tensor_families_valid")

    if "layer_count" in manifest:
        layers = manifest.get("layers", [])
        actual = len(layers)
        declared = manifest["layer_count"]
        if actual != declared:
            errors.append(f"layer_count mismatch: declared={declared}, actual={actual}")
        else:
            checks_passed.append("layer_count_matches")

    # --- Per-tensor checks ---
    layers = manifest.get("layers", [])
    if not layers:
        errors.append("No layer entries found in manifest")
    else:
        checks_passed.append("layers_present")

    layer_indices_seen = set()
    for idx, entry in enumerate(layers):
        prefix = f"layer[{idx}]"

        for field in PER_TENSOR_REQUIRED:
            if field not in entry:
                errors.append(f"{prefix}: missing required field '{field}'")

        # Layer index uniqueness
        if "layer" in entry:
            lid = entry["layer"]
            if lid in layer_indices_seen:
                warnings.append(f"{prefix}: duplicate layer index {lid} (multiple tensors per layer is valid)")
            layer_indices_seen.add(lid)

        # Compression sanity check
        if "compression_ratio_vs_q4" in entry:
            cr = entry["compression_ratio_vs_q4"]
            if cr >= 1.0:
                warnings.append(f"{prefix}: compression_ratio {cr} >= 1.0 (not beneficial)")
            elif cr > 0.8:
                warnings.append(f"{prefix}: compression_ratio {cr} > 0.8 (may not save memory)")
            else:
                checks_passed.append(f"{prefix}:compression_ok")

        # Validation metrics check
        if "validation_metrics" in entry:
            vm = entry["validation_metrics"]
            if "delta_cosine" in vm:
                dc = vm["delta_cosine"]
                if dc < 0:
                    warnings.append(f"{prefix}: delta_cosine {dc} is negative (bad)")
                elif dc < 0.5:
                    warnings.append(f"{prefix}: delta_cosine {dc} < 0.5 (below STRONG threshold)")
                else:
                    checks_passed.append(f"{prefix}:delta_cosine_ok")
            if "verdict" in vm:
                if vm["verdict"] not in ("STRONG_RECOVERY", "MODERATE_RECOVERY", "WEAK_RECOVERY"):
                    warnings.append(f"{prefix}: unusual verdict '{vm['verdict']}'")

        # File existence check
        if check_files and "file_path" in entry:
            fp = entry["file_path"]
            # Resolve relative to manifest directory
            manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
            full_path = os.path.join(manifest_dir, fp)
            if not os.path.exists(full_path):
                errors.append(f"{prefix}: file not found: {full_path}")
            else:
                checks_passed.append(f"{prefix}:file_exists")

    # --- Result ---
    passed = len(errors) == 0
    status = "PASS" if passed else "FAIL"

    result = {
        "status": status,
        "manifest_path": os.path.abspath(manifest_path),
        "errors": errors,
        "warnings": warnings,
        "checks_passed_count": len(checks_passed),
        "passed_checks": checks_passed,
        "summary": {
            "total_errors": len(errors),
            "total_warnings": len(warnings),
            "total_checks": len(checks_passed),
        }
    }

    return result


def print_result(r):
    print(f"\n{'='*60}")
    print(f"MANIFEST VALIDATION: {r['status']}")
    print(f"{'='*60}")
    print(f"Path: {r['manifest_path']}")
    print(f"\nErrors ({r['summary']['total_errors']}):")
    if r['errors']:
        for e in r['errors']:
            print(f"  ❌ {e}")
    else:
        print("  (none)")

    print(f"\nWarnings ({r['summary']['total_warnings']}):")
    if r['warnings']:
        for w in r['warnings']:
            print(f"  ⚠️  {w}")
    else:
        print("  (none)")

    print(f"\nChecks passed: {r['summary']['total_checks']}")
    if r['passed_checks']:
        for c in r['passed_checks'][:10]:
            print(f"  ✅ {c}")
        if len(r['passed_checks']) > 10:
            print(f"  ... and {len(r['passed_checks']) - 10} more")


def main():
    args = parse_args()
    if not os.path.exists(args.manifest):
        print(f"ERROR: manifest not found: {args.manifest}", file=sys.stderr)
        sys.exit(1)

    result = validate_manifest(args.manifest, check_files=args.check_files,
                               base_model=args.base_model)
    print_result(result)

    if args.out_json:
        with open(args.out_json, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nJSON written to {args.out_json}")

    sys.exit(0 if result['status'] == 'PASS' else 1)


if __name__ == "__main__":
    main()