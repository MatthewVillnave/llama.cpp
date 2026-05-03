#!/usr/bin/env python3
"""
PRT Sidecar Validator — Phase 12C

Reads a PRT sidecar manifest and verifies that sidecar files exist
in the specified directory with the expected properties.

Usage:
    python3 prt_validate_sidecars.py --manifest prt_sidecar_manifest.example.json --sidecar-dir /tmp/prt_sidecars/

Exit codes:
    0 — all checks passed
    1 — validation failed (see stderr)
    2 — invalid arguments or manifest
"""

import argparse
import hashlib
import json
import os
import sys


def sha256_file(path: str) -> str:
    """Compute SHA256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def validate(args) -> bool:
    # Load manifest
    try:
        with open(args.manifest, "r") as f:
            manifest = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] Manifest not found: {args.manifest}", file=sys.stderr)
        sys.exit(2)
    except json.JSONDecodeError as e:
        print(f"[ERROR] Invalid JSON in manifest: {e}", file=sys.stderr)
        sys.exit(2)

    # Validate manifest format
    required_fields = ["format_version", "target_model", "sidecar_tensor", "policy", "files"]
    for field in required_fields:
        if field not in manifest:
            print(f"[ERROR] Manifest missing required field: {field}", file=sys.stderr)
            sys.exit(2)

    model = manifest["target_model"]
    tensor = manifest["sidecar_tensor"]
    policy = manifest["policy"]
    files = manifest["files"]

    sidecar_dir = args.sidecar_dir.rstrip("/")
    errors = []
    warnings = []

    print(f"=== PRT Sidecar Validator ===")
    print(f"Manifest: {args.manifest}")
    print(f"Sidecar dir: {sidecar_dir}")
    print(f"Model: {model.get('model_family')} {model.get('model_size')} ({model.get('quantization')})")
    print(f"Architecture: {model.get('architecture')}")
    print(f"Layers: {model.get('n_layers')}")
    print(f"Force-native: {policy.get('force_native_layers', [])}")
    print(f"Check SHA256: {args.checksum}")
    print("")

    # Check sidecar_dir exists
    if not os.path.isdir(sidecar_dir):
        print(f"[ERROR] Sidecar directory not found: {sidecar_dir}", file=sys.stderr)
        sys.exit(1)

    # Build expected files from manifest
    force_native = set(policy.get("force_native_layers", []))
    required_by_manifest = set()
    for f in files:
        if f.get("required", True):
            required_by_manifest.add(f["layer"])

    # Check each file in manifest
    present = 0
    missing = []
    size_errors = []
    checksum_errors = []

    for f in files:
        layer = f["layer"]
        filename = f["filename"]
        required = f.get("required", True)
        expected_size = f.get("bytes", tensor.get("file_size_bytes"))
        expected_sha256 = f.get("sha256", "<sha256>") if args.checksum else None
        path = f"{sidecar_dir}/{filename}"

        # Skip force-native layers (optional)
        if layer in force_native:
            if os.path.exists(path):
                print(f"  [SKIP] Layer {layer}: force-native, sidecar present but not required")
            else:
                print(f"  [SKIP] Layer {layer}: force-native, no sidecar — OK")
            continue

        # Check required files
        if required:
            if not os.path.exists(path):
                missing.append(filename)
                print(f"  [MISSING] Layer {layer}: {filename} — REQUIRED")
                continue
            else:
                present += 1

            # Check size
            actual_size = os.path.getsize(path)
            if expected_size and actual_size != expected_size:
                size_errors.append((filename, expected_size, actual_size))
                print(f"  [SIZE ERROR] Layer {layer}: {filename} — expected {expected_size}, got {actual_size}")

            # Check SHA256 if requested
            if args.checksum and expected_sha256 and expected_sha256 != "<sha256>":
                actual_sha256 = sha256_file(path)
                if actual_sha256 != expected_sha256:
                    checksum_errors.append((filename, expected_sha256, actual_sha256))
                    print(f"  [CHECKSUM ERROR] Layer {layer}: {filename} — SHA256 mismatch")
                else:
                    print(f"  [OK] Layer {layer}: {filename} — size OK, SHA256 OK")
            else:
                print(f"  [OK] Layer {layer}: {filename} — size OK")

    print("")
    print(f"=== Results ===")
    print(f"Required files: {len(required_by_manifest)}")
    print(f"Present: {present}/{len(required_by_manifest)}")

    if missing:
        print(f"[FAIL] Missing required sidecars: {', '.join(missing)}")
        errors.append(f"missing: {len(missing)}")

    if size_errors:
        for fn, exp, act in size_errors:
            print(f"[FAIL] Size mismatch: {fn} — expected {exp}, got {act}")
        errors.append(f"size_errors: {len(size_errors)}")

    if checksum_errors:
        for fn, exp, act in checksum_errors:
            print(f"[FAIL] Checksum mismatch: {fn}")
        errors.append(f"checksum_errors: {len(checksum_errors)}")

    print("")
    if errors:
        print("VALIDATION FAILED")
        sys.exit(1)
    else:
        print("VALIDATION PASSED")
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="Validate PRT sidecar files against a manifest."
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to PRT sidecar manifest JSON file",
    )
    parser.add_argument(
        "--sidecar-dir",
        required=True,
        help="Directory containing sidecar binary files",
    )
    parser.add_argument(
        "--checksum",
        action="store_true",
        help="Also verify SHA256 checksums if present in manifest",
    )
    args = parser.parse_args()
    validate(args)


if __name__ == "__main__":
    main()