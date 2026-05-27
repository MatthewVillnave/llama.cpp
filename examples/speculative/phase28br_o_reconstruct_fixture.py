#!/usr/bin/env python3
"""
Phase 28BR-O: reconstruct the deterministic Phase 28BO layer-0 .trit fixture.

This is a thin wrapper around the canonical prt_trit_io.py writer.  It freezes
the shapes, seeds, block geometry, and Phase28Y manifest schema recorded by the
Phase 28BO report.
"""

import argparse
import json
from pathlib import Path

from prt_trit_io import (
    make_synthetic_scales,
    make_synthetic_ternary,
    read_trit,
    validate_trit,
    write_trit,
)


BLOCK_ROWS = 32
BLOCK_COLS = 48
DEFAULT_OUT = Path("/tmp/phase28br_o_layer0_multifamily_trit")

FIXTURES = [
    {"family": "ffn_up", "rows": 4864, "cols": 896, "seed": 1990868534},
    {"family": "ffn_down", "rows": 896, "cols": 4864, "seed": 1792446221},
    {"family": "ffn_gate", "rows": 4864, "cols": 896, "seed": 895829913},
    {"family": "attn_out", "rows": 896, "cols": 896, "seed": 4244712744},
]

EXPECTED_BYTES = {
    "ffn_up": 1645888,
    "ffn_down": 1645760,
    "ffn_gate": 1645888,
    "attn_out": 303216,
}


def expected_scale_count(rows, cols):
    return ((rows + BLOCK_ROWS - 1) // BLOCK_ROWS) * ((cols + BLOCK_COLS - 1) // BLOCK_COLS)


def build_manifest(entries):
    return {
        "format_name": "prt_residual_sidecar",
        "format_version": 1,
        "source_model": "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf",
        "source_model_hash": "unknown_phase28bo_fixture",
        "base_quant": "Q4_K_M",
        "residual_format": "ternary",
        "generator": "examples/speculative/phase28br_o_reconstruct_fixture.py",
        "layer_count": 1,
        "tensor_families": [item["family"] for item in FIXTURES],
        "block_rows": BLOCK_ROWS,
        "block_cols": BLOCK_COLS,
        "total_residual_bytes": sum(e["byte_size"] for e in entries),
        "entries": entries,
    }


def reconstruct(out_dir, dry_run=False, force=False):
    layer_dir = out_dir / "layers" / "layer_000"
    entries = []
    files = []

    for item in FIXTURES:
        family = item["family"]
        rows = item["rows"]
        cols = item["cols"]
        seed = item["seed"]
        rel_path = Path("layers") / "layer_000" / f"{family}.trit"
        trit_path = out_dir / rel_path
        n_scales = expected_scale_count(rows, cols)

        entry = {
            "layer_index": 0,
            "tensor_name": f"blk.0.{family}.weight",
            "tensor_family": family,
            "shape": [rows, cols],
            "rows": rows,
            "cols": cols,
            "dtype": "ternary",
            "block_rows": BLOCK_ROWS,
            "block_cols": BLOCK_COLS,
            "scale_count": n_scales,
            "seed": seed,
            "file_path": str(rel_path),
            "byte_size": EXPECTED_BYTES[family],
            "status": "active",
            "required": True,
        }
        entries.append(entry)

        if not dry_run:
            if trit_path.exists() and not force:
                raise FileExistsError(f"{trit_path} exists; pass --force to overwrite")
            layer_dir.mkdir(parents=True, exist_ok=True)
            ternary = make_synthetic_ternary(rows, cols, seed)
            scales = make_synthetic_scales(ternary, BLOCK_ROWS, BLOCK_COLS)
            info = write_trit(trit_path, ternary, scales, BLOCK_ROWS, BLOCK_COLS)
            valid, msg = validate_trit(trit_path)
            decoded_ternary, decoded_scales, meta = read_trit(trit_path)
            actual_bytes = trit_path.stat().st_size
            if actual_bytes != EXPECTED_BYTES[family]:
                raise ValueError(f"{family}: byte size {actual_bytes} != expected {EXPECTED_BYTES[family]}")
            if not valid:
                raise ValueError(f"{family}: validate_trit failed: {msg}")
            if decoded_ternary.shape != (rows, cols):
                raise ValueError(f"{family}: decoded shape {decoded_ternary.shape} != {(rows, cols)}")
            if len(decoded_scales) != n_scales:
                raise ValueError(f"{family}: decoded scales {len(decoded_scales)} != {n_scales}")
            files.append({
                "family": family,
                "path": str(trit_path),
                "bytes": actual_bytes,
                "validate_trit": msg,
                "payload_offset": meta["payload_offset"],
                "scale_offset": meta["scale_offset"],
                "scale_count": int(len(decoded_scales)),
                "write_info": {k: (str(v) if isinstance(v, Path) else v) for k, v in info.items()},
            })

    manifest = build_manifest(entries)
    manifest_path = out_dir / "manifest.json"
    if not dry_run:
        if manifest_path.exists() and not force:
            raise FileExistsError(f"{manifest_path} exists; pass --force to overwrite")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    return {
        "phase": "28BR-O",
        "dry_run": dry_run,
        "output_dir": str(out_dir),
        "manifest_path": str(manifest_path),
        "expected_total_bytes": sum(EXPECTED_BYTES.values()),
        "manifest": manifest,
        "generated_files": files,
        "decoder_contract": {
            "header_bytes": 32,
            "magic": "TRIT",
            "version": "0.1",
            "payload_offset": 32,
            "encoding": "3_bits_per_trit_row_major",
            "scale_array": "float32[n_scales] at scale_offset",
            "decode_path": "prt_trit_decoder::decode_bytes(file_bytes, file_size, rows, cols, block_rows, block_cols, n_scales, scales)",
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true", help="Print manifest/report without writing fixture files")
    parser.add_argument("--force", action="store_true", help="Overwrite existing manifest/.trit files")
    parser.add_argument("--out-json", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    result = reconstruct(args.out_dir, dry_run=args.dry_run, force=args.force)
    text = json.dumps(result, indent=2)
    print(text)
    if args.out_json:
        args.out_json.write_text(text + "\n")


if __name__ == "__main__":
    main()
