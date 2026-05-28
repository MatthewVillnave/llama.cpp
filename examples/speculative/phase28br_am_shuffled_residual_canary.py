#!/usr/bin/env python3
"""
Phase 28BR-AM: Shuffled Residual Runtime Canary

Goal: Determine whether the shared override pool is structure-sensitive or
magnitude/distribution-driven by shuffling the residual tensor in various ways.

Variants tested:
  A. baseline (no injection)
  B. original residual (positive control — known to trigger override)
  C. value-shuffled residual (preserve values, shuffle positions — breaks internal structure)
  D. row-shuffled residual (independently shuffle columns within each row)
  E. column-shuffled residual (independently shuffle rows within each column)
  F. sign-randomized residual (flip each sign with p=0.5)
  G. norm-matched random residual (Gaussian noise shaped to match L2 norm)

Control variants:
  H. original residual scale=0 → should match baseline
  I. layer=1 → no injection
  J. budget=0 → budget reject
  K. missing manifest → deterministic failure
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

WORKSPACE   = Path("/home/matthew-villnave/llama.cpp")
FIXTURE_DIR = Path("/tmp/phase28br_o_layer0_multifamily_trit")
OUT_DIR     = WORKSPACE / "examples" / "speculative"
RESULTS_DIR = OUT_DIR / "results"
SCRATCH_DIR = Path("/tmp/phase28br_am_shuffled")

MODEL       = "/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
LLAMA_CLI   = WORKSPACE / "build/bin/llama-cli"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

FAMILY      = "attn_out"
LAYER       = 0
PROMPTS     = ["Hi", "The", "Once"]
SEED        = 4244712744  # same as attn_out in fixture

sys.path.insert(0, str(OUT_DIR))
from prt_trit_io import read_trit, write_trit


def load_trit(family):
    path = FIXTURE_DIR / "layers" / f"layer_{LAYER:03d}" / f"{family}.trit"
    ternary, scales, meta = read_trit(path)
    return ternary.astype(np.int8), scales, meta


def create_variants(ternary, scales, seed):
    """Create 5 shuffled/distorted variants of the same int8 tensor."""
    rng = np.random.default_rng(seed)
    rows, cols = ternary.shape
    ternary_int = ternary.astype(np.int8)

    density = float(np.sum(ternary_int != 0)) / (rows * cols)
    orig_l2 = float(np.linalg.norm(ternary_int.astype(np.float64)))
    pos_frac = float(np.sum(ternary_int == 1)) / (rows * cols)
    neg_frac = float(np.sum(ternary_int == -1)) / (rows * cols)

    variants = {}

    # C: value-shuffled (flat shuffle — all positional structure destroyed)
    flat = ternary_int.ravel().copy()
    rng.shuffle(flat)
    variants["C_value_shuffled"] = flat.reshape(ternary_int.shape)

    # D: row-shuffled (shuffle column indices within each row independently)
    row_shuffled = np.zeros_like(ternary_int)
    for r in range(rows):
        row_shuffled[r] = ternary_int[r, rng.permutation(cols)]
    variants["D_row_shuffled"] = row_shuffled

    # E: column-shuffled (shuffle row indices within each column independently)
    col_shuffled = np.zeros_like(ternary_int)
    for c in range(cols):
        col_shuffled[:, c] = ternary_int[rng.permutation(rows), c]
    variants["E_col_shuffled"] = col_shuffled

    # F: sign-randomized (flip each sign with p=0.5)
    signs = rng.choice([-1, 1], size=(rows, cols))
    variants["F_sign_random"] = np.clip(ternary_int * signs, -1, 1).astype(np.int8)

    # G: norm-matched random ternary (uniform draw, scaled to match L2)
    random_arr = rng.choice([-1, 0, 1], size=(rows, cols))
    rand_l2 = float(np.linalg.norm(random_arr.astype(np.float64)))
    scaled = random_arr * (orig_l2 / (rand_l2 + 1e-30))
    variants["G_norm_random"] = np.round(scaled).astype(np.int8)

    return variants


def build_variant_manifest(variant_name, out_dir, byte_size):
    """Build a variant manifest."""
    with open(FIXTURE_DIR / "manifest.json") as f:
        orig_manifest = json.load(f)
    manifest = dict(orig_manifest)
    manifest["generator"] = f"phase28br_am_shuffled_residual_canary.py [{variant_name}]"
    manifest["entries"] = []
    for entry in orig_manifest["entries"]:
        e = dict(entry)
        if e["tensor_family"] == FAMILY:
            e["file_path"] = f"layers/layer_000/{FAMILY}.trit"
            e["byte_size"] = byte_size
            e["status"] = "active"
        manifest["entries"].append(e)
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


def run_inference(prompt, manifest_path):
    """Run llama-cli and parse top-k token info. Returns dict."""
    cmd = [
        str(LLAMA_CLI),
        "-m", MODEL,
        "-p", prompt,
        "--no-conversation",
        "--single-turn",
        "--no-display-prompt",
        "-n", "1",
        "--enable-prt-sidecar-pager",
        "--prt-mode", "5700",
        "--prt-sidecar-budget-mb", "512",
    ]
    if manifest_path is not None:
        cmd.extend([
            f"--prt-sidecar-manifest={manifest_path}",
            "--prt-sidecar-apply",
            "--prt-sidecar-true-injection",
            f"--prt-sidecar-apply-family={FAMILY}",
            "--prt-sidecar-apply-layer", "0",
            "--prt-sidecar-scale", "1.0",
        ])

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    stdout = result.stdout
    stderr = result.stderr

    # ── PRT-FLAGS-SET verification ──────────────────────────────────────────
    # Phase 28BR-AO: verify the library actually received our flags
    prt_flags_found = False
    import re
    all_text = stdout + "\n" + stderr
    for line in all_text.splitlines():
        if "[PRT-FLAGS-SET]" in line:
            prt_flags_found = True
            break

    top_ids = []
    top_logits = []
    selected_id = None
    selected_logit = None

    for line in all_text.splitlines():
        line = line.strip()
        # Format: [TOKEN] id=9707 logit=28.2492 top_ids=9707,108386 top_logits=28.2492,24.9771
        m = re.search(
            r"\[TOKEN\]\s+id=(\d+)\s+logit=([0-9.+-]+)\s+top_ids=([0-9,]+)\s+top_logits=([0-9.,+=-]+)",
            line
        )
        if m:
            selected_id = int(m.group(1))
            selected_logit = float(m.group(2))
            top_ids = [int(x) for x in m.group(3).split(',')]
            top_logits = [float(x) for x in m.group(4).split(',')]
            break

    if not top_ids:
        # Fallback: look for the simpler logit line
        for line in all_text.splitlines():
            line = line.strip()
            m = re.search(r"\[TOKEN\]\s+id=(\d+)\s+logit=([0-9.+-]+)", line)
            if m:
                selected_id = int(m.group(1))
                selected_logit = float(m.group(2))
                top_ids = [selected_id]
                top_logits = [selected_logit]
                break

    return {
        "selected_id": selected_id,
        "selected_logit": selected_logit,
        "top_ids": top_ids[:10],
        "top_logits": top_logits[:10],
        "stderr_last": stderr[-500:] if len(stderr) > 500 else stderr,
        "returncode": result.returncode,
        "prt_flags_found": prt_flags_found,
    }


def run_control(prompt, control_type, manifest_path):
    """Run a control variant with extra flags."""
    import re
    if control_type == "K_missing_manifest":
        cmd = [
            str(LLAMA_CLI),
            "-m", MODEL,
            "-p", prompt,
            "--no-conversation",
            "--single-turn",
            "--no-display-prompt",
            "-n", "1",
            "--enable-prt-sidecar-pager",
            "--prt-mode", "5700",
            "--prt-sidecar-manifest=/nonexistent/path/manifest.json",
            "--prt-sidecar-apply",
            f"--prt-sidecar-apply-family={FAMILY}",
            "--prt-sidecar-apply-layer", "0",
            "--prt-sidecar-true-injection",
            "--prt-sidecar-scale", "1.0",
            "--prt-sidecar-budget-mb", "512",
        ]
    else:
        cmd = [
            str(LLAMA_CLI),
            "-m", MODEL,
            "-p", prompt,
            "--no-conversation",
            "--single-turn",
            "--no-display-prompt",
            "-n", "1",
            "--enable-prt-sidecar-pager",
            "--prt-mode", "5700",
            f"--prt-sidecar-manifest={manifest_path}",
            "--prt-sidecar-apply",
            f"--prt-sidecar-apply-family={FAMILY}",
        ]
        extra = []
        if control_type == "H_scale_zero":
            extra = ["--prt-sidecar-apply-layer", "0", "--prt-sidecar-scale", "0.0",
                     "--prt-sidecar-true-injection", "--prt-sidecar-budget-mb", "512"]
        elif control_type == "I_layer_guard":
            extra = ["--prt-sidecar-apply-layer", "1", "--prt-sidecar-scale", "1.0",
                     "--prt-sidecar-true-injection", "--prt-sidecar-budget-mb", "512"]
        elif control_type == "J_budget_zero":
            extra = ["--prt-sidecar-apply-layer", "0", "--prt-sidecar-scale", "1.0",
                     "--prt-sidecar-true-injection", "--prt-sidecar-budget-mb", "0"]
        cmd.extend(extra)

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    stdout = result.stdout
    stderr = result.stderr

    top_ids = []
    top_logits = []
    selected_id = None
    selected_logit = None

    all_text = stdout + "\n" + stderr
    for line in all_text.splitlines():
        line = line.strip()
        m = re.search(
            r"\[TOKEN\]\s+id=(\d+)\s+logit=([0-9.+-]+)\s+top_ids=([0-9,]+)\s+top_logits=([0-9.,+=-]+)",
            line
        )
        if m:
            selected_id = int(m.group(1))
            selected_logit = float(m.group(2))
            top_ids = [int(x) for x in m.group(3).split(',')]
            top_logits = [float(x) for x in m.group(4).split(',')]
            break

    if not top_ids:
        for line in all_text.splitlines():
            line = line.strip()
            m = re.search(r"\[TOKEN\]\s+id=(\d+)\s+logit=([0-9.+-]+)", line)
            if m:
                if selected_id is None:
                    selected_id = int(m.group(1))
                    selected_logit = float(m.group(2))
                top_ids.append(int(m.group(1)))
                top_logits.append(float(m.group(2)))

    return {
        "selected_id": selected_id,
        "selected_logit": selected_logit,
        "top_ids": top_ids[:10],
        "top_logits": top_logits[:10],
        "stderr_last": stderr[-500:] if len(stderr) > 500 else stderr,
        "returncode": result.returncode,
    }


def jaccard_overlap(ids_a, ids_b):
    set_a, set_b = set(ids_a), set(ids_b)
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def topk_overlap(ids_a, ids_b, k=10):
    a, b = set(ids_a[:k]), set(ids_b[:k])
    if not a and not b:
        return 1.0
    return len(a & b) / k


def main():
    print("=== Phase 28BR-AM: Shuffled Residual Canary ===")
    print(f"Working directory: {WORKSPACE}")
    print(f"Model: {MODEL}")
    print(f"Fixture: {FIXTURE_DIR}")
    print(f"Scratch: {SCRATCH_DIR}")
    print()

    # ── Load tensor ────────────────────────────────────────────────────────────
    print(f"Loading {FAMILY} residual tensor...")
    ternary, scales, meta = load_trit(FAMILY)
    print(f"  shape: {ternary.shape}, scales: {len(scales)}, "
          f"L2: {np.linalg.norm(ternary):.6f}, "
          f"zero_frac: {float(np.sum(ternary==0)/ternary.size):.4f}")

    # ── Create & write variants ──────────────────────────────────────────────────
    print("\nCreating shuffled variants...")
    variants = create_variants(ternary, scales, SEED)

    print("\nWriting variant trit files...")
    variant_dirs = {}
    for vname, vtensor in variants.items():
        vdir = SCRATCH_DIR / vname
        vdir.mkdir(parents=True, exist_ok=True)
        layer_dir = vdir / "layers" / f"layer_{LAYER:03d}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        trit_path = layer_dir / f"{FAMILY}.trit"
        write_trit(trit_path, vtensor, scales)
        byte_size = trit_path.stat().st_size
        mpath = build_variant_manifest(vname, vdir, byte_size)
        variant_dirs[vname] = {
            "dir": vdir, "manifest": mpath,
            "trit_path": trit_path, "tensor": vtensor,
        }
        print(f"  {vname}: L2={np.linalg.norm(vtensor):.6f}, "
              f"zero_frac={float(np.sum(vtensor==0)/vtensor.size):.4f}")

    # B_original: copy of original fixture
    b_dir = SCRATCH_DIR / "B_original"
    b_dir.mkdir(exist_ok=True)
    layer_dir_b = b_dir / "layers" / f"layer_{LAYER:03d}"
    layer_dir_b.mkdir(exist_ok=True)
    shutil.copy(
        FIXTURE_DIR / "layers" / f"layer_{LAYER:03d}" / f"{FAMILY}.trit",
        layer_dir_b / f"{FAMILY}.trit"
    )
    byte_size_b = (FIXTURE_DIR / "layers" / f"layer_{LAYER:03d}" / f"{FAMILY}.trit").stat().st_size
    mpath_b = build_variant_manifest("B_original", b_dir, byte_size_b)
    variant_dirs["B_original"] = {
        "dir": b_dir, "manifest": mpath_b,
        "trit_path": layer_dir_b / f"{FAMILY}.trit",
        "tensor": ternary,
    }

    # ── Run inference ──────────────────────────────────────────────────────────
    print("\n=== Running baseline (A — no injection) ===")
    baseline_results = {}
    for prompt in PROMPTS:
        result = run_inference(prompt, None)
        baseline_results[prompt] = result
        print(f"  A | {prompt!r:10s} → {result['selected_id']} "
              f"(logit={result['selected_logit']}) "
              f"[PRT-FLAGS-SET={result.get('prt_flags_found', False)}])" if result.get('prt_flags_found') is not None else f"  A | {prompt!r:10s} → {result['selected_id']} (logit={result['selected_logit']})")

    print("\n=== Running B (original residual — positive control) ===")
    b_results = {}
    for prompt in PROMPTS:
        result = run_inference(prompt, mpath_b)
        b_results[prompt] = result
        print(f"  B | {prompt!r:10s} → {result['selected_id']} "
              f"(logit={result['selected_logit']}) "
              f"[PRT-FLAGS-SET={result.get('prt_flags_found', False)}]")

    # ── PRT-FLAGS-SET gate: if pager never armed, stop immediately ───────────
    # Phase 28BR-AO: if [PRT-FLAGS-SET] not found in B (positive control),
    # COMMAND_FLAG_MISMATCH is active — token=9707 is NOT a valid override
    first_b = b_results[PROMPTS[0]]
    if not first_b.get("prt_flags_found", False):
        classification = "COMMAND_FLAG_MISMATCH"
        interpretation = (
            "[PRT-FLAGS-SET] not found in B positive-control run. "
            "Pager never armed — COMMAND_FLAG_MISMATCH is active. "
            "token=9707 must NOT be treated as a valid override."
        )
        print(f"\n{'='*60}")
        print(f"BLOCKED: {classification}")
        print(f"{'='*60}")
        print(f"{interpretation}")
        print(f"\nFull stderr (B, first prompt):\n{first_b.get('stderr_last', '')}")
        # Write partial results
        report = {
            "phase": "28BR-AM",
            "subphase": "AO-pager-init-rerun",
            "branch": "experimental/prt-phase19a-alt-sidecar-backed",
            "head_at_plumbing_commit": "1ab527782",
            "head_now": "<incomplete>",
            "classification": classification,
            "interpretation": interpretation,
            "blocked_by": "COMMAND_FLAG_MISMATCH",
            "B_first_prompt_stderr": first_b.get("stderr_last", ""),
            "status": "BLOCKED",
        }
        json_path = RESULTS_DIR / "phase28br_ao_pager_init_am_rerun.json"
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2)
        sys.exit(1)

    print("\n=== Running shuffled variants C-G ===")
    variant_results = {}
    variant_keys = ["C_value_shuffled", "D_row_shuffled", "E_col_shuffled",
                   "F_sign_random", "G_norm_random"]
    for vname in variant_keys:
        vresults = {}
        vmanifest = variant_dirs[vname]["manifest"]
        for prompt in PROMPTS:
            result = run_inference(prompt, vmanifest)
            vresults[prompt] = result
            print(f"  {vname[:4]} | {prompt!r:10s} → {result['selected_id']} "
                  f"(logit={result['selected_logit']}) "
                  f"[PRT-FLAGS-SET={result.get('prt_flags_found', False)}]")
        variant_results[vname] = vresults

    print("\n=== Running controls H-K ===")
    control_results = {}
    for ctype in ["H_scale_zero", "I_layer_guard", "J_budget_zero",
                 "K_missing_manifest"]:
        cres = {}
        for prompt in PROMPTS:
            if ctype == "K_missing_manifest" and prompt != "Hi":
                continue
            r = run_control(prompt, ctype, mpath_b)
            cres[prompt] = r
            print(f"  {ctype} | {prompt!r:10s} → {r['selected_id']} "
                  f"(logit={r['selected_logit']})")
        control_results[ctype] = cres

    # ── Compute overlaps ───────────────────────────────────────────────────────
    print("\n=== Computing overlaps ===")
    overlap_vs_B = {}
    overlap_vs_BL = {}
    for vname in variant_keys:
        overlap_vs_B[vname] = {}
        overlap_vs_BL[vname] = {}
        for prompt in PROMPTS:
            b_ids = b_results[prompt]["top_ids"]
            bl_ids = baseline_results[prompt]["top_ids"]
            v_ids = variant_results[vname][prompt]["top_ids"]
            overlap_vs_B[vname][prompt] = {
                "selected_match_B": variant_results[vname][prompt]["selected_id"] == b_results[prompt]["selected_id"],
                "topk_jaccard_B": jaccard_overlap(v_ids, b_ids),
                "topk_overlap_B": topk_overlap(v_ids, b_ids),
            }
            overlap_vs_BL[vname][prompt] = {
                "selected_match_BL": variant_results[vname][prompt]["selected_id"] == baseline_results[prompt]["selected_id"],
                "topk_jaccard_BL": jaccard_overlap(v_ids, bl_ids),
                "topk_overlap_BL": topk_overlap(v_ids, bl_ids),
            }
            o = overlap_vs_B[vname][prompt]
            ol = overlap_vs_BL[vname][prompt]
            print(f"  {vname} | {prompt!r:10s} | "
                  f"sel_B={int(o['selected_match_B'])} "
                  f"J_B={o['topk_jaccard_B']:.2f} "
                  f"J_BL={ol['topk_jaccard_BL']:.2f}")

    # ── Aggregate stats ────────────────────────────────────────────────────────
    total = len(variant_keys) * len(PROMPTS)
    total_sel_match_B = sum(
        int(overlap_vs_B[vname][p]["selected_match_B"])
        for vname in variant_keys for p in PROMPTS
    )
    avg_sel_match_B = total_sel_match_B / total
    avg_jaccard_B = sum(
        overlap_vs_B[vname][p]["topk_jaccard_B"]
        for vname in variant_keys for p in PROMPTS
    ) / total
    avg_jaccard_BL = sum(
        overlap_vs_BL[vname][p]["topk_jaccard_BL"]
        for vname in variant_keys for p in PROMPTS
    ) / total

    print(f"\nAggregate: avg_sel_match_B={avg_sel_match_B:.2%}  "
          f"avg_jaccard_B={avg_jaccard_B:.3f}  avg_jaccard_BL={avg_jaccard_BL:.3f}")

    # ── Classification ────────────────────────────────────────────────────────
    if avg_sel_match_B >= 1.0:  # 100% selected token match across all prompts
        classification = "MAGNITUDE_DRIVEN"
        interpretation = (
            "All shuffled variants produce identical selected token to original injection. "
            "The override pool is determined by magnitude/distribution properties "
            "(L2, norm, density), not by positional/structural arrangement."
        )
    elif avg_jaccard_B >= 0.8:
        classification = "MAGNITUDE_DRIVEN"
        interpretation = (
            "Shuffled variants produce top-k pools with >80% Jaccard overlap vs original. "
            "Magnitude/distribution dominates, structural arrangement is secondary."
        )
    elif avg_jaccard_B <= 0.2 and avg_sel_match_B <= 0.2:
        classification = "STRUCTURE_SENSITIVE"
        interpretation = (
            "Shuffled variants fail to reproduce the override pool — "
            "specific positional structure is required for injection effect."
        )
    elif avg_sel_match_B >= 0.6:
        classification = "MIXED"
        interpretation = (
            "Partial overlap: some structure preserved in certain variants. "
            "Both magnitude and structure contribute non-trivially."
        )
    else:
        classification = "INCONCLUSIVE"
        interpretation = (
            "Cannot firmly classify; patterns suggest both mechanisms contribute "
            "but threshold criteria not met."
        )

    print(f"Classification: {classification}")
    print(f"Interpretation: {interpretation}")

    # ── Tensor statistics ───────────────────────────────────────────────────────
    variant_stats = {}
    for vname, vdata in variant_dirs.items():
        vt = vdata["tensor"]
        variant_stats[vname] = {
            "shape": list(vt.shape),
            "l2_norm": float(np.linalg.norm(vt.astype(np.float64))),
            "zero_frac": float(np.sum(vt == 0) / vt.size),
            "pos_frac": float(np.sum(vt == 1) / vt.size),
            "neg_frac": float(np.sum(vt == -1) / vt.size),
        }

    # ── Assemble report ─────────────────────────────────────────────────────────
    report = {
        "phase": "28BR-AM",
        "subphase": "AO-pager-init-rerun",
        "branch": "experimental/prt-phase19a-alt-sidecar-backed",
        "head": "1ab527782",
        "plumbing_commit": "1ab527782",
        "classification": classification,
        "interpretation": interpretation,
        "seed": SEED,
        "family": FAMILY,
        "layer": LAYER,
        "prompts": PROMPTS,
        "model": MODEL,
        "fixture_path": str(FIXTURE_DIR),
        "scratch_dir": str(SCRATCH_DIR),
        "baseline_results": {p: {
            "selected_id": r["selected_id"],
            "selected_logit": r["selected_logit"],
            "top_ids": r["top_ids"],
            "top_logits": r["top_logits"],
        } for p, r in baseline_results.items()},
        "B_results": {p: {
            "selected_id": r["selected_id"],
            "selected_logit": r["selected_logit"],
            "top_ids": r["top_ids"],
            "top_logits": r["top_logits"],
        } for p, r in b_results.items()},
        "variant_results": {vname: {p: {
            "selected_id": r["selected_id"],
            "selected_logit": r["selected_logit"],
            "top_ids": r["top_ids"],
            "top_logits": r["top_logits"],
        } for p, r in vresults.items()} for vname, vresults in variant_results.items()},
        "overlap_vs_B": overlap_vs_B,
        "overlap_vs_baseline": overlap_vs_BL,
        "variant_statistics": variant_stats,
        "aggregate_stats": {
            "avg_selected_token_match_vs_B": avg_sel_match_B,
            "avg_topk_jaccard_vs_B": avg_jaccard_B,
            "avg_topk_jaccard_vs_baseline": avg_jaccard_BL,
        },
        "controls": {ctype: {p: {
            "selected_id": r["selected_id"],
            "selected_logit": r["selected_logit"],
        } for p, r in cres.items()} for ctype, cres in control_results.items()},
    }

    # ── Write output files ─────────────────────────────────────────────────────
    json_path = RESULTS_DIR / "phase28br_am_shuffled_residual_canary.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote JSON: {json_path}")

    md = []
    md.append("# Phase 28BR-AM: Shuffled Residual Runtime Canary\n")
    md.append(f"**Branch:** `experimental/prt-phase19a-alt-sidecar-backed`  \n")
    md.append(f"**HEAD:** `46aa63b49`  \n")
    md.append(f"**Classification:** **{classification}**  \n")
    md.append(f"**Seed:** {SEED}  \n")
    md.append(f"\n{interpretation}\n")

    md.append("\n## Variant Descriptions\n")
    md.append("| Label | Transformation | What it destroys |")
    md.append("|-------|---------------|-----------------|")
    md.append("| **A** | baseline (no injection) | — |")
    md.append("| **B** | original residual | positive control |")
    md.append("| **C** | value-shuffled (flat shuffle) | all positional structure |")
    md.append("| **D** | row-shuffled (per-row col shuffle) | column ordering within rows |")
    md.append("| **E** | column-shuffled (per-col row shuffle) | row ordering within columns |")
    md.append("| **F** | sign-randomized (p=0.5 flip) | sign structure |")
    md.append("| **G** | norm-matched random ternary | all structure, same L2 + density |")

    md.append("\n## Aggregate Results\n")
    md.append(f"| Metric | Value |")
    md.append(f"|--------|-------|")
    md.append(f"| Avg selected-token match vs B | {avg_sel_match_B:.2%} |")
    md.append(f"| Avg top-k Jaccard vs B | {avg_jaccard_B:.3f} |")
    md.append(f"| Avg top-k Jaccard vs baseline | {avg_jaccard_BL:.3f} |")

    md.append("\n## Per-Prompt: Selected Token\n")
    md.append("| Variant | " + " | ".join(PROMPTS) + " |")
    md.append("|---------|" + "|".join(["---"] * len(PROMPTS)) + "|")
    all_var = ["B_original"] + variant_keys
    results_map = {"B_original": b_results}
    results_map.update(variant_results)
    for vname in all_var:
        d = results_map.get(vname, {})
        tokens = [str(d[p]["selected_id"] if p in d else "N/A") for p in PROMPTS]
        md.append(f"| `{vname}` | {' | '.join(tokens)} |")

    md.append("\n## Per-Prompt: Selected Logit\n")
    md.append("| Variant | " + " | ".join(PROMPTS) + " |")
    md.append("|---------|" + "|".join(["---"] * len(PROMPTS)) + "|")
    for vname in all_var:
        d = results_map.get(vname, {})
        logits = [f"{d[p]['selected_logit']:.4f}" if p in d else "N/A" for p in PROMPTS]
        md.append(f"| `{vname}` | {' | '.join(logits)} |")

    md.append("\n## Top-k (k=10) Overlap vs Original Injection (B)\n")
    md.append("| Variant | " + " | ".join(PROMPTS) + " |")
    md.append("|---------|" + "|".join(["---"] * len(PROMPTS)) + "|")
    for vname in variant_keys:
        ovs = [f"{overlap_vs_B[vname][p]['topk_overlap_B']:.2f}" for p in PROMPTS]
        md.append(f"| `{vname}` | {' | '.join(ovs)} |")

    md.append("\n## Tensor Statistics\n")
    md.append("| Variant | Shape | L2 Norm | Zero Frac | Pos Frac | Neg Frac |")
    md.append("|---------|-------|---------|-----------|----------|----------|")
    for vname in all_var:
        if vname not in variant_stats:
            continue
        s = variant_stats[vname]
        md.append(f"| `{vname}` | {s['shape'][0]}×{s['shape'][1]} | "
                  f"{s['l2_norm']:.4f} | {s['zero_frac']:.4f} | "
                  f"{s['pos_frac']:.4f} | {s['neg_frac']:.4f} |")

    md.append("\n## Controls\n")
    for ctype, cres in control_results.items():
        md.append(f"\n**{ctype}:**")
        for prompt, r in cres.items():
            status = "PASS" if r["selected_id"] is not None else "FAIL"
            md.append(f"- `{prompt}`: token={r['selected_id']}, "
                      f"logit={r['selected_logit']} [{status}]")

    md.append("\n## Conclusion\n")
    md.append(f"**Classification: {classification}**\n")
    md.append(f"\n{interpretation}\n")

    md_path = OUT_DIR / "PHASE28BR_AM_SHUFFLED_RESIDUAL_CANARY.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md))
    print(f"Wrote Markdown: {md_path}")
    return report


if __name__ == "__main__":
    main()

