#!/usr/bin/env python3
"""
Phase 28BR-AU: AT invariant sanity + AP magnitude rerun after determinism fix.

AT invariant: B=observe-only, C=shadow-only, D=scale=0 short-circuit
              should all match A=baseline for deterministic prompts.

After c4e4395ba (28BR-AT), the observe_path and scale_zero short-circuit
should no longer mutate the graph — providing true determinism guarantees.
"""

import json, subprocess, sys, time
from pathlib import Path
from collections import defaultdict

WORKSPACE   = Path("/home/matthew-villnave/llama.cpp")
MODEL       = "/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
LLAMA_CLI   = WORKSPACE / "build/bin/llama-cli"
MANIFEST    = "/tmp/phase28br_l_sidecars/manifest.json"
RESULTS_DIR = WORKSPACE / "examples" / "speculative" / "results"

PROMPTS_DET = ["Hi", "2+2="]
PROMPTS_MRG = ["The", "Once", "def"]
SCALES      = [0.0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 1.5, 2.0]

FAMILY      = "attn_out"
LAYER       = 0

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def run_inference(prompt, mode="baseline", scale=None, layer=0, budget=512, n=1):
    """Run llama-cli. modes: baseline, observe, shadow, scale0, true_inj, wrong_layer"""
    cmd = [
        str(LLAMA_CLI),
        "-m", MODEL,
        "-p", prompt,
        "--enable-prt-sidecar-pager",
        "--prt-mode", "5700",
        "--prt-sidecar-budget-mb", str(budget),
        "--prt-sidecar-manifest", MANIFEST,
        "--no-conversation",
        "--single-turn",
        "-n", str(n),
        "--log-disable",
    ]

    if mode == "baseline":
        pass  # no injection flags
    elif mode == "observe":
        cmd += ["--enable-prt-sidecar-pager"]
        # observe-only: pager enabled but no apply
    elif mode == "shadow":
        cmd += ["--prt-sidecar-apply", "--prt-sidecar-shadow-path"]
    elif mode == "scale0":
        cmd += ["--prt-sidecar-apply", "--prt-sidecar-true-injection",
                "--prt-sidecar-apply-layer", str(layer),
                "--prt-sidecar-apply-family", FAMILY,
                "--prt-sidecar-scale", "0.0"]
    elif mode == "true_inj":
        cmd += ["--prt-sidecar-apply", "--prt-sidecar-true-injection",
                "--prt-sidecar-apply-layer", str(layer),
                "--prt-sidecar-apply-family", FAMILY]
        if scale is not None:
            cmd += ["--prt-sidecar-scale", str(scale)]
    elif mode == "wrong_layer":
        cmd += ["--prt-sidecar-apply", "--prt-sidecar-true-injection",
                "--prt-sidecar-apply-layer", "1",
                "--prt-sidecar-apply-family", FAMILY]
        if scale is not None:
            cmd += ["--prt-sidecar-scale", str(scale)]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return result.stdout + "\n" + result.stderr, result.returncode


def parse_output(output):
    info = {
        "selected_token": None,
        "selected_logit": None,
        "top10_ids": [],
        "top10_logits": [],
        "prt_flags_found": False,
    }
    for line in output.splitlines():
        line = line.strip()
        if "PRT-FLAGS-SET" in line:
            info["prt_flags_found"] = True
        if "[TOKEN]" in line:
            parts = line.split()
            for p in parts:
                if p.startswith("id="):
                    try: info["selected_token"] = int(p[3:])
                    except: pass
                elif p.startswith("logit="):
                    try: info["selected_logit"] = float(p[6:])
                    except: pass
                elif p.startswith("top_ids="):
                    info["top10_ids"] = [int(x) for x in p[8:].split(",") if x]
                elif p.startswith("top_logits="):
                    info["top10_logits"] = [float(x) for x in p[11:].split(",") if x]
    return info


def jaccard(a, b, k=10):
    if not a or not b: return 0.0
    s_a = set(a[:k])
    s_b = set(b[:k])
    if not s_a or not s_b: return 0.0
    return len(s_a & s_b) / len(s_a | s_b)


def run_sanity(prompts, modes, runs=3):
    """AT invariant sanity: check B/C/D match A for deterministic prompts."""
    results = {}
    for prompt in prompts:
        results[prompt] = {}
        for mode in modes:
            mode_results = []
            for r in range(runs):
                out, rc = run_inference(prompt, mode=mode)
                info = parse_output(out)
                info["rc"] = rc
                mode_results.append(info)
                time.sleep(0.1)
            results[prompt][mode] = mode_results
    return results


def analyze_sanity(results):
    """Check whether observe/shadow/scale0 modes match baseline."""
    modes = ["baseline", "observe", "shadow", "scale0"]
    summary = {}
    for prompt, mode_data in results.items():
        baseline_tokens = [mode_data["baseline"][r]["selected_token"] for r in range(len(mode_data["baseline"]))]
        baseline_top10_lists = [mode_data["baseline"][r]["top10_ids"] for r in range(len(mode_data["baseline"]))]
        baseline_deterministic = len(set(baseline_tokens)) == 1

        entry = {
            "baseline_tokens": baseline_tokens,
            "baseline_top10": baseline_top10_lists[0] if baseline_top10_lists else [],
            "baseline_deterministic": baseline_deterministic,
            "modes": {},
        }

        for mode in modes[1:]:  # skip baseline
            tokens = [mode_data[mode][r]["selected_token"] for r in range(len(mode_data[mode]))]
            top10_lists = [mode_data[mode][r]["top10_ids"] for r in range(len(mode_data[mode]))]
            tokens_match_baseline = all(t == baseline_tokens[0] for t in tokens)
            jacc = jaccard(top10_lists[0] if top10_lists else [], entry["baseline_top10"])
            prt_flags = [mode_data[mode][r]["prt_flags_found"] for r in range(len(mode_data[mode]))]

            entry["modes"][mode] = {
                "tokens": tokens,
                "tokens_match_baseline": tokens_match_baseline,
                "topk_jaccard_vs_baseline": round(jacc, 4),
                "prt_flags_found": prt_flags,
            }

        summary[prompt] = entry
    return summary


print("=== PHASE 28BR-AU: AT Invariant Sanity + AP Magnitude Rerun ===")
print(f"HEAD: c4e4395ba (28BR-AT: fix pager path determinism)")
print(f"Model: {MODEL}")
print(f"Manifest: {MANIFEST}")
print(f"Scales: {SCALES}")
print(f"Deterministic prompts: {PROMPTS_DET}")
print(f"Marginal prompts: {PROMPTS_MRG}")
print()

# ── STEP 1: AT invariant sanity ──────────────────────────────────────────────
print("=== STEP 2: AT Invariant Sanity ===")
print("Modes: A=baseline, B=observe-only, C=shadow-only, D=scale=0 short-circuit")
print("Expectation: B/C/D should match A for deterministic prompts after AT fix")
print()

modes = ["baseline", "observe", "shadow", "scale0"]
all_prompts = PROMPTS_DET + PROMPTS_MRG

sanity_results = run_sanity(all_prompts, modes, runs=3)
sanity_analysis = analyze_sanity(sanity_results)

print("AT INVARIANT RESULTS:")
for prompt, entry in sanity_analysis.items():
    print(f"\n  Prompt: {prompt!r}  deterministic={entry['baseline_deterministic']}")
    print(f"    Baseline tokens: {entry['baseline_tokens']}")
    for mode, md in entry["modes"].items():
        match = "✓" if md["tokens_match_baseline"] else "✗"
        j = md["topk_jaccard_vs_baseline"]
        prt = any(md["prt_flags_found"])
        print(f"    {mode:12s}: tokens={md['tokens']} jaccard={j} prt_flags={prt} {match}")

print()

# ── STEP 3: AP magnitude rerun for deterministic prompts ────────────────────
print("=== STEP 3: AP Magnitude Sweep ===")
print("Deterministic prompts first (1 clean run):")
ap_results = {"raw": {}, "analysis": {}}

for prompt in PROMPTS_DET:
    print(f"\n  Prompt: {prompt!r}")
    ap_results["raw"][prompt] = {}
    
    # Baseline
    out, rc = run_inference(prompt, mode="baseline")
    info = parse_output(out)
    info["rc"] = rc
    ap_results["raw"][prompt]["baseline"] = info
    print(f"    baseline: token={info['selected_token']} logit={info['selected_logit']}")

    # Sweep scales
    for scale in SCALES:
        out, rc = run_inference(prompt, mode="true_inj", scale=scale)
        info = parse_output(out)
        info["rc"] = rc
        has_nan = any(x != x for x in info["top10_logits"]) if info["top10_logits"] else False
        has_inf = any(abs(x) == float('inf') for x in info["top10_logits"]) if info["top10_logits"] else False
        ap_results["raw"][prompt][str(scale)] = info

        token_changed = info["selected_token"] != ap_results["raw"][prompt]["baseline"]["selected_token"]
        jacc = jaccard(info["top10_ids"], ap_results["raw"][prompt]["baseline"]["top10_ids"])
        status = "nan" if has_nan else "inf" if has_inf else ("CHG" if token_changed else "ok")
        print(f"    scale={scale:4}: token={info['selected_token']} jaccard={jacc:.3f} {status}")
        time.sleep(0.1)

# Marginal prompts: 5 runs each
print(f"\nMarginal prompts (5 runs each):")
marginal_classification = {}
for prompt in PROMPTS_MRG:
    print(f"\n  Prompt: {prompt!r}")
    ap_results["raw"][prompt] = {}
    
    # Baseline 5 runs
    baseline_tokens = []
    for r in range(5):
        out, rc = run_inference(prompt, mode="baseline")
        info = parse_output(out)
        info["rc"] = rc
        ap_results["raw"][prompt][f"baseline_run{r}"] = info
        baseline_tokens.append(info["selected_token"])
        time.sleep(0.1)
    
    unique_baseline = len(set(baseline_tokens))
    is_deterministic = unique_baseline == 1
    marginal_classification[prompt] = {
        "baseline_tokens": baseline_tokens,
        "unique_tokens": unique_baseline,
        "is_deterministic": is_deterministic,
        "classification": "DETERMINISTIC" if is_deterministic else f"MARGINAL({unique_baseline} variants)",
    }
    print(f"    baseline tokens over 5 runs: {baseline_tokens} → {marginal_classification[prompt]['classification']}")
    
    if is_deterministic:
        # Run scale sweep
        for scale in SCALES:
            out, rc = run_inference(prompt, mode="true_inj", scale=scale)
            info = parse_output(out)
            info["rc"] = rc
            has_nan = any(x != x for x in info["top10_logits"]) if info["top10_logits"] else False
            has_inf = any(abs(x) == float('inf') for x in info["top10_logits"]) if info["top10_logits"] else False
            ap_results["raw"][prompt][str(scale)] = info
            token_changed = info["selected_token"] != baseline_tokens[0]
            jacc = jaccard(info["top10_ids"], ap_results["raw"][prompt]["baseline_run0"]["top10_ids"])
            status = "nan" if has_nan else "inf" if has_inf else ("CHG" if token_changed else "ok")
            print(f"    scale={scale:4}: token={info['selected_token']} jaccard={jacc:.3f} {status}")
            time.sleep(0.1)

# ── CLASSIFICATION ───────────────────────────────────────────────────────────
print("\n=== STEP 4: CLASSIFICATION ===")

# Deterministic prompt analysis
det_analysis = {}
for prompt in PROMPTS_DET:
    baseline = ap_results["raw"][prompt]["baseline"]
    baseline_token = baseline["selected_token"]
    
    first_token_change = None
    first_topk_jacc_below_08 = None
    first_topk_jacc_below_05 = None
    last_stable_scale = None
    
    for scale in SCALES:
        entry = ap_results["raw"][prompt].get(str(scale), {})
        if entry.get("selected_token") is None:
            continue
        token_changed = entry["selected_token"] != baseline_token
        jacc = jaccard(entry["top10_ids"], baseline["top10_ids"])
        has_nan = any(x != x for x in entry["top10_logits"]) if entry["top10_logits"] else False
        has_inf = any(abs(x) == float('inf') for x in entry["top10_logits"]) if entry["top10_logits"] else False
        
        if not has_nan and not has_inf:
            last_stable_scale = scale
        
        if first_token_change is None and token_changed:
            first_token_change = scale
        if first_topk_jacc_below_08 is None and jacc < 0.8:
            first_topk_jacc_below_08 = scale
        if first_topk_jacc_below_05 is None and jacc < 0.5:
            first_topk_jacc_below_05 = scale

    det_analysis[prompt] = {
        "baseline_token": baseline_token,
        "first_token_change_scale": first_token_change,
        "first_topk_jacc_below_08": first_topk_jacc_below_08,
        "first_topk_jacc_below_05": first_topk_jacc_below_05,
        "last_stable_scale": last_stable_scale,
    }
    print(f"  {prompt!r}: baseline_token={baseline_token}, first_token_chg={first_token_change}, "
          f"topk<0.8@{first_topk_jacc_below_08}, last_stable={last_stable_scale}")

print()
print(f"=== MARGINAL CLASSIFICATION ===")
for prompt, cls in marginal_classification.items():
    print(f"  {prompt!r}: {cls['classification']}")

print()
print("=== SCALE=0 CLEANLINESS (after AT fix) ===")
for prompt in PROMPTS_DET:
    s0 = sanity_results[prompt]["scale0"]
    tokens = [r["selected_token"] for r in s0]
    baseline_token = ap_results["raw"][prompt]["baseline"]["selected_token"]
    clean = all(t == baseline_token for t in tokens)
    prt_flags = [r["prt_flags_found"] for r in s0]
    print(f"  {prompt!r}: scale0_tokens={tokens} baseline={baseline_token} clean={clean} prt_flags={prt_flags}")

print()
print("=== TOP-K POOL STABILITY ===")
for prompt in PROMPTS_DET:
    baseline_topk = ap_results["raw"][prompt]["baseline"]["top10_ids"]
    scale_topk_all = [ap_results["raw"][prompt].get(str(s), {}).get("top10_ids", []) for s in SCALES]
    all_topk = [baseline_topk] + scale_topk_all
    
    # Compute Jaccard between consecutive scales
    jaccs = []
    prev = baseline_topk
    for tk in scale_topk_all:
        jaccs.append(jaccard(tk, prev))
        prev = tk
    
    print(f"  {prompt!r}: baseline_topk={baseline_topk}")
    for i, (scale, j) in enumerate(zip(SCALES, jaccs)):
        tk = scale_topk_all[i]
        print(f"    scale={scale}: jacc={j:.3f} topk={tk}")

# ── SAVE RESULTS ─────────────────────────────────────────────────────────────
output_json = {
    "phase": "28BR-AU",
    "title": "AP magnitude rerun after determinism fix",
    "branch": "experimental/prt-phase19a-alt-sidecar-backed",
    "old_head": "3992730cd",
    "new_head": "c4e4395ba",
    "fix_commit": "c4e4395ba",
    "fix_description": "AT: observe_path guard, scale_zero short-circuit, shadow_path cleanliness",
    "sanity": {
        "modes_tested": ["baseline", "observe", "shadow", "scale0"],
        "prompts_tested": all_prompts,
        "runs_per_mode": 3,
        "results": sanity_analysis,
    },
    "ap_sweep": {
        "scales": SCALES,
        "deterministic_prompts": PROMPTS_DET,
        "marginal_prompts": PROMPTS_MRG,
        "marginal_classification": marginal_classification,
        "raw_results": ap_results["raw"],
        "det_analysis": det_analysis,
    },
    "classification": {
        "deterministic_prompts": {p: det_analysis[p] for p in PROMPTS_DET},
        "marginal_prompts": marginal_classification,
    },
    "scale_zero_cleanliness": {},
    "topk_pool_stability": {},
    "previous_ap_conclusions_confirmed": None,
    "verdict": None,
    "codex_subagent_used": False,
}

# Determine scale=0 cleanliness
output_json["scale_zero_cleanliness"] = {
    prompt: {
        "tokens_match_baseline": all(
            sanity_results[prompt]["scale0"][r]["selected_token"] == 
            ap_results["raw"][prompt]["baseline"]["selected_token"]
            for r in range(3)
        ),
        "prt_flags": [sanity_results[prompt]["scale0"][r]["prt_flags_found"] for r in range(3)],
    }
    for prompt in PROMPTS_DET
}

# Determine top-k pool stability
for prompt in PROMPTS_DET:
    baseline_topk = ap_results["raw"][prompt]["baseline"]["top10_ids"]
    output_json["topk_pool_stability"][prompt] = {}
    for scale in SCALES:
        tk = ap_results["raw"][prompt].get(str(scale), {}).get("top10_ids", [])
        j = jaccard(tk, baseline_topk)
        output_json["topk_pool_stability"][prompt][str(scale)] = round(j, 4)

# Determine if previous AP conclusions are confirmed
# Previous AP (3992730cd) showed first_token_change at various scales
# Check if we see similar or different patterns
prev_ap_summary = {}  # from the previous run
output_json["previous_ap_conclusions_confirmed"] = "UNDER_REVIEW"

# Determine verdict
# Check if scale=0 is now truly clean (matches baseline with prt_flags=True but no graph mutation)
scale0_clean = all(
    output_json["scale_zero_cleanliness"][p]["tokens_match_baseline"]
    for p in PROMPTS_DET
)
# Check observe and shadow match baseline
observe_match = all(
    all(
        sanity_results[p]["observe"][r]["selected_token"] == 
        ap_results["raw"][p]["baseline"]["selected_token"]
        for r in range(3)
    )
    for p in PROMPTS_DET
)
shadow_match = all(
    all(
        sanity_results[p]["shadow"][r]["selected_token"] == 
        ap_results["raw"][p]["baseline"]["selected_token"]
        for r in range(3)
    )
    for p in PROMPTS_DET
)

output_json["verdict"] = "PASS" if (scale0_clean and observe_match and shadow_match) else "PARTIAL"

json_path = RESULTS_DIR / "phase28br_au_ap_rerun_after_determinism_fix.json"
with open(json_path, "w") as f:
    json.dump(output_json, f, indent=2)
print(f"\nJSON saved to {json_path}")
print(f"Verdict: {output_json['verdict']}")
print(f"scale0_clean={scale0_clean} observe_match={observe_match} shadow_match={shadow_match}")