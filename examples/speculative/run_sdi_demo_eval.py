#!/usr/bin/env python3
"""
Run the standalone SDI demo eval on the Phase 26O fixture set.

Generated outputs are written outside the repo by default.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
FIXTURE_DIR = BASE_DIR / "fixtures" / "sdi_packet_eval"
RUNTIME = BASE_DIR / "sdi_packet_runtime.py"
GUARD = BASE_DIR / "sdi_memory_guard.py"

SCENARIOS = [
    "1_pinned_early_fact",
    "2_long_filler_constraint",
    "3_open_loop_continuation",
    "4_conflicting_recent_vs_old",
    "5_tool_result_preservation",
    "6_realistic_project_handoff",
    "7_debug_trace_root_cause",
    "8_tool_output_exactness",
    "9_constraint_under_contradiction",
    "10_realistic_open_loop",
]


def run_json(cmd: list[str]) -> dict:
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if proc.returncode not in (0, 2):
        print(proc.stdout, end="")
        print(proc.stderr, end="", file=sys.stderr)
        raise SystemExit(proc.returncode)
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        print(proc.stdout, end="")
        print(proc.stderr, end="", file=sys.stderr)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Run standalone SDI demo eval")
    parser.add_argument("--model", default="qwen2.5:0.5b")
    parser.add_argument("--out-dir", default=os.environ.get("PRT_SCRATCH", "/tmp") + "/sdi_demo_eval")
    parser.add_argument("--all-baselines", action="store_true")
    parser.add_argument("--timeout-s", type=int, default=60)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    guard = run_json([
        sys.executable,
        str(GUARD),
        "--model",
        args.model,
        "--active-context-target",
        "4096",
    ])
    if not guard.get("safe"):
        print(json.dumps({"blocked": True, "guard": guard}, indent=2))
        return 2

    baselines = ["no_packet", "recent_only", "simple_summary", "sdi_packet", "auto"] if args.all_baselines else ["auto"]
    rows = []
    for scenario in SCENARIOS:
        scenario_dir = FIXTURE_DIR / scenario
        for baseline in baselines:
            meta_path = out_dir / f"{scenario}-{baseline}.json"
            out_path = out_dir / f"{scenario}-{baseline}.txt"
            cmd = [
                sys.executable,
                str(RUNTIME),
                "--conversation",
                str(scenario_dir / "conversation.txt"),
                "--pinned",
                str(scenario_dir / "pinned_facts.json"),
                "--expected",
                str(scenario_dir / "expected.json"),
                "--tier",
                "auto",
                "--backend",
                "ollama",
                "--model",
                args.model,
                "--baseline",
                baseline,
                "--packet-style",
                "compact",
                "--active-context-target",
                "4096",
                "--timeout-s",
                str(args.timeout_s),
                "--out",
                str(out_path),
                "--meta",
                str(meta_path),
            ]
            result = run_json(cmd)
            rows.append({
                "scenario": scenario,
                "baseline": baseline,
                "effective": result.get("effective_baseline", baseline),
                "selected_policy": result.get("selected_policy"),
                "score": result.get("score", {}).get("score"),
                "token_reduction_pct": result.get("prompt", {}).get("token_reduction_pct"),
                "swap_delta_mb": result.get("swap_delta_mb"),
            })

    summary = {
        "model": args.model,
        "out_dir": str(out_dir),
        "guard": guard,
        "rows": rows,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("| Scenario | Baseline | Effective | Score | Token reduction | Swap delta |")
    print("| -------- | -------- | --------- | ----- | --------------- | ---------- |")
    for row in rows:
        print(
            f"| {row['scenario']} | {row['baseline']} | {row['effective']} | "
            f"{row['score']} | {row['token_reduction_pct']}% | {row['swap_delta_mb']} MB |"
        )
    print(f"\nSummary JSON: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
