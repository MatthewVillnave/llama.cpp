#!/usr/bin/env python3
"""
Phase 26J: SDI Packet Builder Synthetic Evaluation Suite

Usage:
    python3 examples/speculative/evaluate_sdi_packet_builder.py \
        --fixtures examples/speculative/fixtures/sdi_packet_eval \
        --out /tmp/phase26j_eval_summary.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))
from sdi_packet_builder import build_sdi_packet, token_estimate


# --------------------------------------------------------------------------- #
# Core evaluation logic
# --------------------------------------------------------------------------- #

def evaluate_scenario(
    scenario_dir: Path,
    verbose: bool = False,
) -> dict[str, Any]:
    """Evaluate one scenario. Returns result dict."""
    scenario_name = scenario_dir.name
    conv_file = scenario_dir / "conversation.txt"
    pinned_file = scenario_dir / "pinned_facts.json"
    expected_file = scenario_dir / "expected.json"

    if not conv_file.exists():
        return {"scenario": scenario_name, "error": "conversation.txt missing"}
    if not pinned_file.exists():
        return {"scenario": scenario_name, "error": "pinned_facts.json missing"}
    if not expected_file.exists():
        return {"scenario": scenario_name, "error": "expected.json missing"}

    with open(conv_file) as f:
        conversation = f.read()
    with open(pinned_file) as f:
        pinned_data = json.load(f)
    with open(expected_file) as f:
        expected = json.load(f)

    pinned_facts = (
        pinned_data
        if isinstance(pinned_data, list)
        else pinned_data.get("pinned_facts", [])
    )
    open_loops = (
        pinned_data.get("open_loops", [])
        if isinstance(pinned_data, dict)
        else []
    )
    hard_constraints = (
        pinned_data.get("constraints_verbatim", [])
        if isinstance(pinned_data, dict)
        else []
    )

    packet = build_sdi_packet(
        conversation=conversation,
        pinned_facts=pinned_facts,
        task_type="question",
        target_model="qwen2.5-7b",
        context_budget=4096,
        tier=None,
        open_loops=open_loops,
        hard_constraints=hard_constraints,
    )

    result = {
        "scenario": scenario_name,
        "packet_text": packet.packet_text,
        "packet_tokens": packet.estimated_tokens,
        "packet_words": packet.estimated_words,
        "tier": packet.tier,
        "dropped_sections": packet.dropped_sections,
        "checks": {},
        "passed": 0,
        "failed": 0,
        "errors": [],
    }

    # ---- must_include checks ----
    must_include = expected.get("must_include", [])
    must_pass = 0
    for fact in must_include:
        if fact in packet.packet_text:
            must_pass += 1
        else:
            result["errors"].append(f"must_include MISSING: {fact!r}")
    result["checks"]["must_include"] = {
        "total": len(must_include),
        "passed": must_pass,
        "rate": round(must_pass / len(must_include), 3) if must_include else 1.0,
        "missing": [f for f in must_include if f not in packet.packet_text],
    }
    result["passed"] += must_pass
    result["failed"] += len(must_include) - must_pass

    # ---- must_not_include checks ----
    must_not = expected.get("must_not_include", [])
    not_pass = 0
    for fact in must_not:
        if fact not in packet.packet_text:
            not_pass += 1
        else:
            result["errors"].append(f"must_not_include FOUND: {fact!r}")
    result["checks"]["must_not_include"] = {
        "total": len(must_not),
        "passed": not_pass,
        "rate": round(not_pass / len(must_not), 3) if must_not else 1.0,
        "found": [f for f in must_not if f in packet.packet_text],
    }
    result["passed"] += not_pass
    result["failed"] += len(must_not) - not_pass

    # ---- should_include checks ----
    should = expected.get("should_include", [])
    should_pass = 0
    for fact in should:
        if fact in packet.packet_text:
            should_pass += 1
    result["checks"]["should_include"] = {
        "total": len(should),
        "passed": should_pass,
        "rate": round(should_pass / len(should), 3) if should else 1.0,
        "missing": [f for f in should if f not in packet.packet_text],
    }
    result["passed"] += should_pass
    result["failed"] += len(should) - should_pass

    # ---- open loop preservation ----
    expected_loops = expected.get("expected_open_loops", [])
    loops_pass = 0
    for loop in expected_loops:
        if loop in packet.packet_text:
            loops_pass += 1
    result["checks"]["open_loop_preservation"] = {
        "total": len(expected_loops),
        "passed": loops_pass,
        "rate": round(loops_pass / len(expected_loops), 3) if expected_loops else 1.0,
        "missing": [l for l in expected_loops if l not in packet.packet_text],
    }
    result["passed"] += loops_pass
    result["failed"] += len(expected_loops) - loops_pass

    # ---- filler dropped check ----
    expected_dropped = expected.get("expected_dropped_filler", "none")
    if expected_dropped == "none":
        dropped_check = {"pass": True, "note": "no filler expected"}
    elif expected_dropped == "byzantine":
        dropped_check = {
            "pass": "Byzantine" not in packet.packet_text,
            "note": "Byzantine filler should be dropped",
        }
    elif expected_dropped == "database_and_testing_filler":
        # Check that key database/testing phrases are NOT in output
        db_phrases = ["Database management systems", "ACID compliance", "Test-driven development"]
        dropped_check = {
            "pass": not any(p in packet.packet_text for p in db_phrases),
            "note": "Database/testing filler should be dropped",
            "found": [p for p in db_phrases if p in packet.packet_text],
        }
    else:
        dropped_check = {"pass": True, "note": f"unknown expected_dropped: {expected_dropped}"}
    result["checks"]["filler_dropped"] = dropped_check
    if not dropped_check.get("pass", True):
        result["failed"] += 1
        result["errors"].append(f"filler_dropped FAIL: {dropped_check.get('note')}")
    else:
        result["passed"] += 1

    # ---- pinned facts preservation (from included_facts in metadata) ----
    meta = packet.to_meta()
    included = meta.get("included_facts", [])
    crit_facts = must_include
    crit_pass = sum(1 for f in crit_facts if any(f in inc for inc in included))
    result["checks"]["pinned_facts_preserved"] = {
        "total": len(crit_facts),
        "passed": crit_pass,
        "rate": round(crit_pass / len(crit_facts), 3) if crit_facts else 1.0,
    }

    # ---- token reduction ----
    original_tokens, _ = token_estimate(conversation)
    result["token_reduction"] = {
        "original_tokens": original_tokens,
        "packet_tokens": packet.estimated_tokens,
        "reduction_pct": round(
            (original_tokens - packet.estimated_tokens) / original_tokens * 100, 1
        ) if original_tokens else 0,
    }

    # ---- overall ----
    total = must_pass + not_pass + loops_pass
    result["total_checks"] = total
    result["all_passed"] = result["failed"] == 0

    if verbose:
        print(f"\n  {scenario_name}: {result['passed']}/{total} passed")
        if result["errors"]:
            for e in result["errors"]:
                print(f"    ERROR: {e}")

    return result


def run_evaluation(fixtures_dir: Path, verbose: bool = False) -> list[dict]:
    """Evaluate all scenarios in fixtures_dir."""
    results = []
    for scenario_dir in sorted(fixtures_dir.iterdir()):
        if scenario_dir.is_dir():
            r = evaluate_scenario(scenario_dir, verbose=verbose)
            results.append(r)
    return results


def summarize(results: list[dict]) -> dict[str, Any]:
    """Aggregate results into a summary."""
    total_passed = sum(r.get("passed", 0) for r in results)
    total_failed = sum(r.get("failed", 0) for r in results)
    total_checks = total_passed + total_failed

    scenario_results = []
    for r in results:
        sr = {
            "scenario": r["scenario"],
            "passed": r.get("passed", 0),
            "total": r.get("total_checks", 0),
            "all_passed": r.get("all_passed", False),
            "errors": r.get("errors", []),
            "token_reduction": r.get("token_reduction", {}),
            "filler_dropped": r.get("checks", {}).get("filler_dropped", {}),
        }
        scenario_results.append(sr)

    summary = {
        "total_scenarios": len(results),
        "scenarios_passed": sum(1 for r in results if r.get("all_passed", False)),
        "total_checks": total_checks,
        "checks_passed": total_passed,
        "checks_failed": total_failed,
        "pass_rate": round(total_passed / total_checks * 100, 1) if total_checks else 0,
        "scenarios": scenario_results,
    }
    return summary


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Phase 26J: SDI Packet Evaluator")
    parser.add_argument(
        "--fixtures",
        default=str(BASE_DIR / "fixtures" / "sdi_packet_eval"),
        help="Path to fixtures directory",
    )
    parser.add_argument(
        "--out",
        help="Output path for summary JSON",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print per-scenario results",
    )
    args = parser.parse_args()

    fixtures_path = Path(args.fixtures)
    if not fixtures_path.exists():
        print(f"ERROR: fixtures directory not found: {fixtures_path}")
        sys.exit(1)

    results = run_evaluation(fixtures_path, verbose=args.verbose)
    summary = summarize(results)

    if args.out:
        # Remove packet_text from each scenario result to keep output small
        for s in summary.get("scenarios", []):
            s.pop("token_reduction", None)
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary written to: {args.out}")

    # Print summary table
    print(f"\n{'='*60}")
    print(f"Phase 26J — SDI Packet Builder Evaluation")
    print(f"{'='*60}")
    print(f"Scenarios: {summary['total_scenarios']} total, {summary['scenarios_passed']} passed")
    print(f"Checks:    {summary['checks_passed']}/{summary['total_checks']} passed ({summary['pass_rate']}%)")
    print()
    print(f"{'Scenario':<45} {'Checks':>10} {'Token Red.':>12}")
    print(f"{'-'*60}")
    for s in summary.get("scenarios", []):
        tr = s.get("token_reduction", {})
        pct = f"{tr.get('reduction_pct', 0):.0f}%" if tr else "N/A"
        status = "PASS" if s.get("all_passed") else "FAIL"
        print(f"  [{status}] {s['scenario']:<38} {s['passed']}/{s['total']:>4} {pct:>12}")

    if summary["checks_failed"] > 0:
        print(f"\nFailed checks:")
        for s in summary.get("scenarios", []):
            for err in s.get("errors", []):
                print(f"  - {s['scenario']}: {err}")

    print(f"\n{'='*60}")
    if summary["checks_failed"] == 0:
        print("ALL CHECKS PASSED")
        sys.exit(0)
    else:
        print(f"FAILURES: {summary['checks_failed']} check(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()