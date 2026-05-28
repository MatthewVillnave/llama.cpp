#!/usr/bin/env python3
"""
Phase 29B-R: Real Sidecar Memory Audit
Measures actual RSS/VmRSS/activation metrics for real .trit sidecars.
"""
import subprocess
import time
import re
import os
import json
import sys
from pathlib import Path

LLAMA = "/home/matthew-villnave/llama.cpp/build/bin/llama-cli"
MODEL = "/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
SIDECAR = "/tmp/prt_sidecars_0_5b_layer0"
LOG_DIR = Path("/tmp/phase29b_r_logs")
LOG_DIR.mkdir(exist_ok=True)

def run_test(name, n_predict, extra_flags=None, budget_mb=10):
    """Run a single test and collect memory/performance metrics."""
    if extra_flags is None:
        extra_flags = []

    logfile = LOG_DIR / f"{name}_n{n_predict}.log"

    cmd = [
        "/usr/bin/time", "-v", str(LLAMA),
        "-m", MODEL,
        "--single-turn", "--seed", "42", "-t", "0", "--log-disable",
        "-p", "Hello world, how are you today?",
        "-n", str(n_predict),
        "--enable-prt-sidecar-pager",
        "--prt-sidecar-dir", SIDECAR,
        "--prt-sidecar-manifest", f"{SIDECAR}/manifest.json",
        "--prt-sidecar-budget-mb", str(budget_mb),
        "--prt-mode", "1",
    ] + extra_flags

    wall_start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    wall_end = time.time()
    wall_time = wall_end - wall_start

    # time -v output goes to stderr
    time_output = result.stderr

    # llama output goes to stdout
    llama_output = result.stdout

    # Combined log
    full_output = llama_output + "\n" + time_output
    logfile.write_text(full_output)

    # Parse /usr/bin/time metrics
    max_rss = 0
    minor_pf = 0
    major_pf = 0
    for line in time_output.splitlines():
        if "Maximum resident set size" in line:
            m = re.search(r'(\d+)', line)
            if m: max_rss = int(m.group(1))
        elif "Minor (reclaiming a page)" in line:
            m = re.search(r'(\d+)', line)
            if m: minor_pf = int(m.group(1))
        elif "Major (requiring I/O)" in line:
            m = re.search(r'(\d+)', line)
            if m: major_pf = int(m.group(1))

    # Parse PRT metrics from llama output
    prt_flags = ""
    activation_attempts = 0
    activation_successes = 0
    budget_rejects = 0
    resident_bytes = 0
    injection_successes = 0
    sidecar_math = 0

    for line in llama_output.splitlines():
        if "PRT-FLAGS-SET" in line:
            prt_flags = line.strip()
        m = re.search(r'activation_attempts=(\d+)', line)
        if m: activation_attempts = max(activation_attempts, int(m.group(1)))
        m = re.search(r'activation_successes=(\d+)', line)
        if m: activation_successes = max(activation_successes, int(m.group(1)))
        m = re.search(r'budget_rejects=(\d+)', line)
        if m: budget_rejects = max(budget_rejects, int(m.group(1)))
        m = re.search(r'resident_bytes=(\d+)', line)
        if m: resident_bytes = max(resident_bytes, int(m.group(1)))
        m = re.search(r'injection_successes=(\d+)', line)
        if m: injection_successes = max(injection_successes, int(m.group(1)))
        if "sidecar_math_influenced_output=1" in line:
            sidecar_math = 1

    data = {
        "name": name,
        "n_predict": n_predict,
        "exit_code": result.returncode,
        "wall_time_s": round(wall_time, 3),
        "max_rss_kb": max_rss,
        "minor_page_faults": minor_pf,
        "major_page_faults": major_pf,
        "activation_attempts": activation_attempts,
        "activation_successes": activation_successes,
        "budget_rejects": budget_rejects,
        "resident_bytes": resident_bytes,
        "injection_successes": injection_successes,
        "sidecar_math_influenced_output": sidecar_math,
        "prt_flags": prt_flags,
        "budget_mb": budget_mb,
    }

    jsonfile = LOG_DIR / f"{name}_n{n_predict}.json"
    jsonfile.write_text(json.dumps(data, indent=2))

    print(f"  [{name}] n={n_predict} exit={result.returncode} wall={wall_time:.2f}s "
          f"RSS={max_rss}KB attempts={activation_attempts} succ={activation_successes} "
          f"rejects={budget_rejects} resid={resident_bytes}", flush=True)

    return data

def main():
    results = []

    # A: Baseline native (no pager, no sidecars)
    for n in [1, 8, 32]:
        r = run_test("A_baseline", n)
        results.append(r)

    # B: Pager observe-only (pager enabled, manifest, --prt-sidecar-apply but NO true-injection)
    for n in [1, 8, 32]:
        r = run_test(f"B_pager_observe", n, extra_flags=["--prt-sidecar-apply"])
        results.append(r)

    # C: attn_out layer0 true injection
    for n in [1, 8, 32]:
        r = run_test("C_attn_out", n, extra_flags=[
            "--prt-sidecar-apply",
            "--prt-sidecar-apply-family", "attn_out",
            "--prt-sidecar-apply-layer", "0",
            "--prt-sidecar-true-injection",
            "--prt-sidecar-scale", "1.0",
        ])
        results.append(r)

    # D: ffn_up layer0 true injection
    for n in [1, 8, 32]:
        r = run_test("D_ffn_up", n, extra_flags=[
            "--prt-sidecar-apply",
            "--prt-sidecar-apply-family", "ffn_up",
            "--prt-sidecar-apply-layer", "0",
            "--prt-sidecar-true-injection",
            "--prt-sidecar-scale", "1.0",
        ])
        results.append(r)

    # E: ffn_down layer0 true injection
    for n in [1, 8, 32]:
        r = run_test("E_ffn_down", n, extra_flags=[
            "--prt-sidecar-apply",
            "--prt-sidecar-apply-family", "ffn_down",
            "--prt-sidecar-apply-layer", "0",
            "--prt-sidecar-true-injection",
            "--prt-sidecar-scale", "1.0",
        ])
        results.append(r)

    # G: Budget enforcement (attn_out layer0, n_predict=8, varying budget)
    for budget in [0, 1, 8, 32, 512]:
        r = run_test(f"G_budget_{budget}", 8, extra_flags=[
            "--prt-sidecar-apply",
            "--prt-sidecar-apply-family", "attn_out",
            "--prt-sidecar-apply-layer", "0",
            "--prt-sidecar-true-injection",
            "--prt-sidecar-scale", "1.0",
        ], budget_mb=budget)
        results.append(r)

    # Save combined results
    all_json = LOG_DIR / "all_results.json"
    all_json.write_text(json.dumps(results, indent=2))
    print(f"\nAll results saved to {all_json}")
    print(f"All logs saved to {LOG_DIR}")

if __name__ == "__main__":
    main()
