#!/usr/bin/env python3
"""
Phase 13AK: Phase 12 Speedup Reproduction
Run native and PRT with/without KV cache settings.
"""
import subprocess
import json
import time
import os
import statistics

MODEL = "/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-3B-Instruct-Q4_K_M.gguf"
PROMPT = "The capital of France is"
SIDECARS = "/tmp/prt_sidecars_3b"
BASEDIR = os.path.dirname(os.path.abspath(__file__))

RUNNER = os.path.join(os.path.dirname(BASEDIR), "phase13o_pty_argv_runner.py")
LLAMA = os.path.join(os.path.dirname(BASEDIR), "..", "build", "bin", "llama-cli")

def run_mode(label, extra_args, timeout):
    """Run a single inference mode. Returns dict with timing."""
    cmd = [
        "python3", RUNNER, "--timeout", str(timeout), "--tail-bytes", "262144", "--"
    ] + [
        LLAMA,
        "-m", MODEL, "-p", PROMPT, "-n", "80",
        "--temp", "0", "-c", "256", "-t", "4",
        "--no-display-prompt", "--single-turn",
    ] + extra_args

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout+10)
    wall = time.time() - start

    # Parse JSON output
    data = {}
    try:
        data = json.loads(result.stdout)
    except:
        pass

    elapsed = data.get("elapsed_sec", wall)

    # Extract generation speed from tail
    tail = data.get("tail_text", "")
    gen_speed = None
    for line in tail.split("\n"):
        if "Generation:" in line and "t/s" in line:
            try:
                gen_speed = float(line.split("Generation:")[1].split("t/s")[0].strip())
            except:
                pass

    # Extract prompt speed
    prompt_speed = None
    for line in tail.split("\n"):
        if "Prompt:" in line and "t/s" in line:
            try:
                prompt_speed = float(line.split("Prompt:")[1].split("t/s")[0].strip())
            except:
                pass

    # Check for Paris
    has_paris = "Paris" in tail or "paris" in tail.lower()

    return {
        "label": label,
        "returncode": data.get("returncode", result.returncode),
        "wall": wall,
        "elapsed_sec": elapsed,
        "gen_tok_s": gen_speed,
        "prompt_tok_s": prompt_speed,
        "has_paris": has_paris,
    }

def run_batch(label, modes, n_runs=3):
    """Run n_runs for each mode."""
    results = []
    for mode_name, extra_args, timeout in modes:
        for i in range(n_runs):
            print(f"  {label}/{mode_name} run {i+1}/{n_runs}...", flush=True)
            r = run_mode(f"{label}/{mode_name}", extra_args, timeout)
            r["run"] = i+1
            results.append(r)
            print(f"    wall={r['wall']:.3f}s gen={r['gen_tok_s']} t/s paris={r['has_paris']}", flush=True)
            time.sleep(0.5)
    return results

def summarize(results, label):
    """Summarize results by mode."""
    by_mode = {}
    for r in results:
        key = r["label"]
        if key not in by_mode:
            by_mode[key] = []
        by_mode[key].append(r)

    print(f"\n=== {label} Summary ===")
    summary = {}
    for key, runs in by_mode.items():
        walls = [r["wall"] for r in runs]
        gens = [r["gen_tok_s"] for r in runs if r["gen_tok_s"]]
        speedups = []
        for i in range(0, len(runs), 3):  # pairs of native/prt
            pass  # just report per-mode

        avg_wall = statistics.mean(walls) if walls else 0
        std_wall = statistics.stdev(walls) if len(walls)>1 else 0
        avg_gen = statistics.mean(gens) if gens else 0
        std_gen = statistics.stdev(gens) if len(gens)>1 else 0

        print(f"  {key}: wall={avg_wall:.3f}±{std_wall:.3f}s gen={avg_gen:.1f}±{std_gen:.1f} t/s")
        summary[key] = {"avg_wall": avg_wall, "std_wall": std_wall,
                        "avg_gen": avg_gen, "std_gen": std_gen,
                        "n": len(walls)}
    return summary

if __name__ == "__main__":
    print("=== Phase 13AK: Phase 12 Speed Reproduction ===\n")

    # Modes to test
    MODES = [
        ("Native_clean", [], 300),           # no cache, no PRT
        ("PRT_clean", [                            # no cache, PRT
            "--prt-mode", "5700",
            "--prt-force-native", "11,15",
            "--prt-sidecar-dir", SIDECARS,
            "--prt-sidecar-mmap",
            "--prt-log-file", "/tmp/phase13ak_prt_clean.log",
            "--prt-log-level", "quiet",
        ], 420),
        ("Native_cache", [                        # KV cache, no PRT
            "--cache-type-k", "q8_0",
            "--cache-type-v", "f16",
        ], 300),
        ("PRT_cache", [                            # KV cache, PRT
            "--cache-type-k", "q8_0",
            "--cache-type-v", "f16",
            "--prt-mode", "5700",
            "--prt-force-native", "11,15",
            "--prt-sidecar-dir", SIDECARS,
            "--prt-sidecar-mmap",
            "--prt-log-file", "/tmp/phase13ak_prt_cache.log",
            "--prt-log-level", "quiet",
        ], 420),
    ]

    # AK-A: Clean (no cache) - 3 runs each
    print("=== AK-A: Clean (no KV cache) ===")
    clean_results = run_batch("clean", MODES[:2], n_runs=3)

    time.sleep(2)

    # AK-B: With Phase 12 cache settings - 5 runs each
    print("\n=== AK-B: Phase 12 cache settings ===")
    cache_results = run_batch("cache", MODES[2:], n_runs=5)

    # Summarize
    print("\n=== Final Summary ===")
    all_results = clean_results + cache_results

    # Compute speed ratios
    native_clean = [r for r in all_results if r["label"] == "clean/Native_clean"]
    prt_clean = [r for r in all_results if r["label"] == "clean/PRT_clean"]
    native_cache = [r for r in all_results if r["label"] == "cache/Native_cache"]
    prt_cache = [r for r in all_results if r["label"] == "cache/PRT_cache"]

    def avg(lst, key): return statistics.mean([r[key] for r in lst]) if lst else 0
    def std(lst, key): return statistics.stdev([r[key] for r in lst]) if len(lst)>1 else 0

    print(f"\nNative clean:    wall={avg(native_clean,'wall'):.3f}±{std(native_clean,'wall'):.3f}s gen={avg(native_clean,'gen_tok_s'):.1f}±{std(native_clean,'gen_tok_s'):.1f} t/s")
    print(f"PRT clean:       wall={avg(prt_clean,'wall'):.3f}±{std(prt_clean,'wall'):.3f}s gen={avg(prt_clean,'gen_tok_s'):.1f}±{std(prt_clean,'gen_tok_s'):.1f} t/s")
    print(f"Native cache:    wall={avg(native_cache,'wall'):.3f}±{std(native_cache,'wall'):.3f}s gen={avg(native_cache,'gen_tok_s'):.1f}±{std(native_cache,'gen_tok_s'):.1f} t/s")
    print(f"PRT cache:       wall={avg(prt_cache,'wall'):.3f}±{std(prt_cache,'wall'):.3f}s gen={avg(prt_cache,'gen_tok_s'):.1f}±{std(prt_cache,'gen_tok_s'):.1f} t/s")

    # Speed ratios
    nc_wall = avg(native_clean, 'wall')
    pc_wall = avg(prt_clean, 'wall')
    nk_wall = avg(native_cache, 'wall')
    pk_wall = avg(prt_cache, 'wall')

    print(f"\nSpeed ratios (PRT/Native):")
    print(f"  Clean:  {pc_wall/nc_wall:.3f}x (wall), gen={avg(prt_clean,'gen_tok_s')/avg(native_clean,'gen_tok_s'):.3f}x")
    print(f"  Cache:  {pk_wall/nk_wall:.3f}x (wall), gen={avg(prt_cache,'gen_tok_s')/avg(native_cache,'gen_tok_s'):.3f}x")

    print(f"\nCache effect on native:")
    print(f"  Native with cache: {nk_wall:.3f}s vs clean: {nc_wall:.3f}s = {nk_wall/nc_wall:.3f}x")
    print(f"  PRT with cache:   {pk_wall:.3f}s vs clean: {pc_wall:.3f}s = {pk_wall/pc_wall:.3f}x")

    # Output JSON for report
    report = {
        "phase": "13AK",
        "native_clean": {"wall": round(avg(native_clean,'wall'),3), "gen": round(avg(native_clean,'gen_tok_s'),2), "n": len(native_clean)},
        "prt_clean": {"wall": round(avg(prt_clean,'wall'),3), "gen": round(avg(prt_clean,'gen_tok_s'),2), "n": len(prt_clean)},
        "native_cache": {"wall": round(avg(native_cache,'wall'),3), "gen": round(avg(native_cache,'gen_tok_s'),2), "n": len(native_cache)},
        "prt_cache": {"wall": round(avg(prt_cache,'wall'),3), "gen": round(avg(prt_cache,'gen_tok_s'),2), "n": len(prt_cache)},
        "speed_ratio_clean": round(pc_wall/nc_wall, 3),
        "speed_ratio_cache": round(pk_wall/nk_wall, 3),
        "cache_effect_native": round(nk_wall/nc_wall, 3),
        "cache_effect_prt": round(pk_wall/pc_wall, 3),
        "paris_all": all(r.get("has_paris",False) for r in all_results),
    }
    print(f"\nJSON: {json.dumps(report, indent=2)}")