#!/usr/bin/env python3
"""
PRT Phase 13AI: Warm harness measurement via llama-cli subprocess loop.
Measures per-request latency and throughput when model+sidecars are loaded once
vs. one-shot per-call (each invocation loads from scratch).
"""
import subprocess, json, time, sys, os, re

MODEL = "/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-3B-Instruct-Q4_K_M.gguf"
SIDECARS = "/tmp/prt_sidecars_3b"
LLAMA_CLI = os.path.expanduser("~/llama.cpp/build/bin/llama-cli")
PROMPT = "The capital of France is"
N_TOKENS = 80
CTX = 256
THREADS = 4
N_WARM = 5

def run_one_shot(prt=False, n=1):
    """Run one llama-cli invocation, return timing dict."""
    args = [
        LLAMA_CLI, "-m", MODEL,
        "-p", PROMPT, "-n", str(N_TOKENS),
        "-c", str(CTX), "-t", str(THREADS),
        "--no-display-prompt", "--single-turn",
    ]
    log_file = f"/tmp/phase13ai_oneshot_{('prt' if prt else 'nat')}_{n}.log"
    if prt:
        args += [
            "--prt-mode", "5700", "--prt-force-native", "11,15",
            "--prt-sidecar-dir", SIDECARS, "--prt-sidecar-mmap",
            "--prt-log-file", log_file, "--prt-log-level", "quiet",
        ]
    else:
        args += ["--prt-mode", "0"]

    t0 = time.perf_counter()
    result = subprocess.run(args, capture_output=True, text=True, timeout=240)
    t1 = time.perf_counter()
    wall = t1 - t0

    output = result.stdout + result.stderr
    tail = output[-262144:] if len(output) > 262144 else output

    # Parse timing from output
    gen_match = re.search(r'Gen.*?:\s+([\d.]+)\s+t/s', tail)
    prompt_match = re.search(r'Prompt.*?:\s+([\d.]+)\s+t/s', tail)
    sidecar_ms = None
    if prt:
        m = re.search(r'sidecar_load_ms=([\d.]+)', tail)
        if m: sidecar_ms = float(m.group(1))

    gen_toks = re.search(r'toks.*?(\d+)', tail)
    paris = "Paris" in tail

    return {
        "wall": wall,
        "gen_tok_s": float(gen_match.group(1)) if gen_match else None,
        "prompt_tok_s": float(prompt_match.group(1)) if prompt_match else None,
        "sidecar_ms": sidecar_ms,
        "paris": paris,
        "exit": result.returncode,
    }

def run_warm_batch(prt=False):
    """
    Run N_WARM sequential requests in the SAME llama-cli process.
    This requires a special input mode or using the interactive server.
    
    Instead: run llama-cli once with --log-disable and feed via stdin.
    Or: spawn a server, hit it N times.
    
    Simplest valid approach: run llama-cli N times back-to-back in the same
    process by sending N prompts via a heredoc or by using the server mode.
    
    We use server mode: start llama-server, then hit it N times.
    """
    import socket, threading, time as t2

    # Find an available port
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
    sock.close()

    server_args = [
        LLAMA_CLI, "-m", MODEL,
        "-c", str(CTX), "-t", str(THREADS),
        "--port", str(port), "--log-disable",
    ]
    if prt:
        server_args += [
            "--prt-mode", "5700", "--prt-force-native", "11,15",
            "--prt-sidecar-dir", SIDECARS, "--prt-sidecar-mmap",
        ]

    server_proc = subprocess.Popen(server_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Wait for server to start (check if port is open)
    import urllib.request
    for _ in range(50):
        try:
            t2.sleep(0.2)
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
            break
        except:
            if server_proc.poll() is not None:
                # server exited
                _, stderr = server_proc.communicate()
                raise RuntimeError(f"Server exited early: {stderr[:500]}")
    else:
        server_proc.terminate()
        raise RuntimeError("Server did not start in time")

    results = []
    for i in range(N_WARM):
        req_body = json.dumps({
            "prompt": PROMPT,
            "n_predict": N_TOKENS,
            "temperature": 0,
            "n_threads": THREADS,
            "cache": True,  # try to keep cache
        }).encode()
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/completion",
                data=req_body,
                headers={"Content-Type": "application/json"},
            )
            t0 = t2.time()
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
            t1 = t2.time()
            wall = t1 - t0
            content = data.get("content", "")
            tokens = data.get("tokens_predicted", 0)
            tok_s = tokens / wall if wall > 0 and tokens > 0 else 0
            results.append({
                "idx": i,
                "wall": wall,
                "gen_tok_s": tok_s,
                "n_tokens": tokens,
                "clean": "Paris" in content and "/tmp" not in content,
                "output": content[:100],
            })
        except Exception as e:
            results.append({"idx": i, "error": str(e)})

    server_proc.terminate()
    server_proc.wait()
    return results

def main():
    print("=== Phase 13AI-A: One-shot baselines ===\n")

    # Native one-shot (3 runs)
    print("Native one-shot (3 runs):")
    nat_results = []
    for i in range(1, 4):
        r = run_one_shot(prt=False, n=i)
        nat_results.append(r)
        print(f"  run {i}: wall={r['wall']:.3f}s gen={r['gen_tok_s']} Paris={r['paris']}")

    print("\nPRT one-shot (3 runs):")
    prt_results = []
    for i in range(1, 4):
        r = run_one_shot(prt=True, n=i)
        prt_results.append(r)
        print(f"  run {i}: wall={r['wall']:.3f}s sc={r['sidecar_ms']}ms gen={r['gen_tok_s']} Paris={r['paris']}")

    nat_avg_wall = sum(r['wall'] for r in nat_results) / 3
    nat_avg_gen = sum(r['gen_tok_s'] for r in nat_results) / 3
    prt_avg_wall = sum(r['wall'] for r in prt_results) / 3
    prt_avg_gen = sum(r['gen_tok_s'] for r in prt_results) / 3
    prt_avg_sc = sum(r['sidecar_ms'] for r in prt_results) / 3

    print(f"\nNative avg: wall={nat_avg_wall:.3f}s gen={nat_avg_gen} t/s")
    print(f"PRT avg: wall={prt_avg_wall:.3f}s sc={prt_avg_sc}ms gen={prt_avg_gen} t/s")

    print("\n=== Phase 13AI-B/C: Warm server mode ===\n")

    # Try server mode for warm measurement
    print("Testing server availability...")
    try:
        print("Running warm native batch (5 requests)...")
        nat_warm = run_warm_batch(prt=False)
        for r in nat_warm:
            if "error" in r:
                print(f"  req {r['idx']}: ERROR {r['error']}")
            else:
                print(f"  req {r['idx']}: wall={r['wall']:.3f}s tok/s={r['gen_tok_s']:.1f} Paris={r['clean']}")
    except Exception as e:
        print(f"Warm native FAILED: {e}")
        nat_warm = None

    print()
    try:
        print("Running warm PRT batch (5 requests)...")
        prt_warm = run_warm_batch(prt=True)
        for r in prt_warm:
            if "error" in r:
                print(f"  req {r['idx']}: ERROR {r['error']}")
            else:
                print(f"  req {r['idx']}: wall={r['wall']:.3f}s tok/s={r['gen_tok_s']:.1f} Paris={r['clean']}")
    except Exception as e:
        print(f"Warm PRT FAILED: {e}")
        prt_warm = None

    # Summary
    print("\n=== SUMMARY ===")
    print(f"Native one-shot avg: {nat_avg_wall:.3f}s wall, {nat_avg_gen} gen t/s")
    print(f"PRT one-shot avg:    {prt_avg_wall:.3f}s wall, {prt_avg_gen} gen t/s")
    if nat_warm and all("error" not in r for r in nat_warm):
        first_nat = nat_warm[0]['wall']
        later_nat = sum(r['wall'] for r in nat_warm[1:]) / max(1, len(nat_warm)-1)
        print(f"Native warm first:   {first_nat:.3f}s, later avg: {later_nat:.3f}s")
    if prt_warm and all("error" not in r for r in prt_warm):
        first_prt = prt_warm[0]['wall']
        later_prt = sum(r['wall'] for r in prt_warm[1:]) / max(1, len(prt_warm)-1)
        print(f"PRT warm first:      {first_prt:.3f}s, later avg: {later_prt:.3f}s")

    # JSON output
    out = {
        "phase": "13AI",
        "native_oneshot": {
            "runs": 3,
            "per_run": [{"wall": r['wall'], "gen_tok_s": r['gen_tok_s']} for r in nat_results],
            "avg_wall": nat_avg_wall,
            "avg_gen_tok_s": nat_avg_gen,
        },
        "prt_oneshot": {
            "runs": 3,
            "per_run": [{"wall": r['wall'], "sidecar_ms": r['sidecar_ms'], "gen_tok_s": r['gen_tok_s']} for r in prt_results],
            "avg_wall": prt_avg_wall,
            "avg_sidecar_ms": prt_avg_sc,
            "avg_gen_tok_s": prt_avg_gen,
        },
        "warm_native": nat_warm,
        "warm_prt": prt_warm,
    }
    print("\n" + json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
