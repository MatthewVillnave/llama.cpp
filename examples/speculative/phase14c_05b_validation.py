#!/usr/bin/env python3
"""Phase 14C: 8-prompt quality + timing validation for INT8 sidecars"""
import subprocess, json, sys, os, time

MODEL = "/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
FLOAT_DIR = "/tmp/prt_sidecars/"
INT8_DIR = "/tmp/prt_sidecars_05b_int8/"
LLAMA = "/home/matthew-villnave/llama.cpp/build/bin/llama-cli"

PROMPTS = [
    ("1", "The capital of France is"),
    ("2", "Write a Python function that reverses a list."),
    ("3", "Once upon a time in a"),
    ("4", "Explain CPU inference in one sentence."),
    ("5", "Return JSON with keys name and status."),
    ("6", "The fastest way to sort a list in Python is"),
    ("7", "In two sentences, explain what RAM does."),
    ("8", "Complete this phrase: artificial intelligence is"),
]

def run_mode(prompt_id, mode, prompt_text):
    """mode: 'native', 'float32', 'int8'"""
    log_file = f"/tmp/phase14c_p{prompt_id}_{mode}.log"
    
    argv = [LLAMA, "-m", MODEL, "-p", prompt_text,
            "-n", "40", "--temp", "0", "-c", "256", "-t", "4",
            "--no-display-prompt", "--single-turn"]
    
    extra = []
    if mode == "float32":
        extra = ["--prt-mode", "5700", "--prt-force-native", "11,15",
                 "--prt-sidecar-dir", FLOAT_DIR, "--prt-sidecar-format", "float32",
                 "--prt-sidecar-mmap", "--prt-log-file", log_file, "--prt-log-level", "quiet"]
    elif mode == "int8":
        extra = ["--prt-mode", "5700", "--prt-force-native", "11,15",
                 "--prt-sidecar-dir", INT8_DIR, "--prt-sidecar-format", "int8",
                 "--prt-log-file", log_file, "--prt-log-level", "quiet"]
    
    argv.extend(extra)
    
    start = time.time()
    result = subprocess.run(argv, capture_output=True, text=True, timeout=120)
    elapsed = time.time() - start
    
    # Parse generation speed from stderr (format: "[ Prompt: X t/s | Generation: Y t/s ]")
    gen_tok_s = 0.0
    prompt_tok_s = 0.0
    for line in result.stderr.splitlines():
        if "Generation:" in line:
            import re
            m = re.search(r"Prompt:\s*([\d.]+)\s*t/s.*Generation:\s*([\d.]+)\s*t/s", line)
            if m:
                prompt_tok_s = float(m.group(1))
                gen_tok_s = float(m.group(2))
    
    # Extract clean output (lines after prompt line, before timing line)
    output_lines = []
    in_output = False
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("The capital") or stripped.startswith("Write a") or \
           stripped.startswith("Once upon") or stripped.startswith("Explain") or \
           stripped.startswith("Return JSON") or stripped.startswith("The fastest") or \
           stripped.startswith("In two") or stripped.startswith("Complete"):
            in_output = True
            continue
        if "Prompt:" in line or "Generation:" in line:
            break
        if in_output and stripped:
            output_lines.append(stripped)
    
    clean_output = " ".join(output_lines).strip()
    if not clean_output:
        # fallback: get last non-empty non-timing line from stdout
        for line in reversed(result.stdout.splitlines()):
            line = line.strip()
            if line and "Generation:" not in line and "Prompt:" not in line and \
               not line.startswith("build") and not line.startswith("model") and \
               not line.startswith("modalities") and not line.startswith("available"):
                clean_output = line.strip()
                break
    
    # Check log file for PRT info
    prt_shape = ""
    sidecar_format = ""
    sidecars_loaded = 0
    if os.path.exists(log_file):
        with open(log_file) as f:
            log = f.read()
        for line in log.splitlines():
            if "PRT_SHAPE" in line:
                prt_shape = line.strip()
            if "PRT_FORMAT" in line and "sidecar_format" in line:
                sidecar_format = line.strip()
            if "sidecars_loaded" in line:
                # Extract N/M from "sidecars_loaded=X/Y"
                import re
                m = re.search(r"sidecars_loaded=(\d+)/(\d+)", line)
                if m:
                    sidecars_loaded = int(m.group(1))
    
    return {
        "mode": mode,
        "exit_code": result.returncode,
        "elapsed": round(elapsed, 2),
        "gen_tok_per_s": gen_tok_s,
        "prompt_tok_per_s": prompt_tok_s,
        "output": clean_output[:200],
        "prt_shape": prt_shape,
        "sidecar_format": sidecar_format,
        "sidecars_loaded": sidecars_loaded,
    }

# Run 8-prompt suite
print("=== Phase 14C: 8-prompt quality suite ===")
all_results = {}
for pid, prompt_text in PROMPTS:
    print(f"\n--- Prompt {pid}: {prompt_text[:50]} ---")
    row = {}
    for mode in ["native", "float32", "int8"]:
        print(f"  Running {mode}...", end="", flush=True)
        r = run_mode(pid, mode, prompt_text)
        row[mode] = r
        print(f" gen={r['gen_tok_per_s']:.1f} t/s exit={r['exit_code']} output={r['output'][:60]}")
    all_results[pid] = {"prompt": prompt_text, "modes": row}

# Timing run: 5x each mode with prompt "The capital of France is"
print("\n=== Phase 14C: Timing (5 runs × 3 modes, n=80) ===")
TIMING_PROMPT = "The capital of France is"
timing_results = {}
for mode in ["native", "float32", "int8"]:
    runs = []
    for i in range(5):
        print(f"  {mode} run {i+1}/5...", end="", flush=True)
        log_file = f"/tmp/phase14c_timing_{mode}_run{i}.log"
        argv = [LLAMA, "-m", MODEL, "-p", TIMING_PROMPT,
                "-n", "80", "--temp", "0", "-c", "256", "-t", "4",
                "--no-display-prompt", "--single-turn"]
        extra = []
        if mode == "float32":
            extra = ["--prt-mode", "5700", "--prt-force-native", "11,15",
                     "--prt-sidecar-dir", FLOAT_DIR, "--prt-sidecar-format", "float32",
                     "--prt-sidecar-mmap", "--prt-log-file", log_file, "--prt-log-level", "quiet"]
        elif mode == "int8":
            extra = ["--prt-mode", "5700", "--prt-force-native", "11,15",
                     "--prt-sidecar-dir", INT8_DIR, "--prt-sidecar-format", "int8",
                     "--prt-log-file", log_file, "--prt-log-level", "quiet"]
        argv.extend(extra)
        
        start = time.time()
        result = subprocess.run(argv, capture_output=True, text=True, timeout=180)
        elapsed = time.time() - start
        
        gen_tok_s = 0.0
        for line in result.stderr.splitlines():
            if "Generation:" in line:
                import re
                m = re.search(r"Generation:\s*([\d.]+)\s*t/s", line)
                if m:
                    gen_tok_s = float(m.group(1))
        
        # Get clean output
        output_lines = []
        in_output = False
        for line in result.stdout.splitlines():
            stripped = line.strip()
            if "The capital" in stripped:
                in_output = True
                continue
            if "Generation:" in line or "Prompt:" in line:
                break
            if in_output and stripped:
                output_lines.append(stripped)
        clean_output = " ".join(output_lines).strip()
        
        runs.append({
            "run": i+1,
            "elapsed": round(elapsed, 2),
            "gen_tok_per_s": gen_tok_s,
            "output": clean_output[:100],
            "exit_code": result.returncode,
        })
        print(f" {elapsed:.1f}s gen={gen_tok_s:.1f} t/s")
    
    timing_results[mode] = runs

# Write combined results
with open("/tmp/phase14c_results.json", "w") as f:
    json.dump({"quality": all_results, "timing": timing_results}, f, indent=2)
print("\n=== Done. Results in /tmp/phase14c_results.json ===")