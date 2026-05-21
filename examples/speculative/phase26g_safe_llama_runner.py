#!/usr/bin/env python3
"""
Phase 26G Safe llama-cli Runner v3
Simple shell-pipe approach: timeout + grep filter + bounded capture.
"""

import subprocess
import time
import os
import tempfile
import signal

STDOUT_CAP = 128 * 1024   # 128 KB
TIMEOUT_SEC = 180

LLAMA_CLI = "/home/matthew-villnave/llama.cpp/build/bin/llama-cli"
MODEL_7B = "/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-7B-Instruct-Q4_K_M.gguf"

def get_mem_state():
    memavailable = swapfree = None
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                memavailable = int(line.split()[1])
            elif line.startswith("SwapFree:"):
                swapfree = int(line.split()[1])
    return memavailable, swapfree

def run_safe(prompt_text, ctx_size, n_tokens, model_path=None, timeout_sec=TIMEOUT_SEC):
    """Run llama-cli with timeout. Uses shell pipe to clean output."""
    model = model_path or MODEL_7B
    
    pre_mem, pre_swapfree = get_mem_state()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as pf:
        pf.write(prompt_text)
        pf.flush()
        prompt_file = pf.name
    
    # Build command
    cmd = [
        LLAMA_CLI,
        "-m", model,
        "-f", prompt_file,
        "--ctx-size", str(ctx_size),
        "-n", str(n_tokens),
        "--temp", "0",
        "-t", "1",
        "--log-disable",
    ]
    
    # grep filter to remove noise
    filter_cmd = "grep -v '^\[PRT' | grep -v '^> ' | grep -v '^build\|^modalities\|^  /\|^-\|/\|^▄\|^██\|^$\|^  \\s' | grep -v 'memory_breakdown\|llama_memory' | grep -v '^\\\[0m'"
    
    shell_cmd = f"""exec 2>&1
{cmd[0]} {' '.join(cmd[1:])}
"""
    
    start = time.time()
    
    try:
        # Run via shell with grep filter
        full_cmd = f"""exec 2>&1
{cmd[0]} {' '.join(cmd[1:])} | {filter_cmd}
"""
        
        proc = subprocess.Popen(
            full_cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            cwd=os.path.dirname(cmd[0])
        )
        
        timed_out = False
        killed = False
        output_lines = []
        total_chars = 0
        
        try:
            while True:
                if proc.poll() is not None:
                    # Process exited
                    break
                
                if time.time() - start > timeout_sec:
                    proc.send_signal(signal.SIGTERM)
                    timed_out = True
                    break
                
                line = proc.stdout.readline()
                if line == "":
                    break
                
                if total_chars < STDOUT_CAP:
                    output_lines.append(line)
                    total_chars += len(line)
                    
        except KeyboardInterrupt:
            proc.send_signal(signal.SIGTERM)
            killed = True
        
        # Final cleanup
        if proc.poll() is None:
            try:
                proc.send_signal(signal.SIGTERM)
                time.sleep(1)
                if proc.poll() is None:
                    proc.kill()
                    killed = True
            except OSError:
                pass
        
        proc.wait()
        exit_code = proc.returncode
        
        # Drain any remaining output
        try:
            while total_chars < STDOUT_CAP:
                line = proc.stdout.readline()
                if not line:
                    break
                output_lines.append(line)
                total_chars += len(line)
        except:
            pass
        
        elapsed = time.time() - start
        
    finally:
        os.unlink(prompt_file)
    
    post_mem, post_swapfree = get_mem_state()
    
    # Parse output: get first real text line after prompt
    completion_lines = []
    in_answer = False
    for line in output_lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("Loading model") or stripped.startswith("model ") or \
           stripped.startswith("available commands") or stripped.startswith("real ") or \
           stripped.startswith("user ") or stripped.startswith("sys "):
            continue
        if stripped.startswith("The capital of France"):  # skip prompt echo
            continue
        if not in_answer:
            in_answer = True
        if in_answer:
            completion_lines.append(stripped)
    
    result = {
        "exit_code": exit_code,
        "elapsed_sec": round(elapsed, 2),
        "timed_out": timed_out,
        "killed": killed,
        "stdout_chars": total_chars,
        "pre_memavailable_kb": pre_mem,
        "post_memavailable_kb": post_mem,
        "swap_delta_kb": (pre_swapfree or 0) - (post_swapfree or 0),
        "pre_swapfree_kb": pre_swapfree,
        "post_swapfree_kb": post_swapfree,
        "completion": " ".join(completion_lines[:50]),
    }
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <prompt_file> <ctx_size> [n_tokens] [timeout_sec]")
        sys.exit(1)
    
    with open(sys.argv[1]) as f:
        prompt_text = f.read()
    
    ctx_size = int(sys.argv[2])
    n_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    timeout_sec = int(sys.argv[4]) if len(sys.argv) > 4 else TIMEOUT_SEC
    
    result = run_safe(prompt_text, ctx_size, n_tokens, timeout_sec=timeout_sec)
    
    print(f"exit={result['exit_code']} elapsed={result['elapsed_sec']}s timed_out={result['timed_out']}")
    print(f"swap_delta_kb={result['swap_delta_kb']}")
    print(f"stdout_chars={result['stdout_chars']}")
    print(f"--- COMPLETION ---")
    print(result["completion"][:2000])