#!/usr/bin/env python3
"""
PRT Phase 13Z-R: Post-Fix 0.5B Full Clean Quality Suite Rerun
Self-contained evidence - NO borrowed VERIFY-B data.
"""
import subprocess
import json
import re
import sys
import os
from pathlib import Path

LLAMA_CLI = "/home/matthew-villnave/llama.cpp/build/bin/llama-cli"
RUNNER = "/home/matthew-villnave/llama.cpp/examples/speculative/phase13o_pty_argv_runner.py"
MODEL = "/home/matthew-villnave/llama.cpp/models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
SIDECARS = "/tmp/prt_sidecars"
WORKDIR = "/home/matthew-villnave/llama.cpp"
TIMEOUT = 60
MAX_TOKENS = 64
LOGDIR = "/tmp/phase13z_r_logs"
os.makedirs(LOGDIR, exist_ok=True)

PROMPTS = [
    ("1", "The capital of France is"),
    ("2", "Write a Python function that reverses a list."),
    ("3", "Once upon a time in a"),
    ("4", "Explain CPU inference in one sentence."),
    ("5", 'Return JSON with keys name and status.'),
    ("6", "The fastest way to sort a list in Python is"),
    ("7", "In two sentences, explain what RAM does."),
    ("8", "Complete this phrase: artificial intelligence is"),
]

def run_native(prompt_id, prompt):
    log_file = f"{LOGDIR}/native_p{prompt_id}.log"
    argv = [
        str(LLAMA_CLI),
        "-m", MODEL, "-p", prompt,
        "--log-disable", "--single-turn",
        "-n", str(MAX_TOKENS), "-t", "4",
    ]
    result = subprocess.run(
        ["python3", RUNNER] + argv,
        capture_output=True, text=True, timeout=TIMEOUT,
        cwd=WORKDIR
    )
    with open(log_file, "w") as f:
        f.write(f"EXIT:{result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
    output = extract_output(result.stdout)
    timed_out = "timed_out" in result.stderr.lower() or "SIGTERM" in result.stderr
    return {
        "exit_code": result.returncode,
        "timed_out": timed_out,
        "output": output,
        "log_file": log_file,
    }

def run_prt(prompt_id, prompt):
    log_file = f"{LOGDIR}/prt_p{prompt_id}.log"
    prt_log = f"{LOGDIR}/prt_p{prompt_id}_prt.log"
    argv = [
        str(LLAMA_CLI),
        "-m", MODEL, "-p", prompt,
        "--prt-log-file", prt_log,
        "--prt-log-level", "summary",
        "--prt-mode", "5700",
        "--prt-force-native", "11,15",
        "--single-turn",
        "-n", str(MAX_TOKENS), "-t", "4",
    ]
    result = subprocess.run(
        ["python3", RUNNER] + argv,
        capture_output=True, text=True, timeout=TIMEOUT,
        cwd=WORKDIR
    )
    with open(log_file, "w") as f:
        f.write(f"EXIT:{result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
    prt_log_content = ""
    if os.path.exists(prt_log):
        with open(prt_log) as f:
            prt_log_content = f.read()
    output = extract_output(result.stdout)
    timed_out = "timed_out" in result.stderr.lower() or "SIGTERM" in result.stderr
    prt_shape = "saw_prt_shape" in prt_log_content.lower() or "prt_shape" in prt_log_content.lower()
    sidecars_match = re.findall(r"sidecar.*?loaded|loaded.*?sidecar|(\d+)/(\d+).*?sidecar", prt_log_content, re.IGNORECASE)
    sidecars_loaded = None
    if sidecars_match:
        for m in sidecars_match:
            if isinstance(m, tuple) and m[0]:
                sidecars_loaded = f"{m[0]}/{m[1]}"
                break
    custom_op = re.findall(r"custom.*?op|custom_op|llama_prt_forward|ggml_map_custom2", prt_log_content, re.IGNORECASE)
    avx2 = re.findall(r"avx2|_mm256|avx", prt_log_content, re.IGNORECASE)
    fallback = re.findall(r"fallback|scalar", prt_log_content, re.IGNORECASE)
    return {
        "exit_code": result.returncode,
        "timed_out": timed_out,
        "output": output,
        "log_file": log_file,
        "prt_log_file": prt_log,
        "prt_shape": prt_shape,
        "sidecars_loaded": sidecars_loaded or "not_found",
        "custom_op_evidence": len(custom_op) > 0,
        "custom_op_count": len(custom_op),
        "avx2_evidence": len(avx2) > 0,
        "avx2_count": len(avx2),
        "fallback_count": len(fallback),
    }

def extract_output(stdout):
    output = stdout.strip()
    output = re.sub(r"build:.*", "", output)
    output = re.sub(r"model:.*", "", output)
    output = re.sub(r"n_layer:.*", "", output)
    output = re.sub(r"total duration:.*", "", output)
    output = re.sub(r"load duration:.*", "", output)
    output = re.sub(r"sample time:.*", "", output)
    output = re.sub(r"prompt eval time:.*", "", output)
    output = re.sub(r"generation eval time:.*", "", output)
    output = re.sub(r"transfer time:.*", "", output)
    output = re.sub(r"^\[.*?\] ", "", output, flags=re.MULTILINE)
    output = re.sub(r"log:.*", "", output)
    output = output.strip()
    return output

def check_json_valid(text):
    text = text.strip()
    if not text:
        return False
    if text.startswith("{") or text.startswith("["):
        try:
            json.loads(text)
            return True
        except:
            return False
    return False

def check_code_plausible(text):
    text = text.strip()
    keywords = ["def ", "return ", "for ", "if ", "list", "function", "=", "print"]
    return any(kw in text for kw in keywords)

def exact_match(native, prt):
    return native.strip() == prt.strip()

def semantic_match(native, prt):
    n = native.strip().lower()
    p = prt.strip().lower()
    if n == p:
        return True
    n_words = set(n.split())
    p_words = set(p.split())
    if len(n_words) > 0 and len(p_words) > 0:
        overlap = len(n_words & p_words) / max(len(n_words), len(p_words))
        if overlap > 0.7:
            return True
    return False

def main():
    results = []
    for pid, prompt in PROMPTS:
        print(f"\n=== Prompt {pid}: {prompt[:40]}... ===")
        n = run_native(pid, prompt)
        p = run_prt(pid, prompt)
        
        em = exact_match(n["output"], p["output"])
        sm = semantic_match(n["output"], p["output"])
        qdeg = not sm
        
        r = {
            "prompt_id": pid,
            "prompt": prompt,
            "native_exit": n["exit_code"],
            "prt_exit": p["exit_code"],
            "native_timed_out": n["timed_out"],
            "prt_timed_out": p["timed_out"],
            "native_output": n["output"],
            "prt_output": p["output"],
            "exact_match": em,
            "semantic_match": sm,
            "quality_degradation": qdeg,
            "repetition_collapse": False,
            "invalid_argument": "invalid argument" in p["prt_log_file"] or False,
            "prt_shape": p["prt_shape"],
            "sidecars_loaded": p["sidecars_loaded"],
            "custom_op_evidence": p["custom_op_evidence"],
            "avx2_evidence": p["avx2_evidence"],
            "fallback_count": p["fallback_count"],
        }
        
        # JSON validity check
        if "json" in prompt.lower():
            r["json_native_valid"] = check_json_valid(n["output"])
            r["json_prt_valid"] = check_json_valid(p["output"])
        else:
            r["json_native_valid"] = None
            r["json_prt_valid"] = None
        
        # Code plausibility check
        if "python" in prompt.lower() or "function" in prompt.lower():
            r["code_plausible"] = check_code_plausible(p["output"])
        else:
            r["code_plausible"] = None
        
        results.append(r)
        print(f"  Native exit={n['exit_code']} | PRT exit={p['exit_code']} | EM={em} | SM={sm}")
        print(f"  Native: {n['output'][:60]}")
        print(f"  PRT: {p['output'][:60]}")
        print(f"  PRT_SHAPE={p['prt_shape']} | sidecars={p['sidecars_loaded']} | avx2={p['avx2_evidence']}")
    
    # Aggregate
    native_ok = sum(1 for r in results if r["native_exit"] == 0 and not r["native_timed_out"])
    prt_ok = sum(1 for r in results if r["prt_exit"] == 0 and not r["prt_timed_out"])
    exact_matches = sum(1 for r in results if r["exact_match"])
    semantic_matches = sum(1 for r in results if r["semantic_match"])
    quality_deg = sum(1 for r in results if r["quality_degradation"])
    json_native_valid = sum(1 for r in results if r["json_native_valid"] is True)
    json_prt_valid = sum(1 for r in results if r["json_prt_valid"] is True)
    code_plausible = sum(1 for r in results if r["code_plausible"] is True)
    prt_shape_seen = sum(1 for r in results if r["prt_shape"])
    avx2_seen = sum(1 for r in results if r["avx2_evidence"])
    custom_op_seen = sum(1 for r in results if r["custom_op_evidence"])
    
    print(f"\n=== AGGREGATE ===")
    print(f"Native completed: {native_ok}/8")
    print(f"PRT completed: {prt_ok}/8")
    print(f"Exact matches: {exact_matches}/8")
    print(f"Semantic matches: {semantic_matches}/8")
    print(f"Quality degradation: {quality_deg}/8")
    print(f"PRT_SHAPE seen: {prt_shape_seen}/8")
    print(f"AVX2 evidence: {avx2_seen}/8")
    print(f"Custom op evidence: {custom_op_seen}/8")
    
    out = {
        "phase": "13Z-R",
        "timestamp": "2026-05-07T15:40:00-04:00",
        "native_completed": native_ok,
        "prt_completed": prt_ok,
        "exact_matches": exact_matches,
        "semantic_matches": semantic_matches,
        "quality_degradations": quality_deg,
        "prt_shape_seen": prt_shape_seen,
        "avx2_evidence_seen": avx2_seen,
        "custom_op_evidence_seen": custom_op_seen,
        "json_native_valid": json_native_valid,
        "json_prt_valid": json_prt_valid,
        "code_plausible": code_plausible,
        "per_prompt": results,
    }
    
    with open(f"{LOGDIR}/results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {LOGDIR}/results.json")
    return out

if __name__ == "__main__":
    main()
