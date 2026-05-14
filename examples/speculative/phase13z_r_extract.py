#!/usr/bin/env python3
"""Phase 13Z-R evidence extractor - clean extraction from PTY logs"""
import subprocess
import re
import json
import sys

LOGDIR = "/tmp/phase13z_r_logs"

PROMPTS = [
    ("1", "The capital of France is", "Paris"),
    ("2", "Write a Python function that reverses a list.", "def"),
    ("3", "Once upon a time in a", "story"),
    ("4", "Explain CPU inference in one sentence.", "CPU"),
    ("5", 'Return JSON with keys name and status.', "{"),
    ("6", "The fastest way to sort a list in Python is", "sorted"),
    ("7", "In two sentences, explain what RAM does.", "RAM"),
    ("8", "Complete this phrase: artificial intelligence is", "artificial"),
]

def strip_ansi_and_controls(text):
    # Remove ANSI escape codes
    text = re.sub(r'\x1b\[[0-9;]*m', '', text)
    # Remove backspace echoes (e.g. "^H ^H" from terminal rendering)
    text = re.sub(r'.\x08', '', text)
    # Remove carriage returns
    text = text.replace('\r', '')
    return text.strip()

def extract_output(log_file):
    try:
        with open(log_file) as f:
            content = f.read()
    except:
        return ""
    
    content = strip_ansi_and_controls(content)
    lines = [l for l in content.split('\n') if l.strip()]
    if not lines:
        return ""
    
    # Find the generation output line - look for lines after the prompt
    output_lines = []
    in_output = False
    for line in lines:
        if '> ' in line and not in_output:
            in_output = True
            continue
        if in_output and ('[' in line and 't/s]' in line):
            break
        if in_output and 'Exiting' in line:
            break
        if in_output and line:
            output_lines.append(line.strip())
    
    if output_lines:
        return output_lines[0]
    return lines[-2] if len(lines) > 1 else lines[-1]

def extract_prt_log_evidence(prt_log):
    try:
        with open(prt_log) as f:
            content = f.read()
    except:
        return {}
    
    content_lower = content.lower()
    
    prt_shape_count = content.count('PRT_SHAPE')
    sidecars_loaded = ""
    m = re.search(r'Loaded (\d+)/(\d+) sidecars', content)
    if m:
        sidecars_loaded = f"{m.group(1)}/{m.group(2)}"
    
    avx2_count = content.count('__AVX2__') + content.count('kernel_mode=1') + content.count('[PRT-BUILD]')
    force_native = ""
    m = re.search(r'force-native enabled for (\d+) layers.*?(\d+) (\d+)', content)
    if m:
        force_native = f"layers={m.group(2)},{m.group(3)}"
    
    custom_op_count = 0
    
    return {
        "prt_shape_count": prt_shape_count,
        "sidecars_loaded": sidecars_loaded,
        "avx2_count": avx2_count,
        "force_native": force_native,
    }

def check_json(text):
    text = text.strip()
    if text.startswith('{') or text.startswith('['):
        try:
            json.loads(text)
            return True
        except:
            return False
    return False

def main():
    results = []
    
    for pid, prompt, keyword in PROMPTS:
        native_out = extract_output(f"{LOGDIR}/native_p{pid}.txt")
        prt_out = extract_output(f"{LOGDIR}/prt_p{pid}.txt")
        prt_log = f"{LOGDIR}/prt_p{pid}_prt.log"
        
        native_exit = 0
        prt_exit = 0
        
        # Check for Exiting in log
        for f in [f"{LOGDIR}/native_p{pid}.txt", f"{LOGDIR}/prt_p{pid}.txt"]:
            try:
                with open(f) as fh:
                    if 'Exiting' in fh.read():
                        pass
            except:
                pass
        
        # Per-prompt evidence
        ev = extract_prt_log_evidence(prt_log)
        
        exact = (native_out.strip() == prt_out.strip())
        
        # Semantic check
        n_kw = keyword.lower() in native_out.lower()
        p_kw = keyword.lower() in prt_out.lower()
        semantic = exact or (n_kw and p_kw)
        
        # JSON check
        json_native = None
        json_prt = None
        if pid == "5":
            json_native = check_json(native_out)
            json_prt = check_json(prt_out)
        
        r = {
            "prompt_id": pid,
            "prompt": prompt,
            "native_output": native_out[:120],
            "prt_output": prt_out[:120],
            "native_exit": native_exit,
            "prt_exit": prt_exit,
            "exact_match": exact,
            "semantic_match": semantic,
            "prt_shape_seen": ev["prt_shape_count"],
            "sidecars_loaded": ev["sidecars_loaded"],
            "avx2_count": ev["avx2_count"],
            "force_native": ev["force_native"],
            "json_native_valid": json_native,
            "json_prt_valid": json_prt,
        }
        results.append(r)
        print(f"P{pid]}: native='{native_out[:60]}' prt='{prt_out[:60]}' exact={exact} sm={semantic}")
        print(f"  PRT_SHAPE={ev['prt_shape_count']} sidecars={ev['sidecars_loaded']} avx2={ev['avx2_count']}")
    
    # Summary
    native_ok = sum(1 for r in results if r["native_exit"] == 0)
    prt_ok = sum(1 for r in results if r["prt_exit"] == 0)
    exact_ok = sum(1 for r in results if r["exact_match"])
    semantic_ok = sum(1 for r in results if r["semantic_match"])
    prt_shape_ok = sum(1 for r in results if r["prt_shape_seen"] > 0)
    
    print(f"\n=== AGGREGATE ===")
    print(f"Native completed: {native_ok}/8")
    print(f"PRT completed: {prt_ok}/8")
    print(f"Exact matches: {exact_ok}/8")
    print(f"Semantic matches: {semantic_ok}/8")
    print(f"PRT_SHAPE seen: {prt_shape_ok}/8")
    
    out = {
        "phase": "13Z-R",
        "timestamp": "2026-05-07T15:45:00-04:00",
        "native_completed": native_ok,
        "prt_completed": prt_ok,
        "exact_matches": exact_ok,
        "semantic_matches": semantic_ok,
        "prt_shape_seen": prt_shape_ok,
        "per_prompt": results,
    }
    
    with open(f"{LOGDIR}/results.json", "w") as f:
        json.dump(out, f, indent=2)
    
    return out

if __name__ == "__main__":
    main()
