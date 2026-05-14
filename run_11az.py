#!/usr/bin/env python3
"""Phase 11AZ Part 1: 20-prompt quality suite"""
import subprocess
import re
import os

BIN = "/home/matthew-villnave/llama.cpp/build/bin/llama-prt-posix"
MODEL = "/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-3B-Instruct-Q4_K_M.gguf"
WORKDIR = "/home/matthew-villnave/llama.cpp"
os.chdir(WORKDIR)
os.environ["LD_LIBRARY_PATH"] = "/home/matthew-villnave/llama.cpp/build/bin"

PROMPTS = [
    "What is the capital of France?",
    "XYZ",
    "The company is a large",
    "Tell me a story about a dragon",
    "Once upon a time in a distant galaxy",
    "Write a Python function to reverse a list.",
    "Explain photosynthesis in one paragraph.",
    "Give me three bullet points about CPU inference.",
    "Translate hello world to Spanish.",
    "Solve: 12 * 17.",
    "def quick_sort(arr):",
    "The meaning of life is",
    "In a small village near the mountains",
    "List five colors.",
    "Why is the sky blue?",
    "Write a haiku about winter.",
    "Summarize the benefits of exercise.",
    "Complete this sentence: Artificial intelligence is",
    "Give me a JSON object with name and age.",
    "What comes after Monday?",
]

def run_prompt(prompt, mode, extra=""):
    cmd = [
        BIN,
        "-m", MODEL,
        "-p", prompt,
        "-n", "20",
        "--prt-mode", str(mode)
    ]
    if extra:
        cmd.extend(extra.split())
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        timeout=120
    )
    return result.stderr

def get_token_ids(output):
    # Extract token_id from [GEN] step lines
    tokens = []
    for line in output.split('\n'):
        if '[GEN] step' in line and 'token_id=' in line:
            match = re.search(r'token_id=(\d+)', line)
            if match:
                tokens.append(int(match.group(1)))
    return tokens[:10]  # First 10 tokens

results = []

for idx, prompt in enumerate(PROMPTS, 1):
    print(f"--- Prompt {idx}: {prompt[:40]}...", flush=True)
    
    # Mode 0 (native)
    out0 = run_prompt(prompt, 0, "")
    tokens0 = get_token_ids(out0)
    
    # Mode 5435 (scalar)
    out5435 = run_prompt(prompt, 5435, "")
    tokens5435 = get_token_ids(out5435)
    
    # Mode 5435 --prt-kernel 1 (AVX2)
    out5435avx = run_prompt(prompt, 5435, "--prt-kernel 1")
    tokens5435avx = get_token_ids(out5435avx)
    
    results.append({
        'idx': idx,
        'prompt': prompt,
        'm0': tokens0,
        'ms': tokens5435,
        'ma': tokens5435avx
    })
    
    # Print intermediate result
    m0_str = str(tokens0)
    ms_str = str(tokens5435)
    ma_str = str(tokens5435avx)
    print(f"  M0: {m0_str}", flush=True)
    print(f"  MS: {ms_str}", flush=True)
    print(f"  MA: {ma_str}", flush=True)

# Write results
with open("/home/matthew-villnave/llama.cpp/phase11az_results.txt", "w") as f:
    for r in results:
        f.write(f"--- Prompt {r['idx']}: {r['prompt']} ---\n")
        f.write(f"Native tokens (first 10): {r['m0']}\n")
        f.write(f"Scalar tokens (first 10): {r['ms']}\n")
        f.write(f"AVX2 tokens (first 10): {r['ma']}\n")
        f.write("\n")

print("Results written to phase11az_results.txt")