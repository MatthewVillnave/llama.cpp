#!/bin/bash
# Phase 13N: Full 8-prompt PTY validation suite
set -e

MODEL="/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
RUNNER="/home/matthew-villnave/llama.cpp/examples/speculative/phase13m_pty_runner.sh"
TIMEOUT=120
TAIL_BYTES=16384

PROMpts=(
    "The capital of France is"
    "Write a Python function that reverses a list."
    "Once upon a time in a"
    "Explain CPU inference in one sentence."
    "Return JSON with keys name and status."
    "The fastest way to sort a list in Python is"
    "In two sentences, explain what RAM does."
    "Complete this phrase: artificial intelligence is"
)

echo "=== Phase 13N Pre-flight ==="
echo "Model: $MODEL"
echo "Runner: $RUNNER"
echo "LLM build: $(./build/bin/llama-cli --version 2>/dev/null | head -1 || echo unknown)"
echo "Sidecars: $(ls /tmp/prt_sidecars/ | wc -l)"
echo ""

for i in "${!PROMpts[@]}"; do
    idx=$((i+1))
    prompt="${PROMpts[$i]}"
    echo "=== Prompt $idx/8: $prompt ==="
done