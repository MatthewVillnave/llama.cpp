#!/bin/bash
# Phase 11AZ Part 1: 20-prompt quality suite
# Extracts token IDs from [GEN] step lines

BIN="/home/matthew-villnave/llama.cpp/build/bin/llama-prt-posix"
MODEL="/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-3B-Instruct-Q4_K_M.gguf"
WORKDIR="/home/matthew-villnave/llama.cpp"
export LD_LIBRARY_PATH="/home/matthew-villnave/llama.cpp/build/bin"

PROMPTS=(
    "What is the capital of France?"
    "XYZ"
    "The company is a large"
    "Tell me a story about a dragon"
    "Once upon a time in a distant galaxy"
    "Write a Python function to reverse a list."
    "Explain photosynthesis in one paragraph."
    "Give me three bullet points about CPU inference."
    "Translate hello world to Spanish."
    "Solve: 12 * 17."
    "def quick_sort(arr):"
    "The meaning of life is"
    "In a small village near the mountains"
    "List five colors."
    "Why is the sky blue?"
    "Write a haiku about winter."
    "Summarize the benefits of exercise."
    "Complete this sentence: Artificial intelligence is"
    "Give me a JSON object with name and age."
    "What comes after Monday?"
)

run_prompt() {
    local idx=$1
    local prompt="$2"
    local mode="$3"
    local extra="$4"
    local output
    output=$(cd "$WORKDIR" && LD_LIBRARY_PATH="$LD_LIBRARY_PATH" "$BIN" \
        -m "$MODEL" \
        -p "$prompt" \
        -n 20 \
        --prt-mode "$mode" \
        $extra \
        2>&1)
    echo "$output"
}

echo "=== Phase 11AZ Part 1: Starting 20-prompt suite ==="
echo ""

for i in "${!PROMPTS[@]}"; do
    idx=$((i+1))
    prompt="${PROMPTS[$i]}"
    echo "--- Prompt $idx: $prompt ---"

    # Mode 0 (native)
    out0=$(run_prompt $idx "$prompt" 0 "")
    echo "=== MODE 0 (native) ==="
    echo "$out0" | grep "\[GEN\] step" | awk -F'token_id=' '{print $2}' | awk '{print $1}' | head -10 | tr '\n' ' '
    echo ""
    echo "M0_TEXT:"
    echo "$out0" | grep "\[GEN\] step" | head -10 | sed 's/.*str='"'"'\(.*\)'"'"' (replaced.*/\1/' | tr '\n' ' '
    echo ""

    # Mode 5435 (scalar)
    out5435=$(run_prompt $idx "$prompt" 5435 "")
    echo "=== MODE 5435 (scalar) ==="
    echo "$out5435" | grep "\[GEN\] step" | awk -F'token_id=' '{print $2}' | awk '{print $1}' | head -10 | tr '\n' ' '
    echo ""
    echo "MS_TEXT:"
    echo "$out5435" | grep "\[GEN\] step" | head -10 | sed 's/.*str='"'"'\(.*\)'"'"' (replaced.*/\1/' | tr '\n' ' '
    echo ""

    # Mode 5435 --prt-kernel 1 (AVX2)
    out5435avx=$(run_prompt $idx "$prompt" 5435 "--prt-kernel 1")
    echo "=== MODE 5435 AVX2 ==="
    echo "$out5435avx" | grep "\[GEN\] step" | awk -F'token_id=' '{print $2}' | awk '{print $1}' | head -10 | tr '\n' ' '
    echo ""
    echo "MA_TEXT:"
    echo "$out5435avx" | grep "\[GEN\] step" | head -10 | sed 's/.*str='"'"'\(.*\)'"'"' (replaced.*/\1/' | tr '\n' ' '
    echo ""

    echo ""
done