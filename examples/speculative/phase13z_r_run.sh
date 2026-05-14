#!/bin/bash
# Phase 13Z-R runner
LLAMA_CLI="/home/matthew-villnave/llama.cpp/build/bin/llama-cli"
MODEL="/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
SIDECARS="/tmp/prt_sidecars"
LOGDIR="/tmp/phase13z_r_logs"
mkdir -p "$LOGDIR"
MAX_TOKENS=64
TIMEOUT=60

prompts=(
  "1|The capital of France is"
  "2|Write a Python function that reverses a list."
  "3|Once upon a time in a"
  "4|Explain CPU inference in one sentence."
  "5|Return JSON with keys name and status."
  "6|The fastest way to sort a list in Python is"
  "7|In two sentences, explain what RAM does."
  "8|Complete this phrase: artificial intelligence is"
)

echo "Starting Phase 13Z-R"
for entry in "${prompts[@]}"; do
  pid="${entry%%|*}"
  prompt="${entry#*|}"
  
  echo "=== Prompt $pid ==="
  
  # Native run
  timeout $TIMEOUT script -q -c "$LLAMA_CLI -m $MODEL -p '$prompt' --log-disable --single-turn -n $MAX_TOKENS -t 4" /dev/null > "$LOGDIR/native_p${pid}.txt" 2>&1
  native_exit=$?
  native_output=$(cat "$LOGDIR/native_p${pid}.txt")
  
  # PRT run
  timeout $TIMEOUT script -q -c "$LLAMA_CLI -m $MODEL -p '$prompt' --prt-log-file $LOGDIR/prt_p${pid}_prt.log --prt-log-level summary --prt-mode 5700 --prt-force-native 11,15 --single-turn -n $MAX_TOKENS -t 4" /dev/null > "$LOGDIR/prt_p${pid}.txt" 2>&1
  prt_exit=$?
  prt_output=$(cat "$LOGDIR/prt_p${pid}.txt")
  
  echo "Native exit=$native_exit | PRT exit=$prt_exit"
  echo "Native: ${native_output:0:80}"
  echo "PRT: ${prt_output:0:80}"
  echo ""
done

echo "Done. Logs in $LOGDIR"
