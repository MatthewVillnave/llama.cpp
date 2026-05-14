#!/bin/bash
# Phase 13AA: Post-Fix 0.5B Timing Rebaseline
LLAMA_CLI="/home/matthew-villnave/llama.cpp/build/bin/llama-cli"
MODEL="/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
SIDECARS="/tmp/prt_sidecars"
LOGDIR="/tmp/phase13aa_logs"
mkdir -p "$LOGDIR"
PROMPT="The capital of France is"
N_RUNS=5
TIMEOUT=60
MAX_TOKENS=80
N_CTX=256
N_THREADS=4

echo "Phase 13AA: Post-Fix 0.5B Timing Rebaseline"
echo "=========================================="

# Native runs
echo ""
echo "=== NATIVE BASELINE ($N_RUNS runs) ==="
for i in $(seq 1 $N_RUNS); do
  log="$LOGDIR/native_p${i}.txt"
  prt_log="$LOGDIR/native_p${i}_prt.log"
  start=$(date +%s.%N)
  timeout $TIMEOUT script -q -c "$LLAMA_CLI -m $MODEL -p '$PROMPT' --log-disable --single-turn -n $MAX_TOKENS -t $N_THREADS -c $N_CTX --temp 0 --no-display-prompt" /dev/null > "$log" 2>&1
  exit=$?
  end=$(date +%s.%N)
  wall=$(echo "$end - $start" | bc)
  echo "NATIVE $i: exit=$exit wall=${wall}s" | tee -a "$LOGDIR/native_summary.txt"
done

# PRT runs
echo ""
echo "=== PRT FIXED AVX2 ($N_RUNS runs) ==="
for i in $(seq 1 $N_RUNS); do
  log="$LOGDIR/prt_p${i}.txt"
  prt_log="$LOGDIR/prt_p${i}_prt.log"
  start=$(date +%s.%N)
  timeout $TIMEOUT script -q -c "$LLAMA_CLI -m $MODEL -p '$PROMPT' --prt-log-file '$prt_log' --prt-log-level quiet --prt-mode 5700 --prt-force-native 11,15 --single-turn -n $MAX_TOKENS -t $N_THREADS -c $N_CTX --temp 0 --no-display-prompt" /dev/null > "$log" 2>&1
  exit=$?
  end=$(date +%s.%N)
  wall=$(echo "$end - $start" | bc)
  echo "PRT $i: exit=$exit wall=${wall}s" | tee -a "$LOGDIR/prt_summary.txt"
done

echo ""
echo "Logs in $LOGDIR"
