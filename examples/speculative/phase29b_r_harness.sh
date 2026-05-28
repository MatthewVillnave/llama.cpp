#!/bin/bash
# Phase 29B-R Real Sidecar Memory Audit Harness
set -euo pipefail

LLAMA="${LLAMA:-/home/matthew-villnave/llama.cpp/build/bin/llama-cli}"
MODEL="${MODEL:-/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf}"
SIDECAR="${SIDECAR:-/tmp/prt_sidecars_0_5b_layer0}"
LOG_DIR="${LOG_DIR:-/tmp/phase29b_r_logs}"
mkdir -p "$LOG_DIR"

run_test() {
  local name="$1"; shift
  local n_pred="$1"; shift
  local extra_flags=("$@")

  local logfile="$LOG_DIR/${name}_n${n_pred}.log"
  local jsonfile="$LOG_DIR/${name}_n${n_pred}.json"

  echo "=== RUN: $name n_predict=$n_pred ===" >&2

  # Build command
  local cmd=(
    /usr/bin/time -v "$LLAMA"
      -m "$MODEL"
      --single-turn --seed 42 -t 0 --log-disable
      -p "Hello world, how are you today?"
      -n "$n_pred"
      --enable-prt-sidecar-pager
      --prt-sidecar-dir "$SIDECAR"
      --prt-sidecar-manifest "$SIDECAR/manifest.json"
      --prt-sidecar-budget-mb 10
      --prt-mode 1
      "${extra_flags[@]}"
  )

  # Capture wall time
  local wall_start=$(date +%s.%N)

  # Run and capture output
  local exit_code=0
  local time_output
  time_output=$({ { stdbuf -oL "${cmd[@]}" 2>&1; echo "EXIT:$?" >&3; } | tee "$logfile"; } 3>&1) || exit_code=$?

  local wall_end=$(date +%s.%N)
  local wall_time=$(echo "$wall_end - $wall_start" | bc)

  # Extract /usr/bin/time metrics
  local max_rss=$(echo "$time_output" | grep -E '^Maximum resident set size' | sed 's/.*: *//')
  local minor_pf=$(echo "$time_output" | grep -E '^Minor \(reclaiming a page\)' | sed 's/.*: *//')
  local major_pf=$(echo "$time_output" | grep -E '^Major \(requiring I/O\)' | sed 's/.*: *//')
  local wall_time_line=$(echo "$time_output" | grep -E '^Elapsed' | sed 's/.*: *//')

  # Extract PRT metrics from log
  local activation_attempts=$(grep -oP 'activation_attempts=\K\d+' "$logfile" | awk '{s+=$1}END{print s+0}')
  local activation_successes=$(grep -oP 'activation_successes=\K\d+' "$logfile" | awk '{s+=$1}END{print s+0}')
  local budget_rejects=$(grep -oP 'budget_rejects=\K\d+' "$logfile" | awk '{s+=$1}END{print s+0}')
  local resident_bytes=$(grep -oP 'resident_bytes=\K\d+' "$logfile" | tail -1)
  local prt_flags=$(grep 'PRT-FLAGS-SET' "$logfile" | head -1)
  local sidecar_load=$(grep 'PRT-PAGER-LAZY' "$logfile" | head -1)

  # Extract exit code from log
  local actual_exit=$(grep -oP 'EXIT:\K\d+' "$logfile" | tail -1)
  actual_exit=${actual_exit:-$exit_code}

  echo "  wall_time=$wall_time exit=$actual_exit RSS=$max_rss KB" >&2
  echo "  activation_attempts=$activation_attempts activation_successes=$activation_successes budget_rejects=$budget_rejects" >&2
  echo "  resident_bytes=$resident_bytes" >&2

  # Write JSON
  cat > "$jsonfile" <<EOF
{
  "name": "$name",
  "n_predict": $n_pred,
  "exit_code": ${actual_exit:-0},
  "wall_time": $wall_time,
  "max_rss_kb": ${max_rss:-0},
  "minor_page_faults": ${minor_pf:-0},
  "major_page_faults": ${major_pf:-0},
  "activation_attempts": ${activation_attempts:-0},
  "activation_successes": ${activation_successes:-0},
  "budget_rejects": ${budget_rejects:-0},
  "resident_bytes": ${resident_bytes:-0},
  "prt_flags": "${prt_flags:-}",
  "sidecar_load_line": "${sidecar_load:-}"
}
EOF
  echo "  JSON: $jsonfile" >&2
}

# Test matrix
# A: Baseline native (no pager)
run_test "A_baseline" 1
run_test "A_baseline" 8
run_test "A_baseline" 32

# B: Pager observe-only (no apply)
run_test "B_pager_observe" 1 --prt-sidecar-apply
run_test "B_pager_observe" 8 --prt-sidecar-apply
run_test "B_pager_observe" 32 --prt-sidecar-apply

# C: attn_out layer0 injection
run_test "C_attn_out" 1 \
  --prt-sidecar-apply \
  --prt-sidecar-apply-family attn_out \
  --prt-sidecar-apply-layer 0 \
  --prt-sidecar-true-injection \
  --prt-sidecar-scale 1.0

run_test "C_attn_out" 8 \
  --prt-sidecar-apply \
  --prt-sidecar-apply-family attn_out \
  --prt-sidecar-apply-layer 0 \
  --prt-sidecar-true-injection \
  --prt-sidecar-scale 1.0

run_test "C_attn_out" 32 \
  --prt-sidecar-apply \
  --prt-sidecar-apply-family attn_out \
  --prt-sidecar-apply-layer 0 \
  --prt-sidecar-true-injection \
  --prt-sidecar-scale 1.0

# D: ffn_up layer0 injection
run_test "D_ffn_up" 1 \
  --prt-sidecar-apply \
  --prt-sidecar-apply-family ffn_up \
  --prt-sidecar-apply-layer 0 \
  --prt-sidecar-true-injection \
  --prt-sidecar-scale 1.0

run_test "D_ffn_up" 8 \
  --prt-sidecar-apply \
  --prt-sidecar-apply-family ffn_up \
  --prt-sidecar-apply-layer 0 \
  --prt-sidecar-true-injection \
  --prt-sidecar-scale 1.0

run_test "D_ffn_up" 32 \
  --prt-sidecar-apply \
  --prt-sidecar-apply-family ffn_up \
  --prt-sidecar-apply-layer 0 \
  --prt-sidecar-true-injection \
  --prt-sidecar-scale 1.0

# E: ffn_down layer0 injection
run_test "E_ffn_down" 1 \
  --prt-sidecar-apply \
  --prt-sidecar-apply-family ffn_down \
  --prt-sidecar-apply-layer 0 \
  --prt-sidecar-true-injection \
  --prt-sidecar-scale 1.0

run_test "E_ffn_down" 8 \
  --prt-sidecar-apply \
  --prt-sidecar-apply-family ffn_down \
  --prt-sidecar-apply-layer 0 \
  --prt-sidecar-true-injection \
  --prt-sidecar-scale 1.0

run_test "E_ffn_down" 32 \
  --prt-sidecar-apply \
  --prt-sidecar-apply-family ffn_down \
  --prt-sidecar-apply-layer 0 \
  --prt-sidecar-true-injection \
  --prt-sidecar-scale 1.0

# G: Budget enforcement (attn_out layer0, n_predict=8, varying budget)
for budget in 0 1 8 32 512; do
  run_test "G_budget_${budget}" 8 \
    --prt-sidecar-apply \
    --prt-sidecar-apply-family attn_out \
    --prt-sidecar-apply-layer 0 \
    --prt-sidecar-true-injection \
    --prt-sidecar-scale 1.0 \
    --prt-sidecar-budget-mb "$budget"
done

echo "=== ALL RUNS COMPLETE ===" >&2
