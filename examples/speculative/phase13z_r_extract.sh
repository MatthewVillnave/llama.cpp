#!/bin/bash
# Phase 13Z-R evidence extractor
# Reads /tmp/phase13z_r_logs/ and generates per-prompt evidence

LOGDIR="/tmp/phase13z_r_logs"
OUT="$LOGDIR/evidence.json"

prompts=(
  "1|The capital of France is|Paris"
  "2|Write a Python function that reverses a list.|def"
  "3|Once upon a time in a|story"
  "4|Explain CPU inference in one sentence.|CPU"
  "5|Return JSON with keys name and status.|{"
  "6|The fastest way to sort a list in Python is|sorted"
  "7|In two sentences, explain what RAM does.|RAM"
  "8|Complete this phrase: artificial intelligence is|artificial"
)

echo "{"
echo '  "phase": "13Z-R",'
echo '  "timestamp": "2026-05-07T15:45:00-04:00",'
echo '  "per_prompt": ['

first=1
for entry in "${prompts[@]}"; do
  pid="${entry%%|*}"
  rest="${entry#*|}"
  prompt="${rest%%|*}"
  keyword="${rest##*|}"
  
  native_file="$LOGDIR/native_p${pid}.txt"
  prt_file="$LOGDIR/prt_p${pid}.txt"
  prt_log="$LOGDIR/prt_p${pid}_prt.log"
  
  # Extract native output
  native_out=$(cat "$native_file" | sed 's/\x1b\[[0-9;]*m//g' | grep -v "^$" | tail -5 | head -1 | sed 's/^[[:space:]]*|//;s/[[:space:]]*$//')
  prt_out=$(cat "$prt_file" | sed 's/\x1b\[[0-9;]*m//g' | grep -v "^$" | tail -5 | head -1 | sed 's/^[[:space:]]*|//;s/[[:space:]]*$//')
  
  native_exit=$(tail -3 "$native_file" | grep -c "Exiting" || echo 0)
  prt_exit=$(tail -3 "$prt_file" | grep -c "Exiting" || echo 0)
  
  # Check PRT log
  prt_shape=$(grep -c "PRT_SHAPE" "$prt_log" 2>/dev/null || echo 0)
  sidecars=$(grep "Loaded.*sidecars" "$prt_log" 2>/dev/null || echo "0/0")
  avx2=$(grep -c "AVX2\|__AVX2__\|kernel_mode=1" "$prt_log" 2>/dev/null || echo 0)
  force_native=$(grep "force-native" "$prt_log" 2>/dev/null || echo "none")
  
  # Exact match
  if [ "$native_out" = "$prt_out" ]; then
    exact=1; semantic=1
  else
    exact=0
    # Simple semantic check - contains keyword or similar length
    nk=$(echo "$native_out" | grep -c "$keyword" || echo 0)
    pk=$(echo "$prt_out" | grep -c "$keyword" || echo 0)
    [ $nk -gt 0 ] && [ $pk -gt 0 ] && semantic=1 || semantic=0
  fi
  
  # JSON check for prompt 5
  json_native=0; json_prt=0
  if [ "$pid" = "5" ]; then
    echo "$native_out" | python3 -c "import sys,json; json.load(sys.stdin); print(1)" 2>/dev/null && json_native=1
    echo "$prt_out" | python3 -c "import sys,json; json.load(sys.stdin); print(1)" 2>/dev/null && json_prt=1
  fi
  
  sep=""
  [ $first -eq 0 ] && sep=","
  first=0
  
  echo "  $sep{"
  echo "    \"prompt_id\": $pid,"
  echo "    \"prompt\": \"$prompt\","
  echo "    \"native_output\": \"$(echo "$native_out" | head -1 | sed 's/"/\\"/g')\","
  echo "    \"prt_output\": \"$(echo "$prt_out" | head -1 | sed 's/"/\\"/g')\","
  echo "    \"native_exit\": $native_exit,"
  echo "    \"prt_exit\": $prt_exit,"
  echo "    \"exact_match\": $exact,"
  echo "    \"semantic_match\": $semantic,"
  echo "    \"prt_shape_seen\": $prt_shape,"
  echo "    \"sidecars_loaded\": \"$sidecars\","
  echo "    \"avx2_evidence\": $avx2,"
  echo "    \"force_native_layers\": \"$force_native\""
  echo -n "  }"
  
  echo ""
done

echo "  ],"
echo '  "summary": {'
echo '    "note": "AVX2 and custom op evidence requires debug-level log (--prt-log-level debug). This run used summary level."'
echo '  }'
echo "}"
