#!/bin/bash
#
# PTY-safe llama-cli runner for PRT Phase 13M
# Uses `script -q -c` to run commands under a pseudo-terminal
#
# Usage:
#   phase13m_pty_runner.sh --timeout SEC --tail-byTES BYTES -- CMD...
#
# Output: JSON to stdout

set -e

TIMEOUT=60
TAIL_BYTES=262144

while [[ $# -gt 0 ]]; do
    case "$1" in
        --timeout) TIMEOUT="$2"; shift 2 ;;
        --tail-bytes) TAIL_BYTES="$2"; shift 2 ;;
        --) shift; break ;;
        *) break ;;
    esac
done

if [[ $# -eq 0 ]]; then
    echo '{"error": "no command provided"}'
    exit 1
fi

# Run under PTY and capture output
START=$(date +%s.%N)

OUTPUT=$(timeout "$TIMEOUT" script -q -c "$*" /dev/null 2>&1) || true
EXIT_CODE=$?

END=$(date +%s.%N)
ELAPSED=$(echo "$END - $START" | bc 2>/dev/null || echo "0")

# Keep only tail
if [[ ${#OUTPUT} -gt $TAIL_BYTES ]]; then
    TAIL="${OUTPUT: -$TAIL_BYTES}"
else
    TAIL="$OUTPUT"
fi

# Detect patterns
CONTAINS_PRT_SHAPE=false
CONTAINS_SIDECAR=false
CONTAINS_FLAG_ECHO=false
CONTAINS_PATH_FRAG=false

if echo "$TAIL" | grep -qE "PRT_SHAPE|n_layer.*M="; then
    CONTAINS_PRT_SHAPE=true
fi
if echo "$TAIL" | grep -qiE "sidecar|loaded.*/"; then
    CONTAINS_SIDECAR=true
fi
if echo "$TAIL" | grep -qE "\-\-prt\-"; then
    CONTAINS_FLAG_ECHO=true
fi
if echo "$TAIL" | grep -qE "/tmp/prt_|/llama.cpp/build"; then
    CONTAINS_PATH_FRAG=true
fi

# Build JSON output
cat << EOF
{
  "exit_code": $EXIT_CODE,
  "timed_out": $(if [[ $EXIT_CODE -eq 124 ]] || [[ $EXIT_CODE -eq 137 ]]; then echo "true"; else echo "false"; fi),
  "elapsed_sec": $ELAPSED,
  "tail_bytes": ${#TAIL},
  "raw_bytes": ${#OUTPUT},
  "contains_prt_shape": $CONTAINS_PRT_SHAPE,
  "contains_sidecar_logs": $CONTAINS_SIDECAR,
  "contains_flag_echo": $CONTAINS_FLAG_ECHO,
  "contains_path_fragment": $CONTAINS_PATH_FRAG,
  "tail_text": $(echo "$TAIL" | head -c 16384 | python3 -c "import sys,json; print(json.dumps(sys.stdin.read()))")
}
EOF