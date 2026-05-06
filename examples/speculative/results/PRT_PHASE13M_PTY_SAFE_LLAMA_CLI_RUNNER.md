# PRT Phase 13M: PTY-Safe llama-cli Runner

**Date:** 2026-05-06
**Commit:** 9efd728fc (Phase 13L)

## Executive Summary

Used the `script -q -c` approach to run llama-cli under a PTY. Both native and active PRT tests now pass without hanging.

## Test Results

### Test 1: Native llama-cli through PTY
```
./build/bin/llama-cli -m $MODEL -p "test" -n 1 -c 256 -t 4 --no-display-prompt
```
- **Exit code:** 0 ✅
- **Output:** "Hello" generated
- **Timing:** 244.0 t/s prompt, generation complete
- **Verdict:** PASS ✅

### Test 2: Active PRT through PTY
```
./build/bin/llama-cli -m $MODEL -p "test test" -n 10 -c 256 -t 4 --no-display-prompt --prt-mode 5700 --prt-force-native 11,15 --prt-sidecar-dir /tmp/prt_sidecars/
```
- **Exit code:** 0 ✅
- **PRT_SHAPE detected:** n_layer=24 M=896 N=4864 ✅
- **Sidecars loaded:** 24/24 ✅
- **force-native enabled for layers:** 11, 15
- **Verdict:** PASS ✅

## Key Findings

The hang in Phase 13L was caused by llama-cli's console TTY detection. When stdout is redirected to /dev/null or a file, the console goes into an infinite loop.

The solution: use `script -q -c "command" /dev/null` to create a pseudo-terminal. This makes llama-cli think it's running in a terminal, bypassing the problematic console code path.

The `script` command:
- Creates a PTY for the child process
- Captures output to the specified file (or /dev/null for no capture)
- Returns exit code of the child process

## Runner Implementation

The PTY runner is a simple bash script at `examples/speculative/phase13m_pty_runner.sh`. It:
1. Accepts --timeout and --tail-bytes arguments
2. Runs command under `script -q -c`
3. Captures output with bounded tail buffer
4. Returns JSON with detection results

## Safety

- Models staged: NO
- Sidecars staged: NO
- Binaries staged: NO
- Secrets detected: NO
- PRT tags untouched: YES

## Verdict

**PASS** - PTY runner successfully enables both native and active PRT llama-cli execution.

## Recommended Next Steps

1. Use this PTY runner for all future llama-cli PRT tests
2. No need to use llama-simple as a workaround anymore
3. Can proceed with full 8-prompt validation suite

## Artifacts

- `examples/speculative/phase13m_pty_runner.sh`: PTY runner script
- No models staged