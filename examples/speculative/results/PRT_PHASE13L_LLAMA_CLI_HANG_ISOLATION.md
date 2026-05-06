# PRT Phase 13L: llama-cli Native Hang Isolation

**Date:** 2026-05-06
**Commit:** 5bd61ea27 (Phase 13K)

## Executive Summary

llama-cli hangs when stdout is redirected to /dev/null or a file. This is a TTY detection issue in the console output path, NOT a PRT bug or model corruption.

## Test Results

### Control: llama-simple
```
simple_exit=0
llama_perf_context_print: load time = 215.46 ms
llama_perf_context_print: prompt eval time = 44.80 ms / 6 tokens
llama_perf_context_print: eval time = 367.16 ms / 31 runs
llama_perf_context_print: total time = 589.00 ms / 37 tokens
```
✅ Works perfectly (0.62s real time)

### llama-cli Variants (stdout → /dev/null)

| Variant | Flags | Exit | Result |
|---------|-------|------|--------|
| A | baseline (c=256, --no-display-prompt) | 124 | ❌ TIMEOUT |
| B | -no-display-prompt removed | 124 | ❌ TIMEOUT |
| C | --simple-io | 124 | ❌ TIMEOUT |
| D | --no-conversation | 124 | ❌ TIMEOUT |
| E | -b 32 -ub 32 | 124 | ❌ TIMEOUT |
| F | -t 1 -b 32 | 124 | ❌ TIMEOUT |
| G | no temp override | 124 | ❌ TIMEOUT |

All 7 variants with stdout→/dev/null TIMEOUT.

### llama-cli Variants (stdout to terminal, no redirect)

| Variant | Context | Exit | Result |
|--------|---------|------|--------|
| H | c=32 | 0 | ✅ PASS |
| I | c=64 | 0 | ✅ PASS |
| J | c=128 | 0 | ✅ PASS |
| K | c=256 | 0 | ✅ PASS |
| L | c=512 | 0 | ✅ PASS |

All variants work when stdout goes to terminal.

### Additional Tests

- **stdout → file:** Prints thousands of `> ` characters (infinite spinner loop), then times out
- **stdout → pipe to cat:** Also problematic
- **PTY:** Not tested thoroughly due to Python constraints

## Root Cause Analysis

The hang is caused by llama-cli's **console TTY detection logic**:

1. In `common/console.cpp`, `console::init()` checks if stdout is a terminal
2. When NOT a TTY (redirection case), the console path differs:
   - Opens `/dev/tty` for cursor movement (POSIX)
   - Starts a background spinner thread
   - Uses `fprintf(out, ...)` + `fflush(out)` in the spinner loop

3. When stdout is redirected to /dev/null or a file:
   - The spinner thread writes to a non-TTY file descriptor
   - The console state machine gets confused
   - Infinite loop of prompt characters (`> `) or blocking on fflush

**This is NOT a PRT issue** - it's how llama-cli handles non-TTY output environments.

## Verdict

**Classification:** `LLAMA_CLI_TTY_DETECTION_HANG`

- Root cause: Console/spinner output handling depends on TTY detection
- Trigger: stdout redirection to /dev/null or file in subprocess
- Impact: PRT test harness cannot use stdout→/dev/null with llama-cli
- Workaround: Use PTY or avoid stdout redirection

## Recommended Next Steps

1. **For PRT testing:** Find alternative method that doesn't redirect stdout
   - Could use a PTY wrapper
   - Or accept that llama-cli needs TTY for proper operation

2. **Alternative:** Use `llama-simple` as the clean-frontend control (different code path)

3. **Long-term:** Fix llama-cli console to handle non-TTY gracefully (outside PRT scope)

## Safety

- Models staged: NO
- Sidecars staged: NO
- Binaries staged: NO
- Secrets detected: NO
- PRT tags untouched: YES

## Artifacts

- `/tmp/phase13l_*_stderr.txt`: Test stderr captures
- No models or binaries staged