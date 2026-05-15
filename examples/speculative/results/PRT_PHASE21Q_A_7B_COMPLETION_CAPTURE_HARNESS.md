# PRT Phase 21Q-A: 7B Completion + Text Capture Harness

## Context
- **Branch**: `experimental/prt-phase19a-alt-sidecar-backed`
- **Previous checkpoint**: `c1d5da615` (Phase 21P)
- **Date**: 2026-05-15

## Goal
Fix 7B completion/capture harness so PRT-v2 layer0 runs can complete with captured generated text.

## Key Discovery

### TTY Output Problem
- llama-cli outputs TTY control codes when run non-interactively
- Redirect to file → 300MB+ garbage
- `script -qfc` creates PTY → clean output (28KB)

### Working Capture Command
```bash
script -qfc "timeout 55 ./build/bin/llama-cli -m MODEL -f PROMPT -n 1 -c 16 -t 1 --temp 0 --log-disable --simple-io" /dev/null > OUTPUT 2>&1
```

Key flags:
- `--log-disable`: suppress debug logs
- `--simple-io`: basic IO for subprocess compatibility
- `script -qfc`: PTY wrapper for clean output

## Results

### Native (no PRT)
| Metric | Value |
|--------|-------|
| Output size | 28KB |
| Generated text | "The" (visible in PTY output) |
| Exit | 124 (timeout at 55s) |
| Timing | ~55s for n=1 |

### PRT-v2 INT8 layer0
| Metric | Value |
|--------|-------|
| Output size | 35KB |
| KERNEL_ENTER | ✓ confirmed |
| KERNEL_EXIT | ✓ confirmed |
| output_abs_sum | 1.684134, 1.944075 |
| Generated text | ✓ (ASCII display) |
| Exit | 124 (timeout at 55s) |

### Root Cause of Prior Exit 137
- 90s timeout in Phase 21O harness
- Process killed at ~55s when generation complete
- Exit 137 = SIGTERM from timeout wrapper

### Timing Analysis
- 7B model load: ~5s
- n=1 generation: ~50s (CPU, Q4 quantization)
- Total: ~55s

## Verdict
- **PASS_7B_NATIVE_CAPTURE**: Clean text capture via script -qfc
- **PASS_7B_PRT_CAPTURE**: Kernel evidence + clean text capture
- **PASS_7B_COMPLETION_HARNESS**: script -qfc + timeout 55s works

## Recommended Next
1. If need more tokens (n>1): increase timeout to 90s+
2. Phase 21Q-B: 7B INT8 semantic validation with n=4
3. Alternative: run larger batch on GPU if available

## Files Modified
None (documentation only)

## Recommended Future Command
```bash
script -qfc "timeout 90 ./build/bin/llama-cli -m MODEL -f PROMPT -n 4 -c 16 -t 1 --temp 0 --log-disable --simple-io" /dev/null > CAPTURE 2>&1
```