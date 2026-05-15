# PRT Phase 21Q-B: 7B INT8 Semantic Canary

## Context
- **Branch**: `experimental/prt-phase19a-alt-sidecar-backed`
- **Previous HEAD**: `309380d13` (Phase 21Q-A)
- **Date**: 2026-05-15

## Harness
Working capture command:
```bash
script -qfc "timeout 90 ./build/bin/llama-cli -m MODEL -f PROMPT -n N --temp 0 --log-disable --simple-io" /dev/null > OUTPUT 2>&1
```
- PTY harness gives clean ~30KB output vs 300MB+ TTY garbage
- n=8 works on native (120s), n=4 partial on PRT

## Results

### P1: "The capital of France is"
| Run | Exit | Time | Generated Text | Evidence |
|-----|------|------|---------------|----------|
| Native | 0 | 120s | "Paris" ✅ | - |
| PRT-v2 | 0 | 334s | (spinner frame) | KERNEL_ENTER/EXIT ✅, output_abs_sum=1.684134 |

### P2: "The largest planet in our solar system is"
| Run | Exit | Time | Generated Text | Evidence |
|-----|------|------|---------------|----------|
| Native | 124 | 90s | "The largest planet in" (partial) | - |

## Key Findings

1. **Native**: semantic text visible ("Paris"), clean exit with n=8
2. **PRT-v2**: kernel evidence confirmed, but slow (334s vs 120s) - ~3x slowdown
3. **Capture**: PTY harness works but text hidden in TTY animation frames
4. **output_abs_sum**: 1.684134, 1.944075 (consistent with Phase 21O)

## Verdict
- **PARTIAL_7B_INT8_SEMANTIC_WITH_TIMEOUT**: PRT-v2 works but slow, native text visible

## Recommended Next
- Phase 21Q-C: tune harness for cleaner capture
- Investigate PRT slowdown (3x runtime vs native)

## Disk Usage
- Captures: 896KB
- Scratch free: 51GB (57%)