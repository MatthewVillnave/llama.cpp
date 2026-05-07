# PRT Phase 13Y: Runtime Verification - CANARY

**Date:** 2026-05-06  
**Status:** PARTIAL_RUNTIME_VERIFY

---

## Test Environment

- Model: Qwen2.5-0.5B-Instruct-Q4_K_M.gguf
- Sidecars: /tmp/prt_sidecars/ (24 layers, M=896 N=4864)
- Binary: b8767-3d6111eab

## Test Results

| Test | Prompt | Native Exit | PRT Exit | Output Match |
|------|--------|-------------|----------|--------------|
| 1 | The capital of France is | 0 | 0 | ✓ Paris |
| 2 | Write a Python function... | 0 | 0 | ✓ Code match |
| 3 | Return JSON with keys... | 0 | 0 | ✓ JSON match |
| 4 | Explain CPU inference... | 0 | 0 | ✓ Text match |

## Timing Analysis

- Native: ~95 t/s (tokens/second)
- PRT: ~42 t/s (~2.2× slower, suggests PRT overhead)

## PRT Logs

- [PRT_SHAPE] n_layer=24 M=896 N=4864: ✓ Present
- [PRT-BUILD] __AVX2__=defined: ✓ Present
- Sidecars loaded: ✓ (24/24)

## Key Findings

1. Output quality PRESERVED - all 4 prompts produce correct, coherent output
2. Native and PRT outputs are IDENTICAL (same tokens)
3. System runs stably (no SIGKILL after process isolation)
4. Slower PRT timing suggests different code path execution

## Notes

- saw_prt_true_replacement=False in all 4 PRT runs (log detection issue)
- Output matches native exactly → suggests fallback OR correct PRT execution
- Dimension mismatch: 0.5B model (unknown FFN) vs sidecar N=4864

## Recommended Next

1. Test with 3B model to verify dimension compatibility  
2. Add debug logging to confirm replacement hooks fire
3. Run full 8-prompt quality suite once dimensions verified

## Verdict

PARTIAL_RUNTIME_VERIFY:
- Fixed AVX2 path runs cleanly on 0.5B model
- 4-prompt mini-suite passes  
- Quality preserved (output identical)
- BUT: Cannot confirm PRT custom op replacement vs native fallback
