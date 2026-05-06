# PRT Phase 13N-C: Fix PTY Runner Stdin/Exit Behavior

**Verdict: PASS_PTY_RUNNER_READY**

## Summary

Python argv PTY runner now works correctly with `--single-turn` flag. All tests exit cleanly without timeout, capture full output including PRT logs, and preserve argv with spaces intact.

## Fix Applied

**stdin strategy:** `/dev/null` stdin + `--single-turn` llama-cli flag

The root cause was two-fold:
1. `/dev/null` stdin alone doesn't make llama-cli exit (it still enters interactive waiting mode)
2. `--single-turn` tells llama-cli to exit after one generation pass

Combined fix: devnull stdin + --single-turn = clean exit

## Test Results

| Test | Command | Exit | Elapsed | saw_generation_timing | saw_prt_shape |
|------|---------|------|---------|----------------------|---------------|
| 1 | native short + single-turn | clean | 0.515s | ✓ | ✗ |
| 2 | native full + single-turn | clean | 0.515s | ✓ | ✗ |
| 3 | active PRT full + single-turn | clean | 2.366s | ✓ | ✓ |

## Detailed Metrics

**Test 1 (native short):**
- exit_code: 0, timed_out: false
- elapsed: 0.515s
- raw_bytes: 1132
- saw_prt_shape: false (expected - native)
- saw_generation_timing: true
- saw_invalid_argument: false ✓
- saw_flag_echo: false ✓
- saw_path_fragment: false ✓
- saw_error: false ✓

**Test 2 (native full prompt with spaces):**
- exit_code: 0, timed_out: false
- elapsed: 0.515s
- raw_bytes: 1150
- Prompt "The capital of France is" preserved correctly (no splitting) ✓
- saw_invalid_argument: false ✓ (no "invalid argument: capital")

**Test 3 (active PRT full prompt):**
- exit_code: 0, timed_out: false
- elapsed: 2.366s
- raw_bytes: 37440
- saw_prt_shape: true ✓ (n_layer=24 M=896 N=4864)
- saw_sidecars_loaded: true ✓ (sidecar logs visible)
- saw_prt_true_replacement: true ✓ (PRT-11BB replacement logs)
- saw_generation_timing: true ✓ (24.1 t/s prompt, 19.4 t/s generation)
- saw_flag_echo: false ✓
- saw_path_fragment: true (sidecar paths in PRT logs - expected)
- saw_invalid_argument: false ✓
- saw_error: false ✓

## Pattern Tracking

The runner now tracks these booleans across the full stream (not just tail):
- `saw_prt_shape`: Detects PRT_SHAPE or n_layer.*M=
- `saw_sidecars_loaded`: Detects sidecar/loaded logs
- `saw_prt_true_replacement`: Detects PRT-11BB/ffn_up replacement logs
- `saw_native_fallback`: Detects native_fallback markers
- `saw_callback_overwrites`: Detects callback_overwrites
- `saw_generation_timing`: Detects t/s timing lines
- `saw_flag_echo`: Detects --prt- flags in output
- `saw_path_fragment`: Detects /tmp/prt_ or /llama.cpp/build paths
- `saw_invalid_argument`: Detects invalid argument errors
- `saw_error`: Detects error/Error/Traceback

## Safety Check

- No models staged: ✓
- No secrets leaked: ✓
- Tags untouched: ✓

## Recommended Next

Runner is ready. Run Phase 13N full 8-prompt validation suite.

**Status: PTY runner unblocked** ✓