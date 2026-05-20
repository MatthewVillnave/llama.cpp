# PRT Phase 24F-R3: Log Pollution Fix

## Date
2026-05-19

## Context
R2 found that massive output files were being created. Investigating whether `PRT_V2_QUIET=1` suppresses all PRT-NATIVE logs.

## Findings

### Existing Guard Already Works
The code at `src/llama-graph.cpp:1690` already has the guard:
```cpp
if (!prt_v2_quiet_native_prints() && g_prt_log_level >= 2) prt_logf("[PRT-NATIVE] IL=%d ...", ...);
```

And the helper function:
```cpp
static inline bool prt_v2_quiet_native_prints(void) {
    const char * v = std::getenv("PRT_V2_QUIET");
    return v != nullptr && std::strcmp(v, "1") == 0;
}
```

### Test Results
With `PRT_V2_QUIET=1`:
```
grep -c "[PRT-NATIVE]" stdout stderr
=> 0
```
Logs successfully suppressed ✅

### Remaining Issue: Runner Binary
- `llama-cli` defaults to conversation mode (infinite output)
- `llama-completion --no-conversation` gives bounded output
- This is NOT a PRT bug, just harness choice

## Verdict: PASS_LOG_GUARD_WORKING ✅

- PRT_V2_QUIET=1 already works
- No additional code changes needed
- Choice of runner is user discretion

## Recommended Next
Resume Phase 24F with rule:
- Use `llama-completion --no-conversation` for bounded tests
- OR understand `llama-cli` conversation mode behavior

---
Commit: 9dbcd59aa