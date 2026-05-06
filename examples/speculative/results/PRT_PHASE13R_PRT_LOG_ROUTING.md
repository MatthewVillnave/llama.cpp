# PRT Phase 13R: PRT Debug Log Routing

**Verdict: PASS_LOG_ROUTING** ✅

## Summary

Added `--prt-log-file FNAME` CLI flag that routes ALL PRT debug/auth/logging to a separate file while keeping generated stdout clean. This enables clean quality comparisons between native and PRT-active inference.

## Implementation

### Files Changed (5)

| File | Change |
|------|--------|
| `common/common.h` | Added `std::string prt_log_file` to `llama_params` struct |
| `common/arg.cpp` | Added `--prt-log-file FNAME` CLI argument |
| `src/llama.cpp` | Added `llama_set_prt_log_file()` function + routed 5 fprintf calls |
| `src/llama-graph.cpp` | Added `g_prt_log_file` global + `prt_logf()` helper + routed 4 fprintf calls |
| `examples/speculative/prt_graph_replace.h` | Routed 4 fprintf calls in custom op |
| `tools/cli/cli.cpp` | Declared externs, set log file at startup, routed PRT_SHAPE |

### Key Design

```
g_prt_log_file = nullptr  (global in llama-graph.cpp)
     |
     +-- prt_logf(...) helper in llama-graph.cpp
     |       routes to g_prt_log_file if set, else stderr
     |
     +-- llama_set_prt_log_file(path) 
     |       sets g_prt_log_file = fopen(path, "w")
     |       called by CLI BEFORE llama_set_prt_debug_mode()
     |
     +-- All PRT fprintf(stderr, "[PRT...] calls now use prt_logf()
```

### Rerouted Calls

**llama-graph.cpp (4 calls):**
- `[PRT-11BB-AUTH] IL=%d cur=[...]` → `prt_logf()`
- `[PRT-11BG] IL=%d FORCE-NATIVE` → `prt_logf()`
- `[PRT-11BB-AUTH] IL=%d PRT result ne=[...]` → `prt_logf()`
- `[PRT-11BB-AUTH] IL=%d FALLBACK to native` → `prt_logf()`

**llama.cpp (5 calls):**
- `Sidecar set: layer=%d M=%d N=%d...` → conditional
- `Debug mode set to %d` → conditional
- `[PRT-11BB] kernel mode set to %d` → conditional
- `[PRT-11BG] force-native enabled for %d layers` → conditional
- `[PRT-11BG] force-native cleared` → conditional

**prt_graph_replace.h (4 calls):**
- `[PRT-11BB] ERROR: custom op called without sidecar` → conditional
- `[PRT-11BB] custom op: dst=%s...` → conditional
- `[PRT-11BB] IL=%d hidden=%d...` → conditional

**tools/cli/cli.cpp (1 call):**
- `[PRT_SHAPE] n_layer=%d M=%d N=%d` → conditional

## Test Results

### Test 1 — Native (no PRT, no --prt-log-file)
```
exit: 0, timed_out: False, elapsed: 0.791s
saw_generation_timing: True ✓
stdout_contains_prt_debug: False ✓
```

### Test 2 — PRT active with --prt-log-file
```
exit: 0, timed_out: False, elapsed: 2.324s
stdout_contains_prt_debug: False ✓  (CLEAN!)
Paris in PTY output: True ✓
Generation timing in PTY: True ✓
PRT_LOG_size: 35684 bytes
PRT_LOG_has_shape: True ✓
PRT_LOG_has_sidecar: True ✓
PRT_LOG_has_prt11bb: True ✓
PRT_LOG_has_custom_op: True ✓
```

### Test 3 — PRT backward compatibility (no --prt-log-file)
```
exit: 0, timed_out: False, elapsed: 3.498s
stdout_contains_prt_debug: True (expected — old behavior preserved)
prt_log_file_exists: False (no file created when flag absent)
```

### 2-Prompt Clean Quality Check

**Prompt 1: "The capital of France is"**
| Run | stdout_clean | log_has_SHAPE | log_has_AUTH | log_has_CO |
|-----|-------------|--------------|--------------|------------|
| Native | ✓ | N/A | N/A | N/A |
| PRT+logfile | ✓ | ✓ | ✓ | ✓ |

**Prompt 2: "Return JSON with keys name and status."**
| Run | stdout_clean | JSON_valid | log_has_SHAPE | log_has_AUTH |
|-----|-------------|-----------|--------------|-------------|
| Native | ✓ | YES | N/A | N/A |
| PRT+logfile | ✓ | YES | ✓ | ✓ |

Both produce: `{"name": "Qwen", "status": "Online"}`

## Safety
- No models staged: ✓
- No secrets leaked: ✓
- Tags untouched: ✓
- Backward compat preserved: ✓ (Test 3 passed)