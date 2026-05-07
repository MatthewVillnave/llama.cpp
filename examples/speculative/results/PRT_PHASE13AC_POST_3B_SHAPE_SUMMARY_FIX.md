# PRT Phase 13AC-POST — 3B Shape Summary Sanity Fix

**Date:** 2026-05-07  
**Verdict:** PASS_LOGGING_FIX_ONLY ✅  
**Branch:** `experimental/prt-phase13-model-generalization`

---

## Summary

The summary PRT_SHAPE bug was **logging-only**. The runtime custom op used correct per-layer `g_prt_sidecar_M[]/N[]` values throughout all prior runs. Only the summary log output was wrong — it printed hardcoded 896/4864 instead of reading the actual loaded values.

**Fix:** `tools/cli/cli.cpp` now reads `M` and `N` from `g_prt_sidecar_M[0]` and `g_prt_sidecar_N[0]` (first-loaded sidecar) instead of hardcoded literals.

---

## Bug Investigation

### Where does summary PRT_SHAPE get M/N?

**File:** `tools/cli/cli.cpp` (line ~478)

```cpp
// BEFORE (hardcoded):
fprintf(g_prt_log_file, "[PRT_SHAPE] n_layer=%d M=%d N=%d\n", n_layer,
        (loaded > 0 ? 896 : 0), (loaded > 0 ? 4864 : 0));
```

The summary log used literal values `896` and `4864` — stale 0.5B values.

### Where do per-layer sidecar M/N get stored?

**File:** `src/llama.cpp` (function `llama_set_prt_sidecar()`)

```cpp
void llama_set_prt_sidecar(int layer, const float * data, int M, int N) {
    g_prt_sidecar_M[layer] = M;  // M from actual sidecar
    g_prt_sidecar_N[layer] = N;  // N from actual sidecar
    fprintf(g_prt_log_file, "  [PRT] Sidecar set: layer=%d M=%d N=%d ...\n", ...);
}
```

Per-layer logging correctly shows M=2048, N=11008 for 3B.

### Which M/N does the runtime custom op use?

**File:** `examples/speculative/prt_graph_replace.h` (line ~133)

```cpp
int M = g_prt_sidecar_M[i];   // Per-layer M, correct for any model
int N = g_prt_sidecar_N[i];   // Per-layer N, correct for any model
```

The AVX2 kernel reads per-layer values directly — **always used correct values**.

### Were globals stale?

No. The per-layer arrays `g_prt_sidecar_M[36]` and `g_prt_sidecar_N[36]` were always populated correctly by `llama_set_prt_sidecar()`. Only the summary log printf was stale.

---

## Fix Applied

**File:** `tools/cli/cli.cpp`

```cpp
// AFTER (reads from global arrays):
extern int g_prt_sidecar_M[36];
extern int g_prt_sidecar_N[36];
int M = (loaded > 0 && g_prt_sidecar_M[0] > 0) ? g_prt_sidecar_M[0] : 896;
int N = (loaded > 0 && g_prt_sidecar_N[0] > 0) ? g_prt_sidecar_N[0] : 4864;
fprintf(g_prt_log_file, "[PRT_SHAPE] n_layer=%d M=%d N=%d\n", n_layer, M, N);
```

Rebuilt: `llama-cli` target ✅

---

## Verification

| Check | Expected | Actual | Pass |
|-------|----------|--------|------|
| Summary PRT_SHAPE | `n_layer=36 M=2048 N=11008` | `n_layer=36 M=2048 N=11008` | ✅ |
| Per-layer log first | `layer=0 M=2048 N=11008` | `layer=0 M=2048 N=11008` | ✅ |
| Per-layer log last | `layer=35 M=2048 N=11008` | `layer=35 M=2048 N=11008` | ✅ |
| Sidecars loaded | 36/36 | 36/36 | ✅ |
| No dimension mismatch | none | none | ✅ |
| Runtime generation | not attempted | not attempted | ✅ |

---

## Pass Criteria Met

- ✅ Summary PRT_SHAPE reports `n_layer=36 M=2048 N=11008`
- ✅ Per-layer logs still report `M=2048 N=11008`
- ✅ Sidecars loaded 36/36
- ✅ No dimension mismatch
- ✅ No runtime generation attempted

---

## Root Cause Explained

| Aspect | Detail |
|--------|--------|
| **Symptom** | Summary PRT_SHAPE showed `M=896 N=4864` (0.5B values) while per-layer logs showed `M=2048 N=11008` |
| **Root cause** | `cli.cpp` summary log printf used hardcoded literal `896` and `4864` instead of reading actual global arrays |
| **Impact** | Logging only — cosmetic discrepancy in log output |
| **Runtime effect** | None — PRT kernel always used correct per-layer `g_prt_sidecar_M[i]/N[i]` values |
| **Why no prior runtime impact** | The summary log was a human-readable diagnostic. The actual AVX2 kernel read per-layer M/N correctly. |

---

## What This Phase Proves

- Summary shape logging bug is fixed ✅
- Bug was logging-only, not runtime-affecting ✅
- Runtime shape handling was always correct ✅
- All 3B sidecars load with correct M=2048, N=11008 ✅
- Safe to proceed to Phase 13AD ✅

---

## Recommended Next

**Phase 13AD:** 3B runtime canary + quality validation with corrected summary logging.

---

## Safety

- No `.gguf`/`.bin`/`.safetensors` staged ✅
- No secrets in any changed file ✅
- Sidecar files in `/tmp/` ✅
- PRT log files in `/tmp/` ✅

**Tag:** `PRT_PHASE13AC_POST_3B_SHAPE_FIX`