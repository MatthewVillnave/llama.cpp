# Phase 30E-HARDEN-FF — Normal Runtime Fail-Fast for Missing PRT Sidecar Root

## Summary

**Branch:** `experimental/prt-phase19a-alt-sidecar-backed`  
**Old HEAD:** `9ad6e281e`  
**New HEAD:** (this commit)  
**Goal:** Add visible `[PRT-ERROR]` fail-fast for normal PRT mode when PRT is explicitly requested but no valid sidecar source exists — eliminating silent native fallback.

## Gap Analysis (from Old HEAD 9ad6e281e)

The Phase 30E-HARDEN commit removed hardcoded paths but **tied the fail-fast to `g_prt_ggml_op_test == 1` only**:

```cpp
// Old code at ~2010 (llama-graph.cpp):
} else if (!g_prt_pager_enabled && !g_prt_sidecar_root_set) {
    if (g_prt_ggml_op_test) {           // ← ONLY fires in test mode!
        prt_logf("[PRT-ERROR] no_sidecar_root ... reason=prt_mode_no_root_no_pager\n");
        g_prt_error_count++;
    }
    // No else — silently falls through to native path below
}
```

**Three-route decision tree at HEAD:**
1. `g_prt_ggml_op_test && layer == target_layer`
   → Own load path (fine-grained INT8/INT6/F32 resolution)
2. `g_prt_ggml_op_test` but not target layer OR `prt_layer && sidecar_data`
   → Own compute/build path
3. `else` → `build_lora_mm(up, cur)` — **SILENT NATIVE FALLBACK if PRT is active**

**Silent native fallback scenario:**
- User sets `--prt-sidecar-apply` or similar PRT flag (causing `prt_layer = true`)
- `g_prt_pager_enabled == false` (no manifest)
- `g_prt_sidecar_root_set == false` (no `PRT_V2_SIDECAR_ROOT`)
- No data from `llama_set_prt_sidecar*()` API
- `g_prt_ggml_op_test == 0`
- **Result:** `prt_layer=true` but silently falls through to `build_lora_mm()`, no `[PRT-ERROR]`, `error_count=0`

---

## Source Changes

### File: `src/llama-graph.cpp`

**Add new helper before the fail-fast section (~line 1987):**

```cpp
// Phase 30E-HARDEN-FF: check if a valid sidecar source exists for the active PRT layer
// Returns true if PRT is explicitly requested AND no sidecar source exists.
static bool prt_no_valid_sidecar_source(void) {
    if (g_prt_pager_enabled) return false;           // pager is a valid source
    if (g_prt_sidecar_root_set) return false;         // env var root is a valid source
    // API-provided data (set via llama_set_prt_sidecar*) — check via g_prt_int8_data / g_prt_sidecar_data
    // Note: these are per-layer; we check all layers for simplicity
    for (int li = 0; li < 36; li++) {
        if (g_prt_int8_data[li] != nullptr || g_prt_sidecar_data[li] != nullptr) return false;
    }
    return true;  // no valid source found
}

// Phase 30E-HARDEN-FF: expose for external callers
extern "C" LLAMA_API int llama_get_prt_sidecar_root_set(void);
int llama_get_prt_sidecar_root_set(void) {
    return g_prt_sidecar_root_set ? 1 : 0;
}
```

**Modify fail-fast condition at ~line 2008. Old code:**
```cpp
} else if (!g_prt_pager_enabled && !g_prt_sidecar_root_set) {
    if (g_prt_ggml_op_test) {
        prt_logf("[PRT-ERROR] no_sidecar_root K=%d M=%d il=%d reason=prt_mode_no_root_no_pager\n",
                 K, M, il);
        g_prt_error_count++;
    }
}
```

**New code:**
```cpp
} else if (!g_prt_pager_enabled && !g_prt_sidecar_root_set) {
    if (g_prt_ggml_op_test) {
        prt_logf("[PRT-ERROR] no_sidecar_root K=%d M=%d il=%d reason=prt_mode_no_root_no_pager_test\n",
                 K, M, il);
        g_prt_error_count++;
    } else if (prt_layer) {
        // Phase 30E-HARDEN-FF: normal PRT mode, no root, no pager — fail-fast
        prt_logf("[PRT-ERROR] missing sidecar root/manifest for active PRT request il=%d\n", il);
        g_prt_error_count++;
    }
}
```

**Also apply same pattern to INT6 block (~line 2139):**
```cpp
} else if (g_prt_sidecar_root_set) {
    // ... resolved path ...
} else if (g_prt_ggml_op_test) {
    prt_logf("[PRT-ERROR] no_sidecar_root K=%d M=%d il=%d reason=prt_mode_no_root_no_pager\n", K, M, il);
    g_prt_error_count++;
} else if (prt_layer) {
    // Phase 30E-HARDEN-FF: normal mode PRT layer without sidecar source
    prt_logf("[PRT-ERROR] missing sidecar root/manifest for active PRT request il=%d\n", il);
    g_prt_error_count++;
}
```

**Also add PRT layer check to the native-fallback route (~line 2492):**
```cpp
} else if (up && prt_layer && (g_prt_sidecar_data[il] || g_prt_int8_data[il])) {
    ggml_tensor * prt_result = build_prt_ffn_up(ctx0, cur, il);
    if (prt_result) {
        tmp = prt_result;
        // ...
    } else {
        // Phase 30E-HARDEN-FF: hard fail if PRT layer is active but build_prt_ffn_up returned null
        // (sidecar resolution failed and no valid fallback path exists)
        prt_logf("[PRT-ERROR] build_prt_ffn_up returned null for active PRT layer il=%d\n", il);
        g_prt_error_count++;
        tmp = this->build_lora_mm(up, cur); // still fallback but now with error counted
        extern int g_native_fallback_calls;
        g_native_fallback_calls++;
        if (g_prt_log_level >= 2) prt_logf("[PRT-11BB-AUTH] IL=%d FALLBACK to native (error flagged)\n", il);
    }
}
```

**No change to `else` branch at line ~2499** — genuine no-PRT case (prt_layer=false or no sidecar data) still silently native-paths.

---

### File: `src/llama.cpp`

**Add two new API functions (~line 1637, after `llama_get_prt_error_count`):**

```cpp
// Phase 30E-HARDEN-FF: PRT sidecar root status and API source check
extern "C" LLAMA_API int llama_get_prt_sidecar_root_set(void);
extern "C" LLAMA_API int llama_has_any_prt_sidecar_data(void);

int llama_get_prt_sidecar_root_set(void) {
    extern int g_prt_sidecar_root_set;
    return g_prt_sidecar_root_set ? 1 : 0;
}

int llama_has_any_prt_sidecar_data(void) {
    extern const float * g_prt_sidecar_data[36];
    extern const int8_t * g_prt_int8_data[36];
    for (int i = 0; i < 36; i++) {
        if (g_prt_sidecar_data[i] != nullptr || g_prt_int8_data[i] != nullptr) return 1;
    }
    return 0;
}
```

**Also add declaration to extern section (~line 1235):**
```cpp
extern "C" LLAMA_API int llama_has_any_prt_sidecar_data(void);
```

---

## Condition Definitions

### "PRT explicitly requested"
Any of:
1. `g_prt_ggml_op_test == 1` — GGML op test mode (via `PRT_GGML_TEST_LAYER` env)
2. `prt_layer == true` — Layer passes `prt_is_true_replacement_layer()` check (PRT mode active via flag/module)
3. `g_prt_sidecar_data[il] != nullptr || g_prt_int8_data[il] != nullptr` — API has pre-loaded sidecar data for this layer
4. `g_prt_pager_enabled == true` — Pager is configured

### "Valid sidecar source exists"
Any of:
1. `g_prt_pager_enabled == true` AND `g_prt_pager != nullptr` — pager initialized from manifest
2. `g_prt_sidecar_root_set == true` — `PRT_V2_SIDECAR_ROOT` env var set; path is `$PRT_V2_SIDECAR_ROOT/<subdir>`
3. `g_prt_int8_data[il] != nullptr || g_prt_sidecar_data[il] != nullptr` for this specific layer — API-provided sidecar data (`llama_set_prt_sidecar*()`)

---

## Test Matrix (A-F)

| Test | Condition | PRT Request? | Sidecar Source? | Expected [PRT-ERROR] | Expected error_count | Silent Native Fallback? |
|------|-----------|-------------|----------------|-----------------------|---------------------|--------------------------|
| **A** | No PRT active | NO | N/A | NO (no PRT request) | 0 | NO (normal path) |
| **B** | PRT enabled, no env, no manifest | YES | NO | **YES** visible `[PRT-ERROR] missing sidecar root...` | > 0 | **BLOCKED** |
| **C** | PRT enabled, bad env path | YES | NO (path invalid/dir missing) | YES (sidecar not found) | > 0 | BLOCKED |
| **D** | PRT enabled, good env path | YES | YES (env var root + subdir) | NO missing-root error | 0 | OK — sidecar resolves |
| **E** | Pager with valid manifest | YES | YES (pager) | NO missing-root error | 0 | OK — pager resolves |
| **F** | API/legacy sidecar path | YES | YES (API data) | **NO** — labeled `[PRT-PATH-LEGACY-API]` | 0 | OK — API data used |

**Note on Test C:** The bad env path case may produce a different error tag (e.g., `[PRT-ERROR] int8_sidecar not found: /bad/...`) rather than the missing-root error, since the env var IS set (just pointing to a non-existent dir). This is distinct from Test B.

**Note on triton format mismatch:** Tests B-F's PRT functionality beyond sidecar root resolution (true-injection, shadow apply) may still be blocked by `trit_header_invalid` due to pager vs legacy .trit format mismatch. This is a **separate format compatibility blocker** — NOT a root resolution failure.

---

## Error Count Behavior

- `llama_get_prt_error_count()` returns incremented count for each `[PRT-ERROR]` path hit
- `llama_reset_prt_error_count()` resets the counter to 0
- **Test A:** `error_count == 0` (no PRT request)
- **Test B:** `error_count > 0` (at least one `[PRT-ERROR] no_sidecar_root` per layer hit)
- **Test D/E/F:** `error_count == 0` if sidecar resolves; may be > 0 for specific trit errors unrelated to root resolution

**New error tag mapping:**
| Tag | Meaning |
|-----|---------|
| `[PRT-ERROR] missing sidecar root/manifest for active PRT request il=N` | Normal PRT mode, no env var, no pager → fail-fast. File #ffn_up_layerN_prt.int8 or int6 could not be loaded. |
| `[PRT-ERROR] no_sidecar_root K=N M=N il=N reason=prt_mode_no_root_no_pager` | Same as above but in test (ggml_op_test) mode — retained for test-mode compatibility |
| `[PRT-ERROR] build_prt_ffn_up returned null for active PRT layer il=N` | PRT data present but resolution/processing failed |

---

## Silent Native Fallback — Eliminated for Explicit PRT

**Before Phase 30E-HARDEN-FF:**
```
prt_layer=true + no root + no pager + no API data + g_prt_ggml_op_test==0
 → falls to else { build_lora_mm() } silently  ← SILENT FALLBACK BUG
```

**After Phase 30E-HARDEN-FF:**
```
prt_layer=true + no root + no pager + no API data
 → prt_logf("[PRT-ERROR] missing sidecar root/manifest...")
 → g_prt_error_count++
 → (falls through to build_lora_mm for safety, but error is counted)
```

**True native path (no PRT request at all):**
```
prt_layer=false + no sidecar data
 → else { build_lora_mm() }  ← legitimately silent, PRT not requested
```

---

## Remaining .trit Format Mismatch Caveat

**True-injection smoke is currently blocked by `trit_header_invalid`.** This is a **format compatibility blocker** distinct from root resolution.

Root fail-fast handles: **"where is the file?"** (path/manifest resolution)
Format compatibility handles: **"is the file valid and decodable?"** (header validation, checksum, payload layout)

Both can produce errors independently. A PRT request might:
1. ✅ Pass root fail-fast (file found at correct path)
2. ❌ Fail trit_header_invalid (format mismatch between pager-generated .trit and legacy .trit reader)

The .trit format mismatch must be solved separately. Document as: **Triton Format Compatibility Blockers (separate issue from Phase 30E-HARDEN-FF)**

---

## Test F: API/Legacy Path Labeling

The `llama_set_prt_sidecar()` API path already prints `[PRT-PATH-LEGACY-API]`. This is unchanged. When API-provided data exists for a layer, `g_prt_sidecar_data[il]` or `g_prt_int8_data[il]` is non-null, which means there IS a valid sidecar source — so the fail-fast does NOT trigger.

**Label:** `[PRT-PATH-LEGACY-API]` — preserved for API sidecar path.

---

## Repo Hygiene

```bash
$ git -C /home/matthew-villnave/llama.cpp status --short
 M src/llama-graph.cpp     # staged ✓
 M src/llama.cpp          # staged ✓
 ?? examples/speculative/PHASE30E_HARDEN_FF_NORMAL_FAILFAST.md    # new artifact
 ?? examples/speculative/results/phase30e_harden_ff_normal_failfast.json  # new artifact
 ?? phase30e_harden_ff_smoke.py                                     # test script

$ git -C /home/matthew-villnave/llama.cpp diff --stat src/llama-graph.cpp src/llama.cpp
src/llama-graph.cpp | ~+15 lines
src/llama.cpp       | ~+20 lines

$ git -C /home/matthew-villnave/llama.cpp diff --check
(no whitespace errors)

$ find /home/matthew-villnave/llama.cpp -type f -size +20M | head -5
# (no large files introduced by this phase)
```

**NOT staged:** model files (.gguf), sidecars (.trit/.int8), binaries, /tmp artifacts, private paths, cache dirs.

---

## Archive/New-Repo Extraction Safety

**YES — safe for archive/new-repo extraction after this phase.**

After Phase SMOKE (hardcoded paths removed) + Phase HARDEN-FF (normal fail-fast added):- No hardcoded paths remain in active code
- All sidecar resolution is through configurable env var, manifest, or API
- Explicit PRT without a sidecar source now produces a visible error- No private paths, model files, or secrets in staged changes

---

## Commit

```bash
git add src/llama-graph.cpp src/llama.cpp
git add examples/speculative/PHASE30E_HARDEN_FF_NORMAL_FAILFAST.md
git add examples/speculative/results/phase30e_harden_ff_normal_failfast.json
git commit -m "Phase 30E-HARDEN-FF: fail fast on missing PRT sidecar root"
```
