# PHASE 28BK — Runtime Hook Dry-Run / No-Generation Integration Probe

## Status: ✅ ALL PASS

## Objective

Bridge from the normalized shadow harness path into the actual llama.cpp-adjacent
runtime hook path (`prt_get_residual_view()`), without running generation.
Prove the hook would work end-to-end.

---

## STEP 1: Runtime Hook Entry Point Identification

**File:** `examples/speculative/prt_sidecar_runtime_link.h`
**Function:** `prt_get_residual_view(int layer_idx, const std::string& tensor_family)`
**Compile Flag:** `PRT_SIDECAR_PAGER_EXPERIMENTAL`
**Enabled by:** Defined in `prt_sidecar_runtime_link.h`, guarded by `#ifdef PRT_SIDECAR_PAGER_EXPERIMENTAL`
**Wired in:** `prt_shadow.h` calls `prt_get_residual_view()` from `run_shadow_test()` when `PRT_SIDECAR_PAGER_EXPERIMENTAL` is set

**Routing logic:**
```
prt_get_residual_view(layer_idx, tensor_family):
  → if g_prt_pager != nullptr && g_prt_pager_enabled:
       → g_prt_pager->get_residual(layer_idx, tensor_family)  [pager path]
  → else:
       → check g_sidecars map  [legacy path]
  → else: return null view with reason="not_found"
```

**Pager path connects to:** `prt_sidecar_pager.get_residual()` → layer_states_ → residuals map → raw .trit bytes

**Legacy path connects to:** `g_sidecars[layer].data` (float*, not uint8_t*)

**Current state:** WIRING EXISTS, not stubbed — `prt_get_residual_view()` is live code
that routes to either pager or legacy based on `g_prt_pager_enabled`.

---

## STEP 2: Test Results

### T1 — Runtime Hook Positive
- **Result:** PASS
- Used `phase28bi_pkg_attn_out_l5_full` (layer 0, attn_out, [896,896], 532 blocks)
- `prt_init_pager()` → `activate_layer(0)` → `prt_get_residual_view(0, "attn_out")`
- `raw.is_null = false`, `raw.size = 303,216 bytes`
- `.trit header`: rows=896, cols=896, block_rows=32, block_cols=48, n_scales=532
- **Decode-first verified:** raw .trit (303,216 bytes) is 10.6x smaller than decoded float (3,211,264 bytes)
- `prt_trit_decoder.decode_file()` → `rows=896, cols=896, float_elems=802,816` ✅
- Pager stats: reads=1, cache_misses=1, fallbacks=0
- **pager_hits=1, legacy_hits=0** ✅

### T2 — DISABLED_MODE
- **Result:** PASS
- `prt_shutdown_pager()` → `g_prt_pager_enabled = false`
- `prt_get_residual_view(0, "attn_out")` → `is_null=true, reason="not_found"`
- No legacy sidecar loaded, no false pass ✅

### T3 — MISSING_MANIFEST_OR_SIDECAR
- **Result:** PASS
- `prt_init_pager()` with `/tmp/nonexistent_manifest_28bk.json` → returns `false`
- Deterministic failure, no crash ✅

### T4 — BAD_TENSOR_KEY
- **Result:** PASS
- Pager initialized with valid manifest, layer activated
- `prt_get_residual_view(0, "nonexistent_family")` → `is_null=true, reason="tensor_not_found"`
- Deterministic null for non-existent tensor family ✅

### T5 — RAW_BYTES_NOT_USED_AS_FLOAT
- **Result:** PASS
- Raw .trit size: 303,216 bytes
- Decoded float size (rows×cols×4): 3,211,264 bytes
- **Compression ratio: 10.6:1** — raw bytes are NOT float32
- Wrong elem count if cast as float*: `303,216 / 4 = 75,804` vs correct `802,816`
- Decode produces correct shape (896×896) ✅
- **Decode-first semantics verified** ✅

### T6 — BUDGET_REJECT
- **Result:** PASS
- `max_resident_bytes = 4` (absurdly tiny)
- `activate_layer(0)` → REJECTED
- `budget_rejects=1`, `resident_bytes=0`
- Deterministic rejection with no crash ✅

---

## STEP 3: Key Evidence

### Runtime Hook Entry Point: IDENTIFIED ✅
- **File:** `examples/speculative/prt_sidecar_runtime_link.h`
- **Function:** `prt_get_residual_view(int, const std::string&)`
- **Flag:** `PRT_SIDECAR_PAGER_EXPERIMENTAL`
- **Wired:** YES — called from `prt_shadow.h` `run_shadow_test()` when flag is set

### Actual Runtime Hook Path Used: YES ✅
- T1 exercised the full pager path: init → activate → `prt_get_residual_view()` → decode
- `prt_get_residual_view()` is the **same function** that would be called from the llama.cpp
  forward graph in actual generation

### Decode-First Semantics: VERIFIED ✅
- Raw view: 303,216 packed .trit bytes
- Decoded float: 3,211,264 bytes (10.6x larger)
- Header confirms: rows=896, cols=896, block_rows=32, block_cols=48, n_scales=532
- Never casts raw bytes to float* ✅

### Counters: pager_hits=1, legacy_hits=0 (positive case) ✅

### Controls: ALL DETERMINISTIC ✅
- T2: null when disabled
- T3: false on missing manifest
- T4: null for bad tensor key
- T6: reject on tiny budget

---

## Conclusion

**Phase 28BK: ALL PASS**

- Runtime hook entry point `prt_get_residual_view()` is **fully identified** and **already wired** in `prt_shadow.h` under `PRT_SIDECAR_PAGER_EXPERIMENTAL`
- The actual runtime hook path was exercised in T1 (not a simulation) — init → activate → hook call → decode
- Decode-first semantics: **verified** — raw .trit bytes (303KB) require decoding to produce 3.2MB float buffer; never cast raw to float*
- Pager counter evidence: pager_hits=1, legacy_hits=0 on positive case
- All 6 controls deterministic, no crashes
- Generation: **false** (dry-run only)

**Commit:** `examples/speculative/phase28bk_runtime_hook_dry_run_probe.cpp` + results + report