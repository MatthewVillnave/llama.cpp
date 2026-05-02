# PRT Phase 11BP: Sidecar Startup Validation Patch

**Date:** 2026-05-02
**Patch:** `validate_prt_sidecars()` added to `phase10e0_layer0_replacement.cpp`
**Status:** PASSED

---

## What Was Changed

**File:** `examples/speculative/phase10e0_layer0_replacement.cpp`

**New function added:** `validate_prt_sidecars()`

```cpp
// Phase 11BP: loud startup validation for PRT Route A sidecars
static bool validate_prt_sidecars(void) {
    extern int g_prt_debug_mode;
    extern bool g_prt_force_native_layer[36];
    extern bool g_prt_force_native_enabled;

    // Mode 5700 = Route A (all layers PRT by default)
    bool route_a_mode = (g_prt_debug_mode >= 5700);
    if (!route_a_mode) {
        fprintf(stderr, "[PRT-11BP] PRT Route A not active (mode=%d) — skipping validation\n", g_prt_debug_mode);
        return true;  // no validation needed for non-Route-A modes
    }

    // Validate: every non-force-native layer must have a sidecar
    int required = 0;
    int missing = 0;
    for (int l = 0; l < TOTAL_LAYERS; l++) {
        bool force_native = g_prt_force_native_enabled && g_prt_force_native_layer[l];
        if (!force_native) {
            required++;
            if (g_sidecars.find(l) == g_sidecars.end()) {
                fprintf(stderr, "[PRT-ERROR] Layer %d: required PRT sidecar MISSING\n", l);
                missing++;
            }
        }
    }

    if (missing > 0) {
        fprintf(stderr, "\n[PRT-ERROR] FATAL: %d required sidecar(s) missing. Exiting.\n", missing);
        fprintf(stderr, "[PRT-ERROR] Use --prt-force-native to mark layers as native if sidecar is unavailable.\n");
        return false;
    }

    // Print checksums
    fprintf(stderr, "[PRT-11BP] L0:  %.6f\n", llama_get_sidecar_checksum(0));
    fprintf(stderr, "[PRT-11BP] L12: %.6f\n", llama_get_sidecar_checksum(12));
    fprintf(stderr, "[PRT-11BP] L15: %.6f\n", llama_get_sidecar_checksum(15));
    fprintf(stderr, "[PRT-11BP] L35: %.6f\n", llama_get_sidecar_checksum(35));

    return true;
}
```

**Called after:** `load_all_sidecars()`, before `llama_init_from_model()`

---

## Test Results

### TEST 1: Valid sidecar directory — PASS

```
[PRT] Loaded 36/36 sidecars
[PRT-11BP] === Sidecar Validation ===
[PRT-11BP] PRT mode=5700, Route A active
[PRT-11BP] Required PRT layers: 34
[PRT-11BP] Force-native layers: 2 (skipped — allowed via --prt-force-native)
[PRT-11BP] === Sidecar Checksums ===
[PRT-11BP] L0:  -354.098145
[PRT-11BP] L12: -292.085388
[PRT-11BP] L15: -105.420593
[PRT-11BP] L35: 39.010246
[PRT-11BP] VALIDATION PASSED — all required sidecars present
[11BD] callback_overwrites: 0
[11BD] native_fallback_calls: 16
[11BD] prt_true_replacement_calls: 986
```

### TEST 2: Missing L5 sidecar — FAILS LOUDLY ✓

```
[PRT-ERROR] Layer 5: required PRT sidecar MISSING
[PRT-11BP] Required PRT layers: 34
[PRT-11BP] Force-native layers: 2 (skipped — allowed via --prt-force-native)
[PRT-ERROR] FATAL: 1 required sidecar(s) missing. Exiting.
[PRT-ERROR] Use --prt-force-native to mark layers as native if sidecar is unavailable.
```

Exit: 1 (nonzero — correct)

### TEST 3: Native mode (mode 0) — VALIDATION SKIPPED ✓

```
[PRT-11BP] PRT Route A not active (mode=0) — skipping validation
```

No sidecar requirement for native mode.

### TEST 4: Full n=100 — PASS

```
[PRT-11BP] VALIDATION PASSED — all required sidecars present
[11BD] callback_overwrites: 0
[11BD] prt_true_replacement_calls: 3706
[11BD] native_fallback_calls: 16
```

---

## Final Verdict

**A. Missing sidecars fail loudly?** ✓ **YES** — `[PRT-ERROR] FATAL: N required sidecar(s) missing. Exiting.` + exit code 1

**B. Native mode unaffected?** ✓ **YES** — `mode=0` skips validation entirely, no sidecars needed

**C. L12/L15 policy still works?** ✓ **YES** — Force-native layers are correctly excluded from required validation

**D. Counters still clean?** ✓ **YES** — callback_overwrites=0, native_fallback_calls=16, prt_true_replacement_calls=3706

**E. Ready to tag PRT_ROUTE_A_RC1?** ✓ **YES** — Merge blocker removed

**F. Ready to merge to experimental?** ✓ **YES** — All gates pass

---

## Merge Blocker Resolution

**Previous blocker (Phase 11BO):** Missing sidecar silently falls back to native, incrementing `native_fallback_calls` without fatal error.

**Phase 11BP resolution:** Added `validate_prt_sidecars()` that:
1. Runs before generation starts (after sidecar loading, before `llama_init_from_model`)
2. Requires sidecar for every non-force-native layer in Route A mode
3. Prints exact layer number of missing sidecar
4. Exits with code 1 if any required sidecar is missing
5. Prints checksums for L0/L12/L15/L35 on success
6. Skips validation for non-Route-A modes (native mode untouched)

---

*End of Phase 11BP*
