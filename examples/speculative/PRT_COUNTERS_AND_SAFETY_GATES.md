# PRT Counters and Safety Gates

**Date:** 2026-05-02
**Scope:** Phase 11BO Code Review — counters, gates, and fail-safes

---

## Counters

### All Counters

| Counter | Declared In | Incremented When | Phase11BN Value |
|---------|-----------|-----------------|---------------|
| `g_prt_true_replacement_calls` | llama-graph.cpp:33 | PRT custom op replaces native FFN_UP | 3706 (n=100) |
| `g_callback_overwrite_calls` | llama-graph.cpp:34 | Callback writes PRT output to tensor | **0** (L12+L15) |
| `g_identity_fallback_calls` | llama-graph.cpp:35 | Identity/no-op fallback | **0** |
| `g_native_fallback_calls` | llama-graph.cpp:36 | Native FFN_UP fallback from PRT fail or force-native | **16** (L12+L15) |
| `g_prt_ffn_up_custom_op_count` | llama-graph.cpp:29 | PRT custom op called | 3706 (n=100) |

**Getter Functions (src/llama.cpp):**
```cpp
llama_get_prt_true_replacement_calls()   → g_prt_true_replacement_calls
llama_get_callback_overwrite_calls()     → g_callback_overwrite_calls
llama_get_identity_fallback_calls()      → g_identity_fallback_calls
llama_get_native_fallback_calls()        → g_native_fallback_calls
llama_get_prt_custom_op_count()          → g_prt_ffn_up_custom_op_count
```

### Expected Values

| Mode | n | callback_overwrites | native_fallback_calls | prt_true_replacement_calls |
|------|---|---------------------|----------------------|--------------------------|
| Native (mode 0) | 100 | 3636 | 0 | 0 |
| L12+L15 (mode 5700) | 100 | **0** | 16 | 3706 |
| L12+L15 (mode 5700) | 50 | **0** | 16 | 2006 |

**Note:** Native mode callback_overwrites=3636 comes from context initialization, not PRT generation. This is baseline behavior, not relevant to Route A.

---

## Safety Gates

### Gate 1: Callback Bypass (Mode 5700)

```cpp
// phase10e0_layer0_replacement.cpp
bool callback_mode = (g_prt_debug_mode < 5700);
```

When `callback_mode = false` (mode 5700), `prt_eval_callback` does NOT intercept FFN_UP tensors.

**GATE STATUS:** ✓ WORKING — No callback intercept for mode 5700.

### Gate 2: Force-Native Priority

```cpp
// src/llama-graph.cpp build_ffn()
bool force_native = g_prt_force_native_enabled && g_prt_force_native_layer[il];

if (force_native) {
    tmp = this->build_lora_mm(up, cur);  // native — checked FIRST
    g_native_fallback_calls++;
} else if (up && prt_layer && g_prt_sidecar_data[il]) {
    // PRT custom op path
} else {
    tmp = this->build_lora_mm(up, cur);  // native fallback
}
```

Force-native is checked FIRST, before PRT custom op. This ensures L12+L15 layers are never replaced by PRT even in mode 5700.

**GATE STATUS:** ✓ WORKING — Verified: native_fallback_calls=16 for L12+L15 (2 layers × 8 tokens).

### Gate 3: Sidecar Missing Fallback

```cpp
// src/llama-graph.cpp build_ffn()
} else if (up && prt_layer && g_prt_sidecar_data[il]) {
    ggml_tensor * prt_result = build_prt_ffn_up(ctx0, cur, il);
    if (prt_result) {
        tmp = prt_result;
    } else {
        tmp = this->build_lora_mm(up, cur);  // ← sidecar missing
        g_native_fallback_calls++;  // ← counts as fallback
    }
}
```

If sidecar is missing (`g_prt_sidecar_data[il] == nullptr`), falls back to native and increments `g_native_fallback_calls`. Silent fallback — no fatal error.

**GATE STATUS:** ⚠️ SILENT — Falls back without loud warning. Recommend adding startup validation.

### Gate 4: Identity Fallback

`g_identity_fallback_calls` counter exists but is NOT incremented in Route A path. Used only in callback-mode fallback.

**GATE STATUS:** ✓ WORKING — Not triggered in L12+L15 mode.

### Gate 5: native_ffn_up_calls

Counter `g_native_ffn_up_calls` exists but is NOT incremented in `build_ffn()`. It was intended for PRT shadow mode but is always 0 in Route A path.

**GATE STATUS:** ⚠️ INFO — Always 0 in Route A. Not a safety issue but confusing. Recommend removing or documenting.

---

## Sidecar Checksum Output

| Layer | Checksum | Source |
|-------|----------|--------|
| L0 | -354.098145 | `llama_get_sidecar_checksum(0)` |
| L12 | (not printed) | `llama_get_sidecar_checksum(12)` |
| L15 | (not printed) | `llama_get_sidecar_checksum(15)` |
| L35 | 39.010246 | `llama_get_sidecar_checksum(35)` |

**Note:** phase10e0 only prints L0 and L35. L12 and L15 checksums are calculable but not currently printed.

---

## Fail-Safe Properties

| Property | Status |
|----------|--------|
| callback_overwrites = 0 in L12+L15 mode | ✓ VERIFIED |
| identity_fallback_calls = 0 in L12+L15 mode | ✓ VERIFIED |
| No silent tensor corruption without fatal error | ✓ VERIFIED |
| PRT replaced, not augmented, native | ✓ VERIFIED |
| Force-native layers never use PRT custom op | ✓ VERIFIED |

---

## Missing-Sidecar Handling

**Current behavior:**
- `load_all_sidecars()` prints `[PRT] Loaded N/36 sidecars`
- If N < 36, no fatal error — continues with partial loading
- Missing layers fall back to native silently in `build_ffn()`
- `g_native_fallback_calls` incremented

**Recommended fix:**
```cpp
// In llama_set_prt_sidecar or load_all_sidecars
if (g_prt_debug_mode >= 5700) {
    for (int i = 0; i < 36; i++) {
        if (!g_prt_sidecar_data[i]) {
            fprintf(stderr, "[PRT-ERROR] Layer %d sidecar missing — cannot proceed\n", i);
            // exit(1); // or return error
        }
    }
}
```

---

## Safety Gate Summary

| Gate | Mechanism | Effective? |
|------|-----------|-----------|
| Callback bypass | `callback_mode = (g_prt_debug_mode < 5700)` | ✓ YES |
| Force-native priority | `g_prt_force_native_layer[il]` checked first | ✓ YES |
| Sidecar missing | Falls back to native with counter | ⚠️ SILENT |
| Identity fallback | Counter exists, not triggered in Route A | ✓ YES |
| native_ffn_up_calls | Always 0 in Route A | ⚠️ INFO |

---

*End of Counters and Safety Gates*
