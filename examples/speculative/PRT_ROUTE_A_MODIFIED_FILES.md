# PRT Route A Modified Files

**Branch:** phase11bm-archived / PRT_ROUTE_A_RC1 candidate
**Date:** 2026-05-02
**Scope:** Route A + L12+L15 native fallback policy

---

## Core Modified Files (must merge)

### 1. src/llama-graph.cpp

**What changed:**
- Line 23: Added `const float * g_prt_sidecar_data[36]` array
- Line 25: `g_prt_debug_mode = 0` (was pre-existing)
- Lines 33-36: Added PRT counter globals
- Lines 1090-1135: `build_ffn()` modified with Route A logic

**Key logic added:**
```cpp
// Phase 11BG: check force-native mask BEFORE custom op
bool force_native = g_prt_force_native_enabled && g_prt_force_native_layer[il];

if (force_native) {
    tmp = this->build_lora_mm(up, cur);  // native, no callback
    g_native_fallback_calls++;
} else if (up && prt_layer && g_prt_sidecar_data[il]) {
    ggml_tensor * prt_result = build_prt_ffn_up(ctx0, cur, il);
    if (prt_result) {
        tmp = prt_result;  // PRT custom op
        g_prt_true_replacement_calls++;
    } else {
        tmp = this->build_lora_mm(up, cur);  // sidecar missing fallback
        g_native_fallback_calls++;
    }
} else {
    tmp = this->build_lora_mm(up, cur);  // native path
}
```

**Risk:** HIGH — This is the hot path. Every FFN_UP layer goes through this. The decision tree must be correct.

---

### 2. src/llama.cpp

**What changed:**
- `llama_set_prt_debug_mode(int mode)` (pre-existing, unchanged behavior)
- `llama_set_prt_force_native_layers(int n, int * layers)` — Phase 11BG
- `llama_clear_prt_force_native()` — Phase 11BG
- Counter getter functions (Phase 11BB/11BD)

**New API:**
```cpp
void llama_set_prt_force_native_layers(int n_layers, const int * layer_ids);
// Sets g_prt_force_native_layer[36] for selected layers
// Sets g_prt_force_native_enabled = true

void llama_clear_prt_force_native(void);
// Clears all g_prt_force_native_layer[] to false
// Sets g_prt_force_native_enabled = false
```

**Risk:** HIGH — Public API. Any bug here affects all callers.

---

### 3. examples/speculative/prt_graph_replace.h

**What changed:**
- `prt_is_true_replacement_layer()` — mode 5700 activates PRT all layers
- `build_prt_ffn_up()` — creates GGML custom op replacing native matmul
- `prt_ffn_up_custom_op()` — SIMD/scalar matmul with PRT sidecar

**Key behavior:**
```cpp
// Mode 5700: all layers use PRT
static bool prt_is_true_replacement_layer(int il) {
    if (g_prt_debug_mode >= 5700) return true;
    if (g_prt_debug_mode >= 5600 && g_prt_debug_mode < 5700) {
        return (g_prt_debug_mode == 5600 + il);
    }
    return false;
}
```

**Risk:** HIGH — Contains the custom op that replaces native FFN_UP matmul. Any error here produces wrong output silently.

---

### 4. examples/speculative/prt_avx2_kernel.h

**What changed:**
- AVX2 SIMD matmul kernel (`matmul_prt_avx2`)
- Scalar fallback (`matmul_prt_scalar`)

**Risk:** MEDIUM — SIMD math. Wrong results if AVX2 implementation has bugs.

---

### 5. examples/speculative/phase10e0_layer0_replacement.cpp

**What changed:**
- `--prt-force-native L0,L1,...` CLI flag parsing
- `load_all_sidecars()` — mmap/POSIX loader
- `prt_eval_callback()` — gated by `callback_mode = (g_prt_debug_mode < 5700)`
- Counter reporting at end of run

**Risk:** MEDIUM — Benchmark binary. Has dead code (`load_sidecar_fopen`). Callback path is correctly gated.

---

## Unchanged Files (still used but not modified for RC1)

| File | Role |
|------|------|
| `common/arg.cpp` | `--prt-mode` only, no `--prt-force-native` |
| `include/llama.h` | No new headers needed |
| `examples/speculative/speculative.cpp` | Speculative decoding (unused) |
| `examples/speculative/prt_active.h` | Phase 10C test code (unused) |

---

## Files NOT Modified (kept for reference)

| File | Note |
|------|------|
| `examples/speculative/prt_shadow.h` | Phase 10B shadow test (pre-Route A) |
| `examples/speculative/phase10b_shadow_test.cpp` | Old benchmark |
| `examples/speculative/phase10c_active_test.cpp` | Old benchmark |
| `examples/speculative/phase10d_guard_test.cpp` | Old benchmark |
| `src/llama-graph.cpp.prt_backup` | Backup of original |

---

## Merge Scope Summary

**Files that need review for merge:**
1. `src/llama-graph.cpp` — Route A hook in `build_ffn`
2. `src/llama.cpp` — Force-native API + counter getters
3. `examples/speculative/prt_graph_replace.h` — Custom op + AVX2 kernel
4. `examples/speculative/prt_avx2_kernel.h` — AVX2 SIMD
5. `examples/speculative/phase10e0_layer0_replacement.cpp` — Benchmark binary

**Total modified scope:** 5 files, ~500 lines added across core files.

---

*End of Modified Files List*
