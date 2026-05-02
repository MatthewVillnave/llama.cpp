# PRT Phase 11BO: Code Review — Route A + L12/L15

**Date:** 2026-05-02
**Reviewer:** ELVIS
**Branch:** phase11bm-archived / PRT_ROUTE_A_RC1 candidate
**Binary:** `llama-prt-posix` (examples/speculative/phase10e0_layer0_replacement.cpp)

---

## Task 1: Modified Files

| File | Function(s) Touched | Purpose | Risk |
|------|--------------------|---------|------|
| `src/llama-graph.cpp` | `build_ffn()`, counter globals | Route A custom op hook, PRT counter declarations | **HIGH** |
| `src/llama.cpp` | `llama_set_prt_debug_mode()`, `llama_set_prt_force_native_layers()`, counter getters | PRT mode API, force-native API, counter getter functions | **HIGH** |
| `examples/speculative/phase10e0_layer0_replacement.cpp` | `main()`, `prt_eval_callback()`, `load_all_sidecars()` | Benchmark binary, callback handling, sidecar loading | **MEDIUM** |
| `examples/speculative/prt_graph_replace.h` | `prt_ffn_up_custom_op()`, `build_prt_ffn_up()`, `prt_is_true_replacement_layer()` | GGML custom op + AVX2 kernel for PRT matmul | **HIGH** |
| `examples/speculative/prt_avx2_kernel.h` | `matmul_prt_avx2()` | AVX2 SIMD kernel for PRT matmul | **MEDIUM** |
| `examples/speculative/speculative.cpp` | N/A | Speculative decoding harness (unused in RC branch) | **LOW** |

### Route A Core Files

**src/llama-graph.cpp:**
- `g_prt_debug_mode = 0` (default)
- Counter globals: `g_prt_true_replacement_calls`, `g_callback_overwrite_calls`, `g_identity_fallback_calls`, `g_native_fallback_calls`
- `g_prt_sidecar_data[36]` array
- `build_ffn()`: Route A logic — checks `g_prt_force_native_layer[il]` FIRST, then `prt_layer` condition

**src/llama.cpp:**
- `llama_set_prt_debug_mode(int mode)` — sets `g_prt_debug_mode`
- `llama_set_prt_force_native_layers(int n, int * layers)` — sets per-layer force-native mask
- `llama_clear_prt_force_native()` — clears mask
- Counter getter functions for all 5 counters

**examples/speculative/prt_graph_replace.h:**
- `prt_is_true_replacement_layer()` — mode 5700 = all layers, 5600+L = specific layer
- `build_prt_ffn_up()` — creates GGML custom op tensor replacing native FFN_UP
- `prt_ffn_up_custom_op()` — SIMD/scalar matmul using sidecar
- AVX2 kernel inline

**examples/speculative/phase10e0_layer0_replacement.cpp:**
- `main()`: `--prt-mode N`, `--prt-force-native L0,L1,...` parsing
- `prt_eval_callback()`: callback-mode only (skipped for mode 5700)
- `load_all_sidecars()`: mmap/POSIX loader
- Counter reporting at end of run

---

## Task 2: Mode Flag Audit

| Mode | Description | Route A? | Callback? | Force-Native? |
|------|-------------|---------|-----------|---------------|
| `0` | Native (no PRT) | NO | NO | NO |
| `5700` | Pure Route A, all 36 layers | YES | NO | NO |
| `5700 + --prt-force-native 12,15` | Route A + L12/L15 fallback | YES | NO | YES |
| `5600+L` | Single-layer PRT for layer L | YES | NO | NO |
| `5435` | Callback mode | NO | YES | NO |
| `5605` | Callback mode | NO | YES | NO |

**Audit Results:**
- ✓ Pure native mode untouched (mode 0 bypasses all PRT)
- ✓ Pure Route A available (mode 5700, no force-native)
- ✓ L12+L15 policy explicit (`--prt-force-native 12,15`)
- ✓ Callback path: **silently skipped** in Route A mode 5700 — `prt_eval_callback` checks `g_prt_debug_mode < 5700` before registering callback intercepts
- ✓ Identity fallback: counter `g_identity_fallback_calls` exists but is NOT triggered in Route A path — only used in callback-mode fallback path
- ✓ Force-native mask checked BEFORE custom op in `build_ffn()`

**Hidden Correction Path Check:**
- `prt_eval_callback()` body writes PRT output directly to tensor — only executed if `callback_mode = (g_prt_debug_mode < 5700)`
- For mode 5700: `callback_mode = false` → callback returns `false` immediately → no intercept
- **No hidden correction path** in mode 5700

---

## Task 3: Counter Audit

| Counter | Declared In | Getter | Phase10e0 Printed | Phase11BN Clean? |
|---------|------------|--------|-----------------|-----------------|
| `g_prt_true_replacement_calls` | llama-graph.cpp:33 | `llama_get_prt_true_replacement_calls()` | YES | ✓ |
| `g_callback_overwrite_calls` | llama-graph.cpp:34 | `llama_get_callback_overwrite_calls()` | YES | ✓ |
| `g_identity_fallback_calls` | llama-graph.cpp:35 | `llama_get_identity_fallback_calls()` | YES | ✓ |
| `g_native_fallback_calls` | llama-graph.cpp:36 | `llama_get_native_fallback_calls()` | YES | ✓ |
| `g_prt_ffn_up_custom_op_count` | llama-graph.cpp:29 | `llama_get_prt_custom_op_count()` | YES | ✓ |

**Sidecar checksums:** `llama_get_sidecar_checksum(layer)` — prints L0, L12, L15, L35

**Audit Results:**
- ✓ All counters have getter functions
- ✓ All counters printed in phase10e0 output
- ✓ callback_overwrites = 0 confirmed for all L12+L15 runs
- ✓ native_fallback_calls = 16 (2 layers × 8) consistent

---

## Task 4: Sidecar Loading Audit

| Check | Status |
|-------|--------|
| POSIX/mmap loader only | ✓ (`load_sidecar_posix` / `load_sidecar_mmap`) |
| No fopen/fread in benchmark | ✓ (`load_sidecar_fopen` defined but not used; `PRT_LOADER_TYPE=0`) |
| Sidecar orientation: `sidecar[j*hidden+k]` | ✓ (confirmed in `prt_ffn_up_custom_op`) |
| Sidecar checksums printed | ✓ (L0, L12, L15, L35 in output) |
| Missing sidecar: fail loudly | ⚠️ SILENT — `build_ffn` falls back to native if `g_prt_sidecar_data[il]==nullptr` |

**Concern:** Missing sidecar falls back to native with `g_native_fallback_calls++` — no fatal error, no loud warning in production path. The debug print `[PRT-11BB-AUTH] IL=%d FALLBACK to native` appears but only when `prt_layer==true`. If PRT is enabled but sidecar missing, it silently uses native.

**Recommendation:** Add explicit check at startup: if mode 5700 and any sidecar is null, print `[PRT-ERROR] missing sidecar for layer N — cannot proceed` and exit.

---

## Task 5: Fallback Policy Audit

| Check | Status |
|-------|--------|
| L12 and L15 are force-native | ✓ (`--prt-force-native 12,15`) |
| All other layers use PRT Route A | ✓ (`build_ffn` force-native check, then PRT custom op) |
| Fallback count matches expectation | ✓ (16 = 2 layers × 8 tokens generated) |
| Fallback is NOT prompt-dependent | ✓ (static mask, set once per run) |

**build_ffn decision tree:**
```
if (force_native) → build_lora_mm (native) [+g_native_fallback_calls]
else if (prt_layer && sidecar exists) → build_prt_ffn_up (custom op) [+g_prt_true_replacement_calls]
else → build_lora_mm (native fallback) [+g_native_fallback_calls]
```

**Note:** Force-native check takes PRIORITY over PRT custom op — even if mode 5700 (all layers PRT), L12+L15 use native. This is the correct behavior.

---

## Task 7: Final Verdict

**A. Is the code path understandable?**
YES — `build_ffn` decision tree is clear. Mode 5700 activates PRT all layers. Force-native mask checked first. Custom op replaces native matmul. Callback path is cleanly gated by mode check.

**B. Are there hidden correction paths?**
NO — `prt_eval_callback` body only runs when `g_prt_debug_mode < 5700`. Mode 5700 fully bypasses callback intercept. Force-native check in `build_ffn` is the only deviation from pure Route A.

**C. Are counters reliable?**
YES — All 5 counters increment in specific code paths only. No counter is incremented without corresponding real action. callback_overwrites=0 is mechanically enforced.

**D. Are missing sidecars safely handled?**
CONDITIONAL — Falls back to native silently with `g_native_fallback_calls++`. Loud failure would be better. Not a blocker for RC1 but should be addressed before production.

**E. Is L12+L15 policy explicit?**
YES — `--prt-force-native 12,15` is the documented CLI flag. The mask is checked first in `build_ffn`. The policy doc is in `PRT_ROUTE_A_L12_L15_POLICY.md`.

**F. Is the branch ready to tag as PRT_ROUTE_A_RC1?**
YES — with one caveat: missing-sidecar handling should be documented.

**G. Is it ready to merge into main experimental branch?**
CONDITIONAL — Core code is clean. The modified files are limited and well-scoped. Missing-sidecar silent fallback is the main concern. Recommend adding startup validation before merge.

---

## Concerns

| Severity | Issue |
|----------|-------|
| MEDIUM | Missing sidecar falls back silently (not loud fail). Recommend startup check. |
| LOW | Debug print statements in `build_ffn` (e.g. `[PRT-11BB-AUTH]`) are verbose but harmless. |
| LOW | `load_sidecar_fopen` exists but is unused — dead code. |
| INFO | AVX2 kernel checked at runtime via `#if defined(__AVX2__)` — safe. |

---

*End of Code Review*
