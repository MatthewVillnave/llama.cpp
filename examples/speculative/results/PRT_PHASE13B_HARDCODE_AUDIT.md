# PRT Phase 13B: Dynamic Shape Support Audit

**Date:** 2026-05-03  
**Branch:** `experimental/prt-phase13-model-generalization`  
**Commit:** `5d45a49a6`

---

## Summary of Findings

The PRT sidecar loader in `phase10e0_layer0_replacement.cpp` has **three categories of hardcoding**:
1. **Layer count** — hardcoded to 36
2. **Sidecar M/N dimensions** — hardcoded to 2048×11008 in struct initialization and static zero sidecar
3. **Tensor dimension constants** — present in test/header files but NOT in the runtime loader

The good news: The runtime loader already derives M/N from the GGML tensor dimensions at lines 283-285. The problem is only the sidecar metadata initialization and TOTAL_LAYERS.

---

## Hardcoded Dimension Findings

### Category 1: Hardcoded Layer Count — HIGH RISK

| File | Line | Code | Risk |
|------|------|------|------|
| `phase10e0_layer0_replacement.cpp` | 23 | `static const int TOTAL_LAYERS = 36;` | HIGH — causes out-of-bounds if model has ≠36 layers |
| `phase10e0_layer0_replacement.cpp` | 153 | `for (int l = 0; l < TOTAL_LAYERS; l++)` | HIGH — loop range wrong for non-36-layer models |
| `phase10e0_layer0_replacement.cpp` | 163 | `fprintf(stderr, "[PRT] Loaded %d/%d sidecars\n", loaded, TOTAL_LAYERS);` | MED — wrong count in message |
| `phase10e0_layer0_layer0_replacement.cpp` | 190 | `for (int l = 0; l < TOTAL_LAYERS; l++)` | HIGH — loop range wrong |
| `phase10e0_layer0_replacement.cpp` | 259 | `&& layer < TOTAL_LAYERS` | HIGH — wrong bounds check |
| `phase10e0_layer0_replacement.cpp` | 274 | `layer >= TOTAL_LAYERS` | HIGH — wrong bounds check |

**Fix:** Replace `TOTAL_LAYERS` with `model->n_layer` from `llama_model`.

### Category 2: Sidecar Struct Hardcoding — HIGH RISK

| File | Line | Code | Risk |
|------|------|------|------|
| `phase10e0_layer0_replacement.cpp` | 95 | `g_sidecars[layer] = {layer, data, 2048, 11008};` | HIGH — wrong M/N for non-3B models |
| `phase10e0_layer0_replacement.cpp` | 112 | `g_sidecars[layer] = {layer, data, 2048, 11008};` | HIGH — mmap path |
| `phase10e0_layer0_replacement.cpp` | 125 | `g_sidecars[layer] = {layer, data, 2048, 11008};` | HIGH — fopen path |
| `phase10e0_layer0_replacement.cpp` | 131 | `static float s_zero_sidecar[2048 * 11008] = {0};` | HIGH — wrong size, also WRONG for 0.5B |
| `phase10e0_layer0_replacement.cpp` | 132 | `g_sidecars[layer] = {layer, s_zero_sidecar, 2048, 11008};` | HIGH — static zero sidecar |

**Fix:** Read M/N from the actual tensor dimensions at load time, not hardcoded constants.

### Category 3: Header/Test Constants — LOW RISK (no runtime impact)

| File | Line | Code | Risk |
|------|------|------|------|
| `prt_active.h` | 25 | `const int M = 2048;` | LOW — only used in test/header, not in loader |
| `prt_active.h` | 26 | `const int N = 11008;` | LOW — only used in test/header |
| `phase10c_active_test.cpp` | 21 | `const int M = 2048;` | LOW — test-only |
| `phase10d_guard_test.cpp` | 21 | `const int M = 2048;` | LOW — test-only |

**Fix:** Update for correctness but not blocking.

---

## Existing Dynamic Dimension Detection

The loader already uses tensor dimensions correctly in one place:

```cpp
// phase10e0_layer0_replacement.cpp, lines 283-285
int M = (int)src1->ne[0];  // ffn_dim from tensor (from W_up shape)
int N = (int)t->ne[0];      // hidden_dim from tensor (from W_up shape)
```

This means: **the actual PRT compute already handles dynamic M/N** — it's only the sidecar metadata setup that is hardcoded.

---

## Qwen2.5-0.5B Dimension Mismatch

| Dimension | Qwen2.5-3B (current) | Qwen2.5-0.5B (blocked) | 0.5B sidecar size |
|-----------|---------------------|----------------------|-----------------|
| Layers | 36 | **24** | N/A |
| Hidden (N) | 2048 | **896** | N/A |
| FFN (M) | 11008 | **4864** | N/A |
| Sidecar bytes | 90,113,024 | **17,432,576** | **17.4MB** |
| Total sidecar storage | ~3.2GB | ~418MB | 400MB saved |

The 0.5B sidecars were successfully generated (24 files × 17.4MB = 418MB). The BLOCKED was purely a software issue — wrong hardcoded values.

---

## Risk Assessment

| Fix Category | Risk | Complexity | 
|-------------|------|------------|
| Replace TOTAL_LAYERS with model->n_layer | LOW | ~5 lines |
| Read M/N from tensor at load time | LOW | ~3 places |
| Validate sidecar bytes against expected size | MED | ~10 lines |
| Fix static zero sidecar array size | MED | ~3 lines |
| Update header constants | NONE | ~4 lines |
| Update test constants | NONE | ~6 lines |

**Total: ~30 lines of targeted changes, no algorithmic changes**

---

## Verification Plan

After fix, test with Qwen2.5-0.5B:
1. Sidecars load without crash
2. `callback_overwrites = 0`
3. `identity_fallback_calls = 0`
4. Output coherent on 5 prompts
5. Speedup measurable