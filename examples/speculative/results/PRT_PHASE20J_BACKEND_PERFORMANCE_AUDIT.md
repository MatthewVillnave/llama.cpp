# PRT Phase 20J: Backend / Performance Audit

**Verdict:** `PASS_BACKEND_BOTTLENECK_IDENTIFIED`

---

## Phase 20J-A: Git State

- Branch: `experimental/prt-phase19a-alt-sidecar-backed`
- HEAD: `d800dbc65` (Phase 20I)
- Status: Clean working tree

---

## Phase 20J-B: Execution Path Map

### CLI Args → Compute Flow

```
CLI: --prt-mode 5700 --prt-only-layers 10,20 --prt-sidecar-format int6
    ↓
llama_set_prt_debug_mode(5700) [llama.cpp]
    ↓
Harness loads INT6 sidecar files from /tmp/prt_sidecars_7b_int6_phase15b_packed/
    ↓
llama_set_prt_sidecar_int6(layer, int8_data, scales, M, N) [llama.cpp]
    → g_prt_int8_data[layer] = int8_data
    → g_prt_int8_scales[layer] = scales
    → g_prt_sidecar_format[layer] = 2  (INT6 scalar)
    ↓
Llama-graph build (each forward pass):
    ↓
Layer decision at llama-graph.cpp:1213-1240:
    if (g_prt_only_layers_set[il] && sidecar exists) → build_prt_ffn_up()
    else → build_lora_mm() (native)
    ↓
build_prt_ffn_up() [prt_graph_replace.h:496]
    → Sets PRTUserData with format=2 (INT6)
    → kernel_mode = 0 (scalar, not AVX2)
    → Creates ggml_custom_4d() tensor
    ↓
prt_ffn_up_custom_op() [prt_graph_replace.h:373-386]
    → Scalar INT8*scale multiply loop (NOT AVX2)
    → No vectorization
```

### Format Paths
| Format | Sidecar Data | Kernel | Path |
|--------|------------|--------|------|
| 0 | float32 | AVX2 | Fast (if compiled) |
| 1 | int8 + scales | Scalar | Medium |
| 2 | int8 + scales | **Scalar (INT6)** | **SLOW** ← current |

---

## Phase 20J-D: Benchmark Matrix

Prompt: "The capital of France is"

| Config | Prompt t/s | Gen t/s | Wall | vs Native |
|--------|-----------|--------|------|----------|
| Native | 33.8 | **9.5** | 4.3s | 1.000x |
| Layer 10 only | 16.7 | 7.5 | 6.6s | 0.789x |
| Layer 20 only | 16.7 | 7.3 | 6.6s | 0.768x |
| Layers 10,20 | 11.1 | 6.2 | 7.8s | 0.653x |
| Layers 12,24 | 10.6 | 6.2 | 7.9s | 0.653x |
| Layers 6,18 | 11.0 | 6.2 | 7.8s | 0.653x |

**Observations:**
- Single layer: ~75-79% of native → 21-25% slower
- Two layers: ~65% of native → **35% slower**
- Adding second layer adds ~15% more slowdown (0.79x → 0.65x)

---

## Phase 20J-F: Bottleneck Classification

### Primary Bottleneck: **A. SCALAR_KERNEL_SLOW + G. NATIVE_ALREADY_OPTIMAL**

The INT6 custom op uses a scalar loop:
```cpp
// prt_graph_replace.h:373-386 (format=2)
for (int t = 0; t < n_tokens; t++) {
    for (int j = 0; j < ffn; j++) {
        float s = 0.0f;
        float sc = ud->int8_scales[j];
        for (int k = 0; k < hidden; k++) {
            s += X_t[k] * (float)ud->int8_data[j*hidden + k] * sc;
        }
        Y_t[j] = s;
    }
}
```

Contrast with native llama.cpp Q4_K_M which uses:
- AVX2 vectorization
- Bit-level dequantization
- Fused multiply-add
- optimized ggml_matmul kernels

### Secondary Bottlenecks:

**B. CUSTOM_OP_OVERHEAD** - ggml_custom_4d:
- Forces thread sync (`n_tasks=1`)
- Prevents graph fusion
- Disables ggml scheduler optimizations

**C. MEMORY_LAYOUT** - Sidecar data:
- Row-major INT8: `W[j*K+k]` 
- Input activations: `X[k]`
- Poor cache locality for inner loop

**F. TOO_FEW_LAYERS** - Only 2/28 layers replaced:
- FFN_UP is ~1/3 of total FFN compute
- FFN is ~2/3 of total transformer compute
- Replacing 2/28 layers = 1/14 ≈ 7% of model
- Overhead dominates such small replacement

### Summary

| Bottleneck | Severity | Evidence |
|-----------|----------|----------|
| **A. SCALAR_KERNEL_SLOW** | **HIGH** | format=2 uses scalar path, not AVX2 |
| **G. NATIVE_ALREADY_OPTIMAL** | **HIGH** | Native uses optimized Q4_K_M kernels |
| B. CUSTOM_OP_OVERHEAD | MEDIUM | ggml_custom_4d prevents fusion |
| C. MEMORY_LAYOUT | LOW | Could matter but buried in A |
| F. TOO_FEW_LAYERS | LOW | 2/28 is too small to matter |

---

## Phase 20J-G: Optimization Options

### Option 1: Optimize scalar INT6 kernel
- **Pros:** Quick win, minimal code
- **Cons:** Ceiling limited, not competitive
- **Likely ceiling:** Maybe 10-20% faster

### Option 2: INT8 resident compute buffer (predecode)
- **Pros:** Uses AVX2 path, proven in Phase 19J
- **Cons:** Memory cost (2x sidecar size)
- **Risk:** Higher RAM

### Option 3: Direct INT6 AVX2 kernel
- **Pros:** memory-efficient
- **Cons:** Complex, hard to unpack/pack

### **Option 4: ggml-native PRT backend kernel** ← BEST LONG-TERM
- **Pros:** Clean integration, avoids custom_op overhead
- **Cons:** Larger rewrite
- **Risk:** Medium
- **Expected benefit:** Eliminates B + improves A via proper SIMD

### Option 5: PRT-v2 clean branch
- **Pros:** Removes experimental debt
- **Cons:** Time-consuming
- **Risk:** Low

### Option 6: Stop 7B INT6 work
- **Verdict:** Publish research only
- **Risk:** None

---

## Phase 20J-H: Recommendation

**Recommended: Option 4 — ggml-native PRT op design (Phase 20K)**

**Why:**
- The scalar path is fundamentally slower than native
- Custom-op overhead is a tax on every inference
- A proper backend kernel can fix both (A + B)

**Estimated risk:** Medium (needs ggml backend integration)

**Estimated benefit:** Could reach ~80-90% native speed with fewer layers

**Falsification test:** If new PRT op can't beat (10,20) at 50% native, abandon.

**Smallest next test:** Design document for ggml-native PRT op interface + benchmark of custom_op vs native overhead.

---

## Phase 20J-I: Report

### A. Branch
`experimental/prt-phase19a-alt-sidecar-backed`

### B. Previous HEAD
`d800dbc65` (Phase 20I)

### C. New HEAD
Pending commit

### D. Execution Path
Mapped (see Section B)

### E. Benchmark Matrix
See Section D (all configs tested, clean outputs)

### F. Native Timing
9.5 gen t/s (baseline)

### G. Sparse INT6 Timing
6.2 gen t/s for 2-layer (10,20) = 0.65x native

### H. PRT Compute Overhead
~35% slowdown from scalar path + custom op

### I. Sidecar/Setup Overhead
Negligible (loaded once at startup)

### J. Per-Layer Overhead
Single layer: ~21% slowdown
Each additional layer: ~15% additive

### K. Bottleneck Classification
**A (SCALAR_KERNEL_SLOW) + G (NATIVE_ALREADY_OPTIMAL)** - Primary

### L. Expected vs Observed
Expected: Replacement should be faster (less math)
Observed: Slower (worse implementation + overhead)

### M. Optimization Options
See Section G (6 options ranked)

### N. Recommended Next
Option 4: ggml-native PRT op design

### O. Models/Sidecars Staged?
NO

### P. Secrets Detected?
NONE

### Q. Existing Tags Touched?
NO

---

## Conclusion

The INT6 PRT slowdown is not a mystery. It's caused by:

1. **Using a scalar INT8 loop instead of native AVX2 Q4_K_M** - This is #1
2. **ggml_custom_4d overhead** - #2
3. **Only replacing 2/28 layers** - #3 (overhead dominates small gains)

The fix is either:
- Predecode INT6 to float32 (adds memory, gets AVX2 speed)
- OR rewrite as proper ggml backend kernel (best long-term)

**Publish with corrections and pause 7B optimization unless AVX2 pre-decoding or backend rewrite is planned.**

---

*Phase 20J Complete*
*HEAD: d800dbc65*