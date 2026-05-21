# PRT Phase 24W: Correctness Forensics — Final Verdict

## Status: COMPLETE ✅

**Date:** 2026-05-20
**Branch:** experimental/prt-phase19a-alt-sidecar-backed
**HEAD:** 20737c5b9

---

## Executive Summary

**PRT custom op is BROKEN for 3B layer0.** It runs, produces numerically correct-looking intermediate values, yet generates garbled output ("Parisyne academics/media") while native produces correct output ("Paris"). Prior phase reports claiming "Paris-like" output were observing native fallback, not PRT custom op output.

---

## Test Results

| Mode | Output | Status |
|------|--------|--------|
| Native (no PRT env) | "Paris" | ✅ CORRECT |
| PRT layer0 INT8 | "Parisyne academics/media" | ❌ GARBAGE |

---

## Key Evidence

### 1. PRT Op Is Actually Running
```
[PRT_V2_KERNEL_ENTER] K=2048 M=11008 N=2
[PRT_V2_KERNEL_PTRS] x=0x56559e828e60 w=0x7d1c44e00000 scales=(nil) dst=0x56559e846660
[PRT_V2_KERNEL_TYPES] x=0 w=0 scales=-1 dst=0
[PRT_V2_SHAPE_RUNTIME] backend=scalar K=2048 M=11008 N=2
[PRT_V2_KERNEL_PROGRESS] token=0 j=0 acc=0.794249
[PRT_V2_KERNEL_EXIT] done=1 output_abs_sum_first4=137.820084
```

Custom op is inserted and dispatched. Kernel runs and exits.

### 2. Numeric Values Are Reasonable
```
[PRT_V2_NUMERIC] backend=scalar abs4=137.820084 y0=0.794249 y1=0.536080 y2=-46.865269 y3=2.840473
```

First token first values: y0=0.79, y1=0.54, y2=-46.87, y3=2.84 — within plausible range.

### 3. AVX2 Not Compiled (But Scalar Works)
```
GGML_AVX2:BOOL=OFF  — build has AVX2 disabled
backend=scalar — scalar path is taken (correct behavior with AVX2=OFF)
```

AVX2 is off in cmake but scalar kernel is the correct fallback. Not the issue.

### 4. No Decode-Only Problem
```
[PRT_V2_POLICY] decode_only=1 N=1 action=prt layer=0
[PRT_V2_POLICY] decode_only=1 N=16 action=native_prefill layer=0
```

Phase 24G used DECODE_ONLY=1 which is the standard mode. Same output.

---

## Prior Report Reconciliation

| Phase | Reported Output | Actual Status | Reason |
|-------|-----------------|---------------|--------|
| 24G-R3 | "Parisyne academics..." | ❌ GARBAGE | Was actually PRT output (not fallback) |
| 24G-R7 | "Paris-like" | ❌ MISREAD | Likely saw only prompt token or misread logs |
| 24H | "Paris-like" | ❌ MISREAD | Same — saw native print before PRT printed |
| 24J | "Parisyne academics..." | ❌ GARBAGE | Phase 24J report correctly noted Parisyne |
| 24S | "correct but slow" | ❌ WRONG | Phase 24S didn't verify correctness, only parity |

---

## Architecture Analysis

### Why Correct Numeric ≠ Correct Output

The PRT custom op computes FFN_UP correctly:
- Y[M,N] = W[K,M]^T @ X[K,N]
- Kernel computes y0=0.79 for token 0, first column

BUT — the FFN is not just FFN_UP. The full SwiGLU requires:
1. **FFN_UP** → result (we compute this)
2. **gate** → ggml_silu(gate) (native path uses Q4_K quantized gate weights)
3. **SiLU(gate) × result** (element-wise multiply)
4. **FFN_DOWN** → native Q4_K matmul

When PRT replaces only FFN_UP, layers 1..35 use native gate + native down with a corrupted intermediate from layer0 PRT. This breaks the SwiGLU activation flow at the FIRST layer, causing cascading errors.

**Why 7B might work:** 7B uses the original (correct) FFN layout and the PRT custom op happens to align correctly there.

---

## Root Cause Hypothesis

The PRT custom op in its current form does NOT correctly implement the FFN_UP replacement for Qwen2.5-3B's SwiGLU architecture. The problem is not in the kernel math but in how the PRT op output is consumed by the downstream native layers.

**Possible specific issues:**
1. Output tensor layout mismatch (Y[M,N] vs what downstream expects)
2. Scales not applied correctly or at all (scales=(nil) in kernel)
3. The "silu" activation from native gate path doesn't match what PRT assumes
4. Output is accumulated into tensor that already has residual from attention

**Most likely:** The scales pointer is NULL in the kernel (`scales=(nil)`), so the kernel uses identity scale (1.0) when the FFN_UP weights need proper scaling. Without scales, the PRT output is the raw matmul without quantization compensation.

---

## Forbidden Claims (Corrected)

- ❌ "PRT produces correct output" — IT DOES NOT for 3B layer0
- ❌ "3B INT8 canonical path works" — broken, not working
- ❌ "Paris-like output from PRT" — garbage output
- ❌ "Phase 24S correct but slow" — incorrect, speed comparison meaningless since broken

---

## What's Working

- ✅ Sidecar loading (INT8 decode path works)
- ✅ Selector logic (correctly routes layer0 to PRT)
- ✅ Kernel dispatch (scalar/AVX2 branch selection correct)
- ✅ Decode formula (k*M + j is correct)
- ✅ Native fallback (correct, "Paris")
- ✅ Offline parity (cosine ~0.9999 for INT8 sidecar vs f32 ref)

---

## Next Steps

1. **FIX PRT_CUSTOM_OP_BROKEN** — Debug why custom op produces wrong output
   - Check scales: why is scales=(nil)? Are scales being passed?
   - Check output tensor layout: ne[0]=11008, ne[1]=1 — correct?
   - Compare intermediate outputs: native vs PRT at each FFN stage
   - Instrument gate and silu outputs for layer0

2. **Phase 24X: Instrumented Correctness Test**
   - Add per-stage audit logging (UP output, silu(gate), final SwiGLU)
   - Run both native and PRT paths, compare each stage
   - Identify exactly where PRT path diverges

3. **Don't claim "correct" until logits match native**

---

## Verdict

**BROKEN_CUSTOM_OP_3B_LAYER0** — PRT custom op runs but produces garbage output for Qwen2.5-3B layer0. Prior reports of "Paris-like" output were misread or observing native fallback. The custom op is not a viable replacement for the native FFN path on this model/layer combination.