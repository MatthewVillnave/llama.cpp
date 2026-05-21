# PRT Phase 24Y: Post-INT8-Scale-Fix Regression Check

## Status: COMPLETE ✅

**Date:** 2026-05-20
**Branch:** experimental/prt-phase19a-alt-sidecar-backed
**HEAD:** 8de123d54

---

## Executive Summary

Global INT8 decode fix (`/127` factor) does not regress 3B or 7B. Both models pass correctness validation after the fix. PRT custom-op produces identical output to native on both model sizes.

---

## Test 1: 3B Repeat Validation (3 runs)

**Model:** Qwen2.5-3B-Instruct-Q4_K_M.gguf  
**Env:** PRT_GGML_TEST_LAYER=0, PRT_V2_SIDECAR_FORMAT=int8, DECODE_ONLY=1, AVX2=1, QUIET=1  
**Params:** c=4, n=8, t=1, temp=0, --no-conversation

| Run | Native | PRT | Match |
|-----|--------|-----|-------|
| 1 | "The capital of France is Paris. Paris is located in the north" | "The capital of France is Paris. Paris is located in the north" | ✅ |
| 2 | "The capital of France is Paris. Paris is located in the north" | "The capital of France is Paris. Paris is located in the north" | ✅ |
| 3 | "The capital of France is Paris. Paris is located in the north" | "The capital of France is Paris. Paris is located in the north" | ✅ |

**Proof chain (run 1 sample):**
- sidecar: `prt_sidecars_3b_int8_phase24f/ffn_up_layer0_prt.int8` K=2048 M=11008 ✅
- scales: first5=0.074261/0.145878/0.085927/0.086346/0.086850 ✅
- route: `IL=0 route=ggml_op reason=selected_layer` ✅
- backend: `avx2` (no scalar fallback) ✅
- numeric: abs4=1.085197 y0=0.006254 y1=0.004221 y2=-0.369018 y3=0.022366 (valid floats) ✅
- No FALLBACK logs ✅

**Verdict:** PASS_3B_POST_SCALE_FIX_REPEAT ✅

---

## Test 2: 7B Canonical INT8 Regression

**Model:** Qwen2.5-7B-Instruct-Q4_K_M.gguf  
**Env:** PRT_GGML_TEST_LAYER=0, PRT_V2_SIDECAR_FORMAT=int8, DECODE_ONLY=1, AVX2=1, QUIET=1  
**Params:** c=4, n=4, t=1, temp=0, --no-conversation

| Mode | Output |
|------|--------|
| Native | "The capital of France is Paris. It is" |
| PRT | "The capital of France is Paris. It is" |

**Match:** ✅ IDENTICAL

**Proof chain:**
- sidecar: `prt_sidecars_7b_int8_phase24g_canonical/ffn_up_layer0_prt.int8` K=3584 M=18944 ✅
- scales: first5=0.067231/0.084354/0.204517/0.080930/0.161444 ✅
- route: `IL=0 route=ggml_op reason=selected_layer` ✅
- backend: `avx2` (no scalar fallback) ✅
- numeric: abs4=1.257279 y0=0.032029 y1=-0.143361 y2=0.055961 y3=-0.164855 (valid floats) ✅
- No FALLBACK logs ✅

**Verdict:** PASS_7B_POST_SCALE_FIX_REGRESSION ✅

---

## Summary

| Test | Result |
|------|--------|
| 3B repeat (3 runs) | ✅ PASS — identical output across all runs |
| 3B vs native match | ✅ PASS |
| 7B vs native match | ✅ PASS |
| 7B PRT output sane | ✅ PASS |
| No fallback | ✅ Both models |
| No NaN/Inf | ✅ Both models |

---

## Recommended Next

**Phase 24Z:** Redo 3B timing from scratch now that correctness is proven. Previous timing data (Phase 24S etc.) is meaningless — it was measuring broken op output.

---

## Models/Sidecars/F32 Refs Staged?
No new files staged.

## Secrets Detected?
None.

## Tags Touched?
None.