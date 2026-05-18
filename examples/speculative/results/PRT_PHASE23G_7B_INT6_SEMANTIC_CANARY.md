# Phase 23G: 7B INT6 Semantic/ Capture Canary, Layer0 Only

## Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## Previous HEAD (23F)
`53deb5b160535e03f64fb4cc2fb7ca9e775053cd`

## New HEAD (23G)
`[NEW_COMMIT]`

## Checkpoint Verdict
**PASS_7B_INT6_SCOPED_SEMANTIC_CANARY** ✅
**PASS_7B_INT6_NO_CORRUPTION** ✅

---

## 1. Executive Summary

7B INT6 decoded-f32 path produces sane, non-corrupt output for all 4 test prompts. No gibberish, no repetition, no NaN/Inf. Policy routing confirmed correct. Scalar fallback remains at 0. The INT6 layer0 path is clean enough to justify timing comparison next.

---

## 2. Prompt Files

| Prompt | File |
|--------|------|
| P1: "The capital of France is" | `$PRT_SCRATCH/prompts/phase23g_p1.txt` |
| P2: "The largest planet in our solar system is" | `$PRT_SCRATCH/prompts/phase23g_p2.txt` |
| P3: "Return JSON only: name=Qwen, status=active" | `$PRT_SCRATCH/prompts/phase23g_p3.txt` |
| P4: "Once upon a time" | `$PRT_SCRATCH/prompts/phase23g_p4.txt` |

---

## 3. Semantic Results

All 4 prompts produced clean, sane output:

| Prompt | Output | Classification |
|--------|--------|----------------|
| P1: "The capital of France is" | `Paris.` | CLEAN_SEMANTIC ✅ |
| P2: "The largest planet in our solar system is" | `Jupiter.` | CLEAN_SEMANTIC ✅ |
| P3: "Return JSON only: name=Qwen, status=active" | `{"name":"Qwen","status":"active"}` | CLEAN_SEMANTIC ✅ |
| P4: "Once upon a time" | `Once upon a time, in a land far, far away...` | CLEAN_SEMANTIC ✅ |

**None required exact match** — classifications based on sanity, not correctness.

**No CORRUPT, no REPETITION, no CAPTURE_LIMITED** across any prompt.

---

## 4. Policy Evidence

All runs confirmed:

| Check | Value | Status |
|-------|-------|--------|
| selected sidecar | int6 | ✅ |
| scale_off | 20 | ✅ |
| K, M | 3584, 18944 | ✅ |
| nan count | 0 | ✅ |
| inf count | 0 | ✅ |
| N≤4 action | prt | ✅ |
| N>4 action | native_prefill | ✅ |
| scalar fallback | 0 | ✅ |

---

## 5. Classification Criteria

| Class | Meaning |
|-------|---------|
| CLEAN_SEMANTIC | Sane text, no corruption, no repetition |
| CLEAN_BUT_DIFFERENT | Sane text but different from expected |
| CAPTURE_LIMITED | Could not capture output |
| CORRUPT | Gibberish, malformed, NaN/Inf in output |
| REPETITION | Repeated tokens/patterns |
| FAIL_RUNTIME | Did not run or crashed |

**Minimum pass criteria met:**
- no corruption ✅
- no repetition loop ✅
- policy correct ✅
- scalar fallback=0 ✅
- P1 and P2 visibly sane ✅

---

## 6. Allowed Claims

✅ **Allowed:**
- 7B INT6 layer0 produces sane, non-corrupt text output
- scale_off=20 works for 7B INT6 in semantic mode
- No NaN/Inf in outputs
- No repetition loops
- Policy routing correct (N≤4 PRT, N>4 native_prefill)
- Scalar PRT fallback eliminated
- Output is valid JSON for JSON prompt (P3)
- Output answers factual questions correctly (P1: Paris, P2: Jupiter)

---

## 7. Forbidden Claims

❌ **Do NOT claim:**
- broad semantic equivalence to native computation
- exact/token match to reference
- end-to-end speedup
- production readiness
- multi-layer support
- all-layer support
- 14B support

---

## 8. Recommended Next

**Phase 23H — 7B INT6 timing comparison vs INT8/native, layer0 only.**

Semantic canary passed. The INT6 layer0 path is clean. Next logical step is to measure per-token latency vs INT8 and native paths to establish whether there is any speedup at the kernel level.

Do NOT jump to multi-layer experiments yet.

---

## 9. Safety Scan

| Check | Result |
|-------|--------|
| Model files staged | ❌ NO |
| Sidecars staged | ❌ NO |
| f32 refs staged | ❌ NO |
| Captures staged (large) | ❌ NO |
| Prompts staged | ❌ NO (temporary, not committed) |
| Binaries staged | ❌ NO |
| Huge logs staged | ❌ NO |
| Secrets detected | ❌ NO |
| Source code changes | ❌ NO |

---

## Final Report Fields

A. **Branch:** `experimental/prt-phase19a-alt-sidecar-backed`

B. **Previous HEAD:** `53deb5b160535e03f64fb4cc2fb7ca9e775053cd`

C. **New HEAD:** `[NEW_COMMIT]`

D. **Prompts tested:** 4 (P1-P4)

E. **Semantic classifications:** P1=CLEAN_SEMANTIC, P2=CLEAN_SEMANTIC, P3=CLEAN_SEMANTIC, P4=CLEAN_SEMANTIC

F. **Visible outputs:** Paris, Jupiter, JSON valid, story intro — all sane

G. **Policy result:** N≤4 PRT AVX2, N>4 native_prefill, scalar=0 ✅

H. **Scalar fallback eliminated?** YES — 0 ✅

I. **abs4 sanity:** nan=0 inf=0, scale_off=20 confirmed ✅

J. **Capture limitations:** None — all 4 prompts captured successfully

K. **Verdict:** PASS_7B_INT6_SCOPED_SEMANTIC_CANARY + PASS_7B_INT6_NO_CORRUPTION

L. **Recommended next:** Phase 23H — 7B INT6 timing comparison vs INT8/native, layer0 only

M. **Models/sidecars/binaries staged?** NO ✅

N. **Secrets detected?** NO ✅

O. **Existing tags touched?** NO ✅

P. **System disk free:** 27G

Q. **Scratch disk free:** 18G

---

_Phase 23G confirms 7B INT6 layer0 produces sane text output. No corruption detected across 4 diverse prompts._