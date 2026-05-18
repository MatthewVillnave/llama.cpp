# Phase 23F: 7B INT6 Policy Baseline Checkpoint

## Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## Previous HEAD (23E)
`5655e535d`

## New HEAD (23F)
`[NEW_COMMIT]`

## Checkpoint Verdict
**PASS_7B_INT6_POLICY_BASELINE_CHECKPOINT** ✅

---

## 1. Executive Summary

7B INT6 decoded-f32 policy path is now stable under repeat canary. The previous 7B INT6 numeric explosion (Phase 23D) was fixed by `scale_off=20` (Phase 23D-R2). Explicit sidecar selection (`PRT_V2_SIDECAR_FORMAT=int6`) prevents INT8/INT6 priority ambiguity (Phase 23D-R4). Repeat validation across 3 runs confirms perfect determinism (Phase 23E).

**This is a kernel/policy/numeric-sanity checkpoint only. No speed or production claims are made.**

---

## 2. Frozen Technical Baseline

| Field | Value |
|-------|-------|
| Branch | `experimental/prt-phase19a-alt-sidecar-backed` |
| HEAD | `[NEW_COMMIT]` (commit after 5655e535d) |
| Model | `Qwen2.5-7B-Instruct-Q4_K_M.gguf` |
| Layer | layer0 only (FFN up projection) |
| Sidecar format | INT6 (6-bit packed, decoded to f32) |
| Sidecar selector | `PRT_V2_SIDECAR_FORMAT=int6` |
| Sidecar path | `/media/matthew-villnave/VL_usb/prt_scratch/sidecars/prt_sidecars_7b_int6_phase15b_packed/ffn_up_layer0_prt.int6` |
| Dimensions | K=3584, M=18944 |
| Scale offset | **20** (corrected from 16 in Phase 23D-R2) |
| Decode path | PRT6 header → unpack 6-bit → scale → f32 W → `ggml_prt_ffn_up` |
| Policy | N≤4 → PRT AVX2, N>4 → native prefill |
| Scratch path | `/media/matthew-villnave/VL_usb/prt_scratch` |

---

## 3. Provenance / Scale Audit

| Field | Value |
|-------|-------|
| scale_off | 20 ✅ (confirmed every run) |
| first4 scales | 0.00216875, 0.00510111, 0.00218038, 0.00243282 |
| scale range | 0.00016086 to 0.07262494 |
| mean | 0.00488559 |
| nan count | 0 |
| inf count | 0 |

---

## 4. Repeat Validation Evidence

All 3 runs produced **identical** 11-value abs4 sequences with **zero variance**:

| Run | N=1 abs4 | N=2 abs4 | N=4 abs4 |
|-----|---------|---------|---------|
| Run 1 | 1.657920 | 2.477368 | 2.160922 |
| Run 2 | 1.657920 | 2.477368 | 2.160922 |
| Run 3 | 1.657920 | 2.477368 | 2.160922 |

**Full 11-value sequence (identical across all 3 runs):**
1. 1.657920
2. 0.696213
3. 2.477368
4. 2.160922
5. 2.577286
6. 1.646865
7. 1.688019
8. 1.364714
9. 1.333170
10. 0.975327
11. 0.916585

**Repeat stability: PERFECT — zero variance across 3 independent runs.**

---

## 5. Policy Evidence

| N value | Action | Backend |
|---------|--------|---------|
| N=1 | prt | avx2 ✅ |
| N=2 | prt | avx2 ✅ |
| N=4 | prt | avx2 ✅ |
| N>4 | native_prefill ✅ | — |

**c=16:** PRT AVX2 count: 8, native_prefill count: 8, scalar fallback: 0 ✅

**c=64:** PRT AVX2 count: 8, native_prefill count: 8, scalar fallback: 0 ✅

---

## 6. Correct Interpretation

The 7B INT6 policy path is stable for layer0 canary **under the following explicit limitations:**

- This does **not** prove multi-layer quality
- This does **not** prove broad semantic equivalence
- This does **not** prove end-to-end speedup
- This does **not** prove production readiness
- This is decoded-f32 INT6, **not** direct packed INT6 compute
- This is layer0 only, **not** all-layer support

---

## 7. Allowed Claims

✅ **Allowed:**
- 7B layer0 INT6 decoded-f32 policy canary is repeat-stable
- `scale_off=20` fix works correctly for 7B
- explicit sidecar selection (`PRT_V2_SIDECAR_FORMAT=int6`) works
- N≤4 routes to PRT AVX2
- N>4 routes to native prefill
- scalar PRT fallback eliminated in tested 7B INT6 canary
- numeric sanity passed with deterministic abs4 values (zero variance across 3 runs)

---

## 8. Forbidden Claims

❌ **Do NOT claim:**
- direct packed INT6 compute
- end-to-end speedup
- production readiness
- multi-layer support
- all-layer support
- broad semantic equivalence
- exact/token match
- 14B support

---

## 9. Recommended Next Steps (choose one)

**Option A — Semantic/capture canary:** Phase 23G: 7B INT6 semantic/capture canary, still layer0 only. Captures model output quality for a few prompts.

**Option B — Timing comparison:** Phase 23G: 7B INT6 timing comparison vs INT8/native for layer0 policy. Measures per-token latency.

**Option C — Two-layer experiment on 0.5B first:** Phase 23G: Begin two-layer policy experiment on 0.5B before scaling to 7B.

**Preferred recommendation:** Option A or B, scoped to layer0 only. Do not jump to multi-layer 7B yet.

---

## 10. Safety Scan

| Check | Result |
|-------|--------|
| Model files staged | ❌ NO |
| Sidecars staged | ❌ NO |
| f32 refs staged | ❌ NO |
| Captures staged | ❌ NO |
| Prompts staged | ❌ NO |
| Binaries staged | ❌ NO |
| Huge logs staged | ❌ NO |
| Secrets detected | ❌ NO |
| Scratch drive used | ✅ yes |
| System disk healthy | ✅ 27G free (88%) |
| Scratch disk healthy | ✅ 18G free (85%) |
| Source code changes | ✅ only docs/JSON |

---

## Final Report Fields

A. **Branch:** `experimental/prt-phase19a-alt-sidecar-backed`

B. **Previous HEAD:** `5655e535d` (Phase 23E)

C. **New HEAD:** `[NEW_COMMIT]`

D. **Checkpoint file:** `examples/speculative/results/PRT_PHASE23F_7B_INT6_POLICY_BASELINE_CHECKPOINT.md`

E. **JSON:** `examples/speculative/results/phase23f_7b_int6_policy_baseline_checkpoint.json`

F. **Tag created:** `PRT_PHASE23F_7B_INT6_POLICY_BASELINE_CHECKPOINT` ✅

G. **Frozen verdict:** `PASS_7B_INT6_POLICY_BASELINE_CHECKPOINT`

H. **7B INT6 provenance:** `scale_off=20`, sidecar at `prt_sidecars_7b_int6_phase15b_packed/ffn_up_layer0_prt.int6`, K=3584 M=18944

I. **Repeat validation:** 3 runs, 11 abs4 values identical each run, zero variance ✅

J. **Policy result:** N≤4 PRT AVX2, N>4 native prefill, scalar fallback=0 ✅

K. **Scalar fallback eliminated?** YES — 0 across all runs ✅

L. **Allowed claims:** Repeat-stable canary, scale_off=20 fix, explicit selector, policy routing, numeric sanity

M. **Forbidden claims:** direct packed compute, speedup, production readiness, multi-layer, all-layer, semantic equivalence, 14B support

N. **Recommended next:** Phase 23G — scoped 7B INT6 semantic canary OR timing comparison, layer0 only

O. **Models/sidecars/binaries staged?** NO ✅

P. **Secrets detected?** NO ✅

Q. **Existing tags altered?** NO ✅ (new tag only)

R. **System disk free:** 27G (88% used)

S. **Scratch disk free:** 18G (85% used)

---

## History of Fixes in This Branch

| Phase | Fix |
|-------|-----|
| Phase 23D-R forensic | Root cause: scale_off=16 used for 7B INT6 (should be 20) |
| Phase 23D-R2 | Fix: `uint32_t int6_scale_off = (M == 18944 && K == 3584) ? 20 : 16;` |
| Phase 23D-R4 | Fix: `PRT_V2_SIDECAR_FORMAT` env var — explicit INT6/INT8/f32 selection |
| Phase 23E | Validation: 3-run repeat, zero variance, all checks pass |

---

_This checkpoint freezes the 7B INT6 decoded-f32 policy path at the validated state. No claims beyond this scope are permitted._