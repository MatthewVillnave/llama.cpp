# PRT Phase 24Z: Fresh 3B Timing After INT8 /127 Correctness Fix

## Status: COMPLETE ✅

**Date:** 2026-05-20
**Branch:** experimental/prt-phase19a-alt-sidecar-backed
**HEAD:** 8a8030e62
**Scale-fix commit:** 8de123d54

---

## Executive Summary

First real timing measurement of corrected PRT custom op. PRT layer0 is **~36% slower than native** on Qwen2.5-3B with n_threads=1, c=4, n=8, temp=0. Correctness confirmed — outputs match native identically. Previous timing data (Phase 24S, etc.) was invalid, measuring broken op.

---

## Timing Results

### Raw Numbers (real time, seconds)

| Run | Native | PRT |
|-----|--------|-----|
| 1 | 2.90 | 3.44 |
| 2 | 2.39 | 3.48 |
| 3 | 2.34 | 3.46 |

**Native avg:** (2.90 + 2.39 + 2.34) / 3 = **2.54s**
**PRT avg:** (3.44 + 3.48 + 3.46) / 3 = **3.46s**
**Ratio (native/PRT):** 2.54 / 3.46 = **0.73**
**PRT slower by:** 1 / 0.73 − 1 = **~36%**

### user+sys breakdown

| Run | Native (user+sys) | PRT (user+sys) |
|-----|--------------------|----------------|
| 1 | 1.95 + 0.52 = 2.47 | 2.77 + 0.68 = 3.45 |
| 2 | 2.07 + 0.32 = 2.39 | 2.80 + 0.68 = 3.48 |
| 3 | 1.97 + 0.36 = 2.33 | 2.76 + 0.70 = 3.46 |

The sys time is consistently higher for PRT (~0.68-0.70s vs ~0.32-0.52s) — likely sidecar file I/O showing up as system time.

---

## Correctness During Timing

### n=4 comparison
```
Native: The capital of France is Paris. Paris is
PRT:    The capital of France is Paris. Paris is
```
**IDENTICAL ✅**

### n=8 comparison
```
Native: The capital of France is Paris. Paris is located in the north
PRT:    The capital of France is Paris. Paris is located in the north
```
**IDENTICAL ✅**

### Proof chain (all 3 PRT runs)
- route=ggml_op (custom op, no FALLBACK) ✅
- avx2 backend active — no scalar fallback ✅
- Custom op kernel entered for all decode tokens ✅
- Kernel wrote valid floats (abs4 values, no NaN/Inf) ✅
- Output matches native ✅

---

## Per-Token Kernel Cost

From PRT r1 logs:
- **Prefill kernel (N=2):** ~28ms (K=2048, M=11008, N=2)
- **Decode kernel (N=1):** ~82ms per token (K=2048, M=11008, N=1)

Total kernel time for n=8: 1 prefill + 8 decode tokens × ~82ms/token = ~656ms
Native prefill+decode: ~247ms average real time total

Overhead breakdown estimate:
- Kernel compute: ~656ms (from PRT logs)
- Native equivalent compute: ~400ms (estimated)
- Additional I/O overhead: sys time ~0.18s extra

---

## Interpretation

**PRT is slower than native for this configuration.** This is the first clean measurement post-fix. The overhead is real and measurable.

**Key observations:**
1. Single-threaded measurement (n_threads=1) — results may differ with multithread
2. PRT kernel is fast per call (~82ms N=1) but adds up over 8 decode tokens
3. Sys overhead is higher for PRT (~0.18s more) — sidecar decode I/O
4. The custom op path adds ~36% wall-clock time for this workload

**Do not extrapolate:**
- This is one prompt, one model, n_threads=1
- Not 7B, not all-layers, not production workloads
- Not representative of multithread batch performance

---

## Verdict

| Check | Result |
|-------|--------|
| Fresh timing captured | ✅ 3 native + 3 PRT runs |
| 3B correctness confirmed | ✅ n=4 and n=8 output match native |
| Proof chain complete | ✅ custom op, no fallback, avx2 |
| PRT slower than native | ✅ ~36% slower |
| Verdict | `PARTIAL_PRT_SLOWER_AFTER_FIX` |

---

## Recommended Next

**Phase 25A:** Decide whether to pause PRT speed work or investigate the overhead. Options:
1. **Pause speed work** — PRT is slower than native for single-thread layer0; document as research result and move on
2. **Profile the overhead** — isolate decode I/O cost vs compute cost; test with mmap'd sidecar to remove I/O
3. **Multithread sanity** — re-run with n_threads=4 or higher to see if PRT advantage emerges with batching

**If proceeding:** Run Phase 25A with second prompt and c=32 sanity to get more representative numbers before any optimization work.

---

## Machine State

| Field | Value |
|-------|-------|
| System disk free | 145G |
| Scratch disk free | ~0.5G used / 7.7G tmpfs |
| RAM available | 13Gi |
| Swap | 3.8Gi free |
| Stale processes | None |

---

## Models/Sidecars/F32 Refs Staged?
No new files staged.

## Secrets Detected?
None.

## Tags Touched?
None.