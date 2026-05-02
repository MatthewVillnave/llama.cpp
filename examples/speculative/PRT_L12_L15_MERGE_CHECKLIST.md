# PRT L12+L15 Merge Checklist

**Branch:** phase11bm-archived / PRT_ROUTE_A_RC1 candidate
**Date:** 2026-05-02
**Status:** Ready for review

---

## Pre-Merge Checklist

### Core Behavior
- [x] Route A replaces native FFN_UP matmul in graph (not callback)
- [x] callback_overwrites = 0 confirmed on all L12+L15 runs
- [x] native_fallback_calls = 16 (2 layers × 8) for L12+L15 at n=100
- [x] prt_true_replacement_calls = 3706 (n=100) / 2006 (n=50) correct
- [x] identity_fallback_calls = 0
- [x] No callback intercept for mode 5700

### Mode Flags
- [x] Mode 0 (native) works unchanged
- [x] Mode 5700 (pure Route A) works
- [x] `--prt-force-native 12,15` correctly parsed
- [x] Force-native takes priority over PRT in build_ffn

### Counters
- [x] All 5 counters have getter functions
- [x] All counters printed in phase10e0 output
- [x] Counter values match expected behavior
- [x] Sidecar checksums printed (L0, L35 at minimum)

### Sidecar Loading
- [x] POSIX/mmap loader only (no fopen in benchmark path)
- [x] Sidecar orientation: `sidecar[j*hidden+k]`
- [x] Missing sidecar: fallback to native (silent — recommend fix)

### Quality
- [x] Known failure "Once upon a time in a" fixed at n=100
- [x] Token 0 matches native on all tested prompts
- [x] No collapse/repetition in any tested output
- [x] Coherent output on narrative, code, factual prompts

### Speed
- [x] Average speedup ~1.84x across mini-suite
- [x] n=100 stable (5/6 prompts)
- [x] n=50 JSON completed

### Memory
- [x] RAM stable across 12 benchmark runs
- [x] No memory growth detected
- [x] Available RAM stayed at 11-12Gi

### Documentation
- [x] RC verdict: PRT_PHASE11BM_RC_VERDICT.md
- [x] Policy doc: PRT_ROUTE_A_L12_L15_POLICY.md
- [x] Results: PRT_PHASE11BM_RESULTS.md
- [x] Allowed/forbidden claims: CLAIMS_ALLOWED/FORBIDDEN_PHASE11BM.md
- [x] Next phase plan: NEXT_PHASE_11BN_PLAN.md
- [x] One-page summary: PRT_ROUTE_A_RC_ONE_PAGE.md
- [x] Code review: PRT_PHASE11BO_CODE_REVIEW.md (this file set)
- [x] Modified files: PRT_ROUTE_A_MODIFIED_FILES.md
- [x] Counters/safety: PRT_COUNTERS_AND_SAFETY_GATES.md

---

## Recommended Fixes (Before Production)

### HIGH PRIORITY
1. **Missing sidecar startup check:** If mode 5700 and any sidecar missing, print error and exit. Currently silent fallback.
2. **native_ffn_up_calls counter:** Remove or document that it's always 0 in Route A. Confusing as-is.

### MEDIUM PRIORITY
3. **Dead code removal:** `load_sidecar_fopen()` in phase10e0 is unused. Remove to avoid confusion.
4. **Debug print cleanup:** `[PRT-11BB-AUTH]` prints in build_ffn are verbose. Add runtime log level.

### LOW PRIORITY
5. **L12/L15 checksums not printed:** phase10e0 only prints L0 and L35. Print L12 and L15 too for full coverage.
6. **Additional fallback pairs documented:** L13+L14 also works — document as alternative.

---

## Files to Submit for Merge

### Core Library Changes
```
src/llama-graph.cpp      (~50 lines: counters + build_ffn Route A hook)
src/llama.cpp            (~80 lines: force-native API + counter getters)
include/llama.h         (no new symbols needed — existing API covers it)
```

### Speculative Examples
```
examples/speculative/prt_graph_replace.h    (custom op + build_prt_ffn_up)
examples/speculative/prt_avx2_kernel.h      (AVX2 SIMD kernel)
examples/speculative/phase10e0_layer0_replacement.cpp  (benchmark binary)
```

### Documentation
```
examples/speculative/PRT_ROUTE_A_L12_L15_POLICY.md
examples/speculative/PRT_PHASE11BM_RC_VERDICT.md
examples/speculative/PRT_PHASE11BM_RESULTS.md
examples/speculative/PRT_ROUTE_A_RC_ONE_PAGE.md
```

---

## Merge Path

1. **Tag as PRT_ROUTE_A_RC1** — freeze working branch
2. **Code review** — reviewer checks `src/llama-graph.cpp build_ffn()` logic
3. **Test on clean machine** — re-run Phase 11BN mini-suite to confirm
4. **Merge to experimental** — PR merge with all docs
5. **Continued validation** — JSON/structured full suite on >32GB machine

---

## Sign-Off

| Milestone | Status |
|-----------|--------|
| Mini-suite passes | ✓ DONE |
| Code review complete | ✓ DONE |
| Counters clean | ✓ DONE |
| Documentation complete | ✓ DONE |
| Recommended fixes identified | ✓ DONE |
| **Ready for merge?** | **YES — CONDITIONAL** |

**Conditional:** Missing-sidecar startup check should be added before merge to main. Current behavior (silent fallback) could cause silent quality regressions if sidecars are accidentally missing.

---

*End of Merge Checklist*
