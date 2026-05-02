# PRT Route A RC1: Final Merge Readiness

**Tag:** `prt-route-a-rc1`
**Date:** 2026-05-02
**Branch:** phase11bm-archived (frozen)
**Status:** MERGE-READY ✓

---

## Journey to RC1

| Phase | What Happened |
|-------|-------------|
| Phase 11BB | Route A custom op discovery — true graph replacement |
| Phase 11BD | Counters added, callback path separated |
| Phase 11BG | Force-native mask for selective layer fallback |
| Phase 11BH | L12+L15 discovered as fix for "Once upon a time in a" failure |
| Phase 11BJ | Phase 11BE quality failure, L12+L15 as partial pass |
| Phase 11BL | n=100 stable after Ollama memory cleanup |
| Phase 11BM | ~1.84x speedup confirmed, counters clean |
| Phase 11BN | Full mini-suite: 6/6 prompts, all counters clean |
| Phase 11BO | Code review: merge blocker = silent missing-sidecar fallback |
| Phase 11BP | Sidecar validation patch: missing sidecar → FATAL error |

---

## What's in RC1

### Core Implementation
- `src/llama-graph.cpp` — `build_ffn()` with Route A + force-native priority
- `src/llama.cpp` — force-native API + counter getters
- `examples/speculative/prt_graph_replace.h` — GGML custom op + AVX2 kernel
- `examples/speculative/prt_avx2_kernel.h` — AVX2 SIMD matmul

### Benchmark Binary
- `examples/speculative/phase10e0_layer0_replacement.cpp`
- **With Phase 11BP validation patch**

### Policy
- L12+L15 native fallback for known-failure prompt recovery
- ~1.84x speedup vs native at n=100
- 34/36 layers PRT, 2/36 native

---

## Validation Summary (Phase 11BN + 11BP)

| Metric | Result |
|--------|--------|
| Prompts tested | 6 |
| Average speedup | ~1.84x |
| callback_overwrites | 0 (all runs) |
| native_fallback_calls | 16 (2×8, expected) |
| identity_fallback_calls | 0 |
| prt_true_replacement_calls | 3706 (n=100) |
| Known failure fixed | ✓ |
| No collapse/repetition | ✓ |
| Memory stable | ✓ |
| Missing sidecar = FATAL | ✓ |

---

## Safety Gates (All Passing)

| Gate | Mechanism | Status |
|------|-----------|--------|
| Callback bypass | `mode < 5700` gate | ✓ PASS |
| Force-native priority | `build_ffn` checks force-native first | ✓ PASS |
| Missing sidecar = FATAL | `validate_prt_sidecars()` | ✓ PASS |
| Identity fallback = 0 | Counter, not triggered | ✓ PASS |
| Native mode untouched | `mode=0` skips validation | ✓ PASS |

---

## Missing-Sidecar Behavior: Before vs After

| Scenario | Before Phase 11BP | After Phase 11BP |
|----------|------------------|------------------|
| Required sidecar missing | Silent fallback → native | `[PRT-ERROR] FATAL` + exit 1 |
| Force-native layer sidecar missing | N/A (unused anyway) | No error (skipped) |
| Non-force-native layer missing | Silent quality degradation | `[PRT-ERROR] Layer N: MISSING` |
| All sidecars present | Proceed normally | Proceed + checksums printed |

---

## Documentation

| File | Purpose |
|------|---------|
| `PRT_PHASE11BM_RC_VERDICT.md` | Final verdict |
| `PRT_ROUTE_A_L12_L15_POLICY.md` | Why L12+L15, expected counters |
| `PRT_PHASE11BM_RESULTS.md` | Timing and quality tables |
| `CLAIMS_ALLOWED_PHASE11BM.md` | Allowed claims |
| `CLAIMS_FORBIDDEN_PHASE11BM.md` | Forbidden claims |
| `PRT_ROUTE_A_RC_ONE_PAGE.md` | Plain-language summary |
| `PRT_PHASE11BO_CODE_REVIEW.md` | Full code audit |
| `PRT_ROUTE_A_MODIFIED_FILES.md` | Modified files + risk levels |
| `PRT_COUNTERS_AND_SAFETY_GATES.md` | Counters, gates, fail-safes |
| `PRT_L12_L15_MERGE_CHECKLIST.md` | Pre-merge checklist |
| `PRT_RC1_TAG_NOTES.md` | How to use RC1 tag |
| `PRT_PHASE11BP_SIDECAR_VALIDATION.md` | Phase 11BP patch |
| `PRT_MISSING_SIDECAR_FAILURE_TEST.md` | Missing sidecar test |
| **THIS FILE** | Final merge readiness |

---

## Recommended Fixes (Post-Merge)

Not blocking merge, but should be addressed:

| Priority | Fix |
|----------|-----|
| MEDIUM | `native_ffn_up_calls` always 0 in Route A — remove or document |
| LOW | `load_sidecar_fopen` dead code — remove |
| LOW | Debug print `[PRT-11BB-AUTH]` — add log level |

---

## Final Verdict

**A. Missing sidecars fail loudly?** ✓ **YES**

**B. Native mode unaffected?** ✓ **YES**

**C. L12/L15 policy still works?** ✓ **YES**

**D. Counters still clean?** ✓ **YES**

**E. Ready to tag PRT_ROUTE_A_RC1?** ✓ **YES**

**F. Ready to merge to experimental?** ✓ **YES**

---

## How to Merge

```bash
# 1. Tag the RC
git tag prt-route-a-rc1

# 2. Code review of build_ffn() and validate_prt_sidecars()

# 3. Re-run Phase 11BN on clean machine
./build/bin/llama-prt-posix -m model.gguf -p "Once upon a time in a" -n 100 \
  --prt-mode 5700 --prt-force-native "12,15" \
  2>&1 | grep -E "11BP|callback_overwrites|native_fallback|prt_true_replacement"

# Expected:
# [PRT-11BP] VALIDATION PASSED
# [11BD] callback_overwrites: 0
# [11BD] native_fallback_calls: 16
# [11BD] prt_true_replacement_calls: 3706

# 4. Merge to experimental branch
git merge prt-route-a-rc1
```

---

## Allowed Claims for RC1

- "Release-candidate experimental branch"
- "Production-candidate fallback policy"
- "Approximately 1.84x speedup on tested prompts"
- "Callbacks disabled, no overwrite path"
- "Missing sidecar fails loudly at startup"
- "Validated on 6 prompts including narrative, code, factual, JSON"

## Forbidden Claims for RC1

- "Production-ready"
- "Universal 2x speedup"
- "Validated on all prompts"

---

*End of PRT Route A RC1 Final Merge Readiness*
