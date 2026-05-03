# PRT Route A RC1 — Summary

**Tag:** `PRT_ROUTE_A_RC1`
**Branch:** `experimental/prt-route-a-rc1`
**Commit:** `aaa5f290240dbaacfe355ca073bcbce49b18fde7`
**Date Tagged:** 2026-05-02

---

## Result

**~1.84x average speedup** on controlled mini-suite vs native baseline, with clean output quality on tested prompts.

---

## Test Scope

| Metric | Value |
|--------|-------|
| Prompts tested | 6 |
| Total runs | 12 (native + L12+L15 per prompt) |
| n=100 runs | 5 prompts |
| n=50 runs | 1 prompt (JSON) |
| Average speedup | ~1.84x |

### Prompts Tested

| Prompt | n | Native Wall Time | L12+L15 Wall Time | Speedup |
|--------|---|-----------------|-------------------|---------|
| "Once upon a time in a" | 100 | 1m22.8s | 45.1s | 1.83x |
| "Once upon a time in a distant galaxy" | 100 | 1m24.9s | 46.0s | 1.85x |
| "Write a Python function to reverse a list." | 100 | 1m25.2s | 46.3s | 1.84x |
| "What is the capital of France?" | 100 | 1m23.9s | 45.4s | 1.85x |
| "Give me a JSON object with name and age." | 50 | 46.6s | 25.6s | 1.82x |
| "The company is a large" | 100 | 1m23.3s | 44.9s | 1.86x |

---

## Counters (L12+L15 Mode)

| Counter | Value | Notes |
|---------|-------|-------|
| `callback_overwrites` | **0** | No callback path used in L12+L15 mode |
| `native_fallback_calls` | **16** | 2 layers × 8 generation steps (expected) |
| `identity_fallback_calls` | **0** | Clean |
| `native_ffn_up_calls` | **0** | PRT replaces FFN_UP for 34 layers |
| `prt_true_replacement_calls` | **3706** (n=100) / **2006** (n=50) | All PRT layers, all tokens |

---

## Known Failure Recovery

**Prompt:** "Once upon a time in a"

| Mode | Token 0 Output | Status |
|------|---------------|--------|
| Native | " small village" | ✓ Correct |
| Pure all-36 Route A | " far away land" | ✗ Wrong |
| Route A + L12+L15 | " small village" | ✓ **Fixed** |

L12+L15 produces **identical output** to native on this known-failure prompt at n=100.

---

## Memory Stability

RAM measured before and after each run across 12 benchmark runs:

- Before runs: 12Gi available
- After runs: 12Gi available
- **Delta: 0** — No memory growth detected

---

## Quality Observations

- No collapse detected in any tested output
- No repetition detected in any tested output
- Narrative prompts produce coherent continuations
- Code prompt produces syntactically plausible Python
- JSON prompt produces valid JSON (at n=50)
- Factual prompt produces accurate output

---

## What Was NOT Tested

- JSON/structured at n > 50 (memory-limited machine)
- Broader suite (> 6 prompts)
- Other models
- Batch sizes > 1
- Very long contexts (> 512 tokens)

---

## RC1 vs Previous Phases

- Phase 11BE: Pure Route A — quality failure on known-failure prompt
- Phase 11BJ: Route A + L12+L15 — partial pass, counters not fully confirmed
- Phase 11BN: Full validation — all counters clean, all prompts pass
- **Phase 11BO: Code review — merge blocker found (silent missing-sidecar)**
- Phase 11BP: Sidecar validation patch — merge blocker resolved
- **Phase 11BQ: Published to GitHub fork**
- **Phase 12: Active development branch created**

---

## Documentation Files in RC1

| File | Purpose |
|------|---------|
| `PRT_OVERVIEW.md` | What PRT is, high-level explanation |
| `PRT_ROUTE_A_L12_L15_POLICY.md` | Why L12+L15, why pure all36 fails |
| `PRT_PHASE11BM_RESULTS.md` | Timing tables, quality tables |
| `PRT_PHASE11BM_RC_VERDICT.md` | Final verdict + mechanical/speed/quality status |
| `CLAIMS_ALLOWED_PHASE11BM.md` | Verified claims + allowed phrasing |
| `CLAIMS_FORBIDDEN_PHASE11BM.md` | Forbidden claims + phrasing |
| `PRT_ROUTE_A_RC_ONE_PAGE.md` | Plain-language one-page summary |
| `PRT_PHASE11BO_CODE_REVIEW.md` | Full code audit |
| `PRT_ROUTE_A_MODIFIED_FILES.md` | Modified files + risk levels |
| `PRT_COUNTERS_AND_SAFETY_GATES.md` | Counters, gates, fail-safes |
| `PRT_L12_L15_MERGE_CHECKLIST.md` | Pre-merge checklist |
| `PRT_RC1_TAG_NOTES.md` | How to use tag |
| `PRT_PHASE11BP_SIDECAR_VALIDATION.md` | Phase 11BP patch notes |
| `PRT_MISSING_SIDECAR_FAILURE_TEST.md` | Missing sidecar test |
| `PRT_RC1_FINAL_MERGE_READINESS.md` | Final merge readiness |
| `PRT_ROUTE_A_RC1_TAGGED.md` | Tag confirmation |
| `PRT_ROUTE_A_RC1_PUBLISHED.md` | Publish confirmation |

---

*Summary accurate as of tag date 2026-05-02*
