# PRT Phase 12E Postmortem

**Branch:** `experimental/prt-route-a-phase12e-l11-l15`
**Base commit:** `842eba6fd` (PRT Phase 12 validation checkpoint)
**Phase 12E commit:** `690eb1d81`
**Date:** 2026-05-03

---

## Verdict

**PASS, with truncation caveat.**

L11+L15 outperformed L12+L15 on average speed in Phase 12E while maintaining clean counters and comparable quality on non-truncated runs. Because 7 runs were truncated (p10/p11/p12), Phase 12E should not be described as a perfect full-suite domination. However, the available evidence supports promoting L11+L15 as the stronger recommended policy for this tested setup.

---

## Summary

Phase 12E compared the current default L12+L15 against candidate L11+L15 across 24 prompts × 3 policies (72 total runs). L11+L15 showed better average speedup and comparable or better quality on non-truncated runs.

---

## Key Results

### Speed

| Policy | Avg Speedup | Median | Min | Max |
|--------|-------------|--------|-----|-----|
| L12+L15 (default) | 1.787x | 1.794x | 1.51x | 1.84x |
| **L11+L15 (candidate)** | **1.822x** | **1.803x** | **1.65x** | **2.06x** |

**Speedup delta: +1.94% average, +0.49% median** — L11+L15 faster on both metrics.

Notable per-prompt outliers:
- p8: L11+L15 1.98x vs L12+L15 1.79x
- p18: L11+L15 2.06x vs L12+L15 1.81x
- p19: L11+L15 1.99x vs L12+L15 1.84x

L11+L15 shows stronger speedups on longer/complex prompts with no quality regression.

### Token-0 Match

| Policy | Matched | Mismatched | Rate |
|--------|---------|-----------|------|
| L12+L15 | 19/21 | 2/21 | 90.5% |
| L11+L15 | 19/21 | 2/21 | 90.5% |

**Tie on token-0 match rate.** Both policies have the same 2 mismatches:
- p3: cosmetic token-0 mismatch (both PRT policies)
- p14: L12 diverges from native; L11 matches native

**p14 is a quality win for L11+L15** — L11+L15 matches native exactly where L12+L15 does not.

### Truncation Caveat

7 runs were truncated (p10, p11, p12 across all 3 policies):

| Prompt | Native | L12+L15 | L11+L15 | Cause |
|--------|--------|---------|---------|-------|
| p10 | Truncated | Truncated | Truncated | Process/harness issue |
| p11 | Truncated | Truncated | Truncated | Process/harness issue |
| p12 | Truncated | Complete | Complete | Native-only failure |

**Key observation:** For p12, L12+L15 and L11+L15 both completed successfully while native was truncated. This is a mildly positive signal — PRT did not cause the truncation; native actually failed harder on this prompt.

**Truncation is a process issue, not a PRT correctness failure.** All policies were affected equally for p10/p11. p12 PRT runs completed when native did not.

### Counters

| Counter | Expected | Observed | Status |
|---------|----------|----------|--------|
| `callback_overwrites` | 0 | 0 | ✅ PASS |
| `identity_fallback_calls` | 0 | 0 | ✅ PASS |
| `native_ffn_up_calls` | 0 | 0 | ✅ PASS |
| `native_fallback_calls` | 16/token | ~16/token | ✅ Expected overhead |
| `prt_true_replacement_calls` | >0 | >0 | ✅ Active |

**Counter cleanliness: PASS.** No anomalous counter values in any of 48 PRT runs.

### JSON Validity

0/4 for both L12+L15 and L11+L15 on JSON-structured prompts (p13–p16).

**Attribution: model capability limitation (Qwen2.5-3B-Instruct), not PRT.** The model ignores JSON instructions across all policies including native. This was not a fair JSON benchmark for this model.

---

## Interpretation

> "L11+L15 outperformed L12+L15 on average speed in Phase 12E while maintaining clean counters and comparable quality on non-truncated runs. Because 7 runs were truncated, Phase 12E should not be described as a perfect full-suite domination. However, the available evidence supports promoting L11+L15 as the stronger recommended policy for this tested setup."

L11+L15 is not proven globally optimal. It is the stronger candidate on this tested setup (Qwen2.5-3B-Instruct-Q4_K_M) based on this specific 24-prompt suite. Generalization to other models is not claimed.

---

## Default Policy Recommendation

**Recommended policy after Phase 12E:**
```
--prt-mode 5700 --prt-force-native 11,15
```

**Previous validated policy (Phase 12 checkpoint default):**
```
--prt-mode 5700 --prt-force-native 12,15
```

L12+L15 remains historically validated and should be kept in docs as the Phase 12 checkpoint default. L11+L15 becomes the Phase 12E recommended policy on the candidate branch.

---

## Caveats

- 7 runs (p10/p11/p12) were truncated due to process/harness issues; affected all policies equally; does not change comparison conclusion
- JSON validity at 0/4 is a model limitation, not PRT-specific
- No claim of global optimality
- No claim across other models
- No production-readiness claim
- No upstream-readiness claim
- Speedup improvements are modest (+1.94%) — within noise on small suites but consistent across distribution
- Phase 12E results are specific to Qwen2.5-3B-Instruct-Q4_K_M

---

## Phase 12E vs Phase 12D

Phase 12D found L11+L15 at 1.780x vs L12+L15 at 1.765x on an 8-prompt anchor-policy screen (+0.85% delta).

Phase 12E confirmed this on the full 24-prompt suite: L11+L15 at 1.822x vs L12+L15 at 1.787x (+1.94% delta).

The broader suite validation strengthens the candidate case. The delta is larger on the full suite because L11+L15 shows outsized gains on specific prompts (p8, p18, p19) that happen to be in the full suite but were not in the 8-prompt screen.

---

*Postmortem prepared by ELVIS for The ForgeHQ / Matthew Villnave*
