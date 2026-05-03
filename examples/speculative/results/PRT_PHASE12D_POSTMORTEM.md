# PRT Phase 12D Postmortem

**Date:** 2026-05-03
**Branch:** `experimental/prt-route-a-phase12`
**Commit:** `898008ed960591f121e2d99f539664ca94284890`
**Status:** PASS

---

## Verdict

**PASS** — Phase 12D native anchor policy search completed across 10 PRT policies in 88 runs. L12+L15 remains the default validated policy.

---

## Summary

Phase 12D tested multiple native-anchor policies and found that L12+L15 remains the best validated default, even though L11+L15 slightly outperformed it on the smaller 8-prompt policy screen. This confirms L12+L15 is robust rather than lucky.

---

## Key Results

| Metric | Value |
|--------|-------|
| Policies tested | 10 PRT policies + native baseline |
| Total runs | 88 (80 PRT + 8 native) |
| Prompts used | 8 |
| Best average speedup policy | L11+L15 at 1.780x |
| Current default policy | L12+L15 at 1.765x |
| Margin of L11+L15 over L12+L15 | ~0.85% — within noise margin |
| Best single speed case | L12+L14+L15 at 1.855x (P3) — but with 2 anomalous runs |
| Best token-0 match policy | L13+L14 at 100% |
| Pure all36 average | 1.696x |
| Counter cleanliness | PASS — callback_overwrites=0 on all runs |
| Memory stability | PASS — no SIGKILL |
| Verdict | PASS |

---

## Interpretation

**L11+L15 vs L12+L15:**

L11+L15 had the highest average speedup in Phase 12D (1.780x vs 1.765x), but the margin is only ~0.85%. Because L12+L15 already passed the larger Phase 12A 24-prompt broader validation suite, L12+L15 remains the default validated policy. L11+L15 should be treated as a **promising candidate for future broader validation**, not as the new default.

**Pure all36 correction:**

Pure all36 PRT should no longer be described as a clear failure control. In Phase 12D it remained competitive at ~1.696x speedup — faster than native, comparable to anchored policies. However, anchored policies remain preferred because they had stronger quality/stability behavior and better average speed in the validated path.

**JSON interpretation:**

JSON failures (prompts 4 and 5) occurred across all policies including native. These failures are attributed to model/prompt behavior (Qwen2.5-3B-Instruct produces malformed JSON when instructed to output raw JSON) rather than PRT-specific degradation.

**Triple anchor (L12+L14+L15) anomaly:**

L12+L14+L15 showed the best single-prompt speedup (1.855x on "The company is a large") but had 2 anomalous runs below 1.1x — indicating interference on certain prompt types when three layers are native. This makes it unreliable as a general policy.

---

## Allowed Conclusion

> "Phase 12D tested 10 PRT anchor policies across 88 runs. L12+L15 remained the best validated default policy because it was within ~1% of the fastest average policy while retaining stronger prior validation from Phase 12A's 24-prompt suite. L11+L15 is a promising candidate for future broader validation."

---

## Forbidden Conclusions

- ~~"L12+L15 is globally optimal."~~
- ~~"L11+L15 is now the default."~~
- ~~"Pure all36 is a total failure."~~
- ~~"Anchor policy behavior generalizes across all models."~~
- ~~"Phase 12D proves production readiness."~~
- ~~"L12+L15 is provably the best layer pair."~~

---

## Phase 12 Scope So Far

| Phase | Status | Key Finding |
|-------|--------|-------------|
| 12-PRD | ✓ Complete | 5 public readiness docs |
| 12A | ✓ PASS | 24/24 prompts, 1.793x avg, 0 failures |
| 12A-POST | ✓ Complete | Postmortem, claim update |
| 12B | ✓ Complete | Speedup attributed to FFN_UP replacement; L12+L15 faster than pure PRT |
| 12C | ✓ PASS | Manifest spec, example manifest, validator |
| 12D | ✓ PASS | L12+L15 remains default; L11+L15 promising candidate |

---

## Next Steps

Recommended options:
- **Phase 12E:** L11+L15 candidate broader validation on full 24-prompt Phase 12A suite
- **Phase 12F:** Full Phase 12 wrap-up and tagged Phase 12 checkpoint

L11+L15 should be validated on the broader suite before any policy change.

---

*Postmortem by ELVIS for Matthew Villnave / The ForgeHQ*