# PRT Phase 12 Scope Review

**Date:** 2026-05-03
**Branch:** `experimental/prt-route-a-phase12`
**RC1 Tag:** `PRT_ROUTE_A_RC1` (frozen at `aaa5f290240dbaacfe355ca073bcbce49b18fde7`)

---

## Completed Phases

| Phase | Status | Key Finding |
|-------|--------|-------------|
| 12-PRD | ✓ Complete | 5 public-facing docs: overview, claims, reproduction notes, readiness checklist, RC1 summary |
| 12A | ✓ PASS | 24/24 prompts, 1.793x avg speedup, 0 collapse/repetition, 4/4 JSON valid |
| 12A-POST | ✓ Complete | Postmortem, claim update, broader validation documented |
| 12B | ✓ Complete | Speedup (~1.81x) attributed to PRT replacing FFN_UP; L12+L15 faster than pure PRT by ~5% |
| 12C | ✓ PASS | Manifest spec, example manifest (Qwen2.5-3B), validator script — no binaries committed |
| 12D | ✓ PASS | 10 policies tested; L12+L15 remains default; L11+L15 promising candidate; pure all36 competitive |

---

## Current Strongest Claim

> "PRT Route A + L12/L15 achieved ~1.79x average speedup across a 24-prompt broader validation suite on the tested local CPU/model setup, with clean counters, stable memory, 0 collapse/repetition failures, and 4/4 valid JSON outputs. Phase 12D further tested anchor policies and found no strong reason to replace L12+L15 as the default validated policy."

---

## Current Default Policy

```bash
--prt-mode 5700 --prt-force-native 12,15
```

- 34 PRT layers (94.4% coverage)
- 2 native anchor layers (L12 + L15)
- ~1.79x speedup on 24-prompt suite
- 0 callback_overwrites on all runs
- Memory stable, counters clean

---

## Current Status

- **Experimental documented fork** — published to `matthew-villnave/llama.cpp`
- **NOT production-ready** — do not claim otherwise
- **NOT universal** — tested on Qwen2.5-3B-Instruct-Q4_K_M only
- **NOT upstream-ready** — not submitted to llama.cpp upstream
- **Ready for:** further validation, hardening, documentation refinement

---

## Phase 12D Key Findings

- **L12+L15 (1.765x avg):** Default validated policy — robust, within 1% of best, passed Phase 12A 24-prompt suite
- **L11+L15 (1.780x avg):** Best average speedup in Phase 12D — promising candidate for Phase 12E
- **L13+L14 (1.770x avg):** 100% token-0 match rate — best quality policy
- **L12+L14+L15 (1.575x avg):** Triple anchor shows interference — unreliable
- **Pure all36 (1.696x avg):** Competitive, not a failure — but quality/speed worse than anchored policies
- **Single anchors (L12 or L15):** Viable but slower than dual-anchor policies

---

## Remaining Open Questions

### Validation scope
- Full 24-prompt validation of L11+L15 candidate (Phase 12E if chosen)
- Another model / model-size test (Phase 12E option)
- JSON/structured output at n > 50 (memory-limited on current machine)

### Technical
- Sidecar generation documentation improvement
- Direct runtime manifest integration (`--prt-manifest` flag)
- Upstream sync/rebase strategy (keep RC1 frozen, rebase separate branch)
- Performance profiling beyond coarse timing (per-layer FFN_UP isolation)

### Packaging / polish
- CLI `--prt-sidecar-dir` configurable flag (currently hardcoded to `/tmp/prt_sidecars/`)
- Better error messages for missing/mismatched sidecars
- Installation/build documentation

### Research
- Why L12 and L15 specifically — structural explanation
- Whether the ~0.85% L11+L15 advantage is real or noise
- Generalization to 7B+ models

---

## Recommended Next Phase

**Option A: Phase 12E — L11+L15 Candidate Validation**
- Run L11+L15 on the full 24-prompt Phase 12A suite
- If it passes and beats L12+L15, consider updating default
- If it fails, L12+L15 stays as-is
- This is the rigorous technical path

**Option B: Phase 12F — Full Phase 12 Wrap-Up and Tag**
- Consolidate all Phase 12 documentation
- Tag `PRT_PHASE12` checkpoint
- Review repo for any remaining cleanup
- Prepare public-facing summary

**Recommendation:** If the goal is technical rigor, do Phase 12E. If the goal is a stable public milestone, do Phase 12F wrap-up first, then Phase 12E.

---

## What's Documented and Where

| Document | Location | Purpose |
|----------|----------|---------|
| `PRT_OVERVIEW.md` | examples/speculative/ | High-level PRT explanation |
| `PRT_ROUTE_A_RC1_SUMMARY.md` | examples/speculative/ | Tag, branch, results, counters |
| `PRT_CLAIMS.md` | examples/speculative/ | Allowed and forbidden claims |
| `PRT_REPRODUCTION_NOTES.md` | examples/speculative/ | Setup, flags, sidecar notes |
| `PRT_PUBLIC_READINESS_CHECKLIST.md` | examples/speculative/ | Pre-public safety checklist |
| `PRT_PHASE12_PLAN.md` | examples/speculative/ | Full Phase 12 plan |
| `PRT_PHASE12B_PLAN.md` | examples/speculative/ | Speed attribution plan |
| `PRT_SIDECAR_MANIFEST_SPEC.md` | examples/speculative/ | Sidecar format and policy spec |
| `prt_sidecar_manifest.example.json` | examples/speculative/ | Example manifest |
| `prt_validate_sidecars.py` | examples/speculative/ | Standalone validator |
| `PRT_PHASE12A_BROADER_VALIDATION.md` | results/ | 24-prompt table |
| `phase12a_results.json` | results/ | Structured 24-prompt data |
| `PRT_PHASE12A_POSTMORTEM.md` | results/ | Phase 12A analysis |
| `PRT_PHASE12B_SPEED_ATTRIBUTION.md` | results/ | 3-prompt timing, pure vs anchored |
| `phase12b_timing.json` | results/ | Structured timing data |
| `PRT_PHASE12B_PROFILING_NOTES.md` | results/ | Profiling methodology and limits |
| `PRT_PHASE12C_SIDECAR_PACKAGING.md` | results/ | Packaging report |
| `PRT_PHASE12D_ANCHOR_POLICY_SEARCH.md` | results/ | Full policy table |
| `phase12d_anchor_results.json` | results/ | Structured policy data |
| `PRT_PHASE12D_POLICY_RANKING.md` | results/ | Ranked policies |
| `PRT_PHASE12D_FAILURES.md` | results/ | Failure report |
| `PRT_PHASE12D_VERDICT.md` | results/ | Verdict with A-M metrics |
| `PRT_PHASE12D_POSTMORTEM.md` | results/ | This document |
| `PRT_PHASE12_SCOPE_REVIEW.md` | examples/speculative/ | This document |

---

*Scope review by ELVIS for Matthew Villnave / The ForgeHQ*