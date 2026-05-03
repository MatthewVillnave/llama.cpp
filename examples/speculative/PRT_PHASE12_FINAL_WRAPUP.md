# PRT Phase 12 Final Wrap-Up

**Date:** 2026-05-03
**Branch:** `experimental/prt-route-a-phase12`
**Tag:** `PRT_PHASE12_VALIDATION_CHECKPOINT`
**Commit:** `fff7811b43c01b006fa4c86726001ed4641412fc`

---

## Status

Phase 12 validation checkpoint — complete.

---

## Frozen RC1

`PRT_ROUTE_A_RC1` remains frozen and untouched. It is the permanent RC1 checkpoint at `aaa5f290240dbaacfe355ca073bcbce49b18fde7`. This document does not modify it.

---

## Default Validated Policy

```bash
--prt-mode 5700 --prt-force-native 12,15
```

- 34 PRT layers (94.4% coverage)
- 2 native anchor layers (L12 + L15)
- Average speedup: ~1.79x on 24-prompt suite

---

## Completed Phases

| # | Phase | Status | Key Finding |
|---|-------|--------|-------------|
| 1 | 12-PRD — Public readiness documentation | ✓ Complete | 5 docs: overview, claims, reproduction notes, readiness checklist, RC1 summary |
| 2 | 12A — 24-prompt broader validation | ✓ PASS | 1.793x avg, 0 failures, 4/4 JSON valid |
| 3 | 12A-POST — Postmortem and claim update | ✓ Complete | Postmortem, claim update, broader validation documented |
| 4 | 12B — Speed attribution | ✓ Complete | ~1.81x; speedup from FFN_UP replacement; L12+L15 faster than pure PRT |
| 5 | 12C — Sidecar packaging | ✓ PASS | Manifest spec, example manifest, validator script — no binaries committed |
| 6 | 12D — Native anchor policy search | ✓ PASS | 88 runs, 10 policies; L12+L15 remains default; L11+L15 promising candidate |
| 7 | 12D-POST — Policy interpretation and scope review | ✓ Complete | Interpretation locked; scope review; claims updated |
| 8 | 12F — Final wrap-up and checkpoint | ← This document | |

---

## Key Validation Result (Phase 12A)

| Metric | Value |
|--------|-------|
| Prompts completed | 24/24 |
| Paired native/PRT runs | 24 pairs |
| Average speedup | 1.793x |
| Median speedup | 1.801x |
| Min/max speedup | 1.51x / 1.95x |
| Token-0 matches | 21/24 (87.5%) |
| First-8-token matches | 15/24 (62.5%) |
| Coherent outputs | 23/24 (95.8%) |
| Collapse/repetition failures | 0 |
| JSON validity | 4/4 |
| callback_overwrites | 0 |
| identity_fallback_calls | 0 |
| Memory stability | PASS |

---

## Speed Attribution Result (Phase 12B)

| Metric | Value |
|--------|-------|
| Prompts | 3 (narrative, code, JSON) |
| Total runs | 9 (3 modes × 3 prompts) |
| L12+L15 average speedup | ~1.812x |
| Pure all36 average | ~1.722x |
| L12+L15 advantage over pure PRT | ~5.2% faster |

**Attribution:** Speedup is from PRT replacing FFN_UP matmul with sparse ternary matmul — user CPU time drops ~40% proportionally. Per-layer FFN_UP isolation, graph dispatch overhead, and memory bandwidth attribution were not fully isolated (require invasive llama.cpp core changes).

---

## Sidecar Packaging Result (Phase 12C)

| Deliverable | Status |
|-------------|--------|
| Manifest spec | ✓ `PRT_SIDECAR_MANIFEST_SPEC.md` |
| Example manifest (Qwen2.5-3B) | ✓ `prt_sidecar_manifest.example.json` |
| Validator script | ✓ `prt_validate_sidecars.py` |
| Sidecar binaries committed | ✗ Intentionally not committed |
| Model weights committed | ✗ Intentionally not committed |
| Reproduction | Requires local model + local sidecar generation |

---

## Anchor Policy Result (Phase 12D)

| Metric | Value |
|--------|-------|
| Policies tested | 10 PRT + native |
| Total runs | 88 |
| Best average speedup | L11+L15 at 1.780x |
| Default validated policy | L12+L15 at 1.765x |
| L12+L15 vs L11+L15 margin | ~0.85% — within noise margin |
| Best token-0 match rate | L13+L14 at 100% |
| Pure all36 average | ~1.696x (competitive, not preferred) |
| Single-anchor viability | L12 and L15 both pass token-0 matching |
| Triple-anchor anomaly | L12+L14+L15 had interference — unreliable |

**L12+L15 remains default** because it passed the larger Phase 12A 24-prompt suite. L11+L15 is a promising candidate but only showed ~0.85% advantage on the smaller 8-prompt screen.

---

## Current Strongest Allowed Claim

> "PRT Route A + L12/L15 achieved ~1.79x average speedup across a 24-prompt broader validation suite on the tested local CPU/model setup, with clean counters, stable memory, 0 collapse/repetition failures, and 4/4 valid JSON outputs. Phase 12D tested 10 anchor policies across 88 runs and found no strong reason to replace L12+L15 as the default validated policy."

---

## Forbidden Claims

- ~~"Production-ready"~~
- ~~"Universal CPU acceleration"~~
- ~~"Validated across all models"~~
- ~~"Upstream-ready"~~
- ~~"L12+L15 is globally optimal"~~
- ~~"L11+L15 is now the default"~~
- ~~"L11+L15 is proven superior overall"~~
- ~~"Pure all36 is a total failure"~~
- ~~"Sidecars are plug-and-play for all models"~~
- ~~"No further testing needed"~~
- ~~"Anchor policy behavior generalizes across all models"~~

---

## Open Questions

### Validation
- Full 24-prompt validation of L11+L15 candidate (Phase 12E recommended next)
- Longer generation tests (n > 200)
- Another model or model-size test
- JSON/structured output at n > 50 (memory-limited on current machine)

### Technical
- Direct runtime manifest integration (`--prt-manifest` flag)
- Sidecar generation documentation improvement
- Upstream sync/rebase strategy
- Deeper profiling (per-layer isolation)
- CLI `--prt-sidecar-dir` configurable flag (currently hardcoded)

### Packaging
- Installation/build documentation
- Better error messages for missing/mismatched sidecars
- GitHub repo public visibility decision (currently private)

---

## Recommended Next Branch

**`experimental/prt-route-a-phase12e-l11-l15`**

Purpose: Full 24-prompt broader validation of the L11+L15 candidate without changing the default L12+L15 policy unless L11+L15 wins the broader suite.

Do NOT change the default policy before L11+L15 passes the larger Phase 12A-style validation.

---

## Documentation Index

| Document | Purpose |
|----------|---------|
| `PRT_OVERVIEW.md` | What PRT is, Route A explanation |
| `PRT_ROUTE_A_RC1_SUMMARY.md` | RC1 tag, branch, results, counters |
| `PRT_PHASE12_FINAL_WRAPUP.md` | This document — Phase 12 complete summary |
| `PRT_CLAIMS.md` | Allowed and forbidden claims |
| `PRT_REPRODUCTION_NOTES.md` | How to reproduce, flags, sidecar notes |
| `PRT_PUBLIC_READINESS_CHECKLIST.md` | Pre-public safety checklist |
| `PRT_SIDECAR_MANIFEST_SPEC.md` | Sidecar format and policy spec |
| `prt_sidecar_manifest.example.json` | Example manifest for Qwen2.5-3B |
| `prt_validate_sidecars.py` | Standalone sidecar validator |
| `PRT_PHASE12_SCOPE_REVIEW.md` | Phase 12 scope and remaining open questions |
| `results/PRT_PHASE12A_BROADER_VALIDATION.md` | 24-prompt table |
| `results/phase12a_results.json` | Structured Phase 12A data |
| `results/PRT_PHASE12A_POSTMORTEM.md` | Phase 12A analysis |
| `results/PRT_PHASE12B_SPEED_ATTRIBUTION.md` | Speed attribution results |
| `results/phase12b_timing.json` | Structured timing data |
| `results/PRT_PHASE12C_SIDECAR_PACKAGING.md` | Packaging report |
| `results/PRT_PHASE12D_ANCHOR_POLICY_SEARCH.md` | Full policy table |
| `results/phase12d_anchor_results.json` | Structured policy data |
| `results/PRT_PHASE12D_POLICY_RANKING.md` | Ranked policies |
| `results/PRT_PHASE12D_FAILURES.md` | Failure report |
| `results/PRT_PHASE12D_VERDICT.md` | Phase 12D verdict |
| `results/PRT_PHASE12D_POSTMORTEM.md` | Phase 12D analysis |

---

*Phase 12 final wrap-up by ELVIS for Matthew Villnave / The ForgeHQ*