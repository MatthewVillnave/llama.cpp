# PRT Phase 12 Checkpoint Tag Notes

**Tag:** `PRT_PHASE12_VALIDATION_CHECKPOINT`
**Date:** 2026-05-03
**Commit:** `fff7811b43c01b006fa4c86726001ed4641412fc`
**Branch:** `experimental/prt-route-a-phase12`

---

## Purpose

Marks the end of Phase 12 validation. This tag represents the completion of:

1. Public readiness documentation (Phase 12-PRD)
2. 24-prompt broader validation (Phase 12A)
3. Postmortem and claim update (Phase 12A-POST)
4. Speed attribution profiling (Phase 12B)
5. Sidecar packaging and reproducibility (Phase 12C)
6. Native anchor policy search (Phase 12D)
7. Policy interpretation and scope review (Phase 12D-POST)
8. Final wrap-up and checkpoint (Phase 12F ← this document)

---

## Relationship to RC1

**This tag does NOT replace `PRT_ROUTE_A_RC1`.**

| Tag | Purpose | Commit |
|-----|---------|--------|
| `PRT_ROUTE_A_RC1` | Frozen RC1 checkpoint | `aaa5f290240dbaacfe355ca073bcbce49b18fde7` |
| `PRT_PHASE12_VALIDATION_CHECKPOINT` | Phase 12 validation checkpoint | `fff7811b43c01b006fa4c86726001ed4641412fc` |

`PRT_ROUTE_A_RC1` is the immutable baseline. This checkpoint represents the end of Phase 12 work on the active development branch.

---

## Tag Summary (for GitHub releases or description)

> "PRT Phase 12 validation checkpoint: Route A + L12/L15 default policy, 24-prompt broader validation at ~1.79x average speedup, clean counters, stable memory, sidecar packaging docs, validator script, and anchor policy review. L11+L15 is a promising candidate for future validation. L12+L15 remains the default."

---

## What This Tag Represents

**Validated:**
- ~1.79x average speedup on 24-prompt suite
- callback_overwrites = 0 on all runs
- identity_fallback_calls = 0 on all runs
- 0 collapse/repetition failures
- 4/4 valid JSON outputs
- stable memory, no OOM
- sidecar manifest spec and validator
- public readiness docs
- allowed/forbidden claims documented

**Not Validated:**
- Production readiness
- All-model generalization
- JSON at n > 50
- L11+L15 as default (only ~0.85% faster in smaller screen)
- Upstream merge readiness

---

## How to Use This Tag

```bash
# Checkout the Phase 12 checkpoint
git checkout PRT_PHASE12_VALIDATION_CHECKPOINT

# Or the branch
git checkout experimental/prt-route-a-phase12

# See what's in this checkpoint
git show PRT_PHASE12_VALIDATION_CHECKPOINT --stat
```

---

## Next Steps from This Checkpoint

Recommended: `experimental/prt-route-a-phase12e-l11-l15`
- Full 24-prompt validation of L11+L15 candidate
- Do NOT change default policy until L11+L15 passes broader suite

---

*Checkpoint tag notes by ELVIS for Matthew Villnave / The ForgeHQ*