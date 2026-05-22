# Phase 28H: 14B Load + Generation — Load-Only Test

## Branch & Commit
- Branch: `experimental/prt-phase19a-alt-sidecar-backed`
- HEAD before: `96baf84b5` (Phase 28G)
- HEAD after: `??` (docs only, no tag)

## Objective
Execute Phase 28F preflight design Stages 2–5: pull qwen2.5:14b, load it, run minimal generation at increasing context sizes, and measure memory behavior against predicted budgets.

## Stage 0 — System Pre-Run Check
| Check | Predicted | Actual | Guard | Result |
|-------|-----------|--------|-------|--------|
| Swap used | ≤1GB | 287MB | ≤1024MB | ✅ PASS |
| RAM available | ≥6GB | ~11GB | ≥6GB | ✅ PASS |
| Disk free | ≥5GB | 132GB | — | ✅ PASS |

## Stage 2 — Model Pull
| Item | Value |
|------|-------|
| Model | qwen2.5:14b |
| Digest | 7cdf5a0187d5 |
| Pull size | 9.0 GB |
| Status | ✅ success |
| Time | ~15 min at 13 MB/s |

## Stage 3 — Tiny Gen c=1024
| Metric | Value |
|--------|-------|
| Status | done: true |
| Prompt tokens | 38 |
| Generated tokens | 177 |
| Response | Correct (Rayleigh scattering explanation) |
| RAM pre | 4.3GB used / 10GB avail |
| RAM post | 13GB used / 1.8GB avail |
| Swap pre | 287MB |
| Swap post | 384MB |
| **Swap delta** | **+97MB** ✅ (<250MB guard) |
| Wall time | 38.3s |

**Stage 3 verdict: PASS** ✅ — 14B loads and generates coherently at c=1024

## Stage 4 — Instruction c=2048
| Metric | Value |
|--------|-------|
| Status | done: true |
| Prompt tokens | 61 |
| Generated tokens | 302 |
| Response | Correct reasoning about Friday deadline constraints |
| RAM pre | ~13GB |
| RAM post | ~13GB |
| Swap pre | 384MB |
| Swap post | 384MB |
| **Swap delta** | **0MB** ✅ |
| Wall time | 67.8s |

**Stage 4 verdict: PASS** ✅ — 14B reasoning coherent at c=2048; swap stable

## Stage 5 — Exact Retrieval c=2048
| Metric | Value |
|--------|-------|
| Status | done: true |
| Prompt tokens | ~55 |
| Generated tokens | 28 |
| Response | "The commit that added the SDI auto policy is **7efdab38b**, which corresponds to Phase 26O." |
| Commit correct | ✅ YES |
| Swap pre | 383MB |
| Swap post | 383MB |
| **Swap delta** | **0MB** ✅ |
| Wall time | 11.0s |

**Stage 5 verdict: PASS** ✅ — 14B correctly recalls exact commit hash from context

## Memory Summary
| Stage | RAM RSS | Swap delta | Guard | Verdict |
|-------|---------|-----------|-------|---------|
| Stage 3 c=1024 | ~13GB | +97MB | ≤250MB | ✅ PASS |
| Stage 4 c=2048 | ~13GB | 0MB | ≤250MB | ✅ PASS |
| Stage 5 c=2048 | ~13GB | 0MB | ≤250MB | ✅ PASS |

**Predicted vs actual:** All stages within Phase 28F predicted budgets. c=2048 total (model + KV) predicted at ~12.25GB — actual RSS ~13GB is consistent.

## Classification
| | Result |
|-|--------|
| LOAD_PASS_STABLE | ✅ YES |
| LOAD_PASS_SWAP_RISK | ❌ NO — swap delta 0-97MB, well within guard |
| LOAD_FAIL | ❌ NO |
| Gen swap thrashing | ❌ NO |

## Phase 28I Recommendation
**PROCEED to Phase 28I: Full c=2048 SDI felt-win demo.**
14B loaded and generated at c=2048 with stable swap. No capacity-first PRT needed yet. SDI context layer is the correct next step to establish felt-win on a model that is actually new territory for this hardware.