# Phase 28K: 30B Candidate Gate + Quant Decision

## A. Branch & Commit
- Branch: `experimental/prt-phase19a-alt-sidecar-backed`
- HEAD before: `36085eadf` (Phase 28J)
- HEAD after: `??` (docs only, no tag)

## B. Candidate Classes

### Class A — "Practical Runnable Large Model" (MoE)
| Candidate | Type | Active Params | Est. Disk | Active Footprint | Native Fit | Demo Value |
|-----------|------|--------------|-----------|-----------------|-----------|-----------|
| `qwen3:30b` | MoE 30B-A3B | ~3B per token | ~60GB total | **~2-3GB** active | ⚠️ Maybe fits | High (practical useful model) |

**Note:** MoE with 8 experts, 2 active per token × 3B each = ~6B activated. Active Q4 footprint ~3-4GB. May fit in 16GB if expert weights can be paged out. CPU dispatch inefficiency is known in Ollama.

**Key distinction:** MoE path tests "can a large useful model run" — not "can weight residency solve impossible dense models."

### Class B — "True Impossible Dense Target" (Dense)
| Candidate | Type | Est. Disk | RSS Est. | Native Fit | Demo Value |
|-----------|------|-----------|----------|-----------|-----------|
| `qwen3:32b` | Dense 32B Q4 | ~19GB | ~23GB | ❌ OOM | **Critical** (true impossible-model proof) |
| `nemotron-3-nano:30b` | Dense 30B Q4 | ~18GB | ~22GB | ❌ OOM | High (true impossible-model proof) |

**Key distinction:** Dense path tests "can weight residency make a truly too-large model run on constrained hardware" — this is the SDI/PRT moonshot.

### Why They Must Not Be Confused
- MoE success ≠ proof that dense capacity-first PRT works
- A dense 30B Q4 that fails natively but succeeds via paging/residuals is the correct impossible-model demo
- MoE is a useful side lane, not a substitute for the dense target

## C. Quant Ladder Decision

| Quant | Dense 30B RSS | 16GB Fit? | Quality | PRT Role | Decision |
|-------|--------------|-----------|---------|---------|---------|
| Q4 | ~22GB | ❌ OOM | Best | Reduce via out-of-core paging | **Baseline fail target** — prove native fails |
| Q3 | ~16GB | ❌ Marginal (0-2GB headroom) | Medium | Stabilize + quality baseline | **Transition quant** — may load but risky |
| Q2 | ~13GB | ⚠️ Maybe fits | **High risk** | **Quality recovery via residuals** | **First feasible base + PRT substrate** |

### Answers to Key Questions

**1. Is Q4 the right native-fail baseline?**
**YES.** Dense Q4 at 30B/32B clearly exceeds 16GB (22GB+ RSS). Use qwen3:32b Q4 as the canonical "native fails" baseline.

**2. Is Q2 the right first runnable coarse base?**
**YES.** Q2 (~13GB RSS) is the most likely to actually load on 16GB. If Q2 base produces weak output, PRT residuals recover quality. This is the correct first experiment.

**3. Is Q3 the right middle compromise?**
**MAYBE.** Q3 may fit or may be marginal — worth including as a decision point in the ladder. But Q2+residual overlay is the primary target.

**4. Which quant supports Q2/Q3 base + PRT residual overlay?**
**Q2 base is the substrate.** Q2 quality is weak → PRT residuals add detail. Q3 is a possible alternative if Q2 is too weak even with residuals.

## D. Candidate Decision Table

| Candidate | Dense/MoE | Est. Size | Native Fit Risk | Demo Value | PRT Relevance | Decision |
|-----------|-----------|-----------|-----------------|-----------|---------------|---------|
| `qwen3:30b` | MoE | ~60GB disk / 3GB active | ⚠️ Medium (active fits) | Practical useful model | Low — not weight residency proof | `USE_AS_MOE_SIDE_LANE` — pull only if Matt wants practical large model demo |
| `qwen3:32b` | Dense | ~19GB | ❌ OOM | **Critical** — true impossible proof | **High** — canonical dense 32B target | `USE_AS_DENSE_NATIVE_FAIL_TARGET` — canonical Phase 28L moonshot |
| `nemotron-3-nano:30b` | Dense | ~18GB | ❌ OOM | High — true impossible proof | High — 30B variant | `USE_AS_DENSE_NATIVE_FAIL_TARGET` — alternative to qwen3:32b |
| `llama3.1:70b` | Dense | ~40GB | ❌ OOM | Low — too large for this machine | Low — not a 16GB target | `DO_NOT_PULL` — far beyond scope |

## E. Two Possible Paths

### Path 1 — MoE Practical Path
**Goal:** See if MoE 30B runs semi-usably on 16GB CPU because only active params matter.

| Aspect | Value |
|--------|-------|
| Candidate | qwen3:30b |
| Active footprint | ~3GB Q4 |
| Resident target | ~4GB (active + KV) |
| Pros | May work sooner; useful model; less radical engineering |
| Cons | Not proof of dense weight-residency; CPU MoE inefficiency |
| PRT relevance | Low — MoE is architectural, not weight-residency |

**Verdict:** Side lane, not the moonshot. Pull only if Matt explicitly wants a "practical large model that works" demo.

### Path 2 — Dense Impossible Path (MOONSHOT)
**Goal:** Prove dense 30B/32B Q4 fails natively; demonstrate Q2 base + PRT residuals produce coherent output.

| Stage | Test | Expected |
|-------|------|---------|
| 0 | System preflight | Pass |
| 1 | Pull qwen3:32b Q4 → attempt load | OOM/swap → prove native fail ✅ |
| 2 | Pull qwen3:32b Q2 → load | Maybe loads ✅ |
| 3 | Tiny gen (factual) | Weak output → need PRT residuals |
| 4 | Instruction prompt | Quality check |
| 5 | Same with residual overlay | Quality recovery |
| 6 | Exact retrieval | Correct fact |

**Verdict:** This is the correct SDI/PRT impossible-model demo. Q4 fails → Q2 loads → residuals recover quality.

## F. Recommended Next Phase

**Phase 28L: Dense 30B Q2 feasibility + capacity-first PRT overlay spec**

Split decision gate:
- **Lane 1 (MoE side):** qwen3:30b MoE — practical side lane, pull only with explicit Matt approval for "useful large model" demo
- **Lane 2 (Dense moonshot):** qwen3:32b Q4 native-fail baseline → Q2 base + PRT residual overlay

**Preferred execution order:**
1. Design PRT residual overlay spec first (no model needed, analysis only)
2. Run Q4 native-fail preflight for qwen3:32b (Stage 0, no pull — confirm OOM prediction)
3. If preflight confirms: pull qwen3:32b Q2 base for residual overlay experiment

**Not recommended:** Pull MoE 30B without explicit approval for practical demo. It will confuse the narrative of "dense impossible-model proof."

## G. Safety Checklist
| Item | Status |
|------|--------|
| Models/sidecars/f32 refs staged? | **NO** ✅ |
| Raw logs/captures staged? | **NO** ✅ |
| Secrets detected? | **NO** ✅ |
| Tags touched? | **NO** ✅ |
| Any 30B model pulled? | **NO** ✅ |
| llama.cpp modified? | **NO** ✅ |

## Verdicts
- `PASS_PHASE28K_30B_CANDIDATE_GATE`
- `PASS_DENSE_VS_MOE_SEPARATED`
- `PASS_QUANT_LADDER_DECISION`
- `RECOMMEND_DENSE_Q2_Q3_FEASIBILITY`
- `RECOMMEND_MOE_SIDE_LANE`
- `RECOMMEND_CAPACITY_PRT_OVERLAY_SPEC`
- `BLOCKED_REPO_STATE_CLEAN`