# Phase 26A: Post-PRT SDI Path Selection

**Verdict:** `PASS_PHASE26A_PATH_SELECTION`

**Date:** Wed 2026-05-20 22:58 EDT
**Branch:** `experimental/prt-phase19a-alt-sidecar-backed`
**Current HEAD:** `060a40e1b`

---

## A. Branch
```
experimental/prt-phase19a-alt-sidecar-backed
```

## B. Current HEAD
```
060a40e1b ("PRT Phase 25B: freeze correctness-restored checkpoint")
```

## C. PRT Freeze Verified
| Check | Result |
|-------|--------|
| Branch | ✅ `experimental/prt-phase19a-alt-sidecar-backed` |
| HEAD | ✅ `060a40e1b` |
| Tag exists | ✅ `PRT_PHASE25B_CORRECTNESS_RESTORED_CHECKPOINT` |
| Working tree | ✅ Only `ggml/src/ggml-cpu/ops.cpp` modified (12 lines, /127 fix); untracked phase24R docs harmless |
| Models/sidecars staged | ✅ None |

---

## D. PRT Lesson

### What PRT Proved
- Canonical INT8 sidecar layout is a valid representation artifact ✅
- `/127` dequant fix restored correctness — Python `round(f32/scale*127)` requires C decode `f32 = int8 * scale / 127.0f` ✅
- 3B and 7B layer0 correctness canaries pass after fix ✅
- Current GGML custom-op path is slower than native Q4_K_M in narrow 3B c=4 n=8 smoke (PRT ~36% slower) ✅

### What PRT Did NOT Prove
- ❌ Speedup
- ❌ Multi-layer support
- ❌ All-layer support
- ❌ Production readiness
- ❌ 7B timing (regression pass only)
- ❌ 14B support

### Core Lesson
**PRT is currently a valid representation/correctness artifact, not the active speed path.**

The custom-op INT8 path proves canonical layout and decode formula work. It does not beat native Q4_K_M in the tested configuration. Further speed work on this path is high-risk/low-reward until someone profiles the corrected op and identifies a concrete bottleneck with a specific fix.

---

## E. Evaluated Options

### Option 1 — SpecBenchCPU / Speculative Decoding Path
**Question:** Can we get practical CPU speedups by routing only code-like prompts to speculative decoding?

| Factor | Assessment |
|--------|------------|
| Prior evidence | Had measured code speedups before PRT detour |
| Engineering risk | Low — already built and validated |
| Time-to-demo | Short — existing infrastructure |
| Hardware fit | Good — runs on current 15GB RAM machine |
| SDI thesis compatibility | High — core SDI routing topic |
| Product/storytelling value | Medium — concrete speedup narrative |

### Option 2 — Smart Agent Router + SDI Policy Layer
**Question:** Can we make a real system-level SDI stack using routing, preloading, small/large model selection, speculative mode, context routing, and memory compression?

| Factor | Assessment |
|--------|------------|
| Prior evidence | Router exists; LiteRT integration done |
| Engineering risk | Medium — policy layer is new design |
| Time-to-demo | Medium — needs concrete policy decisions |
| Hardware fit | Good — fits Matt's current stack |
| SDI thesis compatibility | Very high — this IS the SDI stack |
| Product/storytelling value | High — demos well, clear user value |

### Option 3 — ContextOS / MemoryOS Integration
**Question:** Can we reduce compute waste by improving context/memory handling instead of kernel math?

| Factor | Assessment |
|--------|------------|
| Prior evidence | Obsidian-based VaultBrain completed 2026-02-20 |
| Engineering risk | Medium — context compression is novel |
| Time-to-demo | Medium — depends on VaultBrain integration |
| Hardware fit | Good |
| SDI thesis compatibility | High — MemoryOS is a core pillar |
| Product/storytelling value | High — good story, clear customer angle |

### Option 4 — PRT v3 Native Backend Design Only
**Question:** Should PRT continue as a design branch only, not implementation?

| Factor | Assessment |
|--------|------------|
| Prior evidence | Canonical layout proven; custom-op path proven slow |
| Engineering risk | High — native backend op is complex |
| Time-to-demo | Long — no clear path to demo |
| Hardware fit | Neutral |
| SDI thesis compatibility | Medium — representation artifact |
| Product/storytelling value | Low — too early to storytel |

### Option 5 — Local Inference Productization
**Question:** Can we package current local inference stack into something useful sooner?

| Factor | Assessment |
|--------|------------|
| Prior evidence | OpenClaw/ELVIS + Smart Agent Router + LiteRT/Ollama routes exist |
| Engineering risk | Low — repackaging existing work |
| Time-to-demo | Short |
| Hardware fit | Good |
| SDI thesis compatibility | Medium — infrastructure but not SDI core |
| Product/storytelling value | High — concrete tool for users |

---

## F. Ranked Recommendations

### 1. Recommended Next Path: SpecBenchCPU Revalidation + Router SDI Mode

**Why:** The speculative decoding path has prior measured speedups on code prompts, already fits the router architecture, and is the lowest-risk path to a new concrete win. Combining it with router SDI policy (Option 2) creates a real selectable-diversity inference stack.

**First concrete phase (26B):**
1. Verify SpecBenchCPU still passes on current `experimental/prt-phase19a-alt-sidecar-backed`
2. Measure current code-prompt speedup vs native (light, single-run)
3. Design SDI routing policy: code-prompt → speculative mode, other → native
4. Integrate into Smart Agent Router as selectable "SDI mode"

**Success criteria:**
- SpecBenchCPU passes ✅
- Code prompt speculative decoding shows measurable speedup ✅
- Router can route to speculative path based on prompt classification ✅

**Risks:**
- SpecBenchCPU results may regress on current branch (low risk — easy to verify)
- Speedup may be smaller than prior runs due to model/version differences
- Router SDI policy layer adds complexity

---

### 2. Second-Best Path: ContextOS / MemoryOS Integration

**Why:** VaultBrain (Obsidian-based memory) is already completed. MemoryOS is a documented SDI pillar. Integration with local inference could reduce compute waste on repetitive context patterns. Good product story, reasonable engineering effort.

**First concrete phase (26B or 27):**
1. Audit current VaultBrain / ClawVault integration points
2. Design context compression or selective-memory pathway for repeated prompt patterns
3. Measure compute savings on repetitive eval workloads

**Success criteria:**
- ContextOS architecture documented ✅
- At least one compute-saving mechanism identified ✅

**Risks:**
- May overlap with existing ClawVault work
- Hard to measure without a concrete demo workload

---

### 3. Parked / Research-Only: PRT v3 Native Backend Design

**Why:** High complexity, no clear path to demo, currently the slowest path. Revisit only after Option 1 produces a concrete win and someone has a specific native backend hypothesis to test. The canonical INT8 layout knowledge is preserved in the frozen branch; no urgency to continue PRT development right now.

---

## G. Recommended Next Phase

**Phase 26B: SpecBenchCPU Router Revalidation**

Scope:
1. Run SpecBenchCPU on current branch — verify speculative decoding correctness still passes
2. One small code-prompt timing smoke (1 native run, 1 speculative run, no repeat)
3. Document current router architecture + SDI mode design proposal
4. Do NOT make speedup claims yet — just verify the path is alive

Purpose: Confirm speculative decoding path is still good, then design the SDI routing layer on top of it.

---

## H. What NOT to Do Next

- ❌ More PRT speed work on the current custom-op path
- ❌ Generate new sidecars
- ❌ 7B or 14B timing
- ❌ Multi-layer PRT implementation
- ❌ Long benchmark loops
- ❌ Tags or model staging

---

## I. Safety Scan

```
git status --short
 M ggml/src/ggml-cpu/ops.cpp
?? examples/speculative/results/PRT_PHASE24R_NATIVE_MULMAT_PROBE.md
?? examples/speculative/results/phase24r_native_mulmat_probe.json

No models/sidecars/f32 refs/binaries staged.
No secrets found.
```

---

## Verdicts

| Verdict | Value |
|---------|-------|
| `PASS_PHASE26A_PATH_SELECTION` | ✅ |
| `PRT_FREEZE_VERIFIED` | ✅ |
| `RECOMMEND_SPECBENCHCPU_ROUTE` | ✅ #1 |
| `RECOMMEND_SDI_ROUTER_ROUTE` | ✅ #2 |
| `RECOMMEND_CONTEXT_MEMORY_ROUTE` | ✅ #3 |
| `RECOMMEND_PRT_V3_DESIGN_ONLY` | ⏸️ Parked |
| `BLOCKED_REPO_STATE` | ✅ No block |

---

*Phase 26A complete. Awaiting Matt's direction to proceed to Phase 26B or pivot.*