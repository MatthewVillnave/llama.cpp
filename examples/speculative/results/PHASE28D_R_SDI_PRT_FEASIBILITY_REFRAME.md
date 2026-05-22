# Phase 28D-R: Reframe SDI/PRT Around Feasibility, Not Speed

## A. Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## B. Current HEAD
`7eea01fc9`

## C. Corrected Objective

**Old wrong-ish question:**
> Can PRT beat native Q4 speed on small already-runnable models?

**Corrected question:**
> Can SDI/PRT make larger dense models run inside CPU/RAM limits where the native path fails, swaps, or becomes unusable?

**Core framing:** "Any coherent generation is faster than impossible." If native dense inference cannot load, swaps to death, OOMs, or becomes unusable, then a slower-but-working SDI/PRT path is still a win.

### Feasibility Success Criteria

- Model loads or partially streams without OOM
- No swap death
- Produces coherent bounded output
- Quality is not collapsed
- Speed can be slow, but not frozen
- Memory behavior is stable under strict guard

### Feasibility Non-Success

- Output garbage
- Infinite generation / hang
- Swap death (thrashing)
- OOM kill
- Only works by using cloud/GPU
- Only works by pretending a smaller model is the larger model
- Works only by dramatically reducing model quality below usable threshold

## D. Why Speed-First PRT Was the Wrong Battlefield

The old PRT research direction asked: "Can we beat native Q4 speed on small models?"

This was the wrong question for two reasons:

1. **Already-runnable models aren't the target.** If 7B runs fine natively, speed optimization of the already-working path has diminishing returns relative to solving the harder problem.

2. **Speed on working hardware ≠ feasibility on impossible hardware.** A 10% speed improvement on 7B doesn't move the needle on running 14B or 30B inside 16GB RAM.

The right battlefield is **capacity-first**: can we make a model run at all in conditions where it shouldn't? Not faster — just possible.

## E. Current SDI Layer Classification

| Piece | Layer | Role | Sufficient for Impossible-Model Feasibility? |
|-------|-------|------|---------------------------------------------|
| SDI Runtime v0.1.1 | Context-residency | Reduces active prompt/KV/context pressure | No — context selection doesn't solve weight streaming |
| PRT custom-op path | Correctness artifact | Validated kernel correctness, not speed | No — not a capacity solution |
| 7B bounded probes | Hardware behavior map | Shows 7B memory behavior under guard | No — 7B already runs; not impossible-model |
| KV/context memory probes | Measurement layer | External RSS/timing across context sizes | No — not a residency solution |
| Memory guard | Safety layer | Pre-run swap check, abort on pressure | Partial — prevents death but doesn't enable load |

**Bottom line:** Current SDI v0.1.1 is context-residency work. It helps already-runnable models stay stable under context pressure. It does not solve weight residency for models that can't load in the first place.

## F. Capacity-First PRT Definition

**Design concept — not implementation.**

Capacity-first PRT asks: *Can model weights be represented or activated progressively so that a model larger than available comfortable RAM can still generate?*

### Possible Mechanisms

- Progressive residual weight planes (load on demand, not all at once)
- Native-layout compressed sidecars (structured compression, not raw f32)
- Layer/block paging with strict memory budget (never materialize full model)
- Partial activation of FFN/channel blocks (skip what's not needed)
- Low-rank/residual reconstruction on demand (approximate weights cheaply)
- CPU-native sparse/decomposed weight format (leverage sparsity)
- Verification/fallback against native where possible (blend paths)

### Hard Constraints

- Smaller resident footprint than naive full-weight load
- Bounded working set — never exceeds available RAM + guard margin
- No giant f32 decode expansion (f32 decode of Q4 is not a solution)
- No 90MB per-layer materialization if it exceeds the budget
- No sidecar larger than the thing it replaces
- Native/backend layout considered from day one, not retrofitted
- Must degrade gracefully (slower/coherent over swap death/garbage)

### What This Is NOT

- Not the old PRT custom-op speed path
- Not prompt compression or context selection
- Not routing between models
- Not KV cache modification

## G. True Target Model Class

| Model | Hardware | Native Status | Feasibility Target? |
|-------|----------|--------------|-------------------|
| 7B | 16GB RAM | Runs (marginal at high context) | Useful but not impossible |
| 14B | 16GB RAM | Likely marginal/fails at high context | **Best first target** |
| 30B/32B | 16GB RAM | Likely impossible with Q4 | True long-term target |
| 7B at c=16384 | 15GB RAM | Fails or swaps | Already being studied |

### Recommendation: Start with 14B

**14B on 16GB RAM** is the best first feasibility target because:

1. **Truly impossible for naive native path** — 14B Q4 needs ~10-12GB just for weights; add KV cache at high context and it can exceed 16GB comfortably
2. **Not so large as to require radical compression** — feasible to analyze and design for
3. **Real-world relevance** — many 16GB machines exist; 14B is the next step up from 7B
4. **Lower risk than 30B** — if 14B fails we learn why; 30B might just confirm "impossible"

**What to analyze before running anything:**
- Native 14B load estimate: what does Ollama actually report for RSS with 14B loaded?
- Context budget: at what context length does 14B OOM?
- KV pressure at different context lengths
- What would have to be true for capacity-first PRT to help

## H. Feasibility Benchmark Definition

**Design only — do not run yet.**

### Native Baseline
```
Attempt: model load + generation under strict memory guard
Record: OOM / swap death / load failure / unusable speed (tokens/sec below threshold)
Pass: native fails or is unusable
```

### SDI/PRT Candidate
```
Same prompt/task
Same strict memory guard
Record: coherent output check + memory delta + time
Pass: produces coherent bounded answer without swap death
```

### Minimum Viability Criteria (3 tasks)
1. **Short factual answer:** "What is the capital of France?" — coherent single-line answer
2. **Instruction-following:** "Write a 3-sentence summary of: [20-word input]" — coherent multi-sentence output
3. **Exact retrieval:** Retrieve a specific string (commit hash, tracking ID) embedded in the prompt

### Pass Threshold
- Native: fails to load, OOMs, or swap thrashes
- SDI/PRT path: produces coherent answer for all 3 task types without swap death

### Non-Pass
- Native actually works fine → not a true feasibility target
- SDI/PRT produces garbage → capacity solution not working
- Swap death occurs → guard didn't help

## I. Recommended Next Phase

**Phase 28E: 14B feasibility analysis + capacity-first PRT requirements**

No running 14B yet. Analysis and design only.

Scope:
1. Document what native 14B load looks like on this hardware (Ollama RSS estimate)
2. Define the context budget where 14B would OOM
3. Design capacity-first PRT requirements document
4. Define what the 14B feasibility benchmark would need to measure
5. Determine whether context-residency SDI alone helps or whether weight residency is required

This is a checkpoint document, not an implementation. The implementation comes only after the analysis shows feasibility is plausible.

## J. Allowed Claims

- SDI v0.1.1 is context-residency work, not capacity-first solution
- The correct framing is feasibility under hardware limits, not speed optimization
- Capacity-first PRT is about making impossible models run, not making already-working models faster
- "Any coherent generation is faster than impossible"
- 14B on 16GB RAM is the best first true feasibility target

## K. Forbidden Claims

- ❌ Speedup claim on any already-runnable model
- ❌ 14B feasibility demonstrated (design only)
- ❌ PRT v3 implemented
- ❌ Weight residency solved
- ❌ Production readiness
- ❌ KV cache modified
- ❌ 30B support

## L. Models/Sidecars/F32 Refs Staged?
No.

## M. Secrets Detected?
No secrets in any committed files.

## N. Tags Touched?
No tags created, modified, or pushed.

---

## Verdicts
- ✅ `PASS_PHASE28D_R_FEASIBILITY_REFRAME`
- ✅ `PASS_CONTEXT_LAYER_NOT_ENDGAME_DEFINED`
- ✅ `PASS_CAPACITY_FIRST_PRT_DEFINED`
- ✅ `RECOMMEND_14B_FEASIBILITY_ANALYSIS`
- ✅ `RECOMMEND_CAPACITY_FIRST_PRT_DESIGN`
- ✅ `BLOCKED_REPO_STATE` (old untracked reports remain untracked)