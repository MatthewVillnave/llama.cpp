# Phase 28A: KV/Context Memory Mapping Roadmap

## A. Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## B. Current HEAD
`30419c3d8`

## C. SDI v0.1.1 Endpoint

**Frozen state:**
- SDI Runtime v0.1.1 checkpoint exists (`SDI_PHASE26W_RUNTIME_V0_1_1_CHECKPOINT`)
- Public package corrected and committed (Phase 27L)
- Bounded 7B probing chapter closed (Phases 27H–27K)
- qwen2.5:7B bounded probes passed WS-512 through WS-6144 under strict guard
- c=8192 forensics passed all three test types
- No memory/swap cliff found in bounded 7B range
- Swap delta: 0.0 GB across all Phase 27H–27J runs
- PRT speed path remains parked

**Forbidden claims that remain active:**
- ❌ Broad 7B validation
- ❌ Speedup demonstrated
- ❌ Production readiness
- ❌ Long-context solved
- ❌ 14B support
- ❌ KV cache or weight-residency solved

**Why technical probing paused after 27L:**
- The bounded 7B evidence is complete for this backend
- Public package is corrected and consistent
- Next gains require either new measurement data or a design shift
- No clear justification for more 7B runs without new diagnostic evidence

## D. KV/Context Questions

The core question: **How does context length / KV pressure / backend behavior scale on this machine now that SDI context selection is stable?**

### Question 1: RAM as a function of context length
How much RAM does context actually add at c=2048 vs c=4096 vs c=8192 for qwen2.5 models?

Known: Model weights dominate RAM usage (~5.3 GB for 7B in Ollama). KV cache grows with context length but the relationship is not measured on this machine.

### Question 2: Swap trigger conditions
Is swap triggered by context length, prompt structure, model size, or stale process state?

Partial answer from Phase 27H–27J: swap stayed flat at ~0.37 GB across all bounded 7B runs including WS-6144/c=8192. This suggests context length alone does not trigger swap on this machine within the tested range. Prompt structure and runner state were the dominant failure modes.

### Question 3: Ollama memory/context telemetry
Does Ollama expose enough memory/context telemetry to answer Questions 1 and 2 without modifying llama.cpp?

Limited visibility: Ollama's `/api/show` returns model metadata but not live KV/RAM breakdown per context length. External measurement (RAM before/after, swap delta, process RSS) is available via `psutil`. Direct KV measurement requires llama.cpp internals.

### Question 4: Externally measurable without modifying llama.cpp
What can be measured externally?

- Process RSS before/after (via `psutil`)
- Swap usage delta (via `psutil.swap_memory()`)
- Wall time per token (available in Ollama response: `eval_duration / eval_count`)
- `prompt_eval_count` and `eval_count` from Ollama API response
- Ollama server memory footprint via `/api-tags` or `ps aux`
- Timing breakdown: prompt_eval vs token generation

### Question 5: Justification for deeper backend/KV instrumentation
What would justify deeper llama.cpp instrumentation?

- Evidence that KV growth is the actual bottleneck for SDI performance
- Pattern showing output quality degrades at specific context lengths despite correct answers
- Clear SDI policy failure that correlates with context length rather than content type

Currently: no such evidence exists. SDI policies are performing correctly in eval. The bounded 7B runs show no cliff. Deep KV instrumentation is not justified by current data.

## E. Proposed Measurement Plan

**Design only — no runs until next phase is explicitly approved.**

### Proposed future matrix (Phase 28B — design target):

| Model | Contexts | Prompt Types | Metrics |
|-------|----------|-------------|---------|
| qwen2.5:0.5B | c=2048, c=4096, c=8192 | tiny, medium filler | RAM pre/post, swap delta, wall time, output sanity |
| qwen2.5:3B | c=2048, c=4096, c=8192 | tiny, medium filler | same |
| qwen2.5:7B | c=2048, c=4096, c=8192 | tiny, medium filler | same |

**Measurement protocol per run:**
1. Snapshot RAM and swap (via `psutil`)
2. Kill any stale Ollama runners from prior runs
3. Issue Ollama generate request with specified model + context + prompt
4. Capture `prompt_eval_count`, `eval_count`, `eval_duration`, `total_duration`, RAM post, swap post
5. Score output sanity (simple retrieval check)
6. Record delta values

**Pass criteria per run:**
- RAM post > 3 GB available
- Swap delta < 250 MB
- Output sane (retrieval correct)
- No process timeout/stale state

**Fail criteria:**
- Swap delta > 250 MB
- RAM available < 3 GB
- Output not sane
- Runner goes stuck (589% CPU + non-responsive API)

### What this plan is designed to answer:
- RAM usage curve: does it scale linearly, stair-step, or plateau with context?
- Swap threshold: at what context length does swap engage for each model?
- Time-per-token: does it degrade with context length, and at what rate?
- Staleness pattern: is the "stuck runner" problem correlated with context length or model size?

## F. Safety Guard

**Hard limits for any future Phase 28B run:**
- One run at a time; no parallel runs
- No c=16384 yet
- No 14B
- No broad eval suite
- No more than 3 models × 3 contexts × 2 prompt types = 18 runs maximum
- Abort if swap used > 1 GB before a run
- Abort if swap delta > 250 MB after a run
- Abort if available RAM < 3 GB
- Stop on first stuck runner; do not retry same configuration
- Capture full Ollama response including timing and eval counts
- All outputs to `/tmp` only — not staged to repo
- No model files, sidecars, or f32 refs staged

## G. Option Ranking

| Rank | Option | Rationale | Risk |
|------|--------|-----------|------|
| 1 | **External memory/context probe** | Lowest risk. Uses existing Ollama API + `psutil`. Measures what can be measured externally. No llama.cpp changes. | Low |
| 2 | **Small-model context scaling (0.5B + 3B first)** | Establish baseline on small models before touching 7B again. If 0.5B shows no cliff, it strengthens the case that memory isn't the bottleneck. | Low |
| 3 | **Bounded 7B context scaling** | Replicate Phase 27J matrix but with formal memory capture. Useful but requires running 7B again. | Medium |
| 4 | **llama.cpp/KV instrumentation design** | Document how KV hooks could be added later if justified. Parked until evidence justifies it. | High (complex) |
| 5 | **Stop and package work** | 7B chapter is closed. Evidence is complete. Focus on dissemination. | N/A |

## H. Recommended Next Phase

**Phase 28B: External Memory/Context Probe Harness**

Design: Build a harness that runs the small-model matrix (0.5B + 3B at c=2048/4096/8192) with clean memory capture, no new llama.cpp changes. Measure the RAM/context curve without touching 7B.

**Why 0.5B/3B first:** Establish whether the memory/context relationship even shows variation before applying any 7B probe. If 0.5B shows a flat curve (context length doesn't affect RAM meaningfully), that tells us something important about where the pressure point actually is.

**Why not 7B first:** We already have bounded 7B data through WS-6144. The marginal diagnostic value of more 7B runs is low. Small-model data generalizes more and is faster to collect.

## I. Models/Sidecars/F32 Refs Staged?
No.

## J. Secrets Detected?
No secrets in any committed files.

## K. Tags Touched?
No tags created, modified, or pushed.

---

## Verdicts
- ✅ `PASS_PHASE28A_KV_CONTEXT_ROADMAP`
- ✅ `RECOMMEND_EXTERNAL_MEMORY_PROBE`
- ✅ `RECOMMEND_SMALL_MODEL_CONTEXT_SCALING`
- ✅ `PARK_DEEP_KV_INSTRUMENTATION`
- ✅ `BLOCKED_REPO_STATE` (old untracked reports remain untracked)