# Phase 28E: 14B Feasibility Analysis + Capacity-First PRT Requirements

## A. Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## B. Current HEAD
`1e4bd7091`

## C. Impossible-Model Demo Definition

**What we want to demonstrate:**
A CPU-only 16GB RAM system runs a model that normally should not run comfortably there — and produces coherent output without OOM, swap death, or cloud fallback.

**Minimum demo requirements:**
- Model produces coherent bounded output
- No OOM kill
- No swap death (thrashing)
- No runaway generation
- No GPU/cloud fallback
- Speed does not need to beat native small models (slower is fine if coherent)
- Output must be usable enough to count

**Candidate task types (3 minimum):**
1. **Short factual answer:** "What is the capital of France?" → "Paris"
2. **Simple instruction-following:** "Write a 3-sentence summary of: [20-word input]" → coherent multi-sentence output
3. **Exact retrieval:** Retrieve a specific embedded string (commit hash, tracking ID) from noisy context

**Success condition:**
- Native path fails, swaps badly, OOMs, or is unusably unstable
- SDI/PRT capacity path produces coherent bounded output inside strict memory guard

**Failure condition:**
- Native actually works fine → not a true feasibility target
- SDI/PRT produces garbage or crashes → capacity solution not working
- Cloud/GPU required → not a valid demo
- Swap death occurs → guard didn't help

## D. 14B Native Feasibility Estimate

**All figures are estimates based on model size ratios; no 14B model has been pulled or run.**

### Known local facts (measured)
- System: ~16GB RAM total, ~15GB usable with guard margin
- qwen2.5:7B Q4 GGUF: ~4.7GB model file, Ollama RSS ~4.7–5GB loaded
- qwen2.5:3B Q4: ~1.9GB, RSS ~1.9–2GB loaded
- Model weights dominate external RSS; KV cache not directly visible in psutil RSS
- 7B runs at c=8192 on this hardware with stable swap
- Free RAM after system + Ollama idle: ~11–12GB

### 14B estimate (qwen2.5:14B or comparable Q4_K_M)
| Component | Estimate | Basis |
|-----------|----------|-------|
| Model file size | ~8–9GB | 14B Q4_K_M ratio vs 7B (7B=4.7GB, 14B≈3x params but Q4 is efficient, ~8–9GB plausible) |
| Loaded RSS | ~8–9GB | GGML model load, consistent with file size |
| KV cache at c=2048 | ~1.5GB | 2 layers × 512 head × 128 dim × 2 (K+V) × 2 bytes FP16 ≈ 1.5GB |
| KV cache at c=4096 | ~3GB | linear with context |
| KV cache at c=8192 | ~6GB | linear with context |
| Runtime buffers | ~0.5–1GB | attention scores, intermediate tensors |
| OS overhead | ~1–1.5GB | system baseline |

### Memory budget analysis

**At c=2048:** 8.5GB model + 1.5GB KV + 1GB buffers + 1.5GB OS ≈ 12.5GB → ~2.5GB headroom → **marginal but might barely work**

**At c=4096:** 8.5GB + 3GB + 1GB + 1.5GB ≈ 14GB → ~1GB headroom → **risky; context pressure could trigger swap**

**At c=8192:** 8.5GB + 6GB + 1GB + 1.5GB ≈ 17GB → **exceeds 16GB; OOM or swap death likely**

### Feasibility verdict for native 14B

| Context | Native Status | Reason |
|---------|--------------|--------|
| c=512 | Likely works | KV small, total ~10GB |
| c=2048 | Marginal | ~12.5GB, tight headroom |
| c=4096 | Likely fails | ~14GB, minimal headroom |
| c=8192 | OOM/swap | ~17GB, exceeds hardware |

**Conclusion:** Native 14B at c=2048 is the edge of possibility — may load and run a short answer, but any context growth or other processes could push it over. This makes it a valid first feasibility target. The question is not "does it run at all" but "does it run at useful context lengths without swap death."

**What would make it work:**
- KV/context eviction under pressure
- Lower precision (Q5 or Q4_K_S)
- Reduced context budget
- Aggressive memory guard to abort before swap death
- Capacity-first weight representation to reduce the base load

## E. 30B/32B Contrast Estimate

**Not a target yet — contrast only.**

| Component | Estimate |
|-----------|----------|
| Model file size | ~18–22GB+ for Q4 |
| Loaded RSS | ~18–22GB |
| KV at c=2048 | ~3GB (more layers/heads) |
| Total at c=2048 | ~21–25GB+ → **already exceeds 16GB before buffers/OS** |
| Native path | **Impossible on 16GB** — model itself exceeds available RAM |

**Contrast with 14B:** 14B at c=2048 is marginal but potentially loadable. 30B is already over the threshold at model load alone — no context length or KV pressure needed. This is a fundamentally different category of problem.

**Implications:**
- 14B = capacity problem (weights fit but barely; need better KV/context management)
- 30B = weight residency problem (weights don't fit; need out-of-core or radical compression)
- 14B is the correct first target; 30B requires solving weight residency first

## F. What Context SDI Can and Cannot Solve

### What context SDI CAN help with for 14B:

| Problem | How context SDI helps |
|---------|----------------------|
| Context-driven swap | Reduces active prompt length via packetization; keeps only critical facts |
| KV pressure at high context | Selects only high-value tokens to keep in context window |
| Noisy history bloat | Compresses filler/noise while preserving pinned facts and open loops |
| Factual retention under compression | Structured packet format maintains exact values, paths, tracking IDs |
| Active generation memory growth | Memory guard prevents runs when RAM is already pressured |

### What context SDI CANNOT help with for 14B:

| Problem | Why context SDI can't solve it |
|---------|-------------------------------|
| Base model weight size | 8–9GB is 8–9GB regardless of prompt selection |
| Dense per-token weight streaming | SDI doesn't change how weights are loaded per forward pass |
| Model load RAM | Model must be in RAM before generation starts; SDI operates during generation |
| Q4 weight bandwidth | Each token requires reading Q4 weights from RAM; context SDI doesn't reduce this |
| FFN/activation compute | SDI doesn't skip FFN layers or reduce activation memory |
| KV at very high context | If context SDI compresses a 4K prompt to 2K, it reduces KV — but at c=8192 even compressed KV might exceed RAM |

### The Key Distinction

**If 14B fails because:** weights don't fit in RAM at all → **capacity-first PRT required, context SDI won't help**

**If 14B fails because:** context/KV pressure pushes total above available RAM during generation → **context SDI helps**

The real question for 14B feasibility is: does it fail at load (weight footprint) or during generation (KV/context growth)?

**Preliminary assessment:** 14B at c=2048 probably loads (8.5GB model + 1.5GB KV ≈ 10GB — fits in available RAM). The failure mode is likely generation-time KV growth or context length → context SDI could help there. But at c=4096+ the KV alone becomes the bottleneck, and even context compression might not be enough if the base model is already near the limit.

## G. Capacity-First PRT Requirements

**This is NOT the old PRT custom-op speed path. This is weight-residency-first design.**

### Core Requirement: Resident Footprint Must Shrink

The active memory representation of the model must fit under a strict budget. The goal is not to load the full model into RAM — it's to represent the model in a form that fits and activates only what's needed.

**Hard constraints:**
1. **No full f32 expansion** — f32 decode of Q4 is not a solution (doubles size, likely exceeds budget immediately)
2. **No per-layer materialization that exceeds budget** — if a single layer in FP16 exceeds available headroom, that's a failure
3. **No sidecar larger than what it replaces** — compressed format must actually be smaller than the native representation
4. **Bounded working set** — runtime must know exactly what's resident and never exceed it
5. **Graceful degradation** — quality can drop under extreme memory pressure, but not to garbage or swap death

### Six Design Requirements

**1. Resident footprint reduction:**
- Active representation must fit under strict memory budget (~8–10GB for 14B, leaving headroom)
- Consider: low-rank factorization, structured sparsity, quantized residual planes
- Each activated component must be bounded in memory

**2. Progressive activation:**
- Activate only required weight planes/blocks/channels per token/generation step
- Not all parameters needed for every token; sparse activation patterns can skip FFN blocks
- Layer/block-level paging: keep only active blocks resident, fetch others on demand
- Quality degrades gracefully — fewer planes used = slightly lower quality but still coherent

**3. Backend/native layout awareness:**
- Design around GGML/CPU memory layout from day one — not a retrofit
- GGML's tensor layout is optimized for CPU cache lines; custom ops fighting this will lose
- Consider: GGML's existing quantization support (Q4_K_M etc.) may already be the best starting point
- Avoid custom ops unless they give a specific capacity advantage over native GGML path

**4. Bounded working set with explicit budget:**
- Every layer/block/weight page must be explicitly budgeted in the runtime
- Runtime tracks active resident memory in real time
- Memory guard must know when approaching budget and throttle/prefetch/evict before swap occurs
- No hidden allocations — every tensor has a known owner and size

**5. Verification/fallback:**
- Short canary prompts before/during generation to detect collapse or degradation
- Compare outputs against known baseline where possible
- Fallback may be: fewer weight planes, lower precision, or fallback to simpler model
- Never fall back to cloud/GPU — must stay CPU-only

**6. Out-of-core possibility:**
- mmap/paging from fast storage (SSD) if RAM is exceeded
- Explicit prefetch schedule for weight pages based on predicted activation order
- Avoid OS-managed swap — runtime controls what page is where
- Streaming must be controlled by the SDI runtime, not by accidental OS page faults

### What Capacity-First PRT Is NOT

- Not the old custom-op speed path (that was about beating Q4 speed, not enabling impossible models)
- Not context compression or packetization (that's context SDI's job)
- Not KV cache modification (backend-owned)
- Not model routing (runs the same model in a different way)
- Not dynamic quantization at runtime (pre-quantized weights only)

## H. Impossible-Model Benchmark Definition

**Design only — do not run yet. No 14B pull authorized.**

### Native Baseline
```
Environment:
- Strict memory guard (15GB total, ~1GB headroom minimum)
- Shortest safe prompt (~50 chars: "What is the capital of France?")
- Pre-run memory check (abort if RAM > 13GB before start)

Run:
- Attempt model load at c=512 (minimum context for 14B to be usable)
- Generate bounded output (max 20 tokens)
- Strict memory guard active throughout

Record:
- Load success/failure (RSS snapshot at load)
- Swap delta (from pre to post generation)
- Output sanity (correct answer for factual task)
- Wall time (load time + first token time + total generation)
- Abort reason if failed (OOM/kill/swap/OOM-error)

Pass native: completes coherently with <0.1GB swap delta
Fail native: OOM, swap death, load error, or output garbage
```

### SDI/PRT Candidate Path
```
Same environment, same guard
Same prompt and task

If context compression applicable:
- Apply context SDI (packet format, pinned facts, compressed history)

If capacity-first representation available (future):
- Use progressive weight activation
- Bounded output
- Memory guard active

Record:
- Load/RSS behavior
- Swap delta
- Coherent output check (same criteria as native)
- Token rate (if generating)
- Fallback triggered / not triggered

Pass candidate: produces coherent answer without swap death, even if slower
Fail candidate: garbage, OOM, swap death, or only works via cloud/GPU
```

### Pass/Fail Criteria

| Outcome | Native | Candidate | Verdict |
|---------|--------|-----------|---------|
| Coherent + no swap | Pass | Pass | Inconclusive — native works; not a true feasibility target |
| Fails/coherent | Fail | Pass | **Candidate wins — feasibility demo achieved** |
| Fails/garbage | Fail | Fail | Both fail — problem not solved |
| Coherent + heavy swap | Pass | Pass | Both work but candidate has worse memory behavior |
| OOM/kill | Fail | Pass | Candidate avoids death — meaningful win |
| OOM/kill | Fail | Fail | Neither survives — need different approach |

### Minimum viability tasks (3 required):
1. **Short factual:** "What is the capital of France?" → "Paris"
2. **Instruction-following:** "Write a 3-sentence summary of: [20 words]" → coherent 3-sentence output
3. **Exact retrieval:** Retrieve embedded tracking ID from noisy context

All 3 must produce coherent output for the run to count as a feasibility success.

## I. Recommended Next Phase

**Phase 28F: 14B native feasibility preflight design + capacity gap analysis**

Still no 14B pull/run. Design what the preflight check would look like and analyze the capacity gap.

Scope:
1. Design the exact preflight harness: what commands/scripts would be run to probe 14B load behavior
2. Define the exact metrics captured: RSS at load, swap delta, first-token time, output sanity
3. Analyze capacity gap: given 14B memory requirements, what would capacity-first PRT need to achieve to make 14B work?
4. Identify the specific threshold where 14B goes from "marginal" to "fails" — and which PRT mechanism addresses that threshold
5. Determine whether context SDI alone is enough for c=2048 14B or whether weight residency is also required

**This is a checkpoint before any 14B pull.** No 14B model files will be staged.

## J. Allowed Claims

- 14B Q4 on 16GB RAM is estimated to use ~8–9GB for weights + ~1.5–6GB for KV depending on context length
- At c=2048, 14B is marginal (~10GB total); at c=8192 it likely exceeds available RAM
- Context SDI can help with context/KV pressure but not base model weight size
- Capacity-first PRT requires resident footprint reduction, progressive activation, and bounded working set
- 30B likely exceeds 16GB at model load alone — fundamentally different problem from 14B

## K. Forbidden Claims

- ❌ 14B feasibility demonstrated (only estimate, not run)
- ❌ PRT v3 implemented
- ❌ Weight residency solved
- ❌ 14B runs on this hardware
- ❌ Speedup
- ❌ Production readiness
- ❌ 30B support
- ❌ KV cache modified

## L. Models/Sidecars/F32 Refs Staged?
No. No 14B model files staged or pulled.

## M. Secrets Detected?
No secrets in any committed files.

## N. Tags Touched?
No tags created, modified, or pushed.

---

## Verdicts
- ✅ `PASS_PHASE28E_14B_FEASIBILITY_ANALYSIS`
- ✅ `PASS_IMPOSSIBLE_MODEL_DEMO_DEFINED`
- ✅ `PASS_CONTEXT_LAYER_LIMITS_DEFINED`
- ✅ `PASS_CAPACITY_FIRST_PRT_REQUIREMENTS_DEFINED`
- ✅ `RECOMMEND_14B_PREFLIGHT_DESIGN`
- ✅ `RECOMMEND_CAPACITY_PRT_ARCHITECTURE`
- ✅ `BLOCKED_REPO_STATE` (old untracked reports remain untracked)