# Phase 28F: 14B Native Feasibility Preflight Design + Capacity Gap Analysis

## A. Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## B. Current HEAD
`316fa4004`

## C. 14B Feasibility Hypothesis

**Current hypothesis (unverified):**
14B Q4 on 16GB RAM will likely:
- Load successfully (model weights ~8–9GB fit in available RAM with headroom)
- Fail at generation-time due to KV/context pressure pushing total above available RAM
- c=2048 is the edge; c=4096+ is likely unsafe

**Key unknowns to determine:**
1. Does model load actually succeed (RSS within estimate)?
2. Does generation at c=2048 stay stable (swap delta < 250MB)?
3. Is the failure mode KV/context (context SDI can help) or base weight footprint (capacity-first PRT required)?
4. Does Ollama's internal KV management work at all on 14B, or does it fail at load?

**The preflight must determine which failure mode is dominant before any solution is designed.**

## D. Staged Preflight Design

**Design only — do not execute. Explicit Matt approval required for each stage.**

### Stage 0 — Disk/Model Availability Check

**Purpose:** Confirm model exists and is safe to pull before any pull action.

```
Actions:
1. List available Ollama models — find candidate 14B (qwen2.5:14b or comparable)
2. Record model name, file size, quantization
3. Check free disk space — must have >=2x model size free
4. Estimate disk cost of pull

Abort conditions:
- Model not available in Ollama registry
- Model size >10GB (too close to RAM limit)
- Disk free <2x model size

Output: model name, size, disk requirement, Matt approval required before Stage 1
```

### Stage 1 — System Readiness Check

**Purpose:** Confirm hardware is in a clean state before model load.

```
Pre-run checks:
1. Free RAM — must be >=6GB available
2. Swap used — must be <=1GB
3. Stale Ollama processes — kill any leftover runners
4. Other large processes — confirm nothing else consuming RAM
5. Disk free — confirm space for potential model files
6. Ollama daemon — confirm responding

Abort conditions:
- Swap used >1GB
- Available RAM <6GB
- Ollama not responding
- Other large processes consuming >2GB

Output: system ready flag, memory snapshot, Matt approval required before Stage 2
```

### Stage 2 — Model Pull and Load (No Generation)

**Purpose:** Pull model and observe load behavior without generation pressure.

```
Actions:
1. Pull approved 14B model (requires Matt approval)
2. Record disk before/after pull
3. Load model into Ollama (warmup only — no generation)
4. Record RSS immediately after load
5. Record swap delta from pre-load to post-load

Abort conditions:
- Pull fails (network/disk/model not found)
- Load causes OOM kill
- Load causes swap spike >500MB
- RSS after load >11GB (model loaded but no headroom for KV)

Output: model loaded flag, RSS, swap delta, Matt approval required before Stage 3
```

### Stage 3 — Tiny Generation at c=1024 or c=2048

**Purpose:** First generation test with maximum safety margin.

```
Prompt: "Return only the capital of France."
Context: c=1024 (safer than c=2048 for first generation)
Max tokens: 10
Timeout: 60s

Abort conditions:
- Swap delta >250MB during generation
- Available RAM drops below 2GB
- OOM kill
- Generation timeout >60s
- Output is garbage (not coherent short answer)

Output: generation success/failure, swap delta, RAM delta, elapsed time, output sanity
```

### Stage 4 — Short Instruction-Following at c=2048

**Purpose:** Confirm model can follow instructions coherently at target context.

```
Prompt: "Write a 3-sentence summary of: The quick brown fox jumps over the lazy dog near the riverbank."
Context: c=2048
Max tokens: 50
Timeout: 120s

Abort conditions:
- Swap delta >250MB
- Available RAM drops below 2GB
- OOM kill
- Generation timeout >120s
- Output not coherent multi-sentence text

Output: generation success/failure, swap delta, RAM delta, elapsed time, output sanity
```

### Stage 5 — Exact Retrieval at c=2048

**Purpose:** Confirm model can retrieve exact embedded string under noisy context.

```
Prompt format: [2000 char filler] + tracking ID "SDI-2026-Q2-v011" + [question asking for the ID]
Context: c=2048
Max tokens: 64
Timeout: 120s

Abort conditions:
- Swap delta >250MB
- Available RAM drops below 2GB
- OOM kill
- Generation timeout >120s
- Output does not contain "SDI-2026-Q2-v011"

Output: generation success/failure, swap delta, RAM delta, elapsed time, output contains ID

Note: This test also determines whether context SDI packetization could help if this stage fails.
```

### Stage Progression Rules

- Stage 0 must pass before Stage 1
- Stage 1 must pass before Stage 2 (model pull)
- Stage 2 must pass before Stage 3 (generation)
- Stage 3 must pass (c=1024) before Stage 4 (c=2048)
- Stage 4 must pass before Stage 5
- **No c=4096 in first 14B preflight phase**
- **No c=8192 in first 14B preflight phase**
- **Any abort stops the preflight and reports classification**

## E. Candidate 14B Model Criteria

**Do not pull until explicit Matt approval for Stage 2.**

### Preferred characteristics:
- **Qwen family** (qwen2.5:14b) — consistent with prior eval models, Ollama native support
- **GGUF format** — standard, well-understood quantization
- **Q4_K_M quantization** — good balance of size vs quality; not largest Q4 but most stable
- **File size <=9GB** — keeps load within safety margin
- **Ollama registry available** — no manual download required
- **CPU-only runtime** — no GPU requirement

### Documented candidates:

| Model | Size | Quant | Status |
|-------|------|-------|--------|
| qwen2.5:14b | ~9GB (est) | Q4_K_M (default) | Available in Ollama registry — **primary candidate** |
| qwen2.5:14b-instruct | ~9GB (est) | Q4_K_M (default) | Available — alternative |
| other 14B Q4 models | varies | Q4_K_M or similar | Verify Ollama availability before considering |

### Rejection criteria:
- Model size >10GB (insufficient headroom for KV at any context)
- Not available in Ollama (manual pull not allowed without explicit approval)
- Requires GPU (not CPU-only)
- Not a standard GGUF quantization (cannot verify size assumptions)

## F. Memory Budget

### Component Estimates

| Component | Estimate | Notes |
|-----------|----------|-------|
| 14B Q4 weights | ~8–9GB | Based on qwen2.5:7B ~4.7GB, 14B ~3x params but efficient Q4 |
| KV cache at c=512 | ~0.5GB | Conservative — very short context |
| KV cache at c=1024 | ~1GB | Conservative |
| KV cache at c=2048 | ~1.5GB | From Phase 28E analysis |
| KV cache at c=4096 | ~3GB | Linear scaling |
| KV cache at c=8192 | ~6GB | Exceeds available headroom |
| Runtime buffers | ~0.5–1GB | Attention scores, intermediate tensors |
| OS/system baseline | ~1.5GB | Non-negotiable reserve |
| Guard margin | ~1GB | Safety headroom |

### Memory Budget Table

| Scenario | Weights | KV | Buffers | OS | Total | Headroom | Status |
|----------|---------|-----|---------|-----|-------|----------|--------|
| c=512 | 8.5GB | 0.5GB | 0.5GB | 1.5GB | 11GB | ~4GB | 🟢 Likely safe |
| c=1024 | 8.5GB | 1GB | 0.5GB | 1.5GB | 11.5GB | ~3.5GB | 🟢 Safe margin |
| c=2048 | 8.5GB | 1.5GB | 0.75GB | 1.5GB | 12.25GB | ~2.75GB | 🟡 Marginal |
| c=4096 | 8.5GB | 3GB | 1GB | 1.5GB | 14GB | ~1GB | 🔴 Risky |
| c=8192 | 8.5GB | 6GB | 1GB | 1.5GB | 17GB | — | 🔴 OOM/swap |

**All KV estimates are approximate.** Actual KV may be smaller or larger depending on Ollama's internal management, head dimensions, and layer count. The estimates are directionally correct but not precision measurements.

## G. Safety Guard Definitions

### Hard Abort Conditions (any one triggers immediate stop)

| Condition | Threshold | Action |
|-----------|-----------|--------|
| Pre-run swap used | >1GB | Abort before model pull |
| Available RAM before load | <6GB | Abort before model pull |
| Swap delta during load | >500MB | Abort — model load failed |
| RSS after load | >11GB | Abort — model loaded but no headroom for generation |
| Swap delta during generation | >250MB | Abort — generation causing swap |
| Available RAM during generation | <2GB | Abort — headroom exhausted |
| OOM kill | Any | Abort — classify as LOAD_FAIL |
| Machine sluggish | Observable | Abort — system under pressure |
| Generation timeout | >60s (Stage 3), >120s (Stage 4–5) | Abort — hang or extreme slowdown |

### Pre-Run Checklist (must all be true before Stage 2)

```
□ swap used <=1GB
□ available RAM >=6GB
□ no stale Ollama processes
□ Ollama daemon responding
□ disk free >=2x model size
□ no other large processes consuming RAM
```

### During-Generation Checklist (checked continuously)

```
□ swap delta <250MB
□ available RAM >2GB
□ process not OOM killed
□ generation not timing out
□ output coherent (for generation stages)
```

## H. Success/Failure Classification Scheme

### Classification Definitions

**LOAD_FAIL:**
Model cannot be pulled or loaded. OOM during load. Disk insufficient.
→ Implication: Base weight footprint exceeds RAM. Capacity-first PRT required. Context SDI irrelevant.

**LOAD_PASS_GEN_TIMEOUT:**
Model loads successfully but generation times out or hangs.
→ Implication: Model loads but is too slow to be usable. Speed issue, not memory. May be hardware limitation.

**GEN_PASS_SWAP_RISK:**
Generation succeeds but swap grows meaningfully (>250MB delta).
→ Implication: Generation works but is unstable. Context SDI could help by reducing KV. Capacity-first PRT may also help.

**GEN_PASS_STABLE:**
Generation succeeds with stable swap (<50MB delta) and coherent output.
→ Implication: Model works at this context on this hardware. 14B is feasible at c=2048. Next: probe c=4096 carefully.

**CONTEXT_LIMITED:**
c=2048 works but c=4096 would be unsafe by budget estimates.
→ Implication: Context SDI may extend usable context range by compressing active prompt. Capacity-first PRT not required if KV is the only issue.

**CAPACITY_PRT_REQUIRED:**
Base weights/load footprint leave too little headroom even at c=2048.
→ Implication: Weight residency solution required, not just context reduction. Capacity-first PRT must be designed.

**CONTEXT_SDI_SUFFICIENT:**
Base model loads with headroom; only KV/context pressure is risky at higher contexts.
→ Implication: Context SDI alone can extend range. Capacity-first PRT may not be needed for 14B at c=2048.

## I. Implication Matrix

| Classification | Context SDI helps? | Capacity-first PRT needed? | Next action |
|---------------|-------------------|---------------------------|-------------|
| LOAD_FAIL | No | Yes — weight residency required | Design capacity-first PRT |
| LOAD_PASS_GEN_TIMEOUT | No | Maybe — speed, not memory | Accept slower hardware? |
| GEN_PASS_SWAP_RISK | Yes — reduces KV pressure | Possibly — if context alone not enough | Probe context SDI impact |
| GEN_PASS_STABLE | Yes — can extend range | No | Probe higher contexts |
| CONTEXT_LIMITED | Yes — compresses active context | No | Apply context SDI |
| CAPACITY_PRT_REQUIRED | No | Yes — must reduce base load | Design capacity-first PRT |
| CONTEXT_SDI_SUFFICIENT | Yes — primary solution | No | Deploy context SDI |

**The preflight result determines the entire next phase direction.**

## J. Recommended Next Phase

**Phase 28G: 14B model availability + disk/RAM preflight**

Scope:
1. Stage 0 only — confirm which 14B models are available in Ollama registry, their sizes, quantization
2. Record candidate model list with sizes
3. Report disk/RAM preflight status (Stage 1 system readiness)
4. **No model pull without explicit Matt approval**
5. **No generation without explicit Matt approval**

This confirms we have a viable 14B candidate and the system is ready before any pull happens.

**Alternative:** If Matt wants capacity-first PRT architecture work instead of waiting for 14B availability, Phase 28G could be: capacity-first PRT v3 architecture spec. But the preflight approach is recommended to stay grounded in reality.

## K. Allowed Claims

- 14B Q4 estimated to need ~8–9GB for weights
- c=2048 total estimated ~12.25GB with KV+buffers+OS — marginal but possibly loadable
- c=4096 estimated ~14GB — risky
- c=8192 estimated ~17GB — likely OOM/swap on 16GB system
- Context SDI can help with generation-time KV pressure if that is the failure mode
- Capacity-first PRT is required if base weight footprint is the dominant blocker
- All preflight stages have explicit abort guards
- No 14B model has been pulled or run; all figures are estimates

## L. Forbidden Claims

- ❌ 14B feasibility demonstrated (design only)
- ❌ Any 14B model run on this hardware
- ❌ PRT v3 implemented
- ❌ Capacity-first PRT designed
- ❌ Speedup
- ❌ Production readiness
- ❌ 30B support

## M. Models/Sidecars/F32 Refs Staged?
No. No 14B model files staged or pulled.

## N. Secrets Detected?
No secrets in any committed files.

## O. Tags Touched?
No tags created, modified, or pushed.

---

## Verdicts
- ✅ `PASS_PHASE28F_14B_PREFLIGHT_DESIGN`
- ✅ `PASS_MEMORY_BUDGET_DEFINED`
- ✅ `PASS_ABORT_GUARDS_DEFINED`
- ✅ `RECOMMEND_14B_AVAILABILITY_PREFLIGHT`
- ✅ `RECOMMEND_CAPACITY_PRT_ARCHITECTURE` (if 14B estimates show insufficient headroom)
- ✅ `BLOCKED_REPO_STATE` (old untracked reports remain untracked)