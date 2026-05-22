# Phase 28L: Dense Q2 Feasibility + Capacity-First PRT Residual Overlay Spec

## A. Branch & Commit
- Branch: `experimental/prt-phase19a-alt-sidecar-backed`
- HEAD before: `6dd6e5878` (Phase 28K)
- HEAD after: `??` (docs only, no tag)

## B. Dense Q2 Feasibility Hypothesis

**Hypothesis:** A dense 30B/32B Q2 quantized base may fit within 16GB RAM, but quality will be degraded. If it can produce weak/coarse output without swap death, PRT residual overlays can selectively restore quality while staying under memory budget.

**Key architectural distinction — three-layer stack:**
| Layer | Role | What it does |
|-------|------|-------------|
| SDI context layer | KV/context pressure control | Reduces active context size |
| Q2 base model | Capacity/runnability layer | Fits in 16GB, produces coarse output |
| PRT residual overlay | Quality recovery layer | Selectively restores detail under strict memory budget |

**Critical distinction from old PRT:**
- **Old PRT** (Phase 10E path): beat native Q4 speed on models that already ran → speed path
- **New capacity-first PRT**: recover quality for ultra-low-bit models that fit when Q4 does not → capacity path

## C. Native-Fail Baseline Definition

**Future test — do not run in this phase.**

Dense 30B/32B Q4 native path expected failure modes:
| Failure Mode | Evidence |
|-------------|---------|
| RAM/RSS too high at load | 22GB+ RSS > 16GB available |
| First KV allocation failure | c=512 KV pushes total past 16GB |
| Swap explosion | OS begins paging heavily |
| OOM kill | Linux OOM killer terminates process |
| Unusable backend state | Model loads but generation corrupted |

**Purpose:** Establishes why Q2/PRT path is necessary. Q4 must be proven to fail for the demo to have meaning.

## D. Q2 Base Path Design

**Future test only — no execution in this phase.**

Design parameters:
| Parameter | Value |
|-----------|-------|
| Context | c=512 first |
| Prompt | Tiny factual |
| Output bound | ≤128 tokens |
| Guard | swap delta ≤250MB, RAM ≥2GB free |
| No context stress | No long prompts, no multi-turn |
| No broad eval | Single task only |

**Metrics to capture:**
- Load success/failure
- RAM/RSS post-load
- Swap delta
- tokens/sec
- Output coherence (0/1)
- Exact answer correctness (0/1)

**Outcome classifications:**
| Classification | Meaning |
|---------------|---------|
| `Q2_LOAD_FAIL` | Cannot load or immediate OOM |
| `Q2_LOAD_PASS_GARBAGE` | Loads but output is meaningless |
| `Q2_LOAD_PASS_WEAK` | Loads, coherent, but factually wrong |
| `Q2_LOAD_PASS_COHERENT` | Loads, coherent, factually usable |
| `Q2_LOAD_PASS_SWAP_RISK` | Loads but swap delta > 250MB |

## E. PRT Residual Overlay Architecture

### 1. Base Model
- Dense Q2/Q3 model stays resident (or mostly resident with layer paging)
- Provides coarse generation path at low memory cost

### 2. Residual Overlay
- Compressed residual planes/blocks stored separately on disk
- Only selected layers receive residual correction (not all layers)
- Residuals can be mmaped/paged — not all must be resident simultaneously
- Active residual budget is explicit and bounded

### 3. Selection Policy (v0)
- **Fixed selected layers** — no dynamic token routing
- Layer selection based on offline sensitivity/parity tests
- No top-k token/head routing in v0 (per Matt's correction)
- AVX2 practical path, not AVX-512

### 4. Memory Budget
```
Component           Budget
---------           ------
Q2 base model       ~10-13 GB (loaded)
KV (c=512)          ~2 GB
Active residual set ~1-2 GB (selected layers)
OS/headroom         ~1 GB
Daemon overhead     ~4 GB
-----------------
Total              ~16-22 GB  ← exceeds without layer paging
```

**Conclusion: Layer paging required even for Q2 base.** Q2 alone may not fit without paging if KV and residuals are added.

### 5. Execution Modes
| Mode | Active Layers | Residual | Notes |
|------|-------------|----------|-------|
| 0 | Q2 base only | None | Baseline, cheapest |
| 1 | Q2 + N selected | Selected residual layers | Quality improved |
| 2 | Q2 + more | More residual if memory allows | Max quality |
| FALLBACK | Reduce N | Reduce residual set | If memory pressure detected |

### 6. Verification
- Tiny canary prompts before full generation
- Exact answer checks (known factual short outputs)
- Collapse/repetition detection
- Compare against 14B/7B baseline if Q4 30B unavailable

## F. Residual Generation Plan (Design — No Generation Yet)

### Representation Design
```
source reference:  Q4 or higher quality tensor
base:              Q2 quantized tensor
residual:          source - reconstructed_Q2

Compression:
  → ternary (-1, 0, +1) or int2 or int4 planes
  → per-layer metadata stored separately
  → offline cosine similarity / matvec validation before runtime
```

### Hard Requirements
| Requirement | Reason |
|-------------|--------|
| Residual overlay must be smaller than Q4 | Otherwise why not just use Q4 |
| No f32 expansion at runtime | Phase 10E failure mode |
| No sidecar larger than replaced tensor | Memory budget enforcement |
| Layout must be native/backend-aware | Avoid tensor shape conflicts |
| No OS swap as valid tier | Matt's explicit correction |

## G. MVP Experiment (Future)

**Stage 1 — Small model proof-of-concept (if needed)**
- Use qwen2.5:0.5B or qwen2.5:3B as testbed
- Demonstrate Q2 + residual concept at small scale
- Only if it directly informs 30B path
- Not required — skip if it doesn't add information

**Stage 2 — Dense 30B/32B sequence**
1. Q4 native fail baseline (prove it fails under guard)
2. Q2 base load + generation (prove it fits)
3. Q2 + residual overlay — offline parity validation
4. Runtime only after offline parity passes

**Success criteria:**
- Q2 base runs but produces weak/coarse output
- Residual overlay improves targeted outputs or offline parity
- Memory stays under guard (swap delta ≤ 250MB)

**Failure criteria:**
- Q2 cannot load (→ PRT not viable, need other approach)
- Q2 output is garbage even with residuals
- Residual overlay exceeds memory budget
- Residual overlay requires f32 expansion
- Quality recovery is negligible

## H. Recommended Next Phase

**Phase 28M — PRT residual overlay offline math spec**

Rationale: Define the residual representation (ternary/int2/int4 plane math), the parity validation methodology, and the layer selection criteria before touching any 30B model files. Representation must be correct before attempting the 30B path.

**Alternative options:**
- Phase 28M Option B: Q2/Q3 candidate availability preflight (check what's actually available in Ollama)
- Phase 28M Option C: Small-model Q2 residual overlay proof-of-concept (validate concept at small scale first)
- Phase 28M Option D: Dense Q4 native-fail baseline design (prove Q4 fails before designing recovery)

## I. Safety Checklist
| Item | Status |
|------|--------|
| Models/sidecars/f32 refs staged? | **NO** ✅ |
| Raw logs/captures staged? | **NO** ✅ |
| Secrets detected? | **NO** ✅ |
| Tags touched? | **NO** ✅ |
| Any 30B model pulled? | **NO** ✅ |
| llama.cpp modified? | **NO** ✅ |
| OS swap used as valid tier? | **NO** ✅ — explicitly excluded |

## Verdicts
- `PASS_PHASE28L_DENSE_Q2_PRT_OVERLAY_SPEC`
- `PASS_Q2_AS_BASE_DEFINED`
- `PASS_PRT_AS_QUALITY_RECOVERY_DEFINED`
- `PASS_NO_SWAP_AS_TIER`
- `PASS_THREE_LAYER_STACK_DEFINED`
- `RECOMMEND_RESIDUAL_OVERLAY_MATH_SPEC`
- `RECOMMEND_Q2_Q3_CANDIDATE_PREFLIGHT`
- `BLOCKED_REPO_STATE_CLEAN`