# Phase 28J: Out-of-Core Layer Paging Design + 30B Stage 0 Preflight

## A. Branch & Commit
- Branch: `experimental/prt-phase19a-alt-sidecar-backed`
- HEAD before: `0483d7ff4` (Phase 28I)
- HEAD after: `??` (docs only, no tag)

## B. Phase 28I Recap (anchors)
| Finding | Value |
|---------|-------|
| 14B c≤2048 | edge-feasible native baseline (~5 tok/s) |
| 30B/32B native | FAILS — ~18GB file → ~22GB+ RSS exceeds 16GB RAM |
| Best mechanism | Out-of-core layer paging (~0.6GB/layer vs ~18GB full) |
| OS swap | NOT a valid memory tier — treated as failure/poison |
| First path | Q2/Q3 base + progressive residual overlay — simpler than original PRT |

## C. 30B Quant Ladder

### Measured anchors (from 14B):
| Model | File Size | RSS | Context |
|-------|-----------|-----|---------|
| qwen2.5:14B Q4_K_M | 9.0 GB | ~13 GB | c≤2048 works |
| Scaling ratio | 2.1× params → 2.0-2.1× size | 1.44× overhead | — |

### 30B/32B estimates by quant:

| Quant | File Size | RSS Estimate | 16GB Feasibility | Quality Risk | PRT Role |
|-------|-----------|-------------|-----------------|-------------|---------|
| **Q4_K_M** | ~18-19 GB | ~22-23 GB | ❌ OOM at load | Low | Reduce residency via paging |
| **Q3_K_M** | ~12-14 GB | ~16-18 GB | ❌ Likely OOM or marginal | Medium | Stabilize under budget pressure |
| **Q2** | ~9-10 GB | ~13-14 GB | ⚠️ Marginal fit | **High** | Residual quality recovery |
| **Q4 + paging** | ~0.6 GB/layer resident | ~1-2 GB resident | ✅ Paging enables | Low | Overlay residuals for quality |

**Key corrections from Matt's design guidance:**
- OS swap = failure/poison, not a valid tier
- AVX2 path assumed (not AVX-512)
- No token/head top-k routing in first path
- First capacity-first PRT = **Q2/Q3 base + progressive residual overlay** — simpler than Phase 10E-style custom op

### Q4 clearly exceeds RAM?
**YES.** 14B Q4 = 9GB file → ~13GB RSS. 32B Q4 = ~18-19GB file → ~22-23GB RSS. 16GB machine cannot hold 22GB+ plus KV plus OS headroom. Native Q4 fails.

### Could Q3 fit?
**Marginal/no.** ~12-14GB file → ~16-18GB RSS. 16GB RAM leaves 0-2GB headroom. KV allocation or daemon overhead likely triggers OOM. Risky without paging.

### Could Q2 fit but produce weak output?
**Possibly.** ~9-10GB file → ~13-14GB RSS. Might load. But Q2 quality collapse is significant — output likely garbage without residual correction. PRT's role here is **quality recovery**, not just loading.

### PRT becomes quality-recovery layer if Q2/Q3 fits?
**YES — this is the refined view.** If Q2/Q3 base model can load, PRT residuals must then recover quality. This is architecturally different from the Phase 10E speed path.

## D. Stage 0 Candidate Discovery

**No 30B/32B models installed. No pulls made.**

Installed models confirmed:
```
qwen2.5:14b  9.0 GB  ✅ (14B reference)
qwen2.5:7b   4.7 GB
qwen2.5:3b   1.9 GB
qwen2.5:0.5b 397 MB
nomic-embed-text:latest 274 MB
```

### Candidates found (registry only, not pulled):

| Candidate | Class | Quant | Est. Size | OOM Potential | Notes |
|-----------|-------|-------|-----------|---------------|-------|
| `qwen3:30b` | MoE 30B (3B active) | Q4_K_M default | ~60GB disk / ~6GB active | ⚠️ Medium — only 3B activated | MoE; Ollama has GPU utilization issues on CPU |
| `qwen3:32b` | Dense 32B | Q4_K_M default | ~19GB disk | ❌ OOM — 32B dense | Not yet tried in this project |
| `nemotron-3-nano:30b` | Dense 30B | Q4 default | ~18GB | ❌ OOM — 30B dense | In Ollama registry |
| `llama3.1:70b` | Dense 70B | Q4_K_M | ~40GB | ❌ OOM — far too large | Not a target |

### MoE caveat:
Qwen3:30b is MoE with 30B total / 3B active per token. Active-weight footprint is ~3B Q4 (~2-3GB) not 30B. **This changes the capacity math significantly** — MoE may be loadable if the inactive expert weights can be paged out. However, Ollama's CPU MoE implementation has known inefficiency (GPU-optimized dispatch). AVX2 CPU path may not be well-optimized.

### Decision: APPROVE_LATER_PULL
- System is Stage 0 ready (swap 530MB < 1GB, RAM 11GB available)
- `qwen3:30b` MoE is the most interesting candidate — only 3B active
- `qwen3:32b` dense is the true impossible-model target
- Neither pulled — awaiting explicit approval

## E. Out-of-Core Layer Paging Architecture

### Core principle
Keep only active layer/block/chunk resident. Page weights from disk/NVMe as needed. Runtime owns paging schedule, not OS swap.

### Paging unit options
| Unit | Resident Size (32B Q4) | I/O per token | Complexity | Notes |
|------|------------------------|---------------|-----------|-------|
| Whole layer | ~0.5-0.8 GB | 1 full layer | Low | Simple; standard llama.cpp layer loop |
| FFN block | ~0.2-0.3 GB | 2 blocks | Medium | Matches transformer block structure |
| Attention block | ~0.1 GB | 1 block | Medium | Q/K/V/O projections |
| Tensor column group | Variable | Variable | High | Depends on quantization layout |
| Residual plane | Variable | Variable | Very High | Requires plane decomposition |

**Recommended: whole layer** — simplest, matches llama.cpp architecture, sufficient memory reduction.

### Residency budget
| Component | Budget |
|-----------|--------|
| Max resident weights | 2 GB |
| KV budget (c=512) | 2 GB |
| OS/headroom reserve | 1 GB |
| Daemon overhead | ~4 GB |
| **Total** | **~9 GB** (within 16GB) |

Per-layer resident: 32B Q4 has ~48-64 layers. 2GB / 48 layers ≈ **42 MB per layer resident**. Actual Q4 layer size ≈ 0.5-0.8 GB. **Layer paging alone is not sufficient — need Q3 or Q2 base quant.**

### Prefetch schedule (autoregressive)
```
Token N:
  1. Prefetch layer K+1 while layer K computes    [I/O concurrent with compute]
  2. Compute layer K (layer K+1 already in RAM)
  3. Evict layer K-2 after use (if not needed for KV回头)

Token N+1:
  4. Prefetch layer K+2 while layer K+1 computes
  5. Compute layer K+1
  6. Evict layer K-1

Key: I/O latency hiding — overlap disk read with compute
Risk: if I/O slower than compute, CPU stalls
```

### Minimum viable prototype
- Not fast — slow is acceptable for feasibility demo
- One coherent bounded answer (≤128 tokens)
- Memory guard: abort if swap delta > 250MB
- No OS swap death
- Pass: coherent output under strict memory budget

### Failure risks
| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Disk bandwidth too slow | High — 15GB/s NVMe, ~0.5GB layer → 30ms per layer | Overlap with compute; start with tiny model |
| mmap page faults → OS thrash | Medium | madvise(MADV_WILLNEED) prefetch |
| CPU idle waiting on I/O | High without overlap | Async prefetch, thread pool |
| Quality collapse | Medium if Q2 base | Residual overlay |
| Wrong layer evicted for KV backprop | Medium | Track KV reuse; keep recent layers |

## F. Capacity-First PRT Architecture Refinement

**Different from Phase 10E PRT (custom-op speed path). That path = speed. This path = feasibility.**

### Option 1 — Paged native quant (Q3) weights
- Out-of-core Q3 pages, no residuals initially
- Tests feasibility only
- Resident: ~1-2GB layers + KV
- Quality: Q3 medium risk, may need residuals

### Option 2 — Q2/Q3 base + residual planes (PREFERRED)
- Resident: Q2 base (~10GB loaded) + 1 active layer
- Paged: residual correction planes from disk
- Selected layers only for residuals (most quantization-sensitive layers)
- PRT role: **quality recovery under memory budget**
- Complexity: Medium-high (plane decomposition + paging + residual overlay)
- This is the simplest path to a working demo

### Option 3 — Selected-layer residual overlay
- Only layers most sensitive to quantization receive residuals
- Cheaper than full residual model
- Needs layer sensitivity analysis

### Option 4 — Progressive retry
- Run cheap pass first (Q2, minimal layers)
- If output fails canary, rerun with more residual planes
- Best for demo: start fast/cheap, escalate only if needed

### Refined first path (Matt correction):
**Q2 base + progressive residual overlay + strict memory budget**
- Simpler than original Phase 10E custom-op path
- No token/head top-k routing
- AVX2 practical path
- Tests: can Q2 base load + can residuals recover quality

## G. 30B Impossible-Model Benchmark v0

### Native baseline stages:
| Stage | Test | Pass Criterion |
|-------|------|---------------|
| 0 | System preflight (swap ≤1GB, RAM ≥6GB) | Guard check |
| 1 | Pull/load under guard | OOM/swap/fail → native fails |
| 2 | Tiny factual prompt (c=128) | Garbage/OOM → fail |
| 3 | Instruction prompt (c=512) | Garbage/OOM → fail |
| 4 | Exact retrieval (c=512) | Wrong/OOM → fail |

### Candidate path stages:
| Stage | Test | Pass Criterion |
|-------|------|---------------|
| 5 | Same factual prompt via paged/residual path | Coherent bounded answer |
| 6 | Same instruction prompt | Coherent bounded answer |
| 7 | Same exact retrieval | Correct fact |

### Pass: native Q4 fails stages 1-4 AND candidate passes 5-7.
### Partial: Q2/Q3 native runs but weak; PRT residuals improve answer.
### Fail: candidate garbage/OOM/hang in stages 5-7.

## H. Recommended Next Phase

**Preferred: Phase 28K — 30B quant-ladder candidate report + Q2/Q3/Q4 feasibility decision gate**

Specifically:
1. Finalize 30B/32B candidate list with exact quant/size estimates
2. Design Q2 base + residual overlay test (what residual would look like)
3. Decision gate: pull qwen3:30b MoE (smallest active footprint) OR qwen3:32b dense (true impossible target)
4. Stage 0 preflight only — no actual pull without explicit approval

**Not recommended yet:**
- 14B c=4096+ (same model class, not the moonshot)
- More SDI context demos (14B is the edge, not the target)
- Phase 10E restart (Phase 10E blocker still unresolved)

## I. Safety Checklist
| Item | Status |
|------|--------|
| Models/sidecars/f32 refs staged? | **NO** ✅ |
| Raw logs/captures staged? | **NO** ✅ |
| Secrets detected? | **NO** ✅ |
| Tags touched? | **NO** ✅ |
| 30B/32B model pulled? | **NO** ✅ |
| llama.cpp modified? | **NO** ✅ |
| OS swap used as valid tier? | **NO** ✅ — explicitly excluded |

## Verdicts
- `PASS_PHASE28J_OUT_OF_CORE_DESIGN`
- `PASS_30B_QUANT_LADDER_DEFINED`
- `PASS_NO_SWAP_AS_TIER`
- `PASS_CAPACITY_PRT_REFINED`
- `PASS_Q2_BASE_PLUS_RESIDUAL_OVERLAY_IDENTIFIED`
- `RECOMMEND_30B_CANDIDATE_GATE`
- `RECOMMEND_Q2_Q3_FEASIBILITY`
- `BLOCKED_REPO_STATE_CLEAN`