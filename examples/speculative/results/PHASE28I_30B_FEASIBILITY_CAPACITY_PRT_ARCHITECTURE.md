# Phase 28I: 30B/32B Feasibility Estimate + Capacity-First PRT Architecture

## A. Branch & Commit
- Branch: `experimental/prt-phase19a-alt-sidecar-backed`
- HEAD before: `100ab9339` (Phase 28H-R)
- HEAD after: `??` (docs only, no tag)

## B. Phase 28H-R Recap (anchors for estimates)
| Measured Fact | Value |
|--------------|-------|
| qwen2.5:14b model size | ~9.0 GB on disk, ~8.4 GB loaded |
| qwen2.5:14b RSS | ~13 GB total (model + daemon overhead) |
| Native 14B speed | ~5 tok/s at c≤2048 |
| Swap behavior | 0–148 MB delta across all runs |
| 14B verdict | Edge-feasible native baseline, not impossible |

## C. Target Ladder

| Tier | Model | Context | Status | Notes |
|------|-------|---------|--------|-------|
| 1 | 7B | any | **comfortable** | Fast, stable, bounded WS passed |
| 2 | 14B | c≤2048 | **edge-feasible native baseline** | ~5 tok/s, coherent, stable swap |
| 3 | 14B | c=4096+ | **untested pressure zone** | May work; needs explicit approval |
| 4 | 30B/32B | any | **true impossible-model target** | Native Q4 exceeds 16GB RAM before KV |

## D. 30B/32B Native Feasibility Estimate

### Model Size Scaling
| Model | Parameters | Q4 File Size | RSS Load Footprint | KV c=512 | KV c=1024 | KV c=2048 | KV c=4096 |
|-------|-----------|--------------|-------------------|----------|-----------|-----------|-----------|
| qwen2.5:7B | 7B | ~4.7 GB | ~7 GB | ~0.5 GB | ~1 GB | ~2 GB | ~4 GB |
| qwen2.5:14B | 14B | ~9.0 GB | ~13 GB | ~1 GB | ~2 GB | ~3.5 GB | ~7 GB |
| qwen2.5:32B | 32B | **~19 GB** | **~23 GB+** | ~2 GB | ~4 GB | ~8 GB | ~16 GB |
| llama3.1:32B | 32B | **~18 GB** | **~22 GB+** | ~2 GB | ~4 GB | ~8 GB | ~16 GB |

**Anchored from local measurements:**
- 14B at c≤2048: model ~9GB, RSS ~13GB — confirmed ✅
- Scaling ratio 14B→32B: ~2.1× parameters → ~2.1× file size
- RSS-to-file ratio for 14B: 13GB / 9GB = 1.44× (daemon overhead + KV)

### 30B/32B Native Failure Point

| Scenario | Model+KV | Total RSS | Available | Outcome |
|----------|----------|-----------|-----------|---------|
| 32B c=512 | ~21 GB | ~22 GB | 16 GB | **OOM before generation** |
| 32B c=1024 | ~23 GB | ~24 GB | 16 GB | **OOM at load or first token** |
| 32B c=2048 | ~27 GB | ~28 GB | 16 GB | **OOM guaranteed** |
| 32B c=4096 | ~35 GB | ~36 GB | 16 GB | **OOM guaranteed** |

**Estimated failure mode:** OOM kill or severe swap thrashing at load time or first KV allocation. Native path fails before producing any output.

### Why Native Fails
1. **Weight footprint alone exceeds available RAM:** 32B Q4 ~18-19GB file → ~22GB+ RSS. 16GB RAM can't hold that + KV + daemon overhead.
2. **No OS-level out-of-core:** llama.cpp/Ollama loads full model into RAM. No layer streaming by default.
3. **KV allocation triggers OOM:** Even if weights somehow fit, KV at c≥1024 pushes total past 16GB.
4. **OS swap death:** Not graceful degradation — Linux OOM killer or swap thrashing.

### Estimated Constants Used
| Parameter | Value | Basis |
|-----------|-------|-------|
| 32B Q4 file size | ~18-19 GB | Ollama standard quant; 2.1× 14B scaling |
| RSS overhead ratio | 1.44× | Measured 14B: 13GB/9GB |
| KV per token | ~2 MB | Standard Q4 KV overhead; 14B measured ~1GB at c=512 |
| OS headroom needed | ~1 GB | Safety buffer for daemon + buffers |
| Ollama daemon base | ~4.6 GB | Measured 14B: ~13GB - ~8.4GB model |

## E. Impossible-Model Success Definition

**Core principle:** Any coherent generation is faster than impossible.

### Minimum Success Criteria
| Criterion | Requirement |
|-----------|-------------|
| Hardware | CPU-only, 16GB RAM, no GPU, no cloud |
| Swap | No swap death, delta ≤ 250MB |
| Output | Coherent, bounded (≤256 tokens) |
| Correctness | At least one factual answer correct |
| Correctness | At least one instruction-following answer correct |
| Correctness | At least one exact retrieval answer correct |
| Speed | Slow but measurable (>0.1 tok/s) |

### Pass Condition
Native path fails (OOM/swap/unusable) AND candidate capacity-first path produces coherent bounded output under guard.

### Failure Conditions
- Output is garbage or hallucinated
- OOM or swap death
- Process hang or timeout
- Requires smaller model to succeed
- Requires cloud/GPU
- Requires staging files larger than disk budget

## F. Capacity-First PRT Architecture

**This is NOT the old PRT custom-op speed path.** That path was about beating Q4 decode speed. This is about making a model run at all on constrained hardware.

### Design Principles

**1. Resident Footprint First**
- Reduce active resident weights below native Q4 footprint
- Active resident = weights actively in CPU/GPU memory during generation
- Sidecar/compressed representation must be smaller than what it replaces
- No f32 full decode; Q4 or lower only for resident weights
- No per-layer materialization that exceeds budget

**2. Progressive Activation**
- Run with fewer planes/blocks/channels first
- Add residual detail only when memory/time allows
- Quality degrades gracefully, not catastrophically
- More residual planes used as optional enhancement, not requirement

**3. Layer/Block Paging**
- Only keep active layer/block/chunk resident
- Use mmap or explicit runtime paging — do not rely on OS swap
- Runtime owns the paging schedule, not the kernel
- Predictable memory budget, not best-effort OS thrashing

**4. Native-Layout Aware**
- Design around GGML/llama.cpp tensor layout from day one
- Avoid fighting matmul shape rules
- No custom op unless it gives a capacity advantage
- Awareness of memory bandwidth and L1/L2/L3 cache behavior

**5. Verification/Fallback**
- Tiny canaries before full generation
- Collapse detection (output quality below threshold)
- Optional re-run with more planes
- Compare known short outputs where possible
- Fallback to smaller model if capacity-first path fails

**6. Hybrid with SDI Context Layer**
- SDI v0.1: reduces KV/context pressure (above weights)
- Capacity-first PRT: reduces weight-residency pressure (below KV)
- Both required for true impossible-model feasibility
- SDI is context-compression; PRT is weight-compression

### Architecture Diagram
```
[User Prompt]
       ↓
[SDI Context Layer]  ← KV/context compression (already built)
       ↓
[Capacity-First PRT] ← weight-residency layer (new design)
       ↓
[GGML/llama.cpp backend] ← native layout, mmap/paging
       ↓
[Ollama or llama.cpp binary]
```

## G. Candidate Mechanisms

### A. Out-of-Core Layer Paging
| Aspect | Detail |
|--------|--------|
| Resident memory impact | Reduces active weights to ~1 layer at a time (~0.6GB for 32B vs ~18GB full) |
| Implementation complexity | Medium — mmap layer files, explicit prefetch, page scheduling |
| Quality risk | Low — full model computed, just slower |
| Fit to 30B/32B | **High** — direct answer to weight residency problem |
| Why it may work | Only ~1 layer needs to be resident at once; disk is cheap and available |
| Why it may fail | Disk I/O bandwidth may make generation too slow to be useful |

### B. Progressive Residual Planes
| Aspect | Detail |
|--------|--------|
| Resident memory impact | Keep coarse/primary planes resident; page residual planes |
| Implementation complexity | High — requires model architecture knowledge, plane decomposition |
| Quality risk | Medium — quality depends on plane selection |
| Fit to 30B/32B | Medium — architectural assumption may not hold |
| Why it may work | Only a subset of weight planes needed for baseline quality |
| Why it may fail | Residual plane selection is model-specific and hard to validate |

### C. Sparse FFN/Block Activation
| Aspect | Detail |
|--------|--------|
| Resident memory impact | Reduces active FFN blocks/channels |
| Implementation complexity | High — requires identifying which blocks to skip |
| Quality risk | **High** — incorrect block skipping causes hallucinations |
| Fit to 30B/32B | Low — too risky without extensive validation |
| Why it may work | FFN is majority of model size; skipping saves memory |
| Why it may fail | No cheap way to know which blocks are "safe to skip" |

### D. Low-Rank/Residual Reconstruction
| Aspect | Detail |
|--------|--------|
| Resident memory impact | Keep low-rank base (~4-6GB) resident; reconstruct residual on demand |
| Implementation complexity | Very high — needs trained decomposition, reconstruction validation |
| Quality risk | **High** — reconstruction may introduce errors |
| Fit to 30B/32B | Low — no validation path without extensive training |
| Why it may work | Low-rank base captures most of model behavior |
| Why it may fail | Quality collapse on reconstruction, no validation harness |

### E. Native-Layout Compressed Sidecars
| Aspect | Detail |
|--------|--------|
| Resident memory impact | Compressed representation smaller than Q4 it replaces |
| Implementation complexity | Very high — requires understanding Phase 10E failure (ggml_map_custom2 bug) |
| Quality risk | **Very high** — Phase 10E showed memory corruption in mixed layout |
| Fit to 30B/32B | Low — Phase 10E blocked this path; needs clean re-implementation |
| Why it may work | Layout-compatible compression avoids f32 expansion bug |
| Why it may fail | Phase 10E blocker; clean path not yet designed |

### F. Disk/SSD Streaming
| Aspect | Detail |
|--------|--------|
| Resident memory impact | Minimal — model on disk, stream to CPU as needed |
| Implementation complexity | Medium — Ollama/llama.cpp already do this partially |
| Quality risk | None — same compute as native |
| Fit to 30B/32B | **High** — Ollama already supports layer streaming |
| Why it may work | llama.cpp has partial KV offloading; extend to weights |
| Why it may fail | I/O bandwidth bottleneck; current KV offloading only |

## H. Mechanism Ranking

| Rank | Mechanism | Resident Mem | Quality Risk | Impl Complexity | Fit to 30B | Notes |
|------|-----------|-------------|-------------|-----------------|-----------|-------|
| 1 | **A. Out-of-core layer paging** | ~0.6GB/layer | Low | Medium | **High** | Direct; model already supports mmap |
| 2 | F. Disk/SSD streaming | Minimal | None | Medium | **High** | Extend llama.cpp KV offloading |
| 3 | B. Progressive residual planes | ~4-6GB base | Medium | High | Medium | Architectural assumption risky |
| 4 | C. Sparse FFN activation | ~8-10GB | **High** | High | Low | Quality collapse risk too high |
| 5 | D. Low-rank reconstruction | ~4-6GB | **High** | Very High | Low | No validation path |
| 6 | E. Compressed sidecars | Variable | **Very High** | Very High | Low | Phase 10E blocker; needs clean design |

**Best candidate: A (out-of-core layer paging)** — direct weight residency solution, low quality risk, no architectural assumptions.

**Secondary: F (disk/SSD streaming)** — existing llama.cpp infrastructure, low quality risk.

**Both can be combined:** Out-of-core paging for weights + SDI for context.

## I. Recommended Next Phase

**Preferred: Phase 28J — Out-of-core layer paging design + 30B preflight Stage 0**

Specifically:
1. Design out-of-core layer paging architecture for 30B Q4
2. Identify Ollama/llama.cpp layer streaming primitives
3. Define paging schedule and memory budget
4. Stage 0 preflight only: check if 30B model is available in Ollama, don't pull
5. Estimate disk I/O requirements

**Not recommended yet:**
- Compressed sidecars (Phase 10E blocker unresolved)
- Low-rank reconstruction (no validation path)
- Sparse FFN (too risky without validation)
- 14B high-context (same model class, not the moonshot)

## J. Safety Checklist
| Item | Status |
|------|--------|
| Models/sidecars/f32 refs staged? | **NO** ✅ |
| Raw logs/captures staged? | **NO** ✅ |
| Secrets detected? | **NO** ✅ |
| Tags touched? | **NO** ✅ |
| 30B/32B model pulled? | **NO** ✅ |
| llama.cpp modified? | **NO** ✅ |
| PRT implementation used? | **NO** ✅ |

## Verdicts
- `PASS_PHASE28I_30B_FEASIBILITY_ESTIMATE`
- `PASS_CAPACITY_FIRST_PRT_ARCHITECTURE_DEFINED`
- `PASS_14B_RECLASSIFIED_AS_EDGE_FEASIBLE`
- `RECOMMEND_CAPACITY_PRT_SPEC`
- `RECOMMEND_OUT_OF_CORE_LAYER_PAGING`
- `RECOMMEND_30B_STAGE0_PREFLIGHT`
- `BLOCKED_REPO_STATE_CLEAN`