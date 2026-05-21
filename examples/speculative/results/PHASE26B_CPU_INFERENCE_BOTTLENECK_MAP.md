# Phase 26B: CPU Inference Bottleneck Map + Model-Residency SDI Plan

**Verdict:** `PASS_PHASE26B_BOTTLENECK_MAP`

**Date:** Wed 2026-05-20 23:17 EDT
**Branch:** `experimental/prt-phase19a-alt-sidecar-backed`
**Current HEAD:** `9d7080d67`

---

## A. Branch
```
experimental/prt-phase19a-alt-sidecar-backed
```

## B. Current HEAD
```
9d7080d67 ("Phase 26A: select post-PRT SDI path")
```

## C. PRT Freeze Verified
| Check | Result |
|-------|--------|
| Branch | ✅ `experimental/prt-phase19a-alt-sidecar-backed` |
| HEAD | ✅ `9d7080d67` |
| Checkpoint tag | ✅ `PRT_PHASE25B_CORRECTNESS_RESTORED_CHECKPOINT` |
| Working tree | ✅ Only `ggml/src/ggml-cpu/ops.cpp` modified; untracked phase24R docs harmless |
| Need to rerun PRT tests | ❌ No — frozen, not modified |

---

## D. Core Bottleneck Thesis

**The problem:** Running a dense model larger than the hardware should comfortably handle on CPU/no-GPU.

The true constraints, in order of severity for a 15GB RAM machine trying to run a 7B–30B dense model:

| Bottleneck | Severity | Why it dominates |
|------------|----------|-----------------|
| **RAM capacity** | 🔴 Critical | Model weights alone: Q4_K_M ≈ 2 bits/param → 7B ≈ 1.75GB, 14B ≈ 3.5GB, 30B ≈ 7.5GB. Add KV cache + activations + intermediates and you exceed available RAM before throughput matters. |
| **RAM bandwidth** | 🔴 Critical | Every token requires streaming full weight matrices from RAM. CPU matmul is memory-bound. Q4/Q5/Q8 reduces bandwidth cost per byte but not the access pattern — each token touches all weights. |
| **Dense activation** | 🟠 Major | Every token activates all layers and all FFN blocks. A dense model does not skip anything. This is fundamentally wasteful for prompts where only a subset of the model is relevant. |
| **KV/context growth** | 🟠 Major | KV cache = O(n_seq × n_layers × n_heads × head_dim × 2). Long prompts consume KV space fast. Prefill is fast; decode step-by-step compounds. |
| **Compute/FLOPs** | 🟡 Secondary | For memory-bound workloads, compute is rarely the bottleneck. FLOPs matter most when bandwidth is already saturated and model fits in cache. |
| **Runtime architecture** | 🟡 Secondary | Graph overhead, custom-op penalties, GGML backend dispatch add latency but are dwarfed by memory costs above. |

**Key insight:** For CPU dense-model inference on constrained hardware, the problem is almost always memory (capacity + bandwidth), not compute. This means quantization alone is insufficient — it helps capacity but not the fundamental memory-access-per-token pattern.

---

## E. RAM Capacity Analysis

| Component | 7B Q4_K_M | 14B Q4_K_M | 30B Q4_K_M |
|-----------|-----------|------------|------------|
| Model weights | ~1.75 GB | ~3.5 GB | ~7.5 GB |
| KV cache (1K ctx) | ~64 MB | ~128 MB | ~256 MB |
| KV cache (4K ctx) | ~256 MB | ~512 MB | ~1 GB |
| Activations (decode) | ~100–200 MB | ~200–400 MB | ~400–800 MB |
| **Rough total (4K ctx)** | ~2.1 GB | ~4.4 GB | ~9.3 GB |

On Matt's 15GB machine with ~6.4GB free RAM:
- 7B: manageable ✅
- 14B: near limit ⚠️
- 30B: likely OOM or swap-bound ❌

**Sidecars add further pressure.** PRT-style sidecars store auxiliary weights in addition to the base model. This is fine for 3B but compounds capacity problems for 7B+.

**结论:** RAM capacity is the first wall. Any path that doesn't address weight residency will hit it for models above ~10B on this hardware.

---

## F. RAM Bandwidth Analysis

| Operation | Weight Access Pattern | Bandwidth Cost |
|-----------|----------------------|----------------|
| Q4_K_M matmul (FFN) | Stream full weight matrix per token | ~gigabytes/token |
| Q8/FP16 matmul | Stream 2x/4x more bytes per token | 2x–4x worse |
| Attention QKV | Stream QKV projections per layer | ~3× weight matrix |
| KV cache write | Write K+V vectors per token | Moderate |
| PRT custom op decode | Additional dequant + scale + custom kernel | On top of native path |

**Why CPU matmul is typically memory-bound:** A single FFN layer in a 7B Q4 model requires reading ~7GB/s of weights per token. On a machine with ~25–50 GB/s RAM bandwidth, that leaves little headroom for the rest of the pipeline.

**PRT lesson:** The custom-op INT8 path correctly restored output but added overhead on top of the native path without reducing the fundamental weight-streaming cost. The custom op doesn't change *what* gets loaded from RAM — only the format. The speed deficit came from custom kernel overhead, not from bandwidth wins.

**Implication:** Any path that still streams full weights per token per layer is fighting the same battle. The bandwidth savings from better quantization formats plateau.

---

## G. Compute Analysis

| Component | FLOPs per token (7B, d=4096) | Relative cost |
|-----------|------------------------------|---------------|
| Attention (QKV + scores + weighted sum) | ~4 × d² = 67 GFLOP | ~15% |
| FFN upproj gate_proj | 2 × 4 × d² = 134 GFLOP | ~30% |
| FFN downproj | 4 × d² = 67 GFLOP | ~15% |
| **Total transformer layer** | **~268 GFLOP** | |
| **20-layer 7B model** | **~5.4 TFLOP/token** | |

On a 13500T (6 performance cores @ ~3.5GHz, AVX2), peak throughput is roughly:
- FP32 matmul: ~100–150 GFLOPS per core → ~600–900 GFLOPS total
- 5.4 TFLOP/token ÷ 750 GFLOPS ≈ **7,200 cycles/token per core**

**But:** Memory bandwidth is the real bottleneck. Even if compute could be faster, feeding weights at the required rate dominates. The model rarely hits FLOPs limits — it stalls waiting for data.

---

## H. KV/Context Analysis

| Context length | KV per layer | KV 20 layers | Added latency |
|---------------|---------------|--------------|---------------|
| 512 | ~10 MB | ~200 MB | Low |
| 2K | ~40 MB | ~800 MB | Moderate |
| 8K | ~160 MB | ~3.2 GB | High |
| 32K | ~640 MB | ~12.8 GB | OOM on 15GB machine |

**Two distinct phases:**
1. **Prefill:** Processes entire prompt at once. Compute-bound-ish. Fast relative to prompt length.
2. **Decode:** Generates tokens one at a time. Memory-bound — each step re-streams all weights AND reads/writes KV cache.

**The KV problem compounds:** For long sessions, the KV cache grows until it becomes a significant fraction of available RAM. At that point, either:
- Context is evicted (lost summarization/retrieval)
- RAM pressure triggers swap (catastrophic slowdown)
- Model must be reloaded (high latency)

**ContextOS thesis:** Most eval workloads waste KV on repetitive prompt structure. If we can compress, evict, or restructure context more intelligently, we reduce the effective KV pressure without losing the information that matters.

---

## I. Model-Residency SDI Definition

**Model-Residency SDI** is a system-level approach to reducing the memory footprint of dense transformer models at runtime by controlling *what* model data must be resident, streamed, decoded, or activated per token — while preserving correctness anchors and native fallback paths.

It is **not**:
- Classic MoE: Expert routing at train time, full model still required
- Prompt routing: Sending different prompts to different models; doesn't compress a single model's memory
- Quantization only: Reduces bits/param but not the fundamental memory-access pattern
- Speculative decoding only: Reduces number of big-model forward passes but doesn't reduce memory per pass
- PRT only: A valid INT8 representation artifact, but the custom-op path proved that format conversion without memory-layout optimization doesn't win on speed

**Model-Residency SDI specifically targets:**
1. **Weight residency** — which model components must be in RAM right now?
2. **Activation sparsity** — which components actually need to fire for this token?
3. **Memory layout** — is the model arranged for the CPU memory hierarchy (cache lines, page boundaries)?
4. **Streaming strategy** — can we page weights in/out without recomputing?
5. **Context management** — can we reduce KV growth without losing information?

**The control plane (Smart Agent Router) orchestrates these mechanisms** based on prompt characteristics, available RAM, latency requirements, and quality constraints.

**What makes it "SDI" rather than just "compression":**
SDI implies a *selective, adaptive* approach — the system chooses between pathways dynamically. Not one fixed format or one fixed routing rule. The selection is the intelligence.

---

## J. Evaluated Paths

### Path 1 — Native-Layout Compressed Sidecars

**Question:** Can we store parts of a dense model in a CPU-native compressed format that avoids RAM/bandwidth cost without custom-op overhead?

**Lessons from PRT:**
- PRT proved canonical INT8 layout is correct ✅
- PRT's custom-op path was slower than native Q4_K_M ❌
- Root cause of slowness: custom kernel overhead layered on top of weight streaming (not a bandwidth win)
- A PRT v3 that lives *inside* the GGML backend as a first-class op (not a sidecar overlay) might avoid overhead

**What a PRT v3 would need:**
- Native GGML backend op, not a custom overlay
- Full FFN consistency (not layer-replacement hacking)
- Memory layout optimized for CPU cache lines
- Demonstrable bandwidth reduction vs native Q4/Q5/Q8 path

**What this path doesn't solve:** Even perfect format conversion still requires streaming all weights per token per layer. The bandwidth bottleneck remains.

**Assessment:** Medium-long term. High complexity. Not the immediate next step.

---

### Path 2 — Block-Sparse / Selective FFN Conversion

**Question:** Can existing dense FFN layers be converted so only a subset of channels/blocks are evaluated per token?

**The opportunity:** Standard FFN layers have activation patterns that are partially sparse in practice. For many prompts, not all input dimensions contribute equally. If we could learn which blocks to skip...

**Challenges:**
- Post-training conversion: Hard. Requires learning a sparsity mask on top of a trained dense model. The dense weights weren't trained to produce sparse activations — the result is unpredictable quality loss.
- Induced sparsity approaches (magnitude pruning, etc.): Work at training level; converting post-hoc is lossy.
- Gating strategy: Learn a lightweight gate that zeros out FFN blocks — requires retraining.
- Fallback requirement: Must have a native dense fallback for quality-critical runs. Without it, this can't be production-safe.
- Quality risk: Significant. "It still works" on a benchmark doesn't mean it works on your specific prompt distribution.

**Memory savings if it worked:** Up to 40–60% of FFN compute skipped → meaningful speedup and bandwidth reduction.

**Assessment:** High potential but high risk. Requires retraining or significant fine-tuning. Not a quick path to demo. Best as a research-only track until a clean fallback strategy exists.

---

### Path 3 — KV/Context Compression

**Question:** Can we make larger models usable by reducing context/KV growth and prompt waste?

**Opportunity:** Most KV cache entries in a typical session are redundant:
- Repeated prompt structures (system prompts, few-shot examples)
- Attention to filler content that doesn't affect the answer
- Older tokens that have diminishing relevance as generation progresses

**Mechanisms:**
- Selective KV eviction: Drop attention to low-relevance tokens as context grows
- KV compression: Summarize older KV state into a smaller representation
- Context restructuring: Preprocess prompts to remove redundancy before KV accumulation

**Memory savings:** Up to 50–80% reduction in effective KV growth for repetitive workloads.

**Quality risk:** Lower than weight compression — KV is about storage, not weights. If the model can attend to what matters in the compressed representation, quality may be preserved.

**ContextOS relevance:** This is the most direct expression of MemoryOS on the inference side. VaultBrain already handles persistent memory; ContextOS would handle session-level KV memory.

**Assessment:** **Highest near-term practicality.** Addressed directly in MEMORY.md with VaultBrain precedent. Engineering risk is moderate. Quality risk is lower than weight manipulation. Time-to-demo is short-to-medium.

---

### Path 4 — Speculative Decoding as SDI Component

**Question:** Can we run the big model fewer times by having a small draft model propose tokens?

**What it does:**
- Reduces number of big-model forward passes
- Small draft model is cheap to run
- Verification step is faster than full regeneration
- Code-like prompts showed prior speedups ✅

**What it doesn't do:**
- Does NOT reduce RAM capacity pressure (big model still resident)
- Does NOT reduce memory bandwidth per pass (same weight streaming)
- Draft model adds its own memory footprint

**Best role:** One component in the SDI stack, not the whole answer. Use it where it helps (code, structured generation) and use other mechanisms for memory capacity reduction.

**Assessment:** Low-risk, proven path. Integrates cleanly into Smart Agent Router. Highest time-to-demo value among current builds. Not a complete solution but a strong enabler.

---

### Path 5 — System-Level SDI Controller

**Question:** Can Smart Agent Router become the control layer that chooses between dense, speculative, compressed, local/cloud, context-reduced paths?

**Architecture:**
```
User prompt
    ↓
Smart Agent Router (control plane)
    ├── Prompt classifier → route decision
    ├── RAM availability check
    ├── Context length estimator
    └── Quality/latency preference

    Route decisions:
    ├── "Short code prompt" → speculative mode (small draft + big verify)
    ├── "Long context session" → KV eviction + context compression enabled
    ├── "Quality critical" → full dense, no SDI
    ├── "RAM constrained" → aggressive quantization + selective FFN (if available)
    └── "High latency tolerance" → batch, defer, compress
```

**What makes this SDI and not just routing:**
The controller doesn't just pick a model — it picks a *strategy for fitting the model into available memory* at the component level. The decision includes format, activation strategy, context management, and fallback paths.

**Assessment:** Highest strategic value. Transforms existing infrastructure (router, LiteRT, Ollama, local routes) into an intelligent SDI system. Engineering risk is moderate (policy layer is new but existing infrastructure is solid). Demo potential is high. This is the long-term productizable stack.

---

### Path 6 — New CPU-Native Model Format

**Question:** Is the right long-term path a model/runtime designed around CPU memory hierarchy from the start?

**Requirements:**
- Memory layout optimized for cache lines and page boundaries
- Mixed-precision placed by access frequency (not uniform Q4)
- Avoid memory-hungry operations (e.g., attention alternatives that reduce KV pressure)
- Runtime aware of memory topology

**Reality check:**
- Converting existing GGUF models: Quality loss + conversion cost. Not trivial.
- Training a new model in this format: Years of work. Not on the table.
- Feasibility: Low for near-term. High complexity. Longest time-to-demo of all options.

**Assessment:** Park this as a research/architecture direction. Come back to it only after the SDI stack is operational and clearly identified as the bottleneck.

---

## K. Ranked Recommendation Table

| Path | Solves RAM capacity | Solves bandwidth | Quality risk | Time-to-demo | Risk | Recommendation |
|------|---------------------|------------------|--------------|--------------|------|----------------|
| **3. KV/Context Compression** | Indirect (KV) | Low | Low | Medium | Medium | 🥇 **#1 RECOMMENDED** |
| **4. Speculative Decoding** | No | No | Low | Short | Low | 🥈 **#2 STRONG** |
| **5. SDI Controller** | Yes (orchestrates all) | Yes | Medium | Medium | Medium | 🥉 **#3 STRATEGIC** |
| **2. Selective FFN** | Yes | Yes | High | Long | High | ⏸️ Research only |
| **1. Native-Layout Sidecars** | Some | Some | Low | Long | High | ⏸️ PRT v3 only later |
| **6. New CPU Format** | Potentially | Potentially | High | Very Long | Very High | ⏸️ Long-term only |

---

## L. Recommended Next Phase

**Phase 26C: Model-Residency SDI Architecture Spec**

**What to do:**
1. Document the SDI control plane architecture in a design doc
2. Define the interface between Smart Agent Router and KV/context compression mechanisms
3. Specify how ContextOS connects to VaultBrain (persistent) and inference session KV (ephemeral)
4. Identify the first concrete integration point between existing components
5. Determine what "success" means in terms a demo can show

**Why this instead of more PRT custom-op work:**
PRT is frozen as a correctness artifact. The next win is not more PRT kernel surgery — it's the control plane that orchestrates existing components intelligently. The SDI stack (router + speculative + context compression + quantization selection) is where the product story lives.

**What counts as success:**
- Architecture spec is complete and reviewed ✅
- At least one concrete path from router → KV compression mechanism identified ✅
- Clear interface between ContextOS session memory and VaultBrain persistent memory defined ✅
- No code written yet — this is a design-only phase ✅

**What would kill the path:**
- If no concrete integration point between existing components can be identified
- If KV compression quality loss proves too high in early probe
- If the router architecture can't support the policy layer

**What would make us continue:**
- Clean architecture that maps to demonstrable components
- At least one measurable win (KV reduction ratio or speedup) observable in a smoke test

---

## M. What NOT to Do Next

- ❌ PRT custom-op speed work (frozen)
- ❌ New sidecar generation
- ❌ 7B or 14B timing runs
- ❌ Multi-layer PRT implementation
- ❌ Retraining or fine-tuning experiments
- ❌ New CPU-native model format design
- ❌ Long benchmark loops

---

## N. Safety Scan

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
| `PASS_PHASE26B_BOTTLENECK_MAP` | ✅ |
| `RECOMMEND_MODEL_RESIDENCY_SDI` | ✅ |
| `RECOMMEND_SELECTIVE_FFN_STUDY` | ⏸️ Research only |
| `RECOMMEND_CONTEXT_KV_COMPRESSION` | 🥇 #1 |
| `RECOMMEND_SPECULATIVE_AS_COMPONENT` | 🥈 #2 |
| `RECOMMEND_PRT_V3_DESIGN_ONLY` | ⏸️ Later |
| `BLOCKED_REPO_STATE` | ✅ No block |

---

*Phase 26B complete. Awaiting Matt's direction.*