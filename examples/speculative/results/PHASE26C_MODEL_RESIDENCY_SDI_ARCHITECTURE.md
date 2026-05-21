# Phase 26C: Model-Residency SDI Architecture Spec

**Verdict:** `PASS_PHASE26C_ARCHITECTURE_SPEC`

**Date:** Wed 2026-05-20 23:24 EDT
**Branch:** `experimental/prt-phase19a-alt-sidecar-backed`
**Current HEAD:** `96fbb0668`

---

## A. Branch
```
experimental/prt-phase19a-alt-sidecar-backed
```

## B. Current HEAD
```
96fbb0668 ("Phase 26B-R: audit RAM table, correct Q4_K_M size estimates")
```

## C. Corrected Memory Assumptions

From Phase 26B-R:

| Model | Weights (Q4_K_M) | KV @ 4K ctx | Runtime OH | Total @ 4K | Verdict |
|-------|-----------------|-------------|------------|-----------|---------|
| 0.5B | 0.38 GB | 0.46 GB | 0.20 GB | ~1.0 GB | ✅ Comfortable |
| 3B | 1.8 GB | 0.58 GB | 0.25 GB | ~2.6 GB | ✅ Comfortable |
| 7B | 4.4 GB | 0.46 GB | 0.30 GB | ~5.2 GB | ⚠️ Tight — ~0.8 GB headroom |
| 14B | 8.4 GB | 0.66 GB | 0.30 GB | ~9.4 GB | ❌ OOM |
| 32B | ~21 GB | 0.79 GB | 0.30 GB | ~22 GB | ❌ OOM |

**Machine:** 15.3 GB total, ~6 GB free at session start, ~13 GB MemAvailable (includes cached pages)

**7B context pressure:**

| Context | Total RAM | Free RAM remaining | Status |
|---------|----------|--------------------|--------|
| 4K | ~5.2 GB | ~0.8 GB | Tight |
| 8K | ~6.1 GB | — | Likely swap |
| 16K | ~7.0 GB | — | Swap guaranteed |
| 32K | ~9.1 GB | — | OOM without eviction |

**Core insight:** 7B at 4K+ context leaves almost no headroom. KV eviction is the critical path, not a nice-to-have.

---

## D. Model-Residency SDI Definition

**Model-Residency SDI** is the subset of SDI focused on making a dense model usable on constrained CPU/no-GPU hardware by reducing:

1. **Model memory residency** — how much model data must be in RAM simultaneously
2. **Per-token weight streaming** — how much memory is touched per token generated
3. **KV/context growth** — how much working memory accumulates over a session
4. **Dense activation waste** — compute that runs but doesn't meaningfully contribute to output quality

**What it is NOT:**
- Classic MoE: Full model still required; expert routing is a training-time decision
- Prompt routing: Sends different prompts to different models; doesn't compress a single model's memory footprint
- Quantization only: Reduces bits/param but not the streaming pattern; Q4_K_M at 5 bits/param is still large
- Speculative decoding only: Reduces big-model forward pass count; doesn't reduce memory per pass
- PRT only: Valid INT8 representation artifact, but the custom-op path proved that format conversion without memory-layout optimization doesn't win on speed

**Model-Residency SDI specifically addresses the memory wall** that quantization alone does not solve for dense models on constrained hardware.

---

## E. Architecture Layers

### Layer 1 — Memory Residency Layer
**Purpose:** Reduce or manage the memory required for model weights, KV cache, runtime buffers, sidecars, and context.

| Mechanism | Description |
|-----------|-------------|
| KV eviction | Selectively drop attention to low-relevance tokens as context grows |
| KV compression | Summarize older KV state into a smaller representation |
| Context summarization | Preprocess/reduce prompt redundancy before KV accumulation |
| Memory-aware context packing | Limit active context based on available RAM |
| Quantization selection | Choose Q4_K_M vs Q5 vs Q8 based on memory budget |
| Model-size-aware routing | Route to smaller model when memory is tight |
| Swap avoidance | Detect memory pressure and throttle before OOM |
| Resident model management | Keep only active model components loaded |

### Layer 2 — Bandwidth Reduction Layer
**Purpose:** Reduce per-token memory streaming cost.

| Mechanism | Description |
|-----------|-------------|
| Compressed weight formats | Formats that reduce bytes streamed per token beyond Q4 |
| Native-layout sidecars | Weights arranged for CPU cache lines, not a custom-op overlay |
| Selective FFN/block execution | Only evaluate FFN blocks relevant to current token |
| Decoded weight caching | Cache dequantized weights across tokens where possible |
| Mixed-precision placement | Hot weights in higher precision, cold in lower |

**Note from PRT:** Custom-op overlay paths add kernel overhead without reducing the streaming pattern. Bandwidth reduction requires format + layout co-design, not just format conversion.

### Layer 3 — Activation Reduction Layer
**Purpose:** Avoid activating unnecessary dense compute per token.

| Mechanism | Description |
|-----------|-------------|
| Selective FFN | Skip FFN blocks/channels based on learned relevance |
| Early exit | Stop forward pass at earlier layer for easy tokens |
| Token-dependent layer skipping | Route tokens through only relevant layers |
| Attention sparsity | Reduce attention to low-importance tokens |
| Dense fallback anchors | Always have a full-density fallback path |

**Quality risk is highest in this layer.** Selective execution without a trained sparsity mask and a clean fallback strategy risks output degradation.

### Layer 4 — Verification / Correctness Layer
**Purpose:** Preserve output quality as the primary constraint.

| Mechanism | Description |
|-----------|-------------|
| Native fallback | Full dense path always available |
| Speculative verification | Verify speculative tokens against full model |
| Semantic canaries | Detect quality drift via lightweight semantic check |
| Exact-match canaries | Regression test against known-good outputs |
| Regression harnesses | Automated quality tests before SDI deployment |
| Fallback thresholds | Trigger fallback automatically on quality signal |

### Layer 5 — Control Plane
**Purpose:** Choose which SDI mechanism to use, dynamically.

| Mechanism | Description |
|-----------|-------------|
| Smart Agent Router | Orchestrates all SDI mechanisms as a policy engine |
| Prompt classifier | Categorizes prompts to route to appropriate SDI path |
| Speculative routing | Enable speculative decoding for code/structured prompts |
| Context budget policy | Enforce context length limits based on memory |
| Memory pressure monitor | Detect swap/OOM risk before it happens |
| Model selector | Choose model size based on prompt characteristics + memory |
| Local/cloud fallback | Route to cloud when local memory is constrained |

---

## F. Bottleneck-to-Mechanism Map

| Bottleneck | Primary Mechanism | Secondary Mechanism | Near-term? | Risk | Expected Payoff |
|-----------|-----------------|---------------------|------------|------|----------------|
| **RAM capacity (14B+)** | Model-size-aware routing | Quantization selection | ✅ Yes | Low | Enables 14B on constrained hardware |
| **RAM capacity (7B at 4K+)** | KV eviction | Context summarization | ✅ Yes | Medium | Keeps 7B from swapping |
| **RAM capacity (30B)** | Selective FFN + staged loading | Model residency management | ⚠️ Medium | High | Enables >7B on current hardware |
| **RAM bandwidth** | Native-layout sidecars | Selective FFN | ❌ No (requires PRT v3) | High | Significant if it works |
| **KV/context growth** | KV eviction + compression | Context summarization | ✅ Yes | Medium | 50-80% KV reduction |
| **Dense activation waste** | Selective FFN | Early exit | ❌ No (no fallback) | High | Meaningful but risky |
| **Compute per token** | Speculative decoding | Selective FFN | ✅ Yes | Low | 1.3-2x speedup on code prompts |
| **Quality drift** | Speculative verification | Semantic canaries | ✅ Yes | Low | Preserves output quality |
| **Control plane** | Smart Agent Router policy | Memory pressure monitor | ✅ Yes | Medium | Enables adaptive SDI |

---

## G. Ranked First-Build Options

### 🥇 Option A — KV/Context Compression Probe

**Why #1:** Highest urgency. 7B at 4K+ context is already memory-stressed. KV eviction is the only mechanism that can prevent swap at 8K/16K without changing the model or weights. Low quality risk since we're compressing session state, not model weights.

**First concrete phase:**
1. Design KV eviction strategy (which tokens to drop and when)
2. Measure current KV growth vs context length for real prompts
3. Implement lightweight eviction policy (e.g., drop every Nth token or low-attention token)
4. Verify output quality against non-evicted baseline

**Success criteria:** 7B at 8K context runs without swap, <10% quality signal loss on eval prompts.

---

### 🥈 Option B — SDI Controller v1

**Why #2:** Highest strategic value. Smart Agent Router already exists. Adding a policy layer that routes between SDI mechanisms transforms it from a router into an intelligent control plane. Medium complexity, large payoff.

**First concrete phase:**
1. Define prompt classifier categories (code, long context, short QA, creative)
2. Define memory pressure levels (green/yellow/red)
3. Define routing rules: (prompt_type, memory_level) → (path)
4. Implement memory pressure monitor via /proc/meminfo polling
5. Hook into existing speculative decoding path as first SDI mechanism

**Success criteria:** Router correctly routes code prompts to speculative path, long-context prompts to KV-managed path, memory-pressure prompts to smaller model.

---

### 🥉 Option C — SpecBenchCPU Revalidation

**Why #3:** Lowest risk. Speculative decoding already showed code speedups. Re-running the validation harness confirms the path is still good and provides a clean benchmark baseline. Quick win, but limited strategic value beyond confirming existing work.

**First concrete phase:**
1. Run SpecBenchCPU on current branch with Qwen2.5-3B
2. Record speculative vs native tokens/sec on code prompts
3. Validate output quality vs native baseline
4. Document as the SDI "speculative component" baseline

**Success criteria:** SpecBenchCPU passes, code prompts show >1.1x speedup via speculative decoding.

---

### Option D — Model Residency Monitor

**Why:** Useful tool for preventing swap death. Would give early warning before inference starts that a given configuration won't fit. But it's infrastructure, not a win — lower priority than the mechanisms themselves.

**First concrete phase:** Build a small CLI that takes (model_size, context_length, available_ram) and outputs (will_fit, estimated_swap_risk, recommended_max_context).

---

### Option E — PRT v3 Design Only

**Why parked:** High complexity, no clear implementation path yet, and PRT v2 proved that custom-op overlay approaches don't win on speed. A PRT v3 native backend is a 6-12 month effort with no near-term demo. Preserve as design doc only.

---

## H. Recommended First Build

**Phase 26D: KV/Context Compression Probe**

**Rationale:**
1. 7B at 4K+ context is the most urgent near-term problem on Matt's hardware
2. KV eviction is the only mechanism that solves swap risk without modifying model weights or kernel math
3. Quality risk is lower than weight manipulation
4. Success is measurable (swap avoidance + quality eval)
5. Builds directly on VaultBrain/ContextOS work already in progress

**What it is NOT:** A full SDI controller. Just the first mechanism, implemented cleanly.

**Demo scenario:** 7B Q4_K_M at 8K context — before: swap-bound at ~3 tokens/sec. After: KV eviction enabled — runs without swap, maintains ~8-10 tokens/sec.

---

## I. Success Criteria

| Criterion | Threshold |
|-----------|-----------|
| 7B at 8K context | No swap (verified via /proc/meminfo) |
| Output quality | <10% degradation on eval prompts vs no-eviction baseline |
| Latency | <20% regression on per-token decode time |
| KV reduction ratio | ≥30% reduction in active KV at 4K+ context |
| Repeatability | Works across 3+ different prompt types |
| Integration | Can be called from Smart Agent Router policy layer |

---

## J. Kill Criteria

| Criterion | Kill condition |
|-----------|--------------|
| Quality | Any automated semantic canary fails on common prompts |
| Memory savings | KV eviction provides <20% memory reduction |
| Latency | Per-token decode time increases >50% vs no-eviction |
| Complexity | Requires more than 3 new config knobs |
| Integration | Cannot be triggered via Smart Agent Router policy |
| Specific prompts | Fails on real long-context prompts Matt actually uses |

---

## K. What NOT to Do Next

- ❌ PRT custom-op speed work (frozen, paused)
- ❌ 14B or 30B timing runs
- ❌ Multi-layer PRT implementation
- ❌ New sidecar generation
- ❌ Selective FFN without a trained fallback
- ❌ New CPU-native model format design
- ❌ Full SDI controller as first build (scope creep — do KV probe first)
- ❌ Long benchmark loops before KV probe is done

---

## L. Safety Scan

```
git status --short
 M ggml/src/ggml-cpu/ops.cpp
?? examples/speculative/results/PRT_PHASE24R_NATIVE_MULMAT_PROBE.md
?? examples/speculative/results/phase24r_native_mulmat_probe.json

No models/sidecars/f32 refs staged.
No secrets found.
```

---

## Verdicts

| Verdict | Value |
|---------|-------|
| `PASS_PHASE26C_ARCHITECTURE_SPEC` | ✅ |
| `RECOMMEND_KV_CONTEXT_COMPRESSION_FIRST` | 🥇 |
| `RECOMMEND_SDI_CONTROLLER_FIRST` | 🥈 |
| `RECOMMEND_SPECBENCHCPU_REVALIDATION` | 🥉 |
| `RECOMMEND_MODEL_RESIDENCY_MONITOR` | ⏸️ |
| `RECOMMEND_PRT_V3_DESIGN_ONLY` | ⏸️ |

---

*Phase 26C complete. Architecture spec documented. Awaiting Matt's direction to proceed to Phase 26D or pivot.*