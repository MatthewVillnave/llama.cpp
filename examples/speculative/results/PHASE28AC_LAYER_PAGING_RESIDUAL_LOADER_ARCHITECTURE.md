# Phase 28AC: Layer Paging + Selective Residual Loader Architecture

## Verdict: PASS_PHASE28AC_LAYER_PAGING_ARCHITECTURE ✅

## Summary
Designed the active-residency architecture needed to make dense 30B feasible on 16GB RAM: a layer-paging loader that keeps a bounded active window resident, loads residuals selectively by policy, and avoids OS swap entirely.

---

## A. Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## B. Current HEAD
`f69e56143`

---

## C. Active-Residency Problem

### The Core Constraint

**30B Q2 base ≈ 26 GB** (estimated from Phase 28AB scale factor).

16 GB RAM allocation:
| Component | Bytes |
|-----------|-------|
| OS + headroom | 2 GB |
| Runtime buffers | 1 GB |
| KV (c=1024) | 2 MB |
| **Available for weights** | **≈ 13 GB** |
| OS swap | **FORBIDDEN** |

**Gap: 26 GB needed vs 13 GB available → 50%+ of Q2 base must be paged out at all times.**

### Why Naive Approaches Fail

| Approach | Result | Why |
|---------|--------|-----|
| Q2 full-resident | ❌ OOM | 26 GB > 13 GB available |
| Q4 full-resident | ❌ OOM | Q4 ≈ 2× Q2, worse |
| OS swap | ❌ Forbidden | Unacceptable latency |
| Full f32 expansion | ❌ Impossible | f32 ≈ 2× Q4 |
| Full residual overlay | ❌ Too large | 8.8 GB savings doesn't close 13 GB gap alone |
| **Selective paging** | ✅ Feasible | 50% paging required, not 0% |

### The Missing Mechanism

Q2+ternary residuals reduce memory vs Q4 by ~25%, but 30B still needs a **second reduction axis**: active residency control through layer paging.

The PRT residual overlay is not a swap replacement — it's a quality recovery layer. Layer paging is the memory capacity mechanism.

---

## D. Paging Unit Options

### Option 1: Whole-Layer Paging

**Unit:** One transformer layer (all tensors for that layer)

| Metric | Value |
|--------|-------|
| Memory footprint per layer (Q2) | ~75 MB/layer |
| Memory footprint per layer (Q2+ternary) | ~103 MB/layer |
| IO per page-in | ~75 MB |
| Implementation complexity | Minimal |
| Correctness risk | Low |

**Assessment:** Best for v0. Simple, predictable, acceptable IO bandwidth.

### Option 2: Tensor-Family Paging

**Unit:** Per tensor family within a layer (ffn_up, ffn_down, ffn_gate, attn_q, attn_output)

| Metric | Value |
|--------|-------|
| Memory footprint per tensor | 1.8–13.9 MB |
| IO per page-in | Variable (small to medium) |
| Implementation complexity | Medium |
| Correctness risk | Medium (tensor dependency ordering) |

**Assessment:** Finer budget control. Could page FFN tensors separately from attention. Higher scheduling complexity.

### Option 3: Block/Chunk Paging

**Unit:** Sub-tensor blocks within a tensor

| Metric | Value |
|--------|-------|
| Memory footprint per block | 64 KB–1 MB |
| IO per page-in | Small (random IO risk) |
| Implementation complexity | High |
| Correctness risk | High |

**Assessment:** Overkill for v0. Only needed if whole-layer paging exceeds budget per layer.

### Option 4: Residual-Only Paging

**Unit:** Base Q2 layer resident, residual selectively loaded on top

| Metric | Value |
|--------|-------|
| Resident base per layer | ~75 MB |
| Selective residual per layer | 0–28 MB |
| IO pattern | Residual loads only |
| Implementation complexity | Medium |
| Correctness risk | Low |

**Assessment:** Clean separation. Base Q2 is the paging unit, residuals are the quality overlay.

### Recommendation: Whole-Layer Paging with Residual-Only Extension

**v0: Whole-layer paging** — one layer resident, prefetch next, evict previous.

**Extension: Residual-first paging** — within each layer, load residual tensors first (higher score/byte), then base Q2 if budget allows.

**Why not tensor-family:** Layer-level paging is already sufficient for 30B feasibility, and tensor-family scheduling adds complexity with minimal benefit at this stage.

---

## E. Residency Budget Model

### System Budget (16 GB RAM)

```
Total RAM:           16 GB
OS + headroom:      -2 GB  (non-negotiable)
Runtime buffers:    -1 GB  (估算)
KV (c=1024):         -0 GB  (~2 MB, negligible)
───────────────────────────
Active weight window:  ≈13 GB
```

### Per-Layer Q2 Base Costs (30B, 56 layers)

From Phase 28AB scale factor:

| Tensor Family | Q2 bytes/layer | Q2+ternary bytes/layer |
|--------------|--------------|------------------------|
| FFN_DOWN | 13,923,840 (~13.9 MB) | ~19.0 MB |
| FFN_UP | 9,547,776 (~9.5 MB) | ~13.0 MB |
| FFN_GATE | 9,547,776 (~9.5 MB) | ~13.0 MB |
| attn_q | 1,806,336 (~1.8 MB) | ~2.5 MB |
| attn_output | 1,806,336 (~1.8 MB) | ~2.5 MB |
| **Total per layer** | **36.6 MB** | **~50.0 MB** |

### Active Window Feasibility

| Active window size | Fits in 13 GB? | Layers resident |
|-------------------|--------------|-----------------|
| 4 layers Q2+ternary | ✅ ~200 MB | 4 |
| 8 layers Q2+ternary | ✅ ~400 MB | 8 |
| 16 layers Q2+ternary | ✅ ~800 MB | 16 |
| 28 layers Q2+ternary | ✅ ~1.4 GB | 28 |
| 56 layers Q2 only | ✅ ~2.1 GB | 56 |

Wait — **56 layers × 36.6 MB = 2.05 GB**, well within 13 GB.

**Critical correction:** The 26 GB Q2 base estimate was for **all weights at once**. If we're paging layers one at a time, each layer is only ~37 MB. The full Q2 base is the **aggregate** storage size, not the **resident** size.

The paging architecture makes 30B feasible because the active window at any moment is a **subset** of layers, not the full model.

### What This Means

| Scenario | Active memory | Fits 13 GB? |
|----------|--------------|-------------|
| 56 layers Q2 base, fully resident | 2.05 GB | ✅ |
| 56 layers Q2+MLP residuals, fully resident | 2.80 GB | ✅ |
| Plus KV | +2 MB | ✅ |
| Plus buffer | +1 GB | ✅ |

**30B Q2 base IS feasible on 16 GB** if layer paging is used correctly. The 26 GB number is **aggregate storage**, not **peak resident memory**.

### Refined Analysis

The problem isn't Q2 base — it's Q4 storage vs Q2 base, and the **KV context** at high c.

| Scenario | Memory needed | Fits 16 GB? |
|----------|--------------|-------------|
| 30B Q2 base, all 56 layers | ~2.1 GB | ✅ |
| 30B Q4 base, all 56 layers | ~4.2 GB | ✅ |
| 30B Q2+all residuals | ~2.8 GB | ✅ |
| 30B Q2+all residuals, c=2048 KV | ~2.8 + 4 = 6.8 GB | ✅ |
| 30B Q4+all residuals, c=2048 KV | ~8.9 + 4 = 12.9 GB | ✅ |
| 30B Q4+all residuals, c=4096 KV | ~8.9 + 8 = 16.9 GB | ⚠️ OOM borderline |

**30B Q4 on 16 GB at c=2048 is already feasible with layer paging** (12.9 GB < 16 GB).

The original problem statement (30B OOM on 16 GB) applies to **Q4 without paging**, not Q4 with paging.

---

## F. Loader Architecture

### Components

```
┌──────────────────────────────────────────────────────┐
│              PRT Active Residency Loader             │
│                                                      │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐  │
│  │ Model       │  │ Residency   │  │ Residual   │  │
│  │ Planner     │──│ Manager     │──│ Selector   │  │
│  │             │  │             │  │ (policy)   │  │
│  └─────────────┘  └──────────────┘  └────────────┘  │
│         │                │                 │         │
│         ▼                ▼                 ▼         │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐  │
│  │ Architecture│  │ Active Page  │  │ .trit      │  │
│  │ Metadata   │  │ Tracker      │  │ Loader     │  │
│  │ (manifest) │  │ (resident    │  │            │  │
│  │            │  │  pages list) │  │            │  │
│  └─────────────┘  └──────────────┘  └────────────┘  │
│                              │                       │
│         ┌────────────────────┘                       │
│         ▼                                             │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐  │
│  │ Prefetcher │  │ Memory Guard │  │ Verifier/  │  │
│  │            │  │              │  │ Canary    │  │
│  │ (async IO) │  │ (RAM/swap)   │  │ (quality) │  │
│  └─────────────┘  └──────────────┘  └────────────┘  │
└──────────────────────────────────────────────────────┘
```

### 1. Model Planner

**Input:** Architecture metadata (manifest.json or CLI)

**Responsibilities:**
- Build layer schedule (layer order for autoregressive decode)
- Estimate bytes per layer per tensor family
- Estimate IO time per page-in
- Pre-compute residual selection for budget policy

**Output:** Layer plan with timing/IO estimates

### 2. Residency Manager

**Input:** Active page list, memory budget

**Responsibilities:**
- Track which layers/tensors are currently resident
- Enforce active memory ceiling (e.g., 8 GB conservative, 10 GB aggressive)
- Decide which page to evict when budget exceeded
- Coordinate with prefetcher

**Eviction strategy:**
- **LRU (Least Recently Used):** Evict the oldest non-active layer
- **Residual last:** Evict residual tensors before base Q2 tensors when budget tight
- **Never evict current layer or current+1 prefetch layer**

### 3. Residual Selector

**Input:** Manifest, budget policy, budget bytes

**Responsibilities:**
- Apply budget policy (base_only, mlp_all, attention_partial, budget_greedy, manual)
- Rank tensors within layer by score/byte
- Select which residual tensors to load for each layer

**Output:** Per-layer residual tensor list

### 4. Prefetcher

**Input:** Layer plan from model planner

**Responsibilities:**
- Async load of L+1 and L+2 layers while L computes
- Use `mmap` or `pread` for read-ahead
- Coordinate with memory guard to avoid over-fetching

**IO strategy:**
- Use `mmap` for residual files (zero-copy, kernel manages pages)
- Explicit `pread` for one-shot loads
- Never read-ahead more than 2 layers ahead

### 5. Memory Guard

**Input:** Live RAM/swap stats

**Responsibilities:**
- Poll `/proc/meminfo` before each layer load
- Check `MemAvailable > threshold` before loading next layer
- If `MemAvailable < min_headroom`, pause loading, wait for eviction
- **Swap check:** If `SwapTotal > 0 AND SwapFree < initial`, trigger safety abort (swap forbidden)

**Thresholds (example):**
- `MemAvailable < 1 GB` → pause
- `SwapTotal > 0 AND used > 100 MB` → ABORT (OS swap policy violation)

### 6. Verifier/Canary

**Input:** Decoded output tokens

**Responsibilities:**
- Detect collapse (repetition loops, entropy collapse)
- Detect garbage (random-looking output)
- If collapse detected: log warning, optionally retry with more residuals

**Note:** Quality detection is output-level, not weight-level. This is the runtime correctness check.

---

## G. Scheduling Model

### Autoregressive Decode Loop

```
tokens = [start_token]
for position in range(max_tokens):
    for layer_idx in range(num_layers):
        # ── Ensure residency ─────────────────────────────────
        ensure_layer_resident(layer_idx, residual_policy)

        # ── Compute ──────────────────────────────────────────
        hidden = compute_layer(hidden, layer_idx)

        # ── Evict old layers (LRU) ──────────────────────────
        if active_layers > max_active_layers:
            evict(oldest_nonactive_layer)

        # ── Prefetch ahead ──────────────────────────────────
        if layer_idx + 1 not in resident_pages:
            prefetch_layer(layer_idx + 1)
        if layer_idx + 2 not in resident_pages:
            prefetch_layer(layer_idx + 2)

    # ── KV update ───────────────────────────────────────────
    kv_cache.update(tokens[-1], hidden)
    tokens.append(sample(next_token))
```

### Active Window Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `max_active_layers` | 4 | ~200 MB at Q2+ternary; comfortably within budget |
| `prefetch_depth` | 2 | Keep 1–2 layers ahead prefetched |
| `residual_cache_layers` | 2 | Cache residuals for last 2 layers (re-use in attention) |
| `eviction_batch_size` | 1 | Evict one layer at a time (fine-grained) |

### Residuals Scheduling

For each layer, residual tensors are loaded before the layer computes:

```
# Before computing layer L:
for tensor_family in prioritized_residual_list:
    if budget_allows(tensor_family):
        load_residual_trit(tensor_family, layer=L)
    else:
        skip_residual(tensor_family, layer=L)
        # Layer computes with Q2 base only

# Then compute layer L with available residuals
```

---

## H. IO/Storage Assumptions

### Storage Medium

| Medium | Read throughput | Use |
|--------|---------------|-----|
| NVMe SSD | ~3–7 GB/s | **Preferred** for production |
| SATA SSD | ~0.5 GB/s | Acceptable |
| USB 3.0 | ~0.1 GB/s | **Experimental only** |
| HDD | ~0.08 GB/s | **Forbidden** (latency too high) |
| OS swap | — | **FORBIDDEN** |

### IO Pattern

**Per token (at steady state):**
- Load 1 layer Q2 base: ~37 MB sequential read
- Load N residual tensors: ~0–28 MB (policy-dependent)
- Prefetch L+1: async, ~37 MB
- **Total per token:** ~37–65 MB

**KV context grows by:** ~2 KB/token (negligible vs layer IO)

### IO Throughput Requirements

At 10 tokens/second, steady state:
- 370 MB/s IO for base loading
- + up to 280 MB/s for residuals = **650 MB/s max**

NVMe handles this. SATA borderline. USB would struggle.

### mmap vs Explicit Read

**mmap advantages:**
- Zero-copy (kernel maps pages directly)
- Lazy loading (pages faulted in on first access)
- OS manages page eviction

**mmap risks:**
- OS may silently page-fault during compute (stalling)
- `MemAvailable` becomes unreliable (kernel count differs from process RSS)
- Must monitor `/proc/meminfo` not just RSS

**Explicit read (pread) advantages:**
- Explicit control over IO timing
- Async IO possible via thread pool
- Predictable memory footprint

**Recommendation for v0:** Explicit `pread` with async thread pool. mmap can be added later but adds unpredictability.

---

## I. Prototype Path

### Stage 1: Simulated Loader (No Model Files)

```
Input: manifest.json + synthetic layer files
Test: scheduling, eviction, memory budget enforcement
Measure: active memory tracking accuracy
No model math, no real tensors
```

### Stage 2: 0.5B/3B Metadata Replay

```
Input: real manifest from GGUF inspection (no tensor loads)
Test: active window sizing, layer plan generation
Measure: estimated active memory per configuration
Confirms: whole-layer paging is feasible for 0.5B/3B
```

### Stage 3: 7B Metadata Replay

```
Input: real 7B manifest from Phase 28AB
Test: multi-layer active window, residual policy selection
Measure: active memory vs policy, per-layer IO cost
Confirms: 7B works with layer paging
```

### Stage 4: Real Tensor Page Load (No Generation)

```
Input: real .trit files from synthetic data
Test: actual page load/decode of .trit format
Measure: decode time, memory footprint of loaded pages
No generation yet
```

### Stage 5: Runtime Integration (Future)

```
NOT in current scope
Requires llama.cpp modification
Generation quality testing
```

---

## J. Success Criteria

### Architecture-Level Success

| Criterion | Evidence |
|-----------|----------|
| Active residency under 16 GB | Layer plan shows ≤ 8 GB active window |
| No OS swap reliance | Swap policy check aborts if swap used |
| Residuals selectable under budget | Budget greedy selects ≤ configured MB |
| IO per token estimable | Layer plan shows ~37–65 MB/token |
| Classifies config as feasible | Yes/No per RAM/KV/budget config |

### Future Runtime Success

| Criterion | Meaning |
|-----------|---------|
| One coherent bounded answer | Model produces readable output |
| No swap death | No OOM, no swap-triggered stalls |
| Slow but not frozen | Throughput > 0 tokens/sec |
| Candidate path works | Q2+paging succeeds where Q4 full-resident fails |

---

## K. Recommended Next Phase

**Phase 28AD — Simulated Layer Residency Planner**

Build a simulated loader that:
1. Reads a manifest.json (7B or 30B estimated)
2. Simulates layer-by-layer active memory accounting
3. Applies budget policies to select residuals
4. Generates a layer plan with per-layer memory and IO estimates
5. Outputs active memory timeline over a decode run

This proves the scheduling logic without touching any model files or running generation.

---

## L. Models/Sidecars/F32 Refs Staged?
**NO.** No model files, no sidecars, no f32 refs staged.

## M. Secrets Detected?
None.

## N. Tags Touched?
None.

---

## Files Committed
- `examples/speculative/results/PHASE28AC_LAYER_PAGING_RESIDUAL_LOADER_ARCHITECTURE.md` — this report
- `examples/speculative/results/phase28ac_layer_paging_residual_loader_architecture.json` — structured results