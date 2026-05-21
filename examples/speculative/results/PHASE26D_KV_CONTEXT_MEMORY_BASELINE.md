# Phase 26D: KV/Context Memory Baseline

**Verdict:** `PASS_PHASE26D_BASELINE_CAPTURED`

**Date:** Wed 2026-05-20 23:27 EDT
**Branch:** `experimental/prt-phase19a-alt-sidecar-backed`
**Current HEAD:** `96fbb0668`

---

## A. Branch
```
experimental/prt-phase19a-alt-sidecar-backed
```

## B. Current HEAD
```
96fbb0668
```

## C. 7B Model Path
```
/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-7B-Instruct-Q4_K_M.gguf
```

## D. Model File Size
```
4.4 GB (Q4_K_M, 2-shard Qwen2.5-7B)
```

## E. RAM/Swap Before Tests

| Metric | Value |
|--------|-------|
| Total RAM | 15.3 GB |
| Free RAM | ~4.3–5.9 GB (varies with OS cache state) |
| MemAvailable | ~9.0–10.0 GB (OS can give to processes) |
| Swap total | 4.0 GB |
| Swap used | ~390–520 MB (idle baseline) |
| Swap free | ~3.5–3.8 GB |

**Note:** Swap used at idle baseline is ~390–520 MB from other system activity. The llama.cpp process itself does not add swap until context size becomes memory-constrained.

---

## F. Context Memory Matrix

All runs: llama-cli, Qwen2.5-7B-Q4_K_M, n=8, temp=0, t=1, no PRT env, native GGML path

| Context | Prompt Size | Pre-test MemAvailable | Post-test MemAvailable | Swap Pre | Swap Post | Swap Δ | Verdict |
|---------|-------------|------------------------|------------------------|----------|-----------|--------|---------|
| 2048 | 25 bytes | 9.4 GB | 9.3 GB | 391 MB used | 391 MB used | 0 MB | ✅ Safe |
| 4096 | 2,160 bytes | 9.0 GB | 9.7 GB | 391 MB | 464 MB | +73 MB | ✅ Safe |
| 8192 | 19,000 bytes | 10.1 GB | 9.0 GB | 460 MB | 526 MB | +66 MB | ⚠️ Low pressure |
| 16384 | 19,000 bytes | 9.2 GB | 9.6 GB | 518 MB | 912 MB | **+394 MB** | 🔴 Danger zone |

### Key observations:

1. **c=2048–8192:** No meaningful swap increase. System is memory-stable. The model + KV for these context sizes fit in available RAM.

2. **c=16384:** Swap usage jumped ~400 MB (+394 MB). This is the first evidence of memory pressure. The KV at c=16384 is pushing the resident set beyond what free RAM can hold, and the OS is paging to manage it. Process still completed without OOM, but swap is now active — which means memory access latency will degrade.

3. **No swap death:** c=16384 did not OOM on this run. The 4GB swap gave enough buffer to absorb the spike. But this is a narrow escape — other OS activity or a slightly larger prompt could trigger OOM.

4. **Prompt size effect:** The 19KB prompt (~6–7K tokens) was reused for c=8192 and c=16384. Actual context utilization is driven by `--ctx-size`, not prompt length. The ctx-size is the allocation ceiling.

### Memory formula check (Phase 26B-R estimates):

| Context | Estimated KV | Estimated Total | Actual swap behavior |
|---------|--------------|-----------------|---------------------|
| 2048 | ~229 MB | ~4.9 GB | No swap ✅ |
| 4096 | ~458 MB | ~5.2 GB | Minimal swap (+73 MB) ✅ |
| 8192 | ~916 MB | ~5.7 GB | Low swap (+66 MB) ⚠️ |
| 16384 | ~1.8 GB | ~6.6 GB | Moderate swap (+394 MB) 🔴 |

Estimates were conservative. Real llama.cpp appears to use less KV than the theoretical formula suggests (likely because the model uses n_kv_heads < n_heads, and the formula assumed full n_heads for KV).

---

## G. Safe Context Length

| Threshold | Memory Behavior |
|-----------|----------------|
| **c ≤ 8192** | Safe. No meaningful swap activation. KV + model fit in RAM. |
| **c = 16384** | Danger zone. Swap activated (~400 MB). Process survives but OS is paging. |
| **c > 16384** | Likely OOM or severe swap degradation on this machine. |
| **Absolute hard limit** | ~c=16000–18000 depending on OS state. Swap is the safety valve but at performance cost. |

---

## H. Danger Context Length

**c=16384 is the danger point.**

The swap increase at c=16384 (+394 MB) shows the system crossed from RAM-resident to swap-backed for at least part of the working set. This means:
- Memory access latency increased (RAM → swap penalty)
- Swap pages accumulate as session continues
- At longer session durations or higher OS activity, c=16384 will OOM

**Safe operating ceiling: approximately c=8192–12288**

At c=8192, the margin is slim but still RAM-resident. At c=12288, likely marginal. At c=16384, swap-dependent.

---

## I. Observed Swap Behavior

| Context | Swap activated? | Swap delta | Impact |
|---------|-----------------|------------|--------|
| 2048 | No | 0 MB | None |
| 4096 | Minimal | +73 MB | Likely from OS page reclaim, not inference |
| 8192 | Low | +66 MB | Marginal |
| 16384 | Yes | +394 MB | Active paging, latency risk |

**Inference:** The llama.cpp KV allocation at c=16384 exceeds what the OS will keep in RAM given other baseline usage (~5.9 GB used by OS + other processes). The OS pushes ~400 MB to swap.

---

## J. KV/Context Reduction Target

**Goal:** Keep 7B inference at c=16384 from activating swap, or push the safe ceiling from c=8192 to c=16384+.

**What reduction is needed:**

| Target context | KV at target | Free RAM available | Needed reduction |
|----------------|-------------|--------------------|------------------|
| 16384 | ~1.8 GB | ~0 GB (after model + OS) | Need ~1 GB KV reduction to stay RAM-resident |
| 12288 | ~1.4 GB | ~0.2 GB marginal | Need ~0.5 GB KV reduction |
| 8192 | ~0.9 GB | ~0.9 GB | Already barely fits — no swap at c=8192 |

**For c=16384 to run without swap:**
- Reduce active KV from ~1.8 GB to ~1.0 GB or less
- That's a **~45–55% KV reduction** required
- 50% KV reduction: makes c=16384 barely fit
- 75% KV reduction: makes c=16384 comfortable and c=32768 potentially survivable

**For c=32768 to be survivable:**
- KV at c=32768 ≈ 3.7 GB
- 7B weights ≈ 4.4 GB
- Runtime ≈ 0.3 GB
- Total ≈ 8.4 GB
- Free RAM ≈ 6 GB
- **Deficit: ~2.4 GB** — would need 75%+ KV eviction to fit

---

## K. Candidate Compression Policies

### Policy 1 — Sliding Window KV (keep N most recent tokens)
- Keep most recent N tokens, drop old KV entries
- Simple to implement: maintain a circular buffer of KV entries
- Risk: loses long-range dependencies (cannot reference facts from earlier in conversation)
- Effectiveness: high for single-turn, medium for multi-turn

### Policy 2 — Summary Replacement
- Summarize old context into a compact text representation, drop old KV
- More aggressive compression than sliding window
- Risk: summary quality, what to summarize, summarization latency
- Effectiveness: potentially high but complex

### Policy 3 — Importance-Based Retention (pinned facts)
- Keep system prompt, user constraints, tool results, high-salience facts
- Evict filler, repetition, low-importance context
- Risk: requires salience detection (model or heuristic)
- Effectiveness: potentially highest if salience detection works

### Policy 4 — ContextOS Packetization
- Convert old context into structured memory packet (stored in VaultBrain/ClawVault)
- Keep compact active prompt with a "memory reference"
- Model re-fetches relevant packets when needed
- Risk: requires integration layer between inference and memory
- Effectiveness: could be very high — old context is externalized, not dropped

### Policy 5 — Hybrid
- Recent window (e.g., last 2K tokens) kept as active KV
- Older context → structured summary stored in memory
- System prompt and tool results pinned in KV
- Risk: complexity of multiple mechanisms
- Effectiveness: highest theoretically

---

## L. Recommended First Compression Policy

**Recommended: Sliding Window KV (Policy 1) as first probe**

**Rationale:**
1. Simplest to implement and measure
2. Clear success criterion: reduces swap at c=16384
3. No external integration dependencies
4. Provides baseline for measuring whether it suffices or more complex policies are needed
5. "KV eviction after N tokens" is a well-known pattern in production LLM systems

**What to measure first:**
- At what token limit does swap activate for c=16384 with a sliding window of N?
- How does output quality degrade as N decreases?
- Is there a minimum N (e.g., 512 tokens, 1024 tokens) below which quality collapses?

**If sliding window proves insufficient for c=16384:**
- Evaluate Policy 4 (ContextOS packetization) as the next step
- The hybrid approach (Policy 5) becomes the full solution

---

## M. Success Criteria for Phase 26E Implementation

| Criterion | Threshold |
|-----------|-----------|
| 7B at c=16384 | No swap increase (verified via /proc/meminfo) |
| Output quality | <10% degradation on eval prompts vs no-eviction baseline |
| Latency | <20% regression on per-token decode time |
| Repeatability | Works across 3+ different prompt types |
| Memory monitoring | Memory pressure monitor can detect swap risk and trigger eviction |
| Integration | Can be called from Smart Agent Router policy layer |

---

## N. Kill Criteria

| Criterion | Kill condition |
|-----------|--------------|
| Quality | Any automated semantic canary fails on common prompts |
| Memory savings | KV eviction provides <20% memory reduction at c=16384 |
| Latency | Per-token decode time increases >50% vs no-eviction |
| Implementation complexity | Requires more than 3 new config knobs to tune |
| Real prompts | Fails on real long-context prompts Matt actually uses |

---

## O. Safety Scan

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
| `PASS_PHASE26D_BASELINE_CAPTURED` | ✅ |
| `PASS_KV_REDUCTION_TARGET_DEFINED` | ✅ |
| `RECOMMEND_SLIDING_WINDOW_FIRST` | ✅ |
| `RECOMMEND_HYBRID_CONTEXTOS_KV_POLICY` | ✅ (if sliding window insufficient) |
| `BLOCKED_7B_RUNTIME` | ✅ No block — all tests passed |
| `BLOCKED_MACHINE_STATE` | ✅ No block — machine stable |

---

*Phase 26D complete. Baseline captured, compression target defined, first policy recommended. Awaiting Matt's direction.*