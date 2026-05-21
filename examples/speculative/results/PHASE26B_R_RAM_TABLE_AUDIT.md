# Phase 26B-R: RAM Table Audit — Corrected Memory Estimates

**Verdict:** `PASS_PHASE26B_R_RAM_AUDIT`

**Date:** Wed 2026-05-20 23:20 EDT
**Branch:** `experimental/prt-phase19a-alt-sidecar-backed`
**Current HEAD:** `eb9bb9e09`

---

## Context

Phase 26B used idealized 2-bit estimates for Q4_K_M weight sizes:
- 7B → 1.75 GB (idealized 2 bits/param)
- 14B → 3.5 GB
- 30B → 7.5 GB

Matt correctly flagged these as too low. Real Q4_K_M GGUF files are closer to 5 bits/param due to GGUF metadata, quantization overhead, and vocabulary size.

This note corrects the RAM table with actual file sizes and realistic estimates.

---

## Actual GGUF File Sizes (Q4_K_M)

| Model | File(s) | Size on Disk | Bytes/Param | Bits/Param |
|-------|---------|-------------|-------------|------------|
| Qwen2.5-0.5B | single | 380 MB | 0.76 B/p | ~6.1 b/p |
| Qwen2.5-3B | single | 1.8 GB | 0.60 B/p | ~4.8 b/p |
| Qwen2.5-7B | 2 shards | 4.4 GB total | 0.629 B/p | ~5.0 b/p |
| Qwen2.5-14B | single | 8.4 GB | 0.60 B/p | ~4.8 b/p |
| Qwen2.5-32B | 5 shards | ~4.2 GB/shard × 5 ≈ 21 GB est | 0.656 B/p | ~5.2 b/p |
| Bonsai-8B | single | 1.1 GB | 0.138 B/p | ~1.1 b/p (unusual — likely FP16 or Q6) |

**Key finding:** Real Q4_K_M is ~5 bits/param, not ~2 bits/param. The idealized table underestimated weights by 2.5–3×.

---

## Machine RAM State

```
MemTotal:   16053836 kB ≈ 15.3 GB
MemFree:     6287552 kB ≈  6.0 GB
MemAvailable: 13728852 kB ≈ 13.1 GB  ← what OS will actually give processes
Cached:      7469484 kB ≈  7.1 GB  (file cache, reclaimable)
Buffers:      223264 kB ≈  0.2 GB
SwapTotal:   4194300 kB ≈  4.0 GB
SwapFree:    3974272 kB ≈  3.8 GB
```

**Available RAM for inference:** ~13 GB (MemAvailable after OS overhead)

But MemAvailable includes cached file pages. Real *free* at session start is ~6 GB. Under load with model loaded, available drops toward zero and the OS reclaims cached pages.

**Safe working estimate:** ~6–8 GB truly available for model + KV + runtime under active inference.

---

## KV Cache Estimates

Formula: `KV = n_layers × n_kv_heads × head_dim × 2 × n_seq × bytes_per_float16`

**Qwen2.5 parameters (from config):**
- 7B: n_layers=28, n_heads=32, n_kv_heads=8, head_dim=128 → d=4096
- 3B: n_layers=36, n_heads=24, n_kv_heads=8, head_dim=128 → d=4096
- 14B: n_layers=40 (est), n_heads=40 (est), n_kv_heads=8, head_dim=128 → d=4096
- 32B: n_layers=48, n_heads=40 (est), n_kv_heads=8, head_dim=128 → d=4096

**Per-token KV (FP16):**
- 7B: 28 × 8 × 128 × 2 × 2 = 114,688 bytes ≈ 112 KB/token
- 3B: 36 × 8 × 128 × 2 × 2 = 147,456 bytes ≈ 144 KB/token
- 14B: 40 × 8 × 128 × 2 × 2 = 163,840 bytes ≈ 160 KB/token
- 32B: 48 × 8 × 128 × 2 × 2 = 196,608 bytes ≈ 192 KB/token

**Total KV at various context lengths:**

| Context | 7B KV | 3B KV | 14B KV | 32B KV |
|---------|-------|-------|--------|--------|
| 512 | 58 MB | 74 MB | 82 MB | 99 MB |
| 2K | 229 MB | 292 MB | 327 MB | 393 MB |
| 4K | 458 MB | 584 MB | 655 MB | 786 MB |
| 8K | 916 MB | 1.2 GB | 1.3 GB | 1.6 GB |
| 32K | 3.7 GB | 4.7 GB | 5.2 GB | 6.3 GB |

---

## Runtime Overhead Estimate

| Component | Estimate |
|-----------|----------|
| GGML/GGUF internal state | ~50–100 MB |
| Activation buffers (decode) | ~100–200 MB |
| llama-cli process overhead | ~50 MB |
| Sampler/state | ~20 MB |
| **Total runtime overhead** | **~200–350 MB** |

---

## Corrected RAM Table

| Model | Weights (Q4_K_M) | KV (4K ctx) | Runtime OH | Total @ 4K | Available (~6 GB free) | Verdict |
|-------|-----------------|-------------|------------|------------|----------------------|---------|
| **0.5B** | 0.38 GB | ~0.46 GB | 0.2 GB | **~1.0 GB** | ~5 GB free | ✅ Comfortable |
| **3B** | 1.8 GB | ~0.58 GB | 0.25 GB | **~2.6 GB** | ~3.4 GB free | ✅ Comfortable |
| **7B** | 4.4 GB | ~0.46 GB | 0.3 GB | **~5.2 GB** | ~0.8 GB free | ⚠️ Tight — no room for other work |
| **14B** | 8.4 GB | ~0.66 GB | 0.3 GB | **~9.4 GB** | ❌ OOM | ❌ Does not fit |
| **32B** | ~21 GB est | ~0.79 GB | 0.3 GB | **~22 GB** | ❌ OOM | ❌ Far from fitting |

### Updated 30B estimate

Old estimate: 7.5 GB (idealized 2-bit)
**Corrected: ~15–16 GB** (5-bit weights ~7.5 GB × 2 + KV + overhead)

This is a **2× underestimate** on the 30B weight size alone.

---

## What This Changes

| | Phase 26B (old) | Phase 26B-R (corrected) |
|--|--|--|
| 7B weights | 1.75 GB | 4.4 GB |
| 7B total (4K ctx) | ~2.1 GB | ~5.2 GB |
| 7B verdict | ✅ Comfortable | ✅ Fits, but tight |
| 14B total (4K ctx) | ~4.4 GB | ~9.4 GB |
| 14B verdict | ⚠️ Near limit | ❌ OOM |
| 30B weights | 7.5 GB | ~15–16 GB |
| 30B verdict | ❌ Does not fit | ❌ Does not fit (but worse) |

**Strategic conclusion UNCHANGED:** Memory capacity and bandwidth are the core bottleneck. 7B fits but is bandwidth-limited (memory-bound per token). 14B and above do not fit on 15GB RAM without aggressive KV management or offloading.

**What changes:** The urgency on KV/context compression is even higher. At 4K context, 7B already consumes ~5.2 GB leaving only ~0.8 GB of free RAM headroom. Any context growth pushes 7B toward swap. This makes KV eviction and context management the most critical near-term SDI mechanism.

**The Bonsai-8B anomaly:** Bonsai-8B is only 1.1 GB — suggesting FP16 or unusual quantization. Not representative of Q4_K_M behavior.

---

## Updated Recommendation Implication

The corrected numbers make the case for KV/context compression **stronger**:

1. **7B at 4K ctx** leaves only ~0.8 GB free. Long sessions will push into swap. KV eviction is the primary relief valve.
2. **14B cannot load** on this machine without aggressive memory management. Not a target for now.
3. **30B is ~22 GB total**. Way off. Not a near-term target without a fundamental architecture change.

**Near-term SDI target:** Keep 7B inference from swapping at 4K+ context lengths via selective KV eviction and context compression. This is the first measurable win.

**Verification:**
```
Available RAM check: ~6 GB free at session start
7B model load: ~4.4 GB
7B + 4K KV: ~4.9 GB total
Headroom: ~1.1 GB (OS will reclaim cached pages)
At 8K context: KV ≈ 916 MB → 7B + 8K KV = 5.3 GB → borderline
At 16K context: KV ≈ 1.8 GB → 7B + 16K KV = 6.2 GB → likely swap
```

---

## Safety Scan

```
git status --short
 M ggml/src/ggml-cpu/ops.cpp
?? examples/speculative/results/PRT_PHASE24R_NATIVE_MULMAT_PROBE.md
?? examples/speculative/results/phase24r_native_mulmat_probe.json

No models staged.
No secrets found.
```

---

## Verdicts

| Verdict | Value |
|---------|-------|
| `PASS_PHASE26B_R_RAM_AUDIT` | ✅ |
| `CORRECTED_RAM_TABLE` | ✅ |
| `STRATEGIC_CONCLUSION_UNCHANGED` | ✅ |
| `KV_COMPRESSION_URGENCY_INCREASED` | ✅ |

---

*Phase 26B-R complete. Corrected table supersedes Phase 26B RAM section.*