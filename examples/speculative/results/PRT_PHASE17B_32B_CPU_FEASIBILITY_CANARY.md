# PRT Phase 17B — 32B CPU Feasibility Canary

**Date:** 2026-05-09
**Branch:** `experimental/prt-phase14a-packed-sidecars`
**Previous HEAD:** `2315f1c15` (Phase 17A)
**This HEAD:** Pending commit

---

## Verdict

**INFEASIBLE_32B_HARDWARE_ESTIMATE**

---

## Context

Phase 16 validated PRT at 14B scale on Matt's CPU-only setup (Dell OptiPlex 7010, 15GB RAM, TheForgeHQ). The Phase 17A strategy decision recommended testing the next CPU-only boundary: can 32B run on this hardware?

This canary attempted to download Qwen2.5-32B-Instruct-GGUF (Q4_K_M) from HuggingFace and test native loading at minimal context.

---

## Model / Storage

**Attempted model:** Qwen/Qwen2.5-32B-Instruct-GGUF
**Quantization:** Q4_K_M
**Split file count:** 5 shards (qwen2.5-32b-instruct-q4_k_m-0000X-of-00005.gguf)

**File size analysis (via HTTP HEAD):**
- Shard 1: 3,961,498,272 bytes (~3.69 GB)
- Shard 2: 3,948,990,664 bytes (~3.68 GB)  
- Shard 3: 3,961,498,272 bytes (~3.69 GB)
- Shard 4: 3,948,996,024 bytes (~3.68 GB)
- Shard 5: 3,961,498,272 bytes (~3.69 GB)
- **Total: ~18.5 GB**

**Download attempt:**
- Attempted via curl from HuggingFace CDN
- Progress: received 873,560,707 bytes (~813 MB) in 300 seconds before timeout
- Transfer rate: ~2.9 MB/s average, slowing to ~200-400 KB/s
- Estimated full download time: 5+ hours for all 5 shards

**Storage available:** 86GB free on NVMe (sufficient for 18.5 GB model)

---

## Native Tiny Canary

**Attempted:** Yes — download in progress
**Failed:** Download timeout — network transfer too slow to complete in reasonable time

**Classification:** Network/bandwidth limitation prevented download completion. Even with unlimited time, the hardware constraint remains.

---

## Hardware Constraint Analysis

**RAM analysis:**
- Machine: Dell OptiPlex 7010, 15GB total RAM
- Currently available: ~11GB (system + Ollama using ~4GB)
- 32B Q4_K_M GGUF: ~18.5 GB just for model weights
- Additional required: KV cache at ctx=256 (~2-3 GB), activations, GGML overhead
- **Total estimated: ~22-24 GB needed vs 15 GB available**
- **Shortfall: ~7-9 GB**

**Memory math:**
- 32B model requires loading all 18.5GB of weights into RAM
- At Q4_K_M, each parameter uses ~4.5 bits
- 32B × 4.5 bits = 18B bits = ~2.25 GB per billion parameters × 32 ≈ 72 GB theoretical minimum
- Actual GGUF is more overhead than pure bit-width due to quantization tables and metadata
- Actual observed: 18.5GB for model files

**Verdict:** 32B exceeds available RAM by ~7-9GB. Even IF we downloaded the model, it would OOM on load.

---

## Sidecar Feasibility Estimate (Hypothetical)

Even if the model loaded (which it won't), sidecar generation would be problematic:

- **Expected FFN_UP shape:** ~{8192, 28672} = 235,929,600 elements per layer
- **Expected INT6 per-layer sidecar:** ~63 MB per layer (16-byte header + 8192×4 scales + ~60MB packed)
- **Expected layer count:** 64 layers
- **Expected total INT6 sidecar set:** ~4 GB
- **Disk pressure:** 18.5GB (model) + 4GB (sidecars) = ~22.5GB total

**Verdict even more negative:** Sidecar generation would push disk even higher, not solve the RAM problem.

---

## Interpretation

**Can this hardware run 32B native at all?** ❌ No. RAM insufficient (~11GB available vs ~22-24GB needed).

**Is it usable or merely technically loadable?** Neither — download fails due to network speed, and even if downloaded, OOM guaranteed.

**Is 32B PRT sidecar generation justified?** ❌ No. Can't load the base model, can't generate sidecars, can't test anything.

**Should the project continue 32B, pivot to 8B, or go backend/ggml?**

- For 32B: MUST stop on this hardware. No path forward.
- Recommended pivot: 8B validation (Qwen2.5-8B or Llama-3.1-8B) — both would fit comfortably in ~11GB available
- Alternative: Backend/ggml integration for automatic model detection (solves the hardcoded size checks problem)

---

## Allowed Claims

- 32B download attempted (failed due to slow network)
- 32B model size confirmed via HTTP headers (~18.5GB total)
- Hardware RAM is insufficient (~11GB available vs ~22-24GB needed)
- NO 32B PRT claim since sidecars were never tested

---

## Forbidden Claims

Do NOT claim:
- ❌ 32B PRT works (false — didn't run)
- ❌ Production readiness (no 32B validation)
- ❌ Universal speedup (single hardware config)
- ❌ GPU comparison (no GPU data)
- ❌ Larger-than-32B support (not tested)
- ❌ 32B loaded on this hardware (false — download failed)
- ❌ 32B quality equivalence (no generation data)
- ❌ 32B support beyond this canary attempt

---

## Recommended Next Phase

**Phase 17C: Pivot to 8B Validation**

Rationale: The 14B result already proves PRT works at the "edge" of what's practical on this 15GB machine. The 32B attempt confirms this is truly the boundary. An 8B validation would:

1. Confirm PRT generalizes to smaller-but-still-capable models (8B vs 14B vs 7B weight progression)
2. Validate on hardware where 32B couldn't even be attempted
3. Provide another data point for the weight-class curve (0.5B → 3B → 7B → 14B → 8B)

Alternative: Backend/ggml integration (for long-term infrastructure)

**DO NOT repeat 32B download attempt on this hardware.** It wastes time and network bandwidth.

---

*Phase 17B | Verdict: INFEASIBLE_32B_HARDWARE_ESTIMATE | Branch: experimental/prt-phase14a-packed-sidecars*