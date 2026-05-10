# Sub-Dense Inference: Making CPU Inference More Viable

**An experimental SDI path for CPU-native inference using compressed FFN sidecars**

---

## Introduction

Running large language models on CPU-only hardware is hard. Dense transformer inference is memory-bandwidth-bound, and CPU hardware doesn't have the memory bandwidth that GPU hardware has. The standard approaches — quantization, batching, KV cache management — help, but they don't address the root problem: we're asking CPUs to run GPU-shaped dense inference.

What if we changed what gets computed and how weights are represented?

**Sub-Dense Inference (SDI)** is a research direction that explores exactly this. Instead of running the full dense transformer on CPU, SDI paths look for smarter representations and selective computation that are better suited to CPU constraints.

**PRT** (Progressive Residual / Packed-sidecar path) is one concrete SDI implementation: a sidecar-backed FFN replacement path that stores pre-computed compressed weights and swaps them into the inference kernel at runtime.

This document summarizes the Phase 16 result: PRT working on Qwen2.5-14B Q4_K_M on a CPU-only machine.

---

## Why CPU Inference Struggles

The fundamental problem is memory bandwidth, not compute.

A transformer layer does matrix multiplications. On GPU, memory bandwidth is high and the compute is parallelized across thousands of cores. On CPU, memory bandwidth is lower, and the parallelism is more limited.

When you run a 14B Q4_K_M model on a CPU, the weights are decompressed from GGUF and loaded into memory on every forward pass. This creates a massive memory traffic burden — reading ~8GB of weight data on each token generation step. The CPU computes fast enough, but it spends most of its time waiting for memory.

Standard approaches:
- **Lower-bit quantization** (Q4_K_M, Q8_0) reduces weight size but still uses dense representation
- **Smaller models** reduce memory but lose capability
- **Batching** amortizes load across multiple requests but doesn't reduce per-token cost

None of these address the root: dense inference on CPU is memory-bound, not compute-bound.

---

## What SDI Means

SDI means changing the computation model, not just the storage format.

Instead of: Load full GGUF weights → decompress → run dense matmul on every token
SDI asks: Can we pre-compute and store alternate representations that are more CPU-friendly?

The key insight from Phase 14: float32 sidecars preserved behavior but were *too heavy*. The representation mattered as much as the idea. Packed INT8 sidecars recovered near-native throughput by being 4× smaller — the memory bandwidth savings outweighed the decompression cost.

PRT is the concrete test of this idea: can a compressed sidecar path for FFN_UP weights preserve tested behavior while being more CPU-friendly?

---

## What PRT Is

PRT targets the **FFN_UP weight tensor** — the up-projection in transformer FFN layers. This tensor is memory-dominant (not compute-dominant) in the GGUF layout, making it a good candidate for replacement.

**How it works:**

1. **At generation time:** Pre-compute FFN_UP weights from GGUF, quantize to INT6, write as sidecar files (one per layer)
2. **At startup:** Load sidecar files via mmap (not read-every-time)
3. **At runtime:** Use packed INT6 weights instead of decompressing full GGUF weights per token
4. **Fallback:** Two layers (11, 15) use native GGUF to prevent error accumulation

**The sidecar format:** 16-byte header (magic + M + K), per-row scales (M × float32), packed INT6 payload. ~53MB per layer for 14B (vs ~283MB for float32).

---

## The Experiment Path

### Phase 13: Float32 sidecars → speed-negative
Float32 sidecars preserved behavior but were too memory-intensive. Memory traffic dominated, throughput was below native. Lesson: the representation was the bottleneck.

### Phase 14: Packed INT8 sidecars → near-native throughput
INT8 per-row quantized sidecars reduced size ~4× and recovered near-native throughput:
- 0.5B: 8/8 exact matches, 1.81× faster than float32 PRT
- 3B: 8/8 semantic matches, INT8/native ratio 1.000×
- 7B: 6/8 exact, 8/8 semantic, INT8/native ratio 0.993×

The representation fix resolved the speed problem while preserving quality.

### Phase 15: INT6 exploration + optimization
- INT4 per-row: too lossy (offline parity failed consistently) → abandoned
- INT6: passed offline parity and all quality gates
- CLI-level optimizations: provenance logging, manifest cache, mmap loading, LUT4x unpack
- After optimization, INT6 setup overhead is ~913ms, generation is memory-bandwidth-bound

### Phase 16: 14B validation
- Schema bug fixed (20-byte → 16-byte header)
- Loader size-check bug fixed (14B added to known sizes)
- Full 40-layer INT6 sidecar generation (2.0GB total)
- Tiny canary: 40/40 sidecars loaded, "Paris" exact match
- 8-prompt validation: 8/8 semantic matches, 0 degradations
- Longer-gen smoke: n=320 exact match, c=2048 exact match
- Generation throughput: ~0.977–1.009× native in measured suite

---

## What Worked

1. **INT6 packed sidecars on 14B:** All quality gates passed. No collapse, no repetition. Near-native generation throughput.
2. **Schema correctness:** The 16-byte header format works consistently across all models.
3. **mmap loading:** Fast sidecar loading without read-every-time overhead.
4. **Memory stability:** 12GB available throughout all validation, no OOM or swap pressure.
5. **Large-context retrieval:** c=2048 (≈1500 words) exact-match context retrieval confirms the INT6 path doesn't lose information in large contexts.

---

## What Failed

1. **Float32 sidecars:** Too memory-intensive. Representational change was needed.
2. **INT4 per-row:** Too lossy. Cosine scores too low for quality.
3. **14B schema mismatch:** Writer used 20-byte header, runtime expected 16. Fixed.
4. **14B loader bug:** Hardcoded only 7B size. 14B files silently skipped. Fixed.

---

## The 14B Result

**Model:** Qwen2.5-14B Q4_K_M (8.4GB GGUF)
**Hardware:** Dell OptiPlex 7010, 15GB RAM, CPU-only (TheForgeHQ)
**Format:** INT6 packed per-row sidecars
**Sidecar set:** 40 files, 53,139,472 bytes each, ~2.0GB total

| Test | Native | INT6 | Match |
|------|--------|------|-------|
| Tiny (8 tokens) | "Paris" | "Paris" | ✅ Exact |
| 8-prompt suite | 8/8 | 8/8 | ✅ 8/8 semantic |
| n=320 prose | "story" response | "story" response | ✅ Exact |
| c=2048 factual | "Phase 16K showed..." | "Phase 16K showed..." | ✅ Exact |

**Generation throughput:** 0.977–1.009× native (within measurement noise for practical purposes)
**Memory:** Stable at 12GB available, no OOM

---

## Why This Matters

On GPU-less hardware, "usable" inference often means painful tradeoffs: tiny context windows, models quantized to death, generation that feels sluggish. The user experience is fundamentally degraded compared to GPU inference.

PRT suggests that changing the *representation* — not just the quantization depth — can recover near-native behavior. The 14B result means the pipeline scales to model sizes that are meaningfully capable, not just toy examples.

This doesn't mean GPUs are unnecessary. It means CPUs might be more viable for local inference than the standard dense-transformer-on-CPU framing implies, *if* the runtime is designed around CPU constraints.

---

## What This Does Not Prove

- ❌ Production readiness (no production hardening, error handling, or edge-case testing)
- ❌ Universal speedup (only tested in the measured CPU setup, limited prompt suite)
- ❌ GPU comparison (no GPU testing)
- ❌ All-model support (only Qwen2.5 Q4_K_M tested)
- ❌ Broad quality equivalence (only the tested prompt suite validated)
- ❌ Full long-context guarantee (only tested to c=2048)
- ❌ Larger-than-14B validation (30B+, 70B+ not tested)

---

## What Comes Next

**Engineering depth:** Backend/ggml integration — PRT is CLI-level. Automatic model detection, kernel-level INT6 unpacking, and self-healing on layer mismatch would make it a first-class compute path.

**Communication:** A public writeup package documenting the full Phase 14–16 results for researchers and practitioners interested in CPU-native inference.

**Robustness:** Repeatability suite — re-run the full pipeline from a clean state to confirm reproducibility.

**Scale:** Next-model feasibility — what happens at 30B or 70B? The sidecar generation time and loader generality need testing at larger model sizes.

---

## One-Sentence Thesis

> PRT is a credible experimental SDI path: sidecar-backed compressed FFN replacement preserved tested behavior up to Qwen2.5-14B on CPU-only hardware, with near-native generation throughput in the measured suite — suggesting that changing the representation, not just the hardware, may make CPU inference more viable for local AI.

---

*Document version: Phase 16O*
*Branch: experimental/prt-phase14a-packed-sidecars*
*Results scoped to Qwen2.5 models, measured CPU hardware, tested prompt suite*
