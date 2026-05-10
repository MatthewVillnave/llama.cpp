# PRT Phase 16 — Technical FAQ

## What is SDI?

**Sub-Dense Inference (SDI)** is a research philosophy: making CPU inference more viable by changing what gets computed and how weights are represented — not by brute-forcing GPU-shaped dense inference on CPU hardware. The goal is to address the root problem (memory-bandwidth-bound inference) rather than just optimizing around it.

## What is PRT?

**PRT (Progressive Residual / Packed-sidecar path)** is one concrete SDI implementation. Instead of loading full-weight FFN layers on every forward pass, PRT uses pre-computed compressed sidecar files (one per layer) that are loaded at startup and swapped into the inference kernel at generation time. FFN_UP weights are the target because they're memory-dominant, not compute-dominant, in transformer layers.

## Is this a new model?

No. PRT runs on existing models (Qwen2.5-14B Q4_K_M in Phase 16). It changes how weights are stored and loaded at runtime, not the model architecture itself.

## Is this quantization?

Partially — but it's more than just quantization. Standard quantization (Q4_K_M, Q8_0) reduces weight precision uniformly. PRT stores FFN_UP weights in a different format (packed INT6 per-row) that is designed for a specific runtime path (mmap loading + LUT4x unpack). It's representation-aware, not just precision-reduction.

## Is this faster than native?

Generation throughput is **near-native**, not faster. In the tested suite: ~0.977–1.009× native. The point is that PRT doesn't slow things down while using less memory bandwidth — not that it speeds things up. There is still ~1.10–1.15× wall-time overhead from one-time sidecar loading at startup.

## Is this production ready?

**No.** PRT remains experimental. It has not been production-hardened. There is no error handling for malformed sidecars, no graceful fallback for unknown model sizes, no version management for sidecar formats, and no edge-case testing.

## Why sidecars?

Sidecars decouple weight storage from the GGUF model file. This allows:
- A different representation format (INT6 packed) than what GGUF uses
- Pre-computation at generation time vs. per-token decompression
- mmap-based loading (fast, no read-every-time overhead)

The sidecar is the artifact that enables the alternative runtime path.

## Why FFN_UP?

FFN_UP (the up-projection in transformer FFN layers) was selected because:
1. It is memory-dominant in the GGUF layout — not compute-dominant
2. It has a regular structure (matrix multiply, not attention)
3. It is separable — replacing it doesn't require changing the rest of the compute graph
4. It is where the bulk of memory traffic happens during generation

## Why INT6?

INT6 (4:3 packed compression) was chosen over INT8 because:
- It provides ~1.5× compression vs INT8
- It preserves tested behavior (unlike INT4, which was too lossy)
- It has an efficient LUT4x unpack kernel for CPU
- Generation throughput is near-native (unlike float32, which was too heavy)

INT4 was abandoned early — offline parity consistently failed with cosine scores too low for quality output.

## Why did INT4 fail?

INT4 per-row quantization had insufficient precision for the FFN_UP weights at the tested compression ratio. The quantized representations were too far from the original values, causing quality collapse. INT6 trades off more compression than INT4 at acceptable quality loss.

## What does "near-native throughput" mean?

It means the per-token generation rate is approximately the same as native llama.cpp inference on the same hardware. In the Phase 16 test suite: ~0.977–1.009× the native rate. This is within measurement noise for practical purposes — a user wouldn't notice the difference in interactive use.

## Why does wall time still have overhead?

The ~1.10–1.15× wall-time overhead on first run comes from:
- One-time mmap loading of all 40 sidecar files (~5s total)
- Per-file unpack and SHA verification (~90–230ms per layer)

This overhead happens once at startup, not per token. On subsequent runs with mmap reuse, the overhead is mitigated. The per-token generation rate remains near-native.

## What does the 14B result prove?

It proves that:
1. The INT6 sidecar pipeline scales to 14B (2× the largest prior model)
2. All tested behaviors are preserved (canary, 8-prompt, longer-gen, large-context)
3. Memory stays stable at 12GB (no OOM on a 15GB machine)
4. The loader fix generalizes to both 7B and 14B model sizes
5. The 16-byte header schema works for both model sizes
6. Large-context retrieval (c=2048) works correctly with INT6 path

## What does it NOT prove?

It does NOT prove:
- Production readiness (no production hardening)
- GPU comparison (no GPU data)
- Larger-than-14B support (30B+, 70B+ untested)
- Broad quality equivalence (limited prompt suite)
- All-model support (only Qwen2.5 Q4_K_M tested)
- Full long-context guarantee (only tested to c=2048)
- Public reproducibility (no package provided)

## What comes next?

Priority next steps, depending on goal:
1. **Engineering depth:** Backend/ggml integration (automatic model detection, kernel-level unpack, self-healing fallback)
2. **Communication:** Public/internal writeup package for research community
3. **Robustness:** Repeatability suite from clean state
4. **Scale:** Next-model feasibility testing (30B+ or 70B+)

## Can I reproduce this?

Not easily — this is a documentation package, not a reproducibility package. Reproducing requires: Qwen2.5-14B GGUF model, sidecar generation tools, PRT-enabled llama-cli build, and compatible CPU runtime. None of these are packaged here.

## What's the one-line summary?

> PRT is a credible experimental SDI path: sidecar-backed compressed FFN replacement preserved tested behavior up to Qwen2.5-14B on CPU-only hardware, with near-native generation throughput — suggesting that changing the representation, not just the hardware, may make CPU inference more viable.

---

*Phase 16O | Branch: experimental/prt-phase14a-packed-sidecars*
