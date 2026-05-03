# PRT Overview

**PRT** = **P**rogressive **R**esidual **T**ernary

---

## What PRT Does

PRT replaces selected FFN_UP (feed-forward up-projection) matrix multiplications inside transformer layers with a sparse ternary-weight approximation computed from pre-extracted sidecar matrices.

In llama.cpp's compute graph, this is a **true graph-node replacement** — the native `build_lora_mm` matmul never enters the graph. Instead, a GGML custom op computes the output directly from sidecar data.

**Goal:** Reduce CPU inference time by replacing expensive float32 matmul with threshold-sparse ternary matmul.

---

## Why CPU Inference

CPU inference is slow because float32 matmul is expensive and CPUs can't batch efficiently. GPU inference uses cuBLAS/cuDNN and gets hardware acceleration. CPU inference gets nothing by default.

PRT is a software-level acceleration: replace the math, keep the hardware.

---

## What Route A Means

There are multiple ways to intercept and replace a computation in a inference engine:

- **Callback overwrite:** Run native computation, capture result, overwrite with PRT result. Fast to implement, but requires running native first — defeating the speedup.
- **Route A (graph replacement):** Replace the node in the compute graph itself. Native never runs. True speedup.

Route A is the clean path: the graph says "compute FFN_UP" and the graph node that actually runs is the PRT custom op, not the native matmul.

---

## The L12/L15 Problem

Early testing revealed a problem: pure all-36-layer PRT produced incorrect output on specific narrative prompts (e.g., "Once upon a time in a" → "far away land" instead of "small village").

The failure was not from a missing layer or a simple bug — it was a **nonlinear layer interaction** in the L12-L17 range. Single layers didn't fix it. More native layers didn't fix it. Only specific layer *pairs* fixed it.

**Fix:** Force layers 12 and 15 to use native computation (`--prt-force-native 12,15`) while keeping all other 34 layers on PRT. The model continues as if those two layers were always native. Output matches native.

This is a fallback policy, not a workaround. It reflects a real limitation of the PRT approximation at certain layer configurations that requires anchor points.

---

## Sidecars

PRT requires pre-extracted weight matrices (sidecars) for each layer. These are:

- **Not committed to the repo** — they are large binary files (~90MB per layer × 36 layers ≈ 3.2GB)
- **Model-specific** — extracted from a specific GGUF model
- **Generated once** from a native reference run, then reused

Sidecars must be generated on the same model and quantization as the target deployment. Sidecars from Qwen2.5-3B-Q4_K_M will not produce correct results on Qwen2.5-7B.

---

## Safety: Missing Sidecars

Prior versions of this code silently fell back to native computation if a sidecar was missing — incrementing a counter and continuing. This could produce incorrect output without any error signal.

**Current state:** If PRT Route A is enabled and a required sidecar is missing, the process exits with a fatal error before generation begins:

```
[PRT-ERROR] FATAL: N required sidecar(s) missing. Exiting.
```

This ensures missing sidecars cannot silently degrade output quality.

---

## Experimental Status

PRT Route A is **experimental research code**. It is not:

- Production-ready software
- A general-purpose CPU inference accelerator
- Validated on all prompt types or models
- Merged into llama.cpp upstream

It is a release-candidate experimental branch with validated speedup on a controlled mini-suite of 6 prompts.

**Scope of validation:**
- 6 prompts tested
- Qwen2.5-3B-Instruct-Q4_K_M model only
- n=100 for most prompts, n=50 for JSON
- ~1.84x speedup observed

**Unknown:**
- JSON/structured full validation (memory-limited machine)
- Generalization to other models
- Behavior on very long contexts
- Whether speedup holds at batch sizes > 1

---

*For claims and reproduction notes, see PRT_CLAIMS.md and PRT_REPRODUCTION_NOTES.md*
