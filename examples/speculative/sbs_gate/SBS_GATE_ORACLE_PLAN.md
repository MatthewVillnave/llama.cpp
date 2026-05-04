# SBS-Gate Oracle Feasibility Probe — Plan

## SBS-Gate Thesis

SBS-Gate proposes to accelerate CPU inference by conditionally skipping contiguous blocks of FFN weight matrix during matrix-vector multiply (GEMV). The key insight:

- Batch=1 LLM inference is memory-bandwidth bound, not compute-bound
- FFN layers (gate/up/down projections) account for ~65% of model weights and are read once per token
- GEMV has no data reuse — each weight byte is touched exactly once
- If some contiguous blocks of the FFN intermediate vector (`hidden = silu(gate) * up`) are near-zero, the corresponding columns of W_down can be skipped entirely
- Skipping means not reading those weight columns from DRAM — the only thing that actually reduces the bandwidth wall

The gate is a small MLP that predicts, from the attention output (residual stream), which blocks will be near-zero for the current token. Skipped blocks are not read.

---

## Why Oracle Test Comes Before Gate Training

Before building a learned gate, we need to answer: **does structured contiguous block sparsity actually exist?**

The learned gate (logistic regression on residual → block active/inactive) can only be as good as the underlying structure. If activation sparsity is diffuse — every block contributes meaningfully, no block is consistently near-zero — then no gate can find safe skips, and the whole direction fails.

The oracle test answers this with a clean mathematical bound: **what is the best-case skip rate at a given quality threshold, computed with perfect knowledge?**

If the oracle shows 40%+ skip rate at cosine similarity ≥ 0.995 and relative L2 ≤ 0.02, then structured sparsity exists and a learned gate has headroom.

If the oracle shows < 10% skip rate at those thresholds, the approach is falsified regardless of gate sophistication.

---

## What Would Make the Idea Promising

- Oracle skip rate ≥ 30% at cosine ≥ 0.995 and rel L2 ≤ 0.02
- Results are stable across many random samples (not cherry-picked)
- Smaller block sizes improve skip rate without making the gate impractical
- Quality degrades gracefully — cosine and L2 improve monotonically as keep-rate increases
- Gate cost (single small GEMV per layer per token) is negligible compared to savings

---

## What Would Falsify It

- Oracle skip rate < 10% at cosine ≥ 0.995 — almost every block must be kept
- Block importance is too diffuse — blocks cannot be ranked by contribution
- Activation sparsity is not structured — high-magnitude activations are evenly distributed across blocks
- Quality thresholds only met at keep-rates > 90% — insufficient headroom
- Block boundaries don't align with natural activation structure

---

## Why This Is Separate from PRT

| | PRT | SBS-Gate |
|---|---|---|
| **Mechanism** | Replace W_ffn_up with precomputed sidecar | Skip blocks of FFN based on predicted sparsity |
| **What changes** | W_ffn_up matmul only | Which weight columns are read |
| **Scope** | FFN_UP projection only | gate/up/down projections |
| **Quality mechanism** | Bit-exact replacement on native layers | Approximation via block drop |
| **Integration** | Requires custom GGML op + sidecar files | Requires gate weights + modified kernel |
| **Relation** | orthogonal/complementary | orthogonal/complementary |

SBS-Gate and PRT are independent acceleration paths. SBS-Gate could compose with PRT — PRT replaces FFN_UP with a fast path; SBS-Gate skips blocks of FFN gate/up/down. They attack the same GEMV bottleneck from different angles.

This branch does NOT modify PRT code, docs, or tags. It lives in a clean branch derived from master.

---

## Probe Order

1. **Mode A (Synthetic):** Random matrices. Establishes probe works, establishes baseline oracle bounds on synthetic data.
2. **Mode B (Real activation):** Requires instrumentation in llama.cpp to capture hidden vectors. Only if Mode A is promising and instrumentation is tractable.

Mode A comes first because it is non-invasive, fast, and provides a clean sanity check before touching any model code.

---

## Next Step After Oracle Probe

If promising: instrument llama.cpp to capture FFN hidden activations for real tokens/prompts, then run oracle on real data.

If weak/fail: document, archive, and move on. Do not build the gate or integration.
