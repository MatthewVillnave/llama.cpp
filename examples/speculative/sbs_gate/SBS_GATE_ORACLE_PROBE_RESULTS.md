# SBS-Gate Oracle Probe Results

**Branch:** `experimental/sbs-gate-oracle-probe`  
**Base commit:** `db44417b` (fork/master)  
**Date:** 2026-05-03  
**Mode:** Synthetic (Mode A — non-invasive oracle on random matrices)

---

## What Was Tested

Synthetic oracle probe on random matrices simulating GLU-FFN activation patterns:

- **Hidden dim:** 4096
- **FFN dim:** 11008
- **Block sizes tested:** 128, 256, 512, 1024
- **Samples per config:** 64
- **Quality thresholds:** cosine ≥ 0.995, relative L2 ≤ 0.02
- **Activation model:** Heavy-tailed distribution with ~40% near-zero activations, ~10% heavy bursts

The synthetic activation model is deliberately structured to have many near-zero blocks — this tests whether the oracle probe methodology works before applying it to real data.

---

## Synthetic Results Table

| block_size | n_blocks | avg_skip% | median_skip% | min_skip% | max_skip% | cos@threshold | passed |
|-----------|----------|-----------|--------------|-----------|-----------|---------------|--------|
| 128 | 86 | ? | ? | ? | ? | ? | ? |
| 256 | 43 | ? | ? | ? | ? | ? | ? |
| 512 | 22 | ? | ? | ? | ? | ? | ? |
| 1024 | 11 | ? | ? | ? | ? | ? | ? |

---

## Block Size Sensitivity

(TODO: update after probe run)

- Smaller block sizes → more blocks → finer-grained skipping → higher skip rate
- But smaller blocks also mean more gate output bits → more gate complexity
- Tradeoff: block_size=256 is the recommended sweet spot (43 blocks = 43-bit gate mask)

---

## Interpretation

### If skip rate ≥ 30% on synthetic data:

> Structured sparsity EXISTS in the synthetic activation model. The oracle probe methodology is valid. However, synthetic data is a best-case scenario — the activation distribution was chosen to have structured sparsity. Real FFN activations may not behave the same way. Mode B (real activations) is required before any claims.

### If skip rate 10–30% on synthetic data:

> Some headroom in synthetic mode. This validates the methodology but does not prove real FFN activations have structured sparsity. Need Mode B before proceeding.

### If skip rate < 10% on synthetic data:

> Even in synthetic data with deliberately structured sparsity, <10% skip rate means the methodology needs re-thinking. Block contribution ranking may not be the right metric, or block boundaries don't align with activation structure.

---

## What Remains Unknown

1. **Real FFN activation structure** — we tested synthetic data that approximates GLU-FFN, but real models may behave differently
2. **Which layers have the most structured sparsity** — mid-layers vs late layers may differ
3. **Token-level vs prompt-level variation** — some prompts may be easier to skip than others
4. **Down projection vs up projection skip** — current probe focuses on down projection; gate/up projections also need analysis
5. **Effect of quantization** — Q4_K_M weights may have different activation patterns than FP32 synthetic

---

## Exact Next Step: Real Activation Test

After Mode A (synthetic) completes:

1. **If promising:** Instrument llama.cpp with `ffn_act_callback` (see `SBS_GATE_REAL_ACTIVATION_CAPTURE_PLAN.md`) to capture real `h_hidden` and `out_full` for layers 15 and 25 across 24 prompts
2. **Run oracle on real data** — Mode B of the probe
3. **Decision:** If real oracle skip rate ≥ 20% → proceed to gate bootstrapping. If < 10% → falsify and archive.

---

## Pass/Fail Verdict

**SYNTHETIC MODE — Methodology Validation Only**

- This test validates the oracle probe methodology, NOT SBS-Gate feasibility on real models
- Synthetic mode with structured-sparse activations is a best-case proxy

**Verdict:** PENDING — awaiting Mode A probe results.

After real activation Mode B:  
- PASS: skip rate ≥ 20% on real data → gate bootstrapping  
- WEAK: skip rate 10–20% → more analysis needed  
- FAIL: skip rate < 10% → falsified, archive the direction

---

*Results prepared by ELVIS for The ForgeHQ / Matthew Villnave*
