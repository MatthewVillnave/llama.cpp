# SBS-Gate Oracle 0B: Threshold Sweep + Rescale Diagnostic

**Branch:** `experimental/sbs-gate-oracle-probe`
**Base commit:** `db44417b` (fork/master)
**Date:** 2026-05-03
**Mode:** Synthetic (Mode A — oracle on random matrices)

---

## 1. Summary

The strict dual-threshold oracle (cos≥0.995 AND rel_l2≤0.02) fails everywhere, as found in the initial probe. The 0B sweep extends across a broader threshold grid, multiple block sizes, multiple ranking methods, and a rescale diagnostic. Results are clear:

**Verdict: FAIL for structured contiguous block skip in this form. WEAK signal in the contribution ranking — not enough to build on.**

---

## 2. Why the Strict Test Failed

From the rescale diagnostic (bs=256, contribution_norm):

| keep | skip% | cos_raw | l2_raw | alpha | cos_r | l2_r |
|------|-------|---------|--------|-------|-------|------|
| 43/43 | 0% | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 |
| 40/43 | 7% | 0.980 | 0.199 | 1.001 | 0.980 | 0.199 |
| 35/43 | 19% | 0.935 | 0.355 | 1.001 | 0.935 | 0.355 |
| 30/43 | 30% | 0.880 | 0.474 | 1.012 | 0.880 | 0.474 |
| 25/43 | 42% | 0.817 | 0.577 | 1.031 | 0.817 | 0.576 |
| 20/43 | 53% | 0.753 | 0.659 | 1.029 | 0.753 | 0.659 |
| 15/43 | 65% | 0.671 | 0.741 | 1.033 | 0.671 | 0.741 |
| 10/43 | 77% | 0.585 | 0.812 | 1.069 | 0.585 | 0.811 |
| 5/43 | 88% | 0.443 | 0.898 | 1.103 | 0.443 | 0.897 |
| 1/43 | 98% | 0.200 | 0.980 | 1.067 | 0.200 | 0.980 |

**Key insight: The error is directional, not just norm.** Alpha stays ≈1.0 across all skip rates, meaning rescaling the skipped output by a scalar does NOT fix the L2 error. This rules out "just a magnitude problem." The degradation is that dropping blocks changes which directions in the output space you hit — and every block contributes meaningfully.

---

## 3. Threshold Grid Results

### Block Size 256 (43 blocks), 8 samples

| method | @0.98 cos | @0.95 cos | strict (cos≥0.98 & l2≤0.10) | l2≤0.15 alone |
|--------|-----------|-----------|-------------------------------|---------------|
| contribution_norm | **5.2%** | 7.0% | 0.0% | 0.0% |
| random | 0.0% | 7.0% | 0.0% | 0.0% |
| reverse_contribution | 0.0% | 0.0% | 0.0% | 0.0% |

### Block Size 512 (22 blocks)

| method | @0.98 cos | @0.95 cos | strict | l2≤0.15 |
|--------|-----------|-----------|--------|---------|
| contribution_norm | 0.0% | 9.1% | 0.0% | 0.0% |
| random | 0.0% | 8.0% | 0.0% | 0.0% |
| reverse_contribution | 0.0% | 0.0% | 0.0% | 0.0% |

### Block Size 1024 (11 blocks)

| method | @0.98 cos | @0.95 cos | strict | l2≤0.15 |
|--------|-----------|-----------|--------|---------|
| contribution_norm | 0.0% | 9.1% | 0.0% | 0.0% |
| random | 0.0% | 4.5% | 0.0% | 0.0% |
| reverse_contribution | 0.0% | 0.0% | 0.0% | 0.0% |

---

## 4. Block Size Comparison

- Smaller blocks (256) give contribution_norm its best performance: 5.2% at cos≥0.98. Finer granularity means more fine-grained ranking, which helps even on uniform data.
- Larger blocks (512, 1024) give contribution_norm no advantage over random at cos≥0.98. At coarser granularity, each block is more important individually, making skip harder.

---

## 5. Ranking Method Comparison

**Contribution_norm vs random:**
- At cos≥0.98: contribution_norm wins (5.2% vs 0.0% at bs=256). Real signal exists.
- At cos≥0.95: nearly tied (7.0% vs 7.0%). The threshold is loose enough that random also finds passes.
- At cos≥0.90: contribution_norm gets 18.6% vs random's 9.8% (bs=256). Stronger advantage at looser thresholds.

**Reverse_contribution (worst case):**
- 0.0% skip at cos≥0.98 everywhere. Correctly identifies that dropping high-contribution blocks first is catastrophic.
- Confirms the ranking direction is correct (drop low-contribution blocks first).

**Interpretation:**
The oracle ranking IS working correctly — dropping least-contributing blocks first genuinely preserves quality better than random. But the effect is modest: at the strictest useful quality threshold (cos≥0.98), the advantage is only 5.2% skip rate. This is not enough for a practical speedup.

---

## 6. Raw vs Rescaled Error

**Conclusion from rescale diagnostic:** Rescaling does NOT materially improve L2 error. Alpha stays ≈1.0, and both cos_r and l2_r are essentially identical to cos_raw and l2_raw.

This means the approximation error is **directional, not just magnitude**. Skipping blocks doesn't just make the output smaller — it makes it point in a measurably different direction. Rescaling cannot fix this.

---

## 7. Contribution Concentration

| block_size | n_blocks | top_10% energy | eff_block_count | gini |
|-----------|----------|---------------|----------------|------|
| 256 | 43 | 9.8% | 42.3 | 0.070 |
| 512 | 22 | 8.4% | 21.8 | 0.058 |
| 1024 | 11 | 8.6% | 11.0 | 0.033 |

**Interpretation:**
- Gini ≈ 0.07 (bs=256) — contributions are nearly uniformly distributed. 0.0 = perfectly equal, 1.0 = maximally unequal.
- Top 10% of blocks hold only 9.8% of total energy — the top 4 blocks contribute less than 10% combined.
- Effective block count ≈ 42.3 out of 43 — almost every block contributes nearly equally.
- This is the worst-case scenario for structured sparsity: no block is disproportionately important.

**For comparison:** If contributions were perfectly concentrated (1 block = 100% of energy, rest = 0), top_10% energy would be 100% and effective block count would be ≈1. Here we get nearly the inverse.

---

## 8. Verdict

**FAIL**

Criteria from the plan:

- **Real or synthetic oracle skip ≥30% at cosine ≥0.98 and rel_l2_rescaled ≤0.10?** NO — best is 5.2% at bs=256, cos≥0.98.
- **Contribution_norm strongly beats random?** WEAK signal at cos≥0.98 (5.2% vs 0%), stronger at cos≥0.90 (18.6% vs 9.8%). The ranking works but the absolute skip rates are too low.
- **Contribution is concentrated, not diffuse?** NO — gini≈0.07, top10_energy=9.8%, eff_block_count≈42.3/43. Contributions are near-uniform.

Additional findings:
- Error is directional, not just norm — rescaling does not help
- The oracle ranking IS correct (reverse_contribution is worst case)
- Finer block granularity (256 vs 512/1024) helps modestly

---

## 9. Next Step

**Option A: One real activation capture test before killing**
- Collect actual FFN hidden vectors from Qwen2.5-3B via llama.cpp instrumentation
- Run the oracle on real activations
- If real activations show concentration (gini > 0.3, top10_energy > 30%), structured sparsity may exist in trained models even if it doesn't in random matrices
- This is the only remaining path to validate SBS-Gate

**Option B: Kill SBS-Gate fixed contiguous block skip**
- The synthetic oracle provides strong evidence that random FFN weight matrices don't have structured contiguous block sparsity
- The approach requires dense, not sparse, contribution distribution
- Prune, document, and move on

**Option C: Block clustering/reordering fallback**
- Instead of fixed-size contiguous blocks, cluster FFN dimensions by activation co-occurrence patterns
- This could create blocks with genuinely unequal contributions even when raw dimensions are uniform
- More complex, no guarantee it works

**Recommended: Option B (kill)** — with an asterisk that one real activation test is the only remaining uncertainty.

The fundamental issue is not the thresholds or block sizes — it's that FFN weight matrix columns weighted by activation vectors produce near-uniform contribution distributions on random data. Real trained models might differ, but the prior is strong enough that building on this without real data confirmation is not justified.

---

## Key Numbers for Record

- Best raw skip at cos≥0.98: **5.2%** (bs=256, contribution_norm, 8 samples)
- Best raw skip at cos≥0.995: **0.0%** everywhere
- Contribution_norm vs random at cos≥0.98: **5.2% vs 0.0%** (real signal exists, too small)
- Rescale improves L2: **NO** (directional error, not magnitude)
- Gini coefficient: **0.07** (near-uniform contributions)
- Top 10% energy: **9.8%** (each block matters nearly equally)
- Effective block count: **42.3/43** (almost no structured sparsity)
- Strict dual threshold (cos≥0.98 & l2≤0.10): **0.0% everywhere**

*Report prepared by ELVIS for The ForgeHQ / Matthew Villnave*
