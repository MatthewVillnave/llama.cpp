# SBS-Gate Mode B: Real Weight Oracle Probe Results

**Date:** 2026-05-03  
**Branch:** `experimental/sbs-gate-oracle-probe`  
**Model:** Qwen2.5-3B-Instruct-Q4_K_M.gguf (FFN weights pre-extracted via PRT tool)

---

## Summary

| Metric | Synthetic 0B | Mode B (Real Weights) | Delta |
|--------|---------------|-------------------|------|
| Best skip@cos≥0.98 | 5.2% (layer 1, bs=256) | **32.5%** (layer 1, bs=256) | +27.3% |
| Gini | 0.07 | **0.34** | +0.27 |
| Top 10% energy | 9.8% | 9.5% | -0.3% |
| Strict (cos≥0.98 & l2≤0.10) | 0.0% | 12.1% | +12.1% |
| Effective block count | 42/43 | **28/43** | -14 |

Real model weights show significantly **more structured block contributions** than random synthetic weights.

---

## Results by Layer

| Layer | Position | skip@0.98 | strict | gini | top10% | eff_bk | Verdict |
|-------|---------|------------|-------|------|-------|-------|--------|
| 1 | Early | **32.5%** | 12.1% | **0.340** | 9.5% | 28.4/43 | WEAK |
| 15 | Mid | 14.8% | 3.5% | 0.228 | 8.9% | 36.2/43 | WEAK |
| 35 | Late | 15.5% | 4.2% | 0.231 | 9.6% | 36.2/43 | WEAK |

---

## Key Findings

1. **Real weights have more structure than random:**
   - Gini coefficient: 0.07 → 0.34 (early layer)
   - Best skip rate at cos≥0.98: 5.2% → 32.5%

2. **But NOT enough to be practical:**
   - Top 10% blocks still only hold ~9.5% of energy (vs ~10% for uniform)
   - Effective block count still high (~28 of 43 blocks contribute meaningfully)
   - Strict dual-threshold still fails on most layers

3. **Layer variation:**
   - Early layer (1) shows most structure (gini=0.34, skip=32.5%)
   - Mid/late layers show less structure (gini~0.23)
   - This matches intuition: early layers learn more structured input transformations

---

## Verdict: WEAK

The real-weight oracle shows significant improvement over synthetic (+27% skip rate, gini 0.34 vs 0.07), but the concentration metrics (top10 Energy ~9.5%, eff_bk ~28/43) are still too low for practical skipping.

**Not promising enough to build SBS-Gate** — the structured sparsity exists but is not strong enough to justify the implementation complexity.

---

## Next Steps

- **Option A:** Kill SBS-Gate fixed contiguous block skip — the approach doesn't show enough concentration
- **Option B:** If continued, would need clustering/reordering of FFN dimensions based on learned importance (more complex)
- **Option C:** Focus on PRT/Path-based approaches which already show 1.8x speedups

---

## Comparison: Synthetic vs Real Weight Oracle

| Metric | Synthetic (Mode A) | Real Weight (Mode B) | Interpretation |
|-------|-------------------|------------------|---------------|
| skip@cos≥0.98 | 5.2% | 32.5% | Real weights have more structure |
| skip@cos≥0.995 | 0.0% | ~0% | Threshold still too strict |
| gini | 0.07 | 0.34 | Real > 4x more concentrated |
| top10_energy | 9.8% | 9.5% | Similar (near-uniform) |
| strict (dual) | 0.0% | 12.1% | Some headroom at early layer |
| contribution vs random | +5.2% | Real weights beat synthetic by 5x | Real weights more structured |

The key insight: FFN weights ARE more structured than random, but the structured elements are spread throughout the FFN dimension, not concentrated in contiguous blocks. This limits the skip rate for fixed block sizes.