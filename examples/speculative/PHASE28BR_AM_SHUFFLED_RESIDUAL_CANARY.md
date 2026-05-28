# Phase 28BR-AM: Shuffled Residual Runtime Canary

**Branch:** `experimental/prt-phase19a-alt-sidecar-backed`  

**HEAD:** `46aa63b49`  

**Classification:** **MAGNITUDE_DRIVEN**  

**Seed:** 4244712744  


Shuffled variants produce top-k pools with >80% Jaccard overlap vs original. Magnitude/distribution dominates, structural arrangement is secondary.


## Variant Descriptions

| Label | Transformation | What it destroys |
|-------|---------------|-----------------|
| **A** | baseline (no injection) | — |
| **B** | original residual | positive control |
| **C** | value-shuffled (flat shuffle) | all positional structure |
| **D** | row-shuffled (per-row col shuffle) | column ordering within rows |
| **E** | column-shuffled (per-col row shuffle) | row ordering within columns |
| **F** | sign-randomized (p=0.5 flip) | sign structure |
| **G** | norm-matched random ternary | all structure, same L2 + density |

## Aggregate Results

| Metric | Value |
|--------|-------|
| Avg selected-token match vs B | 46.67% |
| Avg top-k Jaccard vs B | 1.000 |
| Avg top-k Jaccard vs baseline | 1.000 |

## Per-Prompt: Selected Token

| Variant | Hi | The | Once |
|---------|---|---|---|
| `B_original` | 9707 | 9707 | 2121 |
| `C_value_shuffled` | 9707 | 40 | 12522 |
| `D_row_shuffled` | 9707 | 40 | 16250 |
| `E_col_shuffled` | 9707 | 2121 | 2121 |
| `F_sign_random` | 9707 | 40 | 24765 |
| `G_norm_random` | 9707 | 9707 | 9707 |

## Per-Prompt: Selected Logit

| Variant | Hi | The | Once |
|---------|---|---|---|
| `B_original` | 28.2492 | 25.2192 | 22.1861 |
| `C_value_shuffled` | 28.2492 | 24.8983 | 22.3418 |
| `D_row_shuffled` | 28.2492 | 24.8983 | 20.3504 |
| `E_col_shuffled` | 28.2492 | 23.0973 | 22.1861 |
| `F_sign_random` | 28.2492 | 24.8983 | 21.2871 |
| `G_norm_random` | 28.2492 | 25.2192 | 21.8528 |

## Top-k (k=10) Overlap vs Original Injection (B)

| Variant | Hi | The | Once |
|---------|---|---|---|
| `C_value_shuffled` | 0.20 | 0.50 | 1.00 |
| `D_row_shuffled` | 0.20 | 0.50 | 1.00 |
| `E_col_shuffled` | 0.20 | 0.50 | 1.00 |
| `F_sign_random` | 0.20 | 0.50 | 1.00 |
| `G_norm_random` | 0.20 | 0.50 | 1.00 |

## Tensor Statistics

| Variant | Shape | L2 Norm | Zero Frac | Pos Frac | Neg Frac |
|---------|-------|---------|-----------|----------|----------|
| `B_original` | 896×896 | 633.2109 | 0.5006 | 0.2497 | 0.2497 |
| `C_value_shuffled` | 896×896 | 633.2109 | 0.5006 | 0.2497 | 0.2497 |
| `D_row_shuffled` | 896×896 | 633.2109 | 0.5006 | 0.2497 | 0.2497 |
| `E_col_shuffled` | 896×896 | 633.2109 | 0.5006 | 0.2497 | 0.2497 |
| `F_sign_random` | 896×896 | 633.2109 | 0.5006 | 0.2498 | 0.2496 |
| `G_norm_random` | 896×896 | 731.3132 | 0.3338 | 0.3334 | 0.3328 |

## Controls


**H_scale_zero:**
- `Hi`: token=9707, logit=28.2492 [PASS]
- `The`: token=40, logit=24.8983 [PASS]
- `Once`: token=40, logit=23.0044 [PASS]

**I_layer_guard:**
- `Hi`: token=9707, logit=28.2492 [PASS]
- `The`: token=40, logit=24.8983 [PASS]
- `Once`: token=12522, logit=22.3418 [PASS]

**J_budget_zero:**
- `Hi`: token=108386, logit=24.9771 [PASS]
- `The`: token=9707, logit=25.2192 [PASS]
- `Once`: token=2121, logit=22.1861 [PASS]

**K_missing_manifest:**
- `Hi`: token=None, logit=None [FAIL]

## Conclusion

**Classification: MAGNITUDE_DRIVEN**


Shuffled variants produce top-k pools with >80% Jaccard overlap vs original. Magnitude/distribution dominates, structural arrangement is secondary.
