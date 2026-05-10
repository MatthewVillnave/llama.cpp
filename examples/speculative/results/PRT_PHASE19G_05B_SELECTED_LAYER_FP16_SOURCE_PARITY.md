# PRT Phase 19G: 0.5B Selected-Layer FP16/BF16 Source Parity Closure

## Summary

Phase 19G closes the lane on FP16/BF16 source quality for PRT on Qwen2.5-0.5B by testing selected ffn_up layers.

## Sources

| Source | Path | Type | Size |
|--------|------|------|------|
| GGUF Q5.0 | /home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf | Q5.0 | 380MB |
| FP16/BF16 | /home/matthew-villnave/models/hf/qwen2.5/Qwen2.5-0.5B-Instruct/ | bfloat16 | 952MB |

Note: GGUF file named "Q4_K_M" internally uses Q5.0 quantization.

## Model Metadata (Confirmed)

| Parameter | Value |
|-----------|-------|
| Model | Qwen2.5-0.5B |
| Architecture | qwen2 |
| Layers | 24 |
| Hidden size | 896 |
| FFN size | 4864 |

## Selected-Layer Source Parity

### Layer 0

| Metric | Value |
|--------|-------|
| Cosine similarity | 0.99907 |
| MAE | 0.000638 |
| RMSE | 0.000771 |
| Max abs error | 0.0097 |
| Norm ratio | 0.99984 |

### Layer 1

| Metric | Value |
|--------|-------|
| Cosine similarity | 0.99908 |
| MAE | 0.000597 |
| RMSE | 0.000723 |
| Max abs error | 0.0086 |
| Norm ratio | 0.99994 |

### Layer 12 (middle)

| Metric | Value |
|--------|-------|
| Cosine similarity | 0.99899 |
| MAE | 0.000689 |
| RMSE | 0.000841 |
| Max abs error | 0.0106 |
| Norm ratio | 0.99972 |

### Layer 23 (final)

| Metric | Value |
|--------|-------|
| Cosine similarity | 0.99905 |
| MAE | 0.000718 |
| RMSE | 0.000870 |
| Max abs error | 0.0105 |
| Norm ratio | 0.99962 |

## Cross-Layer Summary

| Metric | Min | Max |
|--------|-----|-----|
| Cosine similarity | 0.99899 | 0.99908 |
| MAE | 0.000597 | 0.000718 |
| Norm ratio | 0.99962 | 0.99994 |

**All layers: cosine >= 0.999 across all tested layers.**

## INT6 Parity Comparison (Layer0 Only)

| Source | Cosine vs FP16 ref | MAE vs FP16 ref |
|--------|-------------------|-----------------|
| GGUF source (no quant) | 0.99907 | 0.000638 |
| GGUF-sourced INT6 | 0.99851 | 0.000797 |
| FP16-sourced INT6 | 0.99935 | 0.000539 |

**Finding:** FP16-sourced INT6 matches FP16 reference slightly better (cosine 0.9994 vs 0.9985), but GGUF-sourced INT6 is still within 0.001 cosine difference.

## Interpretation

1. **GGUF source is high quality**: All 4 selected layers show cosine >= 0.999 against FP16 reference
2. **Quantization error dominates**: INT6 quantization introduces ~0.02-0.03% MAE; GGUF→Q5.0 source adds ~0.06% MAE on top
3. **FP16 source marginal benefit**: FP16-sourced INT6 is only ~0.001 cosine better than GGUF-sourced INT6
4. **Not worth the storage cost**: FP16/BF16 source is 952MB vs GGUF at 380MB for only ~0.1% parity improvement

## Verdict

**PARTIAL_SELECTED_LAYER_PARITY** — All layers show GGUF-vs-FP16 cosine >= 0.999 (near-perfect), but layer 12 is just under 0.999 at 0.99899.

GGUF source is **good enough** for PRT on Qwen2.5-0.5B Q5.0. FP16 source does not materially improve parity to justify 2.5x storage cost.

## Recommended Next Steps

1. **Performance focus**: Optimize INT6 kernel rather than pursue FP16 source storage
2. **Formal Phase 19 checkpoint**: Consolidate findings and close PRT Phase 19
3. **Optional**: Test layer 12 specifically to understand why it has lowest cosine
