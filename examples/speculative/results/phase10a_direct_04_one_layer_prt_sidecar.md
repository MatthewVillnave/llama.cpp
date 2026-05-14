# Phase 10A — Step 3C: One-Layer PRT_3P Sidecar + Offline Accuracy

**Timestamp:** 2026-04-30 12:10 EDT
**Status:** PASS ✅

## What was tested
- **Weight source:** blk.0.ffn_up.weight from Qwen2.5-3B-Instruct-Q4_K_M.gguf (dequantized via `ggml_get_type_traits->to_float`)
- **PRT_3P:** Correctly applied to input activations X (not weights W) — same as Phase 8 kernel
- **Thresholds:** Plane 0: |x|>2.0, Plane 1: 0.5<|x|≤2.0, Plane 2: 0.1<|x|≤0.5, Pruned: |x|≤0.1
- **Computation:** Y_prt = X @ W using PRT_3P activation decomposition + float32 weights

## Results

### Batch 16
| Metric | Value |
|--------|-------|
| **Cosine similarity** | **0.999511** ✅ |
| Max abs error | 0.099955 |
| Mean abs error | 0.015309 |
| Std abs error | 0.011651 |
| Pass (cosine ≥ 0.95) | **YES** ✅ |

### Batch 17
| Metric | Value |
|--------|-------|
| **Cosine similarity** | **0.999511** ✅ |
| Max abs error | 0.099955 |
| Mean abs error | 0.015314 |
| Std abs error | 0.011651 |
| Pass (cosine ≥ 0.95) | **YES** ✅ |

## Activation Plane Distribution (uniform X ∈ [-1,1])
- Plane 0 (|x|>2.0): 0 non-zero (none exceed threshold on uniform input)
- Plane 1 (0.5<|x|≤2.0): 16,437 / 32,768 (50.2%)
- Plane 2 (0.1<|x|≤0.5): 13,034 / 32,768 (39.8%)
- Pruned (|x|≤0.1): 3,297 / 32,768 (10.1%)

## Per-Row Cosine (batch 16)
```
row[0]: 0.999561
row[1]: 0.999475
row[2]: 0.999502
row[3]: 0.999492
row[4]: 0.999567
```

## Verdict
**PASS** — One real ffn_up layer from actual GGUF model achieves cosine ≥ 0.95 via PRT_3P offline.