# PRT Phase 16D — 14B One-Layer INT6 Parity Test

## Verdict
**PASS_14B_ONE_LAYER_INT6_PARITY** ✅

## Test Summary
Single-layer INT6 quantization/dequantization of blk.0.ffn_up.weight from Qwen2.5-14B-Instruct-Q4_K_M GGUF.

## Model
- Path: `/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-14B-14B.gguf`
- Shape: {5120, 13824} (K=5120, M=13824)
- Elements: 70,778,880
- Type: Q4_K (dequantized via ggml_type_traits->to_float)

## Metrics

### Finite Values
- **Finite:** ✅ 70,778,880 / 70,778,880
- **NaN:** 0
- **Inf:** 0

### Parity Metrics
| Metric | Value | Pass Gate |
|--------|-------|-----------|
| Weight cosine | **0.999259** | ≥ 0.995 ✅ |
| Matvec cosine (mean) | **0.999274** | ≥ 0.995 ✅ |
| Matvec cosine (min) | **0.997336** | ≥ 0.995 ✅ |
| MAE | 0.000704 | — |
| RMSE | 0.000831 | — |
| Max abs error | 0.004610 | — |
| Norm ratio | 1.000738 | — |

### Sidecar File
- Output: `/tmp/ffn_up_14b_layer0_prt_int6.bin`
- Size: **50.68 MB** (expected 50.68 MB)
- Format: INT6 packed, offset-32, per-row float32 scales
- Header: 20 bytes (PRT6 + ver + rows + cols + reserved)
- Scales: 13824 × 4 = 54.6 KB
- Packed data: 13824 × ((5120+3)//4)*3 = ~50.6 MB

## Interpretation
- Q4_K → float32 → INT6 quantization pathway works cleanly
- All parity metrics exceed pass threshold
- No nonfinite values in dequantized output
- Matvec cosine min of 0.997336 (well above 0.995 gate)
- Sidecar file size is correct and reproducible

## Phase 16D Pass Gates (all met)
- ✅ Min weight cosine ≥ 0.995 (actual: 0.999259)
- ✅ Min matvec cosine ≥ 0.995 (actual: 0.997336)
- ✅ No nonfinite values
- ✅ No shape/orientation mismatch
- ✅ Sidecar file size is sane

## Next Recommended Phase
**Phase 16E: Full 40-Layer INT6 Sidecar Generation for 14B**

All preconditions met. One-layer test confirms the full pipeline works. Proceed to generate INT6 sidecars for all 40 ffn_up.weight layers.

## Forbidden Claims
- Cannot claim "14B PRT inference works" — no runtime tested
- Cannot claim "speedup achieved" — no throughput benchmark
- Cannot claim "production ready" — full generation + runtime validation pending