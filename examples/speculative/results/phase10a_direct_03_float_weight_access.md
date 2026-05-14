# Phase 10A — Step 3B: Float Weight Access

**Timestamp:** 2026-04-30 12:05 EDT
**Status:** PASS ✅

## Method
- **ggml API: `ggml_get_type_traits(type)->to_float`** — official llama.cpp dequant path

## Tensor Details
| Field | Value |
|-------|-------|
| name | blk.0.ffn_up.weight |
| shape | {2048, 11008} |
| type | q4_K (GGML_TYPE_Q4_K = 12) |
| elements | 22,544,384 |
| bytes | 12,681,216 (quantized) → 90,177,536 (float) |

## Float Access
| Field | Value |
|-------|-------|
| Success | YES ✅ |
| Method | `ggml_get_type_traits(GGML_TYPE_Q4_K)->to_float(q4_block_ptr, float_out, n_elements)` |
| Output file | `/tmp/ffn_up_layer0_float.bin` |
| Float bytes | 90,177,536 |
| CRC32 | 0xFFFFFFFF (partial) |

## Stats
| Metric | Value |
|--------|-------|
| min | -0.328526 |
| max | 0.360853 |
| mean | -0.000016 |
| avg_abs | 0.017720 |

## Sample Values (first 20)
```
[0] = 0.023056, [1] = 0.023056, [2] = -0.003680, [3] = 0.016372
[4] = -0.003680, [5] = -0.023733, [6] = -0.023733, [7] = 0.023056
[8] = -0.023733, [9] = 0.049792, [10] = -0.023733, [11] = 0.016372
[12] = 0.029740, [13] = 0.016372, [14] = -0.003680, [15] = 0.016372
[16] = 0.009688, [17] = -0.003680, [18] = 0.003004, [19] = -0.010365
```

## Verdict
**PASS** — blk.0.ffn_up.weight is now accessible as float values via official ggml API.

## Next
Step 3C: Build PRT_3P sidecar, run offline accuracy test.