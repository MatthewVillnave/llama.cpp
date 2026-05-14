# PRT_PHASE10A_DIRECT_ONE_LAYER_VERDICT

**Timestamp:** 2026-04-30 12:10 EDT
**Status:** ✅ PASS

## Pass Criteria
| Criterion | Target | Batch 16 | Batch 17 | Status |
|-----------|---------|----------|----------|--------|
| Cosine similarity | ≥ 0.95 | 0.9995 | 0.9995 | ✅ |
| Real ffn_up tensor bytes accessed | YES | YES | YES | ✅ |
| Q4_K dequant path works | YES | YES | YES | ✅ |
| PRT_3P sidecar builds | YES | YES | YES | ✅ |

## What Was Done
1. **Step 3A**: Built in-tree extractor (`tools/prt-ffn-up-extract.cpp`) using `gguf_init_from_file` + `ggml_get_tensor` ✅
2. **Step 3B**: Dequantized Q4_K to float32 via `ggml_get_type_traits->to_float` ✅
3. **Step 3C**: Built PRT_3P sidecar for blk.0.ffn_up, ran offline accuracy test ✅

## Key Files
- Tool: `tools/prt-ffn-up-extract.cpp`
- Float weights: `/tmp/ffn_up_layer0_float.bin`
- PRT sidecar: `/tmp/ffn_up_layer0_prt3p.bin`
- Accuracy JSON: `/tmp/prt_phase10a_batch16_accuracy.json`, `/tmp/prt_phase10a_batch17_accuracy.json`

## Phase 10A: ONE-LAYER COMPLETE ✅

## Next
Phase 10A pass criteria met for one layer. Can proceed to:
- All 28 layers PRT sidecar extraction
- Phase 10B: Real shadow compute in speculative decoder
- Phase 10E: Active mode (requires Phase 10B first)