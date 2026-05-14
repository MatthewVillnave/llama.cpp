# Phase 10A — Step 3B: Dequant One Layer

**Timestamp:** 2026-04-30 11:40 EDT
**Status:** INCONCLUSIVE

## Tensor Details
| Field | Value |
|-------|-------|
| name | blk.0.ffn_up.weight |
| shape | {2048, 11008} |
| type | q4_K (GGML_TYPE_Q4_K = 12) |
| offset | 286435328 |
| bytes | 12681216 |
| elements | 22544384 |

## Extraction Status
- In-tree extractor: ✅ `tools/prt-ffn-up-extract.cpp` builds and runs
- Tensor bytes extracted: ✅ `/tmp/ffn_up_layer0_f32.bin`
- First bytes verified: `f406 cd13 bffa feff...` (matches expected q4_K header)

## Dequantization Status
| Approach | Status |
|---------|--------|
| Runtime backend (ggml_mul_mat) | Available ✅ — happens at compute time |
| Offline via Python | BLOCKED — gguf-py offset mismatch, manual q4_K buggy |
| Offline via C++ | NOT IMPLEMENTED — needs compute graph |

## Next Blocker
- **Manual Q4_K dequant required for offline PRT**
- ggml doesn't expose standalone dequant functions
- gguf-py has offset calculation bug with GGUF v3
- Time spent on Python q4_K dequant = 30+ min with no working solution

## Verdict
- Tensor access: **PASS** (byte-level extraction works)
- Dequant path: **INCONCLUSIVE** (backend available at runtime, not offline)

## Recommendation
- Option 1: Accept runtime-only path (no offline PRT sidecar)
- Option 2: Use float16 reference weights from model conversion
- Option 3: Skip sidecar, use PRT at verification time

This is as close as I can get with the current constraints.