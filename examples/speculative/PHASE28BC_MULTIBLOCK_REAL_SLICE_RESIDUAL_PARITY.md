# Phase 28BC — Multi-Block Real-Slice Residual Parity

**Commit:** `e57ba9306` (pre-commit state)
**Branch:** `experimental/prt-phase19a-alt-sidecar-backed`
**Date:** 2026-05-23
**Model source:** `Qwen2.5-0.5B-Instruct/safetensors`
**Tensor:** `model.layers.0.mlp.up_proj.weight` (ffn_up, layer 0)
**Block size:** block_rows=32, block_cols=48

---

## Proven

- Canonical `.trit` block layout works across multiple blocks on real model slices
- Block scale/indexing survives row splits, column splits, row+column splits, and partial edge blocks
- Decoded real-slice residual overlay matmul matches reference approximation across block boundaries — exact parity on all 4 cases
- No bypass path — canonical `prt_trit_io.py` read/write used throughout

## Test Results

| Case | Shape K×M×N | Block Grid | Blocks | R max_err | Y max_err | Y cosine | Pass |
|------|------------|------------|--------|-----------|-----------|----------|------|
| 1 Row-split | 96×48×4 | 3×1 | 3 | 0.00e+00 | 0.00e+00 | 1.00000000 | ✅ |
| 2 Col-split | 32×144×4 | 1×3 | 3 | 0.00e+00 | 0.00e+00 | 1.00000012 | ✅ |
| 3 Row+col split | 96×144×6 | 3×3 | 9 | 0.00e+00 | 0.00e+00 | 1.00000000 | ✅ |
| 4 Awkward edge | 70×101×2 | 3×3 | 9 | 0.00e+00 | 0.00e+00 | 1.00000012 | ✅ |

**All 4/4 pass. R parity exact across all cases. Y parity exact across all cases.**

## Not Proven

- Full-layer sidecar correctness
- llama.cpp runtime integration
- Runtime generation
- Whole-model quality or inference
- Speedup or throughput
- 30B feasibility
- Production readiness

## Implementation Notes

- **Source:** Qwen2.5-0.5B-Instruct safetensors — `model.layers.0.mlp.up_proj.weight`, bf16→f32, padded to 96×144
- **Block size:** block_rows=32, block_cols=48 (canonical 32-pixel block)
- **Block grids tested:** 3×1 (row split), 1×3 (col split), 3×3 (full grid), 3×3 (partial edge at K=70, M=101)
- **Partial edge blocks:** handled correctly — `min(block_rows, K-r)` and `min(block_cols, M-c)` used in all block loops
- **Scales:** per-block mean_nonzero_abs of residual, matching `make_synthetic_scales` approach from prt_trit_io.py
- **No generation run**

## Files Changed

- `examples/speculative/phase28bc_multiblock_real_slice.py` — test harness
- `examples/speculative/results/phase28bc_multiblock_real_slice_residual_parity.json` — results

## Claim Boundary

Multi-block `.trit` layout and block scale indexing work on real Qwen2.5-0.5B ffn_up slices. Does NOT prove full-layer sidecar correctness, runtime integration, model quality, speedup, or production readiness.