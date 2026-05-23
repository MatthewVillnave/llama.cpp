# Phase 28BB — Real-Slice Decoded Residual Matmul Parity

**Commit:** `cdaf7fbaf` (pre-commit state)
**Branch:** `experimental/prt-phase19a-alt-sidecar-backed`
**Date:** 2026-05-23
**Model source:** `Qwen2.5-0.5B-Instruct/safetensors`
**Tensor:** `model.layers.0.mlp.up_proj.weight` (ffn_up, layer 0)
**Slice shape:** K=32 × M=48 × N=4

---

## Proven

- Decoded `.trit` residual overlay math works on **tiny real model tensor slices** (Qwen2.5-0.5B ffn_up, layer 0)
- Real-slice residual reconstruction matches reference approximation — R_parity is exact (0.00e+00)
- Real-slice overlay matmul matches decoded/reference parity in isolation — Y_parity is exact (0.00e+00) across all 3 test cases
- Canonical `.trit` encoder/decoder path used end-to-end — no bypass

## Test Results

| Case | Description | max_abs_err | RMSE | Pass |
|------|-------------|-------------|------|------|
| 1 | Zero residual (controlled baseline) | 0.00e+00 | 0.00e+00 | ✅ |
| 2 | Deterministic real residual approximation | 0.00e+00 | 0.00e+00 | ✅ |
| 3 | Synthetic W_base + real residual | 0.00e+00 | 0.00e+00 | ✅ |

**R parity:** 0.00e+00 across all cases (exact match)

## Not Proven

- llama.cpp runtime generation
- Full-layer sidecar correctness
- Whole-model quality or inference
- Speedup or throughput improvement
- 30B feasibility
- Production readiness
- Real model inference correctness (this is isolated matmul only)

## Implementation Notes

- **Source tensor:** `model.layers.0.mlp.up_proj.weight` from Qwen2.5-0.5B safetensors (bf16→f32 conversion)
- **Extraction:** Direct slice from safetensors, no GGUF needed for this phase
- **Canonical path:** `prt_trit_io.py write_trit()` → `.trit` file → `prt_trit_io.py read_trit()` → decode
- **Scale:** 1 scale block (block_rows=K=32, block_cols=M=48)
- **No generation run**

## Files Changed

- `examples/speculative/phase28bb_real_slice_matmul.py` — test harness
- `examples/speculative/results/phase28bb_real_slice_decoded_residual_matmul_parity.json` — results

## Claim Boundary

This phase proves that the `.trit` round-trip + matmul works on a real Qwen2.5-0.5B ffn_up tensor slice. It does NOT prove the residual is a good approximation of the real weight delta, that the approximation is useful, or that it integrates into any runtime. Generation must not be attempted.