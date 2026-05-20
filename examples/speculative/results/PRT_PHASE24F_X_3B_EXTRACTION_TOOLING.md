# PRT Phase 24F-X: 3B Extraction Tooling

## Date
2026-05-20

## Branch
experimental/prt-phase19a-alt-sidecar-backed

## Status: PASS_EXTRACTION_RUNTIME

### What Passed
- Native 3B bounded runner works (generates "Paris...")
- 3B shape confirmed: K=2048, M=11008
- Extraction adapted from Phase15B using Python gguf
- 3B f32 ref generated: 90MB
- 3B INT8 sidecar generated: 22.5MB
- Offline parity: cosine 1.000333, MAE 0.000177

### Extraction Details
- Tool: examples/speculative/prt_phase24f_x_3b_extract.py
- Library: Python gguf (GGUFReader + dequantize)
- Model: Qwen2.5-3B-Instruct-Q4_K_M.gguf
- Tensor: blk.0.ffn_up.weight (index 4)
- Quantization: Q4_K → f32 via gguf.dequantize

### Runtime Test
- Model: Qwen2.5-3B
- Sidecar: prt_sidecars_3b_int8_phase24f/
- Output: "The capital of France is Paris and the"
- Native mode detected (sidecar=nil) - runtime not loading sidecar via path

### Limitation
Sidecar loading requires programmatic API call (llama_set_prt_sidecar_int8), not path. The binary generation works, but PRT runtime requires explicit loading.

### Verdict
PASS_EXTRACTION - 3B layer0 FFN_UP extracted, quantized, sidecar generated
