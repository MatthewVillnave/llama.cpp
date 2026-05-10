# PRT Phase 19F: FP16/BF16-Sourced PRT Quality POC

## Status: BLOCKED — FP16/BF16 Source Missing

### Phase 19F-A: Source Location

| Source | Path | Found |
|--------|------|-------|
| GGUF (Q4_K_M) | `/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf` | ✅ Yes |
| FP16/BF16 safetensors | HuggingFace cache only (32B GGUF), no 0.5B FP16 | ❌ No |

### Phase 19F-B: GGUF Metadata

- Model: Qwen2.5-0.5B-Instruct-Q4_K_M
- Architecture: qwen2
- Layers: 24
- Hidden size: 896
- FFN size: 4864
- Attention heads: 14 Q, 2 KV
- Quantization: Q4_K_M
- FFN_UP tensor shape: `[896, 4864]` (hidden × ffn)
- GGUF tensor dtype enum: 6 (Q4_K_M)
- GGUF tensor size: 2,996,224 bytes
- Data offset: 157,166,400

### Phase 19F-C: Source Comparison

Cannot proceed — FP16/BF16 source not available locally.

### Why No FP16/BF16 Source

1. The GGUF model was converted from Qwen2.5-0.5B-Instruct on HuggingFace
2. The original FP16 weights were never stored locally — only the quantized GGUF
3. HuggingFace cache contains only the 32B GGUF variant (from a different download)
4. No Ollama model files for 0.5B available
5. No safetensors files anywhere on the machine for Qwen 0.5B

### What Is Needed (Exact Specification)

```
Model: Qwen/Qwen2.5-0.5B-Instruct
Format: FP16 (float16) safetensors
Files needed:
  - model-00001-of-00001.safetensors (model weights)
  - config.json
  - tokenizer.json
Estimated size: ~1.0 GB (FP16), ~0.5 GB (bf16)
HuggingFace URL: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/main
```

### Phase 19F-D: INT6 Sidecar Generation

Not applicable — blocked at source stage.

### Phase 19F-E: Parity Comparison

Not applicable — blocked at source stage.

### Phase 19F-F: Selected Layers

Not applicable — blocked at source stage.

### Verdict: BLOCKED_FP16_SOURCE_MISSING

### Commit
c1807cf5f (unchanged from Phase 19E)

*Date: 2026-05-10*