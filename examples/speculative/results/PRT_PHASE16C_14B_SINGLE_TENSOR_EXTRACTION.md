# PRT Phase 16C — 14B Single-Tensor FFN_UP Extraction

## Verdict
**PASS_14B_SINGLE_TENSOR_EXTRACTION** ✅

## Context
- Native 14B works at ~4.6 tok/s (confirmed from Phase 16B)
- Full 14B sidecar generation blocked on FFN_UP extraction/dequant
- This phase tests one tensor first

## Model / Metadata
- **Model path:** `/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-14B-14B.gguf`
- **SHA256:** `f80eeedb22b52cfaca9e33551d07322e32ebb8a3fc4176c56c047034e6d9cd1a`
- **File size:** 8.37 GB
- **Layer count:** 40 (`qwen2.block_count = 40`)
- **Hidden size:** 5120 (`qwen2.embedding_length`)
- **FFN intermediate:** 13824 (`qwen2.feed_forward_length`)

## Tensor Extraction Results

### Layer 0 (blk.0.ffn_up.weight)
- **Shape:** {5120, 13824}
- **Type:** q4_K (GGML_TYPE_Q4_K = 12)
- **Elements:** 70,778,880
- **Output:** 283 MB float32
- **Finite values:** ✅ Yes (no NaN/Inf detected)
- **Sample values:** [0.0013, 0.0168, ..., 0.022] — all in normal range

### Layer 1 AND Layer 39
- Extracted successfully with **different checksums** ✅
- Each: 283 MB float32 output

### Key Finding: Q4_K Dequantization WORKS!
The existing llama-prt-ffn-up-extract tool uses:
- `gguf_init_from_file()` to load GGUF
- `ggml_get_tensor()` to get tensor data
- `ggml_type_traits(t->type)->to_float()` to dequantize

This works for 14B Q4_K_M model out of the box — **no custom Q4_K decoder needed**.

## Memory / Performance
- Extraction time per layer: ~5-10 seconds
- Memory usage: ~600 MB (dequantized float data)
- No swap pressure observed

## Interpretation
- **Can 14B FFN_UP extraction/dequant work?** ✅ YES
- **Is Q4_K_M dequant solved through llama.cpp APIs?** ✅ YES (using built-in to_float)
- **One-layer INT6 parity possible?** YES, tool can output float32 → easy to quantize to INT6
- **Is full 14B sidecar generation now justified?** YES - extraction works and is straightforward

## Next Recommended Phase
- **Phase 16D:** All-layer INT6 sidecar generation for 14B (or just few layers for parity test)

## Pass Criteria Met
- ✅ Tensor found (blk.0.ffn_up.weight)
- ✅ Shape makes sense for 14B FFN_UP (5120 × 13824 = 70M elements)
- ✅ Values finite / non-NaN behavior
- ✅ Checksum produced (283 MB output file)
- ✅ No OOM / no swap spiral

## Forbidden Claims
- ❌ Cannot claim "14B PRT works" — only tensor extraction verified
- ❌ Cannot claim "all layers work" — only tested 0, 1, 39
- ❌ Cannot claim "speedup" — no inference test yet
- ❌ Cannot claim "production ready" — need full 40-layer generation first