# Phase 10E-3R: Model Layer Consistency

## 1. Model Layer Count

- **GGUF metadata**: 36 layers (`qwen2.block_count = 36`)
- **Model**: Qwen2.5-3B-Instruct-Q4_K_M.gguf
- **Source**: GGUF V3 metadata dump from phase10e0

## 2. Sidecar Count

- **28 sidecar files** exist in `/tmp/prt_sidecars/`
- **Layers covered**: 0-27 (ffn_up_layer0_prt.bin through ffn_up_layer27_prt.bin)
- **Layers missing**: 28-35 (8 layers, no sidecar files)
- **Coverage**: 28/36 = 77.8%

## 3. Layer-to-Sidecar Map

| Layer | Sidecar | Status |
|-------|---------|--------|
| 0-27 | ffn_up_layer{0-27}_prt.bin | ✅ Exists |
| 28-35 | — | ❌ No sidecar |

## 4. Sidecar Format

- **All 28 sidecars**: 90,177,536 bytes (matches `[2048, 11008] × float32`)
- **Format**: All positive values (absolute value of weight matrix |W|)
- **Magnitude range** (layer 0): min=0.0001, max=0.0743, mean=0.019
- **Construction**: Phase 10A used `fabs(quantized_W)` on dequantized weights

## 5. Phase 10E-3 vs Phase 10E-3R Mismatch

### Phase 10E-3 (FAIL)
- **Scope**: All layers (0-35) got `ggml_map_custom2`
- **Sidecar loaded**: Layer 0 only
- **Replacement count**: 144 (36 layers × 4 eval steps)
- **Problem**: Layers 1-35 used layer 0's sidecar — WRONG WEIGHTS
- **Result**: cosine = 0.0 (garbage)

### Phase 10E-3R (attempted fix)
- **Scope**: Layer 0 only in `llama-graph.cpp`
- **Harness LAYER_SCOPE=0**: Restricts callback to layer 0
- **Problem**: Custom op only called 2 times (during prompt eval, not decode)
- **Decode graph**: Custom op NOT executed during token-by-token decode

## 6. Root Cause

**The llama-graph.cpp custom op fires only during prompt evaluation (prefill), not during per-token decode.**

The decode loop uses a pre-built graph with KV cache. The `ggml_map_custom2` tensor IS in the decode graph, but the compute backend may be computing the matmul directly instead of calling the custom op.

**This is a compute-backend ordering issue**: the custom op is registered in the graph, but when the decode loop runs, the matmul for ffn_up may execute before the custom op is reached in the scheduling order.

## 7. All-Layer PRT Retest: NOT ALLOWED

Using layer 0 sidecar for all 36 layers is weight mismatch. All-layer replacement requires all 36 sidecars. Only 28 exist.

**All-layer PRT retest requires**: 36 sidecars for all layers. Current state: only 28 available.