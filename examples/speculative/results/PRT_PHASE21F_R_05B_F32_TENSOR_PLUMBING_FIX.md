# PRT Phase 21F-R — 0.5B F32 Tensor Plumbing Fix Report

## Summary
**Branch:** `experimental/prt-phase19a-alt-sidecar-backed`  
**Previous HEAD:** `30ef551f2`  
**New HEAD:** `d90e2c265`  
**Status:** ASSERTION_FIXED — Additional Graph Execution Issue Remains

---

## Phase 21F-R Results

| Check | Status |
|-------|--------|
| Assertion fixed (up type) | ✅ Removed wrong GGML_ASSERT |
| M dimension fixed | ✅ 4864 instead of 896 |
| F32 file loads | ✅ /tmp/prt_phase21f_layer0_W_f32.bin (17MB) |
| Per-layer weight loading | ✅ Fixed with array |
| Tensor creates | ✅ ne[0]=896, ne[1]=4864, type=F32 |
| Native baseline runs | ✅ Clean completion |
| PRT route triggers | ✅ [PRT_V2_ROUTE] layer 0 |
| PRT-v2 generates | ⚠️ Hangs/crashes during generation |

### Logs (Working Phase)
```
[PRT_V2_AUTO] enabled via PRT_GGML_TEST_LAYER=0
[PRT_V2_ROUTE] IL=0 route=ggml_op reason=selected_layer
[PRT_V2_CONFIG] ggml_op_test=1 layer=0
[PRT_V2_SHAPE] IL=0 K=896 M=4864 n_tokens=1 (M from model config)
[PRT_V2_TENSOR] native_up_type=6 (model weight, not used as W)
[PRT_V2_TENSOR] IL=0 loaded=1 path=/tmp/prt_phase21f_layer0_W_f32.bin bytes=17432576 K=896 M=4864
[PRT_V2_TENSOR] source=f32_file layer=0
[DEBUG] before ggml_new_tensor K=896 M=4864
[DEBUG] after new_tensor
```

---

## Root Cause Analysis

1. **Assertion fix:** Removed `GGML_ASSERT(up->type == GGML_TYPE_F32)` because up is Q4 quantized weight, not F32 activation

2. **M dimension:** Changed from `up->ne[0]` (incorrect 896) to hardcoded 4864 (Qwen2 0.5B FFN dimension)

3. **Weight loading:** Fixed file size mismatch by using correct K and M, plus per-layer flag array

4. **Remaining issue:** After PRT tensor is created, the graph compute stage hangs/crashes. Likely:
   - Tensor is created but not properly wired into the GGML graph
   - The context allocation may not match expected for the computation
   - Kernel might need explicit backend registration

---

## Verdict: ASSERTION_FIXED

| Good | Issues |
|------|--------|
| F32 file loads | Graph execution hangs |
| Tensor created | Timing out during compute |
| Log shows working path | Root cause in compute stage |

---

## Next (Phase 21F-R-R)
Debug why graph compute hangs after PRT tensor is inserted:
- Check if tensor needs different allocation context
- Try using native `ggml_mul_mat` as fallback instead of GGML_OP_PRT_FFN_UP
- Verify backend registration for the custom op

---

## Files Changed
- src/llama-graph.cpp — assertion fix + dimension fix + per-layer loading + debug traces

---

## Safety Scan
- Models staged: NO ✅
- Sidecars staged: NO ✅  
- Secrets detected: NO ✅
- Tags touched: NO ✅