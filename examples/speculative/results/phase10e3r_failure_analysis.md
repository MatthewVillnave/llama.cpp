# Phase 10E-3R: Failure Analysis

## Summary

Phase 10E-3R attempted layer0-only PRT replacement with correct scoping but failed due to execution semantics in the custom op integration.

## Root Cause: Layer State Bleed

### The Bug
During graph building, `g_prt_ffn_up_layer_last = il` is set for each layer's ffn_up build. After the loop processes all 36 layers, the variable holds the LAST layer processed (35).

When the custom op executes during compute, it reads `g_prt_ffn_up_layer_last = 35`. The PRT activation check `if (g_prt_ffn_up_layer_last == 0)` fails, triggering fallback to identity.

### Evidence
```
prt_op_entry #1: layer=35 sidecar=0x750f2485e010 il_prev=35
PRT_OP ENTRY: layer=35 sidecar=0x750f2485e010
PRT_OP: layer=35 fallback=identity (sidecar=0x750f2485e010 layer=0 wanted=35)
```

The custom op fires (we see prt_op_entry), but with layer=35 instead of layer=0. The sidecar IS loaded (layer 0 sidecar), but the layer check fails.

### Why Layer 35?
The prompt eval graph builds all 36 layers. `g_prt_ffn_up_layer_last` gets set to 0, 1, 2, ... 35. After the final layer (35) is built, it holds 35. During compute of the custom op, it reads 35.

During decode, the graph is also built for all layers (Qwen2.5 architecture). Even though only 1 token is processed, the graph includes all layers. So the same bleed occurs.

## What the Graph Structure Shows

```
[DEBUG] mul_mat: ffn_up-0      ← no custom op (uses ffn_up.prt.layer0)
[DEBUG] mul_mat: ffn_up-1      ← regular matmul (no substitution)
...
[DEBUG] mul_mat: ffn_up-35     ← regular matmul (no substitution)
```

Layer 0 uses `ffn_up.prt.layer0` (custom op tensor). Layers 1-35 use regular `ffn_up` matmul nodes. Both show in the debug output as "mul_mat" (ggml_mul_mat underlying op).

But the custom op for layer 0 (`ffn_up.prt.layer0`) fires during compute... but reads the wrong layer value (35).

## Why Not Earlier Detection?

In Phase 10E-2, the custom op was tested in isolation with `prt_ffn_up_smoke_op` which doesn't use `g_prt_ffn_up_layer_last`. The PRT compute function was added later, and the layer check was assumed to work with the global variable.

The issue is that `g_prt_ffn_up_layer_last` is a global single-value, not a per-op-instance context. The custom op has no way to know WHICH layer's ffn_up it is computing during execution — it only has the tensor shapes and the global variable that holds the last-set value.

## Failure Chain

1. Graph building sets `g_prt_ffn_up_layer_last = il` for each layer
2. After building all layers, last value = 35
3. During compute, custom op fires and reads `g_prt_ffn_up_layer_last = 35`
4. Layer check `il == 0` fails → fallback to identity
5. No PRT modification occurs for any layer
6. Output = normal matmul results → random (cosine against PRT = 0.0)

## Possible Fixes

### Fix 1: Per-Tensor Layer Context (Recommended)
Store layer in tensor `extra` field or use a tensor-to-layer mapping:
```cpp
// During graph building, associate layer with tensor
tmp->opaque = (void*)(intptr_t)il;  // embed layer in tensor

// In custom op, retrieve from tensor
int op_layer = (int)(intptr_t)dst->opaque;
```

### Fix 2: Switch to eval callback approach
The eval callback already correctly identifies layer from tensor name. Keep using the callback for PRT, not custom ops. But Phase 10E-2 already tested this.

### Fix 3: Use GGML custom op's userdata parameter
Pass layer as userdata when creating custom op:
```cpp
// In build_ffn: pass layer index as userdata
int* layer_idx = new int(il);
tmp = ggml_map_custom2(ctx0, matmul_result, cur, prt_ffn_up_prt_op, 1, (void*)layer_idx);
```

Then in op:
```cpp
int op_layer = *(int*)userdata;
```

## Conclusion

Phase 10E-3R FAILED due to a layer state management bug in the custom op approach. The scope restriction (layer 0 only) was correctly implemented in both harness and llama-graph.cpp, but the execution layer broke due to global variable bleed.

**Not a custom op math problem. Not a sidecar problem. A state management problem in the integration layer.**