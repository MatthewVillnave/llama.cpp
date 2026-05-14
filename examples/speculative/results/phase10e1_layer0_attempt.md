# results/phase10e1_layer0_attempt.md
# Phase 10E-1: Layer 0 Replacement Attempt

## Goal
Replace the float ffn_up matmul output for layer 0 only with PRT matmul output.

## What Was Implemented

The graph-level interception was added (see phase10e1_graph_intercept.md), but actual output substitution was NOT implemented in this phase.

### Why Not Implemented

The branch fires correctly, but substituting the output requires answering one key question:

**Can we redirect the graph so downstream ops read PRT(output) instead of float(output)?**

### Feasibility Analysis

The `tmp` tensor returned by `build_lora_mm(up, cur)` is:
- A `ggml_tensor *` with `ne = {11008, batch}` and `backend = w->backend` (GPU/CPU)
- Connected to downstream ops via ggml graph edges
- Used by SiLU activation, gate matmul, and ffn_down

#### Option A: Custom ggml op
Replace `build_lora_mm(up, cur)` with a custom op that computes PRT in a `ggml_map_custom1` wrapper.
**Problem**: Requires defining `GGML_OP_MAP_CUSTOM1` compute function that accesses `up->data` and computes PRT, then returns PRT output. This is complex but possible.

#### Option B: Precomputed tensor + graph surgery
Create a new `ggml_tensor` with precomputed PRT output and replace `tmp` with it.
**Problem**: Cannot easily redirect all downstream edges from `tmp` to the new tensor within `build_ffn()`. Graph edges are already connected to `tmp`.

#### Option C: Backend callback (eval callback, Phase 10E-0)
Intercept after compute, write PRT output to tensor buffer.
**Problem**: This approach was already tried in Phase 10E-0 — callback doesn't fire in our harness.

#### Option D: Larger backend surgery
Modify `ggml_compute_forward_mul_mat` to detect PRT mode and compute differently.
**Problem**: Deep inside ggml, not llama-graph. Requires patching ggml-core.

## What Would Be Needed for Layer 0 Replacement

1. **Intercept at build_ffn**: Currently done ✅
2. **Check layer index**: `if (il == 0 && up)` — currently done ✅
3. **Get input activation**: `cur->data` contains the input activation matrix (M=2048, N=batch)
4. **Get PRT weight**: Load from `/tmp/prt_sidecars/ffn_up_layer0_prt.bin` (2048×11008 float)
5. **Compute PRT**: For each output element Y[i,j], compute `sum(|X[k,j]| > T ? X[k,j] * W[i,k] : 0)`
6. **Insert PRT tensor into graph**: This is the hard part

### The Blocker

The blocker is **Step 6**: inserting the PRT tensor so downstream ops use it instead of the float result.

In ggml, once you call `ggml_mul_mat(ctx, w, cur)`, the result tensor `tmp` has edges pointing TO it from `cur` and `w`. Downstream ops like SiLU reference `tmp` by pointer. Simply overwriting `tmp->data` after compute would work (Option C), but the callback approach failed.

The custom op approach (Option A) is the only viable path that doesn't require post-compute buffer overwriting.

## Current State

| Component | Status |
|-----------|--------|
| Layer 0 detection | ✅ Works (count=2 fires for layer 0) |
| Input activation access | ✅ cur->data accessible |
| PRT weight access | ✅ sidecar loaded |
| PRT computation | ⚠️ Not implemented |
| Output substitution | ❌ Not implemented |

## Next Step for True Replacement

Modify `build_ffn()` to use `ggml_map_custom1` instead of plain `build_lora_mm` for layer 0:

```cpp
if (il == 0 && up) {
    // Use custom op for PRT
    ggml_tensor * prt_result = ggml_map_custom1(ctx0, cur,
        [](struct ggml_tensor * dst, int itask, int nth, void * user_data) {
            // Compute PRT matmul: Y = PRT(X @ W)
            // X = dst->src[0], W = loaded from sidecar
            // Write to dst->data
        }, GGML_UNARY_OP_CUSTOM1, nullptr);
    tmp = prt_result;
} else {
    tmp = build_lora_mm(up, cur);
}
```

This requires defining the `GGML_OP_MAP_CUSTOM1` compute function in ggml, which is a larger patch.