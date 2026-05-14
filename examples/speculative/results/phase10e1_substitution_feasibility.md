# results/phase10e1_substitution_feasibility.md
# Phase 10E-1: Output Substitution Feasibility

## Question
Can PRT output be inserted as a ggml tensor that downstream ops consume instead of the float matmul result?

## Feasibility Matrix

| Method | Viable | Complexity | Notes |
|--------|--------|------------|-------|
| A. ggml custom op | YES | HIGH | ggml_map_custom1 wrapper, custom compute fn |
| B. Precomputed tensor + graph surgery | NO | N/A | Cannot redirect graph edges easily |
| C. Backend callback / post-compute write | NO | N/A | Tried in Phase 10E-0, callback doesn't fire |
| D. ggml-backend compute override | MAYBE | VERY HIGH | Modify ggml_compute_forward_mul_mat |

---

## Method A: ggml Custom Op — RECOMMENDED PATH

### How It Works

Replace `build_lora_mm(up, cur)` with:
```cpp
ggml_tensor * tmp = ggml_map_custom1(ctx0, cur, prt_ffn_up_fn, GGML_UNARY_OP_CUSTOM1, prt_data);
```

Where `prt_ffn_up_fn` is a compute function that:
1. Reads `cur->data` (input activation, M×batch)
2. Loads PRT weight from sidecar (M×N magnitude matrix)
3. Computes PRT matmul: Y = PRT(X @ W)
4. Writes result to `dst->data`

### Why It Works
- `ggml_map_custom1` creates a tensor that can appear anywhere in the graph
- The custom compute function runs during `ggml_graph_compute`
- Downstream ops (SiLU, gate, down) automatically consume `dst`
- The graph topology is preserved — only the compute function changes

### Requirements
1. Define `prt_ffn_up_fn` as `ggml_custom1_fn_t`
2. Store PRT weight pointer in user_data
3. Register compute function with ggml (requires ggml-core patch OR runtime registration)
4. Handle quantization: Q4_K dequantization of weight inside compute fn

### Complexity
- GGML-core patch needed to register custom op compute function
- Or: use existing GGML_OP_MAP_CUSTOM1 if already registered

### Verdict: VIABLE — requires ggml-core patch

---

## Method B: Precomputed Tensor + Graph Surgery — NOT VIABLE

### Why Not
In ggml, tensor edges are pointer-based. When `build_ffn()` does:
```cpp
tmp = build_lora_mm(up, cur);  // tmp->src[0]=up, tmp->src[1]=cur
cb(tmp, "ffn_up", il);
cur = ggml_silu(ctx0, tmp);    // cur->src[0]=tmp
```

You cannot redirect `tmp` to a different tensor after creation without breaking the graph.

### Attempted Approaches
1. **Overwrite tmp->data after compute**: Requires post-compute hook (Method C)
2. **Create new tensor and reassign**: `tmp` already connected to downstream ops
3. **Patch ggml_mul_mat directly**: Method D

### Verdict: NOT VIABLE

---

## Method C: Backend Callback — NOT VIABLE (Phase 10E-0 confirmed)

### What Was Tried
- Set `ggml_backend_sched_eval_callback` via `llama_context_params.cb_eval`
- Expected callback to fire after each tensor compute
- Tried to overwrite `dst->data` with PRT result

### Why It Failed
- Callback never fired (replacement count = 0 in Phase 10E-0)
- The `cb_eval` path is not the same as `params.cb` (graph-building callback)
- `cb_eval` requires `ggml_backend_sched_set_eval_callback()` which is not exposed via llama.cpp public API

### Verdict: NOT VIABLE for this harness

---

## Method D: ggml-backend Compute Override — VERY HIGH COMPLEXITY

### How It Works
Modify `ggml_compute_forward_mul_mat()` in ggml-cpu.c to:
1. Detect ffn_up tensor by name
2. Use PRT computation instead of float GEMM

### Why It's Complex
- Deep inside ggml-core (not llama-graph)
- Tensor name not always available at compute time
- Would affect all mul_mat ops, not just ffn_up
- Requires coordination with quantization (Q4_K dequant happens inside mul_mat)

### Verdict: NOT RECOMMENDED for Phase 10E

---

## Recommended Path for Phase 10E

**Use Method A: ggml custom op**

The approach:
1. Register a custom op in ggml (compute function for GGML_OP_MAP_CUSTOM1)
2. In `build_ffn()`, replace `build_lora_mm(up, cur)` with `ggml_map_custom1(ctx0, cur, prt_fn, ...)` for layer 0
3. The custom compute function:
   - Gets input activation from `dst->src[0]->data`
   - Loads PRT weight from sidecar
   - Computes PRT matmul
   - Writes to `dst->data`

This is the ONLY clean way to substitute PRT output without breaking graph topology.

---

## Step 2 Summary

| Question | Answer |
|----------|--------|
| Can PRT output be inserted as ggml tensor? | YES (via custom op) |
| Can graph edges be redirected? | NO (precomputed tensor approach) |
| Can backend callback overwrite buffer? | NO (Phase 10E-0 confirmed) |
| Can ggml-backend be overridden? | YES but VERY HIGH complexity |
| Is substitution implemented? | NO — mapped only |
| Recommended next step | ggml custom op (Method A) |