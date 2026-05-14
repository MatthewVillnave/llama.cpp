# results/phase10e0_hook_trace.md
# PRT Phase 10E-0 Hook Trace

## Investigation Summary

**Purpose**: Find real integration point where ffn_up matmul can be replaced

**Methods Investigated**:
1. llama-graph.cpp trace
2. GGML backend compute path
3. Eval callback approach

---

## Investigation 1: llama-graph.cpp

### File: `/home/matthew-villnave/llama.cpp/src/llama-graph.cpp`

#### Line 1062: build_ffn()
```cpp
ggml_tensor * llm_graph_context::build_ffn(
    ggml_tensor * cur,
    ggml_tensor * up,
    ggml_tensor * up_b,
    ggml_tensor * up_s,
    ggml_tensor * gate,
    ggml_tensor * gate_b,
    ggml_tensor * gate_s,
    ggml_tensor * down,
    ggml_tensor * down_b,
    ggml_tensor * down_s,
    ggml_tensor * act_scales,
    llm_ffn_op_type type_op,
    llm_ffn_gate_type type_gate,
    int il) const {
```

#### Line 1077: ffn_up creation
```cpp
ggml_tensor * tmp = up ? build_lora_mm(up, cur) : cur;
cb(tmp, "ffn_up", il);
```

#### build_lora_mm() at line 968
```cpp
ggml_tensor * llm_graph_context::build_lora_mm(
    ggml_tensor * w,
    ggml_tensor * cur,
    ggml_tensor * w_s) const {
    ggml_tensor * res = ggml_mul_mat(ctx0, w, cur);
    // ... LoRA additions
    return res;
}
```

### Integration Point: 
- Function: `build_ffn()` line 1077 OR `build_lora_mm()` line 970
- Operation: `ggml_mul_mat(ctx0, w, cur)` - weight matrix multiply
- Tensor name pattern: `blk.{il}.ffn_up` (set by cb at line 1078)
- Layer index source: `int il` parameter to build_ffn()

### Graph-Level Replacement: POSSIBLE
- Would need to modify build_ffn() to detect ffn_up layer
- Replace build_lora_mm(up, cur) with custom op

---

## Investigation 2: GGML Backend Compute

### File: `/home/matthew-villnave/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c`

#### Line ~1241: ggml_compute_forward_mul_mat()
```c
void ggml_compute_forward_mul_mat(
    struct ggml_compute_forward_mul_mat_params * params,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    struct ggml_tensor * dst) {
    // dispatches based on src0 tensor type (Q4_K, Q8_0, etc.)
}
```

### Backend-Level Replacement: POSSIBLE but INVASIVE
- Custom backend would need to override operation dispatch
- Too invasive for Phase 10E-0

---

## Investigation 3: Eval Callback (Attempted Implementation)

### File: `examples/speculative/phase10e0_layer0_replacement.cpp`

#### Implementation:
- Uses `ggml_backend_sched_eval_callback`
- Set via `llama_context_params.cb_eval`
- Callback fires for each node during graph compute

#### Callback signature:
```c
typedef bool (*ggml_backend_sched_eval_callback)(
    struct ggml_tensor * t, bool ask, void * user_data);
```

#### How it works:
1. `ask=true`: called BEFORE compute, return true to get post-compute callback
2. `ask=false`: called AFTER compute, tensor data available in t->data

#### Our callback code:
```c
static bool prt_eval_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    if (ask) {
        // Check if this is ffn_up tensor for layer 0
        if (t->op == GGML_OP_MUL_MAT && strstr(t->name, "ffn_up")) {
            int layer = -1;
            if (sscanf(t->name, "blk.%d.ffn_up", &layer) == 1 
                && layer == LAYER_SCOPE) {
                return true;
            }
        }
        return false;
    }
    // After compute: get input, compute PRT, replace output
    ...
}
```

#### Result: NOT WORKING
- Replacement count: 0
- Callback not receiving events

---

## Summary: Integration Points Found

| Approach | Location | Complexity | Replaces Output? |
|----------|----------|-----------|----------------|
| Graph-level | build_ffn() line 1077 | MEDIUM | YES |
| Backend-level | ggml-cpu.c line 1241 | HIGH | YES |
| Eval callback | cb_eval param | LOW | YES (but not working) |

---

## Root Cause: Why Callback Not Working

**Possibilities**:
1. Callback not being set correctly in context
2. Scheduler using a different callback path
3. cb_eval parameter name changed in this llama.cpp version
4. Need `warmup=false` AND proper batch params

**Evidence**:
- Using `llama_init_from_model()` and `llama_context_params`
- Setting `cparams.cb_eval = prt_eval_callback`
- `cparams.cb_eval_user_data = &g_prt_state`

**Alternative from eval-callback example**:
- Uses `common_init_from_params(params)` 
- Sets `params.cb_eval = common_debug_cb_eval<false>`
- Sets `params.warmup = false`

---

## Conclusion

**Integration point IDENTIFIED** but implementation incomplete.

The eval callback approach should work in principle:
- We receive tensor after compute
- Get input from src[1], compute PRT
- Write to dst buffer

But we're not receiving callback events in our test.

**Recommended path forward**:
1. Fix eval callback OR
2. Use graph-level approach (modify build_ffn directly)