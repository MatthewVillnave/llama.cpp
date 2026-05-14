# results/phase10e2_custom_op_design.md
# Phase 10E-2: Custom Op Design

## Approach

Use `ggml_map_custom1_inplace` to wrap the `build_lora_mm(up, cur)` result for layer 0.

### Why ggml_map_custom1_inplace

- `ggml_map_custom1` with copy creates new tensor with uninitialized data → garbage in output
- `ggml_map_custom1_inplace` modifies the matmul result tensor in-place
- No memory copy issues - operates on the same buffer as the matmul result
- Simpler for smoke test

### Custom Op Function

```cpp
static void prt_ffn_up_smoke_op(
        struct ggml_tensor * dst,
        const struct ggml_tensor * a,
        int ith, int nth, void * userdata) {
    // Only thread 0 does the work
    if (ith != 0) return;
    
    const int64_t n = (int64_t)ggml_nelements(dst);
    float * dst_data = (float *)dst->data;
    
    // Fill with 0.0f - downstream FFN gets zeroed input
    for (int64_t i = 0; i < n; i++) {
        dst_data[i] = 0.0f;
    }
    
    g_prt_ffn_up_custom_op_count++;
    g_prt_ffn_up_replacement_count++;
}
```

### Integration Point

File: `src/llama-graph.cpp`
Function: `llm_graph_context::build_ffn()`
Line: ~1127

```cpp
if (il == 0) {
    // Phase 10E-2 smoke test: use ggml_map_custom1_inplace
    tmp = ggml_map_custom1_inplace(ctx0, matmul_result, prt_ffn_up_smoke_op, 1, nullptr);
    ggml_set_name(tmp, "blk.0.ffn_up.prt_smoke");
}
```

### Data Flow

1. `build_lora_mm(up, cur)` creates matmul result tensor with buffer allocated from ggml context pool
2. `ggml_map_custom1_inplace` wraps it as a custom op tensor
3. During `ggml_graph_compute`:
   - Matmul computes first → result in buffer
   - Custom op runs → fills buffer with 0.0f
4. Downstream SiLU reads zeroed FFN input → FFN output is zero
5. Model continues with zeroed layer 0 FFN

### Shape Verification

- Input (cur): `[seq_len, hidden=2048]`
- Up projection weight: `[hidden=2048, ffn=11008]`
- Output (tmp): `[seq_len=1, ffn=11008]`
- Elements: 11008 confirmed in log

### Exported Symbol

Added `llama_get_prt_replacement_count()` in `llama.cpp` to expose replacement count to harness:

```cpp
// llama.h
LLAMA_API int llama_get_prt_replacement_count(void);

// llama.cpp
int llama_get_prt_replacement_count(void) {
    extern int g_prt_ffn_up_custom_op_count;
    return g_prt_ffn_up_custom_op_count;
}
```

### Variables (exported for harness access)

```cpp
// llama-graph.cpp - non-static for export
int g_prt_ffn_up_custom_op_count = 0;
int g_prt_ffn_up_replacement_count = 0;
int g_prt_ffn_up_layer_last = -1;
const char * g_prt_ffn_up_name_last = nullptr;
```

### Why Output is Garbage

When layer 0 FFN is zeroed:
- Qwen2 FFN = `silu(x @ W_up) * (x @ W_gate)`
- If `x @ W_up` = 0 → `silu(0) = 0`
- FFN output = `0 * gate_result = 0`
- This zero propagates through all subsequent layers
- Final output is garbage because layer 0 is broken

This is EXPECTED for a zero-fill smoke test. The goal was proving integration works, not quality.