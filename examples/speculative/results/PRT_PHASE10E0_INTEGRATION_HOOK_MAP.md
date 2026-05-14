# PRT Phase 10E-0: Integration Hook Trace

## Status: MAYBE — Hook points identified, implementation not complete

**What this phase accomplished:**
- Traced integration points in llama.cpp code
- Identified where ffn_up matmul is created and executed
- Analyzed feasibility of each approach
- Found that true active replacement requires deeper integration than currently wired

---

## 1. Real Integration Point Found: YES

Multiple integration candidates were identified.

## 2. Integration Layer: GRAPH-LEVEL + BACKEND-LEVEL

Two viable integration points found:

### A. llama-graph.cpp build_ffn() — Graph Building Level

**File:** `/home/matthew-villnave/llama.cpp/src/llama-graph.cpp`
**Function:** `llm_graph_context::build_ffn()` (line ~1062)

**ffn_up creation:**
```cpp
ggml_tensor * tmp = up ? build_lora_mm(up, cur) : cur;
cb(tmp, "ffn_up", il);  // named "blk.{il}.ffn_up"

ggml_tensor * res = ggml_mul_mat(ctx0, w, cur);  // w = ffn_up weight
```

**build_lora_mm()** (line ~968):
```cpp
ggml_tensor * res = ggml_mul_mat(ctx0, w, cur);  // w=weight, cur=activation
// LoRA adapters may be added
return res;
```

**Tensor info:**
- Name pattern: `blk.{il}.ffn_up` (layer index `il`)
- Op: `GGML_OP_MUL_MAT`
- src0 = ffn_up weight (Q4_K quantized)
- src1 = input activation ({batch, 2048})
- dst = output ({batch, 11008})

**Approach A1: Modify build_ffn() to use ggml_map_custom3 instead of ggml_mul_mat for ffn_up**
```cpp
// Instead of:
res = ggml_mul_mat(ctx0, w, cur);  // w=ffn_up_weight, cur=input

// Use:
res = ggml_map_custom3(ctx0, cur, w, w_prt, prt_op, n_tasks, &prt_data);
// prt_op computes: output = PRT(input, |W|) where |W| from sidecar
```

**Approach A2: Hook post-compute** — After llama_decode() completes, find the ffn_up output tensor, run PRT, compare, and optionally write back.

**Feasibility:** MEDIUM — requires modifying build_ffn() or adding post-decode hook.

---

### B. ggml-backend.cpp / ggml-cpu.c — Backend Execution Level

**File:** `/home/matthew-villnave/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c`
**Function:** `ggml_compute_forward_mul_mat()` (line ~1241)

**What it does:**
```c
void ggml_compute_forward_mul_mat(...) {
    // dispatches based on src0 tensor type (Q4_K, Q8_0, F16, F32, etc.)
    // calls type-specific kernel (e.g., ggml_compute_forward_mul_mat_q4_0)
}
```

**Hook approach:** 
- Custom ggml backend that wraps the CPU backend
- Intercepts GGML_OP_MUL_MAT ops for tensors matching `blk.{il}.ffn_up`
- Computes PRT output and writes to dst buffer
- Falls back to CPU matmul for all other tensors

**Feasibility:** LOW — requires significant ggml-backend changes

---

## 3. ffn_up Location Details

| Field | Value |
|-------|-------|
| File | `/home/matthew-villnave/llama.cpp/src/llama-graph.cpp` |
| Function | `llm_graph_context::build_ffn()` → `build_lora_mm()` |
| Operation | `ggml_mul_mat(ctx0, w, cur)` where w is ffn_up weight |
| Tensor name pattern | `blk.{il}.ffn_up` |
| Layer index source | `int il` parameter to build_ffn() |
| Weight tensor | `layer.ffn_up` in model, Q4_K quantized |
| Input activation | `cur` tensor from previous layer, float32 |
| Output tensor | dst of ggml_mul_mat, float32 |

---

## 4. Runtime Activations Accessible: YES

After `llama_decode()`:
- Input activations exist as graph tensors in backend buffer
- They are accessible via `tensor->data` pointer
- The challenge is finding WHICH tensor corresponds to ffn_up input at runtime

**To find ffn_up tensors after compute:**
- Use `llama_internal_get_tensor_map()` (line 9312 in llama-model.cpp)
- Or enumerate tensors from the model via `llama_model::get_tensor()`
- Match tensor names containing `blk.{il}.ffn_up`

**Problem:** The ffn_up tensor from model is the WEIGHT (Q4_K), not the output.
The output is a temporary graph tensor created during decode.

---

## 5. Output Replacement Possible: MAYBE

**Challenge:** 
- ffn_up output tensor is created during graph building
- It's a `{batch, 11008}` float tensor in backend memory
- After `llama_decode()` returns, the tensor data is in the buffer
- We could OVERWRITE the output with PRT result, but need to:
  1. Find the tensor (by name or pattern)
  2. Know the batch size
  3. Compute PRT output
  4. Write it back to the buffer
  5. Ensure downstream ops (SiLU, gate matmul) read the modified data

**Key constraint:** In llama.cpp, after `llama_decode()` returns, the logits are ready. If we modified ffn_up output tensor, the decode has already completed for that step. For true replacement, we'd need to intercept BEFORE compute for the NEXT token.

**Most viable approach:** 
1. Use custom op in build_ffn() that computes PRT instead of matmul for ffn_up
2. Or: Run a shadow PRT pass after each decode and compare, without actual replacement

---

## 6. Layer0 Replacement Attempted: NO

Not yet implemented. The code analysis revealed that true active replacement requires:
- Modifying build_ffn() to insert custom op, OR
- Backend-level interception

This was not built in this phase due to complexity.

---

## 7. Generation Smoke: NOT RUN

Cannot run generation smoke without completing the hook implementation.

---

## 8. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Cannot find ffn_up activation tensor at runtime | MEDIUM | HIGH | Use graph node naming convention |
| Modifying graph tensor breaks validation | MEDIUM | HIGH | Use custom op instead of graph surgery |
| ffn_up weight is Q4_K, PRT needs float sidecar | LOW | LOW | Sidecars already built from Phase 10A |
| Backend hook too invasive | MEDIUM | MEDIUM | Use graph-level approach instead |
| Custom op not supported in scheduled backend | LOW | HIGH | Test with simple ggml backend first |

---

## Integration Strategy (Recommended)

**Approach: Post-compute PRT shadow + optional layer-0 replacement**

1. After each `llama_decode()` call:
   - Find all ffn_up output tensors (name pattern `blk.{il}.ffn_up` or similar)
   - Capture input activation tensor
   - Run PRT matmul with sidecar weights
   - Compare PRT vs float output (cosine, error metrics)
   - Log results

2. For true replacement (next phase):
   - Modify `build_ffn()` to use `ggml_map_custom3` instead of `ggml_mul_mat` for ffn_up
   - Pass sidecar data via userdata
   - Custom op computes PRT directly

**Why this is safe:**
- Does not modify model weights
- Does not break graph structure
- Shadow mode just adds PRT computation alongside float (extra overhead)
- No risk of corrupting model state

---

## Required Next Steps for True Active Replacement

1. **Implement custom PRT op** — Use `ggml_map_custom3()` with signature:
   ```cpp
   ggml_tensor * res = ggml_map_custom3(
       ctx,                    // ggml context
       cur,                    // input activation
       ffn_up_weight,          // Q4_K weight (passed but not used)
       W_prt_sidecar,          // float32 |W| sidecar
       prt_matmul_op,          // custom function
       n_threads,              // parallelism
       &prt_data               // {M, N, W_prt_ptr}
   );
   ```

2. **Modify build_ffn()** to detect ffn_up and use custom3 for PRT:
   ```cpp
   if (is_ffn_up && prt_enabled && layer_in_scope) {
       res = ggml_map_custom3(ctx, cur, up, W_prt, prt_op, n_threads, &data);
   } else {
       res = build_lora_mm(up, cur);  // normal path
   }
   ```

3. **Handle fallback** — If PRT produces NaN or cosine < threshold, fall back to build_lora_mm()

4. **Test with generation** — Run short generation with layer 0 only, check for crashes and output quality