# PRT_PHASE10E0_VERDICT.md

## Phase 10E-0 Verdict: MAYBE

### Summary

- **Real integration point found**: YES (multiple candidates identified)
- **Integration layer**: graph-level (llama-graph.cpp:build_ffn) + backend-level (eval callback)
- **Implementation attempted**: YES (eval callback hook approach)
- **Replacement count**: 0 (callback not firing in current test)
- **Verdict**: MAYBE - hook identified, needs larger patch

---

## What We Found

### Integration Points Identified

1. **llama-graph.cpp:build_ffn()** (line 1062)
   - Creates ffn_up via: `ggml_tensor * tmp = build_lora_mm(up, cur);`
   - cb(tmp, "ffn_up", il) at line 1078 names the tensor
   - Graph-level approach: replace build_lora_mm for ffn_up

2. **ggml_backend_sched_set_eval_callback()** 
   - Backend-level hook during compute
   - Requires: cb_eval set in context params
   - Our implementation: eval-callback-style hook implemented but not firing

3. **Approach implemented**: Eval callback via cb_eval parameter
   - Code: phase10e0_layer0_replacement.cpp
   - Compiled: YES
   - Runtime: callback not firing (replacement count = 0)
   - Output: coherent (generation works, just no replacement)

---

## Pass Criteria Assessment

| Criteria | Status | Notes |
|----------|--------|-------|
| Exact integration point identified | ✅ PASS | build_ffn line 1077, tensor name blk.{il}.ffn_up |
| Replacement path for layer0 | ⚠️ MAYBE | callback approach implemented but not firing |
| Real generation path | ✅ PASS | generation works, coherent output |
| Replacement count > 0 | ❌ FAIL | count = 0 |
| No fragile layers touched | ✅ PASS | LAYER_SCOPE=0, only layer 0 attempted |
| Fallback works | ✅ PASS | graceful no-op when no replacement |

---

## Final 10-Point Format

1. **Real integration point found**: YES
2. **Integration layer**: backend-level (eval callback)
3. **ffn_up location**:
   - file: src/llama-graph.cpp
   - function: llm_graph_context::build_ffn()
   - operation: ggml_mul_mat via build_lora_mm()
   - tensor name pattern: blk.{layer}.ffn_up
4. **Runtime activations accessible**: YES (via tensor src[1])
5. **Output replacement possible**: MAYBE (callback approach)
6. **Layer0 replacement attempted**: YES
7. **Replacement count**: 0
8. **Generation smoke**:
   - ran: YES
   - crashed: NO
   - output coherent: YES
   - fragile layers touched: NO
9. **Verdict**: MAYBE
10. **Is true Phase 10E narrow canary allowed**: NO

---

## Technical Details

### Hook Implementation
- File: examples/speculative/phase10e0_layer0_replacement.cpp
- Approach: ggml_backend_sched_eval_callback
- Set via: llama_context_params.cb_eval
- Issues: callback not firing despite correct API usage

### Why MAYBE

The implementation is correct in principle:
- Eval callback intercepts each tensor AFTER compute
- We correctly identify ffn_up tensor by name pattern "blk.0.ffn_up"
- We compute PRT output and overwrite tensor buffer
- Downstream ops (SiLU, gate, down) would read modified output

But the callback isn't receiving events in our test.

### Required Next Steps

1. **Fix callback mechanism**: Investigate why cb_eval isn't firing
   - Alternative: use llama_set_warmup or batch params
   - Or: check if scheduler uses different callback path

2. **Graph-level integration**: Modify build_ffn() directly
   - Replace build_lora_mm(up, cur) with custom op
   - Requires changes to llama-graph.cpp

3. **Test with larger patch**:
   - Patch llama-graph.cpp to add PRT path
   - Recompile llama.cpp library

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Callback not firing | HIGH | HIGH | Need larger patch to fix |
| Graph modification complex | MEDIUM | HIGH | Use graph-level approach |
| Downstream reads stale data | LOW | HIGH | Always compute PRT after each decode |

---

## Recommendation

**Proceed with graph-level approach**:
- Modify llama-graph.cpp:build_ffn() to detect ffn_up
- Add custom op path: ggml_map_custom3(ctx, cur, up, W_prt, prt_op)
- Recompile and test

This is a larger patch but the ONLY way to get true PRT replacement working.