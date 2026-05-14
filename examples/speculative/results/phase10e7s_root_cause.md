# Phase 10E-7S Root Cause: ggml Custom Op Integration Bug

## Summary

PRT custom op integration produces garbage output when sidecars are loaded, even when the custom op does a simple identity copy (src0→dst).

## Root Cause: PARTIAL — IN GGML CUSTOM OP INTEGRATION

### Evidence

1. **Sidecar file audit**: CLEAN ✅ — no path strings, correct size, finite values
2. **Loader pointer audit**: CLEAN ✅ — pointers within allocated range
3. **Standalone PRT compute**: CLEAN ✅ — produces correct output outside llama.cpp
4. **Custom op identity copy**: CORRECT ✅ — src0==dst (verified floats match inside op)
5. **Final output**: CORRUPT ❌ — path string fragments in decoded token

### Key Finding

Inside the custom op, BEFORE copy: `src0[0]=0.235489` (CORRECT)
After copy: `dst[0]=0.235489` (CORRECT, matches src0)

But final decoded token shows: `'公司rt_sidecars/ffn_up_layer35_prt.bin'`

**Conclusion**: The corruption happens in the ggml graph computation BEFORE the custom op is called. The src0 tensor (matmul result) arrives at the custom op with already-corrupted data. This is NOT a bug in our custom op code — it's upstream in how ggml computes the graph when sidecar is loaded.

## Hypothesis

When `llama_set_prt_sidecar` stores the sidecar pointer, something in the ggml tensor allocation or computation graph is being corrupted. The path string "ffn_up_layer35_prt.bin" appears to be leaking from the sidecar loading/storage mechanism into the ggml tensor buffer used for matmul results.

## Comparison

- llama-completion (sidecar=nil): CORRECT output ✅
- llama-phase10e0 (sidecar=0x... loaded): GARBAGE output ❌

The difference is that sidecar data is loaded. Something about loaded sidecar + ggml custom op is incompatible.

## Next Steps

1. Investigate ggml_map_custom2 tensor lifecycle
2. Check if tensor buffer is being reused/overwritten
3. Examine if userdata pointer is interfering