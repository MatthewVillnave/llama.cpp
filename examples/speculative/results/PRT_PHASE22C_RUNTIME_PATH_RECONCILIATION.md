# PRT Phase 22C: Runtime Path Reconciliation

## Branch
experimental/prt-phase19a-alt-sidecar-backed

## Previous HEAD
7f83ca5ba37fec11fb9d685394c6b29338ef0e69 (PRT Phase 22A)

## Reconciliation Method
Modified `build_prt_ffn_up()` in `prt_graph_replace.h` to:
1. For f32 sidecar (format==0): use `ggml_prt_ffn_up(...)` → ops.cpp kernel
2. For INT8/INT6: keep inline custom op path
3. Added route/path logs: `[PRT_V2_PATH] mode=ggml_native_op` / `mode=inline_fallback`

## Key Fixes Applied
- Use `cur->ne[0]` for hidden dimension (not swapped sidecar metadata)
- Use `g_prt_sidecar_M` for ffn dimension (NOT swapped!)
- Dimension formula for f32: hidden = cur->ne[0], ffn = g_prt_sidecar_M

## Dimension Analysis
- f32 file: 17,432,576 bytes = 896*4864*4 bytes = 17,432,576
- Loader reads as M=896, N=4864 → sets g_prt_sidecar_M=896, g_prt_sidecar_N=4864
- Actually file stores K=896 (hidden), M=4864 (ffn) per Phase 21F loader
- **Fix:** hidden = cur->ne[0] = 896, ffn = g_prt_sidecar_M = 4864

## Inline Path Location
examples/speculative/prt_graph_replace.h:493-559 (build_prt_ffn_up function)

## GGML Native Path Location
ggml/src/ggml.c:3328 (ggml_prt_ffn_up constructor)
ggml/src/ggml-cpu/ops.cpp:10836 (ops.cpp kernel)

## Scalar ggml-op Runtime Reached?
NOT YET TESTED (model unavailable for test run)

## AVX2 ggml-op Runtime Reached?  
NOT YET TESTED (model unavailable for test run)

## Inline Fallback Status
PRESERVED for INT8/INT6 format (format != 0)

## Verdict
PARTIAL_GGML_OP_PATH_WIRED - code implemented but runtime not yet proven

## Recommended Next
Phase 22D - test 0.5B with float32 sidecar to verify ops.cpp kernel fires

## Limitations
- Model files not accessible for runtime test in this session
- f32 sidecar must be loaded at runtime for path to activate only

## Files Changed
- examples/speculative/prt_graph_replace.h (added f32 path with ggml_prt_ffn_up, added path logs)
- ggml/src/ggml-cpu/ops.cpp (unchanged from Phase 22A - has AVX2 kernel)

## Build Status
✅ Compiled successfully
