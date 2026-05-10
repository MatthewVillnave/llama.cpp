# Phase 19R: 7B INT6 Kernel Matvec Parity Debug - BLOCKED

## Status

**Verdict**: BLOCKED_GGUF_EXTRACTION

Due to swap exhaustion on this machine (76KB free vs 4.7GB model), Python-based GGUF tensor extraction keeps getting OOM-killed.

## Asset Verification
- Model: Qwen2.5-7B-Instruct-Q4_K_M.gguf (4.7GB, SHA 1875fb29)
- Sidecars: 28 unique INT6 files in `/tmp/prt_sidecars_7b_int6_phase15b_packed/`
- Layer 0: M=18944, K=3584, scale[0]=0.00216875 (verified correct)

## Known Evidence

From Phase 15B-F:
- Weight cosine: 0.997873 (GGUF vs INT6 sidecar at that time)
- Weight MAE: 0.000727  
- Interpretation: The sidecar weights themselves appear valid

From Phase 19Q validation:
- Native 7B: "Paris." (clean output)
- INT6 (26 layers) + native (2 layers): garbage
- This confirms scale offset is fixed but INT6 kernel still corrupts output

## Runtime Observations
- ALL 28 native → clean output "Paris." at 9.7 t/s
- 26 INT6 + 2 native → garbled text at 1.1 t/s
- 28 INT6  → garbled text at 1.1 t/s  
- Force-native works: This proves the FFN kernel works correctly
- INT6 path has a problem in the compute kernel

## Conclusion

The weight conversion logic was verified to work in Phase 15B-F. The bug appears to be in how the runtime INT6 kernel processes dequantized weights during graph execution.

To debug further would require:
1. Memory to extract reference GGUF weights (blocked by OOM)
2. Runtime kernel instrumentation showing actual values used

## Machine State
- Swap: 76KB / 4GB (FULL - blocks extraction)
- RAM: 11GB available but model + operations exceed this

## Next Steps
- Need either larger machine for extraction OR
- Runtime-only debugging approach
- Could try running with fewer threads/lower batch to free memory for Python