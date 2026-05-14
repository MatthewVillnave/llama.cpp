# Phase 10E-7 Failure Notes

## Critical Issue: Garbage Output from PRT

### What Happened
The PRT all-layer generation produced garbage output:
```
amup/prt_sidecars/ffn_up_layer35_prt.bin
.Act/prt_sidecars/ffn_up_layer35_prt.bin
 includedsidecars/ffn_up_layer35_prt.bin
```

Baseline output:
```
CPUs,
```

### Root Cause Analysis
1. The output strings contain what appear to be sidecar path fragments
2. This is a BUFFER CORRUPTION issue — the output buffer `buf` (256 bytes) is being partially overwritten with path-like bytes
3. `llama_token_to_piece` writes directly into `buf[256]`. If the PRT output write goes past the intended tensor bounds and corrupts adjacent memory, the `buf` variable on the stack could be affected
4. Alternatively: the PRT `ggml_backend_tensor_set(t, prt_output.data(), ...)` writes to tensor memory, and something in that write is corrupting the buf stack variable

### Why This Is A Real Bug, Not Expected Behavior
- Normal PRT behavior: different text, still coherent
- This: garbage strings that contain path-like fragments
- The fragment "ffn_up_layer35_prt.bin" appears in the output — this is a strong signal of memory corruption

### What's Working
- Model loads correctly
- All 36 sidecars load
- PRT custom op fires for all 36 layers
- 144 PRT replacements, 0 fallbacks
- No crash, clean exit

### What's Broken
- Output quality: complete garbage, not coherent English
- Memory corruption: path strings in output buffer
- Likely cause: PRT tensor write corrupts buf stack variable

### Next Steps
1. Investigate memory layout: does the PRT custom op write corrupt the stack?
2. Check if `prt_output` buffer is properly allocated
3. Examine `ggml_backend_tensor_set` behavior — is it writing to the right memory location?
4. Verify the PRT custom op output buffer doesn't overlap with stack variables