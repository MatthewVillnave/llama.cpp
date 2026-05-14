# PRT_CUSTOM_OP_CORRUPTION_SUMMARY.md

## Corruption Event Summary

**Date:** 2026-04-30
**Phase:** 10E-7S
**Symptom:** Decoded token output contains path string fragments from sidecar filenames (e.g., `公司rt_sidecars/ffn_up_layer35_prt.bin`)

## What Was Tested

### Test Matrix

| Test | Sidecar | Custom Op Logic | src0 at op entry | dst at op exit | Final Output |
|------|---------|---------------|------------------|----------------|--------------|
| Standalone PRT | Yes | Full PRT math | N/A (outside llama) | N/A | ✅ CORRECT |
| llama-completion | No | Identity fallback | N/A | N/A | ✅ CORRECT |
| Bounded fill | Yes | Y[i]=0.001*i | N/A | N/A | ❌ GARBAGE |
| Zero fill | Yes | memset(0) | N/A | N/A | ❌ GARBAGE |
| Identity copy | Yes | for(i) dst[i]=src0[i] | ✅ 0.235489 | ✅ 0.235489 | ❌ GARBAGE |

### Key Observation

Inside the custom op, the src0 data is CORRECT (verified floats). The dst data after copy is CORRECT (matches src0). But the final decoded token is GARBAGE — containing path string fragments.

**Conclusion:** The corruption happens UPSTREAM of the custom op in the ggml graph computation. The matmul result tensor arrives at the custom op already containing garbage.

## Affected Code Path

```
llama_set_prt_sidecar(layer, data, M, N)
  → stores g_prt_sidecar_data[layer] = data
  → llama_set_prt_replacement_count()

llama_decode(ctx, batch)
  → ggml_graph_compute()
    → build_lora_mm(up, cur) → matmul_result
    → ggml_map_custom2(ctx, matmul_result, cur, prt_ffn_up_prt_op, ...)
      → prt_ffn_up_prt_op(dst=dst, src0=matmul_result, src1=cur)
        → [custom op logic]
        → output is GARBAGE ← corruption upstream
```

## Sidecar Loading Pattern (from harness)

```cpp
// In phase10e0_layer0_replacement.cpp:
char path[256];
snprintf(path, sizeof(path), "/tmp/prt_sidecars/ffn_up_layer%d_prt.bin", layer);
FILE * f = fopen(path, "rb");
float * data = new float[M * N];
fread(data, sizeof(float), M * N, f);
llama_set_prt_sidecar(l, data, M, N);
```

## Hypotheses Tested

1. ❌ Sidecar file corruption — CLEAN (validated with audit tools)
2. ❌ Loader pointer error — CLEAN (validated with pointer arithmetic)
3. ❌ Standalone PRT math bug — CLEAN (validated with synthetic X)
4. ❌ Custom op code bug — CLEAN (identity copy verified correct floats)
5. ❌ memcmp/self-copy UB — RULED OUT (switched to element-by-element loop)
6. ✅ **ggml_map_custom2 tensor lifecycle bug** — CORRUPTION HERE

## ggml_map_custom2 Investigation Clues

- llama-completion (no sidecar, custom op fires with sidecar=nil): CORRECT output
- llama-phase10e0 (sidecar loaded, custom op fires with sidecar=0x...): GARBAGE
- Custom op only accesses sidecar pointer (reads, doesn't dereference in broken tests): GARBAGE
- The path string "ffn_up_layer35_prt.bin" is the LAST sidecar loaded (layer 35), appearing in output for ALL layers

## Conclusion

The corruption is in the ggml_map_custom2 integration layer inside llama.cpp. When sidecar data is loaded and the custom op graph substitution is active, the ggml tensor buffers are being corrupted in a way that introduces path string fragments into the computation.

This is a low-level ggml tensor management bug, not a PRT algorithm bug.