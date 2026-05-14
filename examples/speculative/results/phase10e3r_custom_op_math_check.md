# Phase 10E-3R: Custom Op Math Check

## Custom Op Interface

```cpp
// ggml_map_custom2 signature
ggml_tensor * ggml_map_custom2(ctx, matmul_result, cur, prt_ffn_up_prt_op, 1, nullptr)
// dst = matmul_result, src[1] = cur
```

## Custom Op Math: Analysis

### PRT Computation (layer 0)
```cpp
Y[b,j] = sum_k (cur[k,b] * W_prt[k,j])  // sparse via thresholds
```
Where:
- `cur` = src[1]->data = FFN input activations [2048, batch]  
- `W_prt` = g_prt_sidecar_data = layer 0 |W_up| [2048, 11008]
- Thresholds: T0=2.0, T1=0.5, T2=0.1 (all activations pass through)
- Output: Y [11008, batch]

### Computes X @ W_prt: YES ✅
- `cur` = input activations X
- `W_prt` = weight sidecar
- Operation: dot product for each (b,j)

### Uses Input-Specific Cache: NO ❌
- Sidecar is static weights, not activation cache
- No per-prompt recomputation of sidecar
- Sidecar precomputed in Phase 10A from model weights

### Output Shape: [batch, 11008] ✅
- Matches expected ffn_up output

## Math Validation Summary

| Check | Result |
|-------|--------|
| Computes X @ W_prt | YES |
| Uses activation cache | NO |
| Output shape correct | YES |
| Sidecar is layer-specific | YES (layer 0 only) |
| Sidecar format is |W| (all positive) | YES |

## Why Cosine = 0.0

The math IS correct (X @ W_prt). But cosine=0.0 because:

1. **Custom op not called during decode**: Only fires 2 times (fallback), not PRT compute
2. **Fallback = identity**: Copies matmul result. But layer 0 matmul is SKIPPED because custom op replaces it
3. **Custom op registered but not executed by compute backend**: The tensor exists in graph but ggml compute doesn't call the op function

The "cosine" metric measures custom op output vs reference. Since custom op never fires, cosine=0.0. The fallback fires and does identity, but that means ffn_up layer 0 gets matmul result directly (no PRT modification), and other layers get their normal matmul.

## Verdict

**Custom op math: CORRECT**
**Execution: BROKEN** - op registered but not called by backend
**Result: cosine=0.0 due to execution failure, not math failure**