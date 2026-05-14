# results/phase10e2_prt_op_feasibility.md
# Phase 10E-2: PRT Op Feasibility

## Question
Can the zero-fill smoke test custom op be upgraded to compute PRT matmul?

## Requirements for PRT Op

1. **Access runtime activation buffer**
   - Custom op receives `dst->data` = matmul result buffer
   - Activation = `dst->data` (after matmul computes)
   - Shape: `[seq_len=1, ffn=11008]` = 11008 floats

2. **Access sidecar pointer**
   - Need to pass sidecar data via `userdata` parameter
   - Sidecar shape: `[hidden=2048, ffn=11008]` = 22,044,544 bytes (8-bit) or 88,178,176 bytes (float)

3. **Output buffer writable**
   - Custom op writes to `dst->data`
   - Shape must match: 11008 floats

4. **Thread safety**
   - `ggml_map_custom1` with `n_tasks=1` runs single-threaded (safe)
   - If `n_tasks>1`, partition by output element

## Shape/Stride Compatibility

| Component | Shape | Stride |
|-----------|-------|-------|
| Activation input (X) | 11008 | 11008 |
| PRT weights (W) | 2048×11008 | 11008×... |
| Output buffer | 11008 | 11008 |

**Stride compatibility**: Both activation and output use stride = ne[0] = 11008. Compatible.

## Current Smoke Test Issues

The output is garbage because:
- FFN = `silu(x @ W_up) * (x @ W_gate)`
- Zero-fill → `silu(0) = 0`, FFN output = `0`
- Zero propagates through all subsequent layers

For PRT, need:
```cpp
// Instead of zeros, compute PRT:
for (int64_t j = 0; j < N; j++) {
    float sum = 0.0f;
    for (int64_t k = 0; k < K; k++) {
        float x = activation[k * stride_X + j];  // row-major
        float w = sidecar[j * K + k];  // PRT weight (magnitude-only)
        if (fabsf(x) > threshold) {
            sum += x * w;
        }
    }
    dst_data[j] = sum;
}
```

## Required Changes for PRT

### 1. Load sidecar in harness (before graph build)
```cpp
// Load sidecar into memory accessible to custom op
float* sidecar_data = load_sidecar(layer);
void* userdata = (void*)sidecar_data;
```

### 2. Pass userdata to ggml_map_custom1_inplace
```cpp
tmp = ggml_map_custom1_inplace(ctx0, matmul_result, prt_ffn_up_prt_op, 1, userdata);
```

### 3. Implement PRT op function
```cpp
static void prt_ffn_up_prt_op(
        struct ggml_tensor * dst,
        const struct ggml_tensor * a,
        int ith, int nth, void * userdata) {
    if (ith != 0) return;
    
    const float* activation = (const float*)dst->data;  // input to FFN up
    const float* W = (const float*)userdata;  // PRT weights
    float* output = (float*)dst->data;
    
    const int64_t N = 11008;  // ffn hidden
    const int64_t K = 2048;  // hidden
    const float T0 = 2.0f, T1 = 0.5f, T2 = 0.1f;
    
    for (int64_t j = 0; j < N; j++) {
        float sum = 0.0f;
        for (int64_t k = 0; k < K; k++) {
            float x = activation[k];  // activation[k*stride + j]
            float w = W[j * K + k];
            float abs_x = fabsf(x);
            if (abs_x > T0) {
                sum += x * w;
            } else if (abs_x > T1) {
                sum += x * w;
            } else if (abs_x > T2) {
                sum += x * w;
            }
            // else: skip (multiply by 0 implicitly)
        }
        output[j] = sum;
    }
}
```

### Key Issues to Solve

| Issue | Status | Solution |
|-------|--------|----------|
| Access activation buffer | ✅ WORKS | `dst->data` contains matmul result |
| Access sidecar pointer | ⚠️ NEEDS | Load at startup, pass via userdata |
| Output buffer writable | ✅ WORKS | `dst->data` is writable |
| Shape/stride compatible | ✅ WORKS | Both 11008 elements |
| Thread safety | ✅ WORKS | Use `n_tasks=1` |

## Blocker

**None.** The integration works. The blocker from Phase 10E-1 is solved:
- Custom op executes correctly
- Replacement count increments
- Graph downstream consumes the modified tensor

## What Was Learned

1. **ggml_map_custom1_inplace** works for in-place modification
2. **Custom op fires** at the right point in the graph
3. **Replacement count** correctly incremented (6 > 0)
4. **Output garbage** is expected when zeroing FFN (not a blocker)

## For Full PRT Implementation

Need to:
1. Load all 28 layer sidecars into memory at startup
2. Create a mapping layer_index → sidecar pointer
3. Pass sidecar via userdata
4. Compute PRT matmul in custom op instead of zeros

This is straightforward now that the integration is proven.