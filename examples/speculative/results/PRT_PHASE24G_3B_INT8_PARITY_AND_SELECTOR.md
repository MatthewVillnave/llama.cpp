# PRT Phase 24G-R5: 3B INT8 Output Layout Analysis

## Date
2026-05-20 04:xx

## Branch
experimental/prt-phase19a-alt-sidecar-backed

## HEAD
b572bbfaac56a590c1864b6e3cec6d303ecac08b

## Status: DECODE INDEX BUG FOUND

### What Works
- 3B K/M selector (K=2048, M=11008) ✅
- INT8 scales read order fixed ✅  
- Sidecar loads (g_f32_weights non-null) ✅
- Custom op inserted (GGML_OP_PRT_FFN_UP) ✅
- Scalar kernel executes ✅
- Produces valid logits (not NaN/Inf) ✅

### What Doesn't Work
- Output: "getChild" vs expected "Paris"
- **ROOT CAUSE: Decode formula index mismatch**

### Analysis Results

| Metric | Value |
|--------|-------|
| GGUF tensor shape | [M,K]=[11008,2048] |
| F32 ref shape | [K,M]=[2048,11008] |
| Cosine(GGUF.T, f32_ref) | 1.0000 ✅ |
| Cosine(decoded, f32_ref) | **-0.412** ❌ |

### Root Cause

**Python extractor writes:**
```python
# f32 is [K,M] after transpose
int8_data = np.round(f32 / scales * 127)
# Flat row-major: int8_data[k*M + j]
```

**Runtime C decoder reads (line 1372):**
```c
g_f32_weights[il][k + j * K] = w_val * scales_buf[j];
// Index: k + j*K = k + j*2048
```

**MISMATCH:**
- Python writes index for (k=1,j=0): 1×11008 + 0 = **11008**
- C reads index for (k=1,j=0): 1 + 0×2048 = **1**
- TOTAL MISMATCH when k,j > 0

### Fix Required

Change line 1372 in `src/llama-graph.cpp`:
```c
// FROM:
g_f32_weights[il][k + j * K] = w_val * scales_buf[j];
// TO:
g_f32_weights[il][k * M + j] = w_val * scales_buf[j];
```

### Verification After Fix

Expected:
- Cosine(decoded, f32_ref) > 0.99
- Runtime output "Paris" instead of "getChild"

## Verdict
FAIL_3B_INT8_DECODE_INDEX_MISMATCH

## Recommended Next
Fix line 1372 in llama-graph.cpp, regenerate INT8 sidecar, rerun canary
