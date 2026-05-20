# PRT Phase 24G-R4: 3B INT8 Output Quality Debug

## Date
2026-05-20 03:xx

## Branch
experimental/prt-phase19a-alt-sidecar-backed

## Status: PARTIAL - kernel runs but output wrong

### What Works
- 3B K/M selector (K=2048, M=11008) ✅
- INT8 scales read order fixed ✅  
- Sidecar loads (g_f32_weights non-null) ✅
- Custom op inserted (GGML_OP_PRT_FFN_UP) ✅
- Scalar kernel executes ✅
- Produces valid logits (not NaN/Inf) ✅
- PRT path runs instead of native ✅

### What Doesn't Work
- Output: "getChild" vs expected "Paris"
- Output completely wrong

### Debug Results

Scalar kernel produces valid logits:
```
[PRT_V2_NUMERIC] backend=scalar abs4=168.259842 y0=-12.215832 y1=5.465144 y2=-48.803802 y3=9.317798
```

These ARE logits - they're not random. But they're DIFFERENT from native path.

### Analysis

Likely root cause: Weight matmul formula mismatch
- Kernel formula: Y[j,n] = sum_k X[k,n] * W[k,j]
- But weight storage: might be W[j,k] (transposed)

The decode formula at line 1372 applies scales during load - so decoded weights should already be correct.
Formula mismatch between kernel compute order and weight storage order in W tensor.

### Without Fix (Commit e9da22054)
- Native: "Paris is located" ✅
- PRT: "getChild" ❌

## Verdict
FAIL_OUTPUT_MISMATCH - Kernel formula mismatch

## Recommended Next
Debug weight storage layout in decoded f32 sidecar:
- Check if [K,M] or [M,K] 
- Compare against model original layout
- Fix decode formula to match kernel expected layout

