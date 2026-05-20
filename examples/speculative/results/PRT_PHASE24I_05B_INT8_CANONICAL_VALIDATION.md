# PRT Phase 24I: 0.5B INT8 Canonical Validation

## Date
2026-05-20 12:52

## Branch
experimental/prt-phase19a-alt-sidecar-backed

## HEAD
(pending commit)

## Summary

0.5B selector path was added to point to existing 0.5B INT8 sidecar.

However, the existing 0.5B INT8 sidecar was generated in Phase 22E with different normalization:
- Existing sidecar uses normalized weights (range -1 to 1)
- Not raw GGUF dequantized weights
- This leads to garbled output when decoded with canonical formula

Selector works correctly:
- K=896 M=4864 detected
- Correct sidecar path selected

Runtime output: garbled (similar to 3B before fix)

## What Works
- 3B INT8 canonical path: Paris ✅
- 7B INT8 canonical path: Parisian ✅  
- 0.5B selector added and working (path selected correctly)

## What Doesn't Work
- 0.5B runtime output is garbled because the sidecar was generated with different normalization

## Root Cause
The existing 0.5B INT8 sidecar (from Phase 22E) was generated with normalized weights, not raw GGUF dequantized weights.

## Verdict
PARTIAL_05B_INT8 - selector works, sidecar needs regeneration

## Recommended Next
Regenerate 0.5B INT8 canonical sidecar using same method as 3B/7B:
1. Use gguf.GGUFReader to get layer0 ffn_up tensor
2. Dequantize using correct type 
3. Transpose to [K,M]
4. Compute scales per column
5. Quantize to INT8
6. Save in canonical format
