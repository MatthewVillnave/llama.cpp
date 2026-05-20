# PRT Phase 24G-R6: Regression Test Results

## Date
2026-05-20 12:xx

## Branch
experimental/prt-phase19a-alt-sidecar-backed

## HEAD
676a738b0f65e4b036afaec9d2f6dec71dbef808

## Summary

| Model | Result | Output | Notes |
|-------|--------|--------|-------|
| **3B** | ✅ PASS | "Paris" | Fixed decoder + regenerated sidecar |
| **0.5B** | ✅ PASS | "Paris" | Uses INT6 fallback |
| **7B** | ❌ FAIL | "ParisG" | INT8 sidecar has wrong layout |

## Details

### 3B INT8 (FIXED)
- Decoder fix applied: `k + j*K` → `k*M + j`
- Sidecar regenerated with correct write order
- Output: "Paris" ✅ matches native

### 0.5B INT8 (REGRESSION PASS)
- 0.5B auto-selected INT6 path (file size mismatch for INT8)
- Output: "Paris" ✅ 
- No regression - existing path still works

### 7B INT8 (REGRESSION FAIL)
- 7B sidecar was generated with older layout
- Scales corrupted: `0.0/0.0/garbage/0.0/0.0`
- Output: "ParisG" ❌ (garbled)
- Sidecar format differs from 3B

## Root Cause
Mixed sidecar layouts:
- 3B (new): writes INT8 as [K,M] row-major → works with fixed decoder
- 7B (old): writes INT8 in different order → fails with fixed decoder

## Verdict
- PASS_3B_INT8_POLICY_CANARY ✅
- PASS_INT8_INDEX_FIX_05B_REGRESSION ✅  
- FAIL_INT8_INDEX_FIX_7B_REGRESSION ❌
- FAIL_MIXED_SIDECAR_LAYOUTS

## Recommended Next
1. Regenerate 7B INT8 sidecar using same method as 3B fix
2. OR add layout-version detection in decoder
3. Test regenerated 7B sidecar
