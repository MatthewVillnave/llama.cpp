# PRT Phase 24H: Canonical INT8 Layout Checkpoint

## Date
2026-05-20 12:44

## Branch
experimental/prt-phase19a-alt-sidecar-backed

## HEAD
d945f92343ab40cf0637443c89d95e17a4c07d76

## Executive Summary

Canonical INT8 layout is now defined and validated on 3B and 7B layer0 canaries.

- 3B INT8 canonical path works after decoder index fix.
- 7B INT8 sidecar was regenerated in canonical layout and regression is fixed.
- 0.5B operationally passes via INT6 fallback, but explicit 0.5B INT8 canonical validation remains pending.
- This is a layout/policy/semantic sanity checkpoint, NOT a timing or production checkpoint.

## Canonical INT8 Format

scales first:
- M float32 scales

int8 data second:
- K*M int8 values

indexing:
- q[k*M + j]

scale:
- scale[j]

decode:
- W[k,j] = q[k*M+j] * scale[j]

runtime formula:
- Y[j,n] = sum_k X[k,n] * W[k,j]

## Root Cause Fixed

- Python generator wrote int8[k*M + j]
- C decoder previously read int8[k + j*K]
- This mismatch caused garbled 3B output
- Fix changed C decoder to int8[k*M + j]
- 3B output changed from garbled/getChild to Paris
- Old 7B INT8 sidecar used incompatible layout, so it was regenerated canonical

## Validated Results

### 3B
- K=2048, M=11008
- f32 ref: 90,177,536 bytes
- INT8 sidecar: 22,588,416 bytes
- true cosine: 0.9999614671
- runtime output: Paris-like

### 7B  
- K=3584, M=18944
- canonical sidecar regenerated
- size: 67,971,072 bytes
- runtime output: Parisian

### 0.5B
- operational via INT6 fallback
- explicit INT8 canonical NOT proven
- status: PARTIAL

## Allowed Claims

- Canonical INT8 sidecar layout is defined
- 3B layer0 INT8 canonical canary passes
- 7B layer0 INT8 canonical canary passes after sidecar regeneration
- Decoder index mismatch was found and fixed
- Mixed sidecar layouts were identified
- 0.5B remains operational through INT6 fallback

## Forbidden Claims

- **No timing claims** - NOT a timing checkpoint
- **No speedup claims** - performance not validated
- **No production readiness** - research only
- **No multi-layer support** - layer0 only
- **No 14B support** - not tested
- **No broad semantic equivalence** - sanity check only
- **No exact token match** - similar but may vary

## Risks/Caveats

- Old sidecars may use mixed layouts
- Future generators must use canonical format
- Existing 0.5B INT8 may need regeneration
- No timing claims
- Layer0 only
- INT6 path separate

## Recommended Next

1. Phase 24I: Regenerate 0.5B INT8 canonical sidecar and test explicit 0.5B INT8
2. Phase 24J: 3B INT8 repeat validation
3. Phase 24K: Timing smoke only after repeats pass

## Verdict
PASS_CANONICAL_INT8_LAYOUT_CHECKPOINT
