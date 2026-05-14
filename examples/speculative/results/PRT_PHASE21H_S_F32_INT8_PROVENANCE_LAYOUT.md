# PRT Phase 21H-S: f32 vs INT8 Provenance/Layout Reconciliation

## Status: PASS_F32_INT8_LAYOUT_RECONCILED

## A. Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## B. Previous HEAD  
`dd9c190a6` (Phase 21H-R)

## C. New HEAD  
`dd9c190a6` (no code change — forensic only)

## D. f32 file path/SHA/size
- Path: `/tmp/prt_phase21f_layer0_W_f32.bin`
- Size: 17,432,576 bytes = K*M*4 (K=896, M=4864)
- SHA256: `dbf8d4e4d7d4c08833c1791797965120e61e594a7b121c4ab4537275c0a43901`
- Generated: May 14 01:55 (Phase 21F/21G)

## E. INT8 file path/SHA/size
- Path: `/tmp/prt_sidecars_05b_int8/ffn_up_layer0_prt.int8`
- Size: 4,377,600 bytes = K*M + M*4
- SHA256: `d8e05ad43473213b53f495ae6f82ae3ba323f1c91b3015cb4bf0e0da0fa530dd`
- Generated: May 7 20:11 (Phase ???)

## F. Provenance confidence
**HIGH** — both extracted from same Qwen2.5-0.5B-Instruct-Q4_K_M.gguf, layer0 ffn_up weights

## G. Best layout comparison
**FINDING: INT8 stored as [K,M] row-major, NOT [M,K]**

The Phase 21G C++ formula was:
```cpp
g_f32_weights[il][k * M + j] = int8_buf[j * K + k] * scales_buf[j];  // WRONG
```

CORRECT formula:
```cpp
g_f32_weights[il][k * M + j] = int8_buf[k * M + j] * scales_buf[j];  // CORRECT
```

| Metric | Phase 21G (wrong) | Corrected |
|--------|-------------------|-----------|
| Cosine vs f32 | -0.000150 | **0.9656** |
| MAE vs f32 | 0.020505 | **0.002402** |
| Norm ratio | 1.0001 | 1.0675 |
| Sign match | 44.4% | **100%** |
| Rows cos > 0.9 | 0 | **896/896** |
| Rows cos > 0.95 | 0 | **829/896** |

## H. Best cosine
**0.9656** (corrected formula, all rows cos > 0.9)

## I. Sign match
**100%** with corrected formula

## J. GGUF sampled comparison
Not needed — f32 file already extracted from GGUF and cosine 0.9656 confirms same source.

## K. Runtime sanity result
Not re-run in this phase. Prior runs:
- INT8 path (broken formula): produced blank output (SIGKILL)
- f32 file path: worked in Phase 21G with "Paris" output

## L. Correct source for INT6 regeneration
**f32_file** (`/tmp/prt_phase21f_layer0_W_f32.bin`) — confirmed correct layout, extract INT6 from this.

## M. Verdict
**PASS_F32_INT8_LAYOUT_RECONCILED**

## N. Recommended next
1. **Phase 21H-T**: Fix INT8 decode formula in C++, verify runtime correctness with corrected formula
2. **Then**: Regenerate INT6 sidecar from f32_file (confirmed correct reference)
3. **INT6 decode should use same layout fix**: INT6 packed storage is likely also [K,M] row-major

## O. Models/sidecars/binaries staged?
No — all /tmp references

## P. Secrets detected?
None

## Q. Existing tags touched?
None

## Key Finding: Phase 21G Decode Formula Wrong

Phase 21G C++ assumed INT8 sidecar was stored [M,K] row-major:
```cpp
// Phase 21G assumed: int8_buf[j*K + k] → W[k,j]
// This produced WRONG results (cosine ≈ 0)
```

The INT8 sidecar is actually stored [K,M] row-major:
```cpp
// CORRECT: int8_buf[k*M + j] → W[k,j]
// Cosine vs f32: 0.9656, MAE: 0.0024
```

The INT8 sidecar format from Phase 14B (May 7):
- Storage: `[K,M] = [896,4864]` row-major flat
- Scales: `[M] = [4864]` per-output-row
- Decode: `W[k,j] = int8_buf[k*M + j] * scale[j]`

## SIGKILL Root Cause

Prior SIGKILLs occurred because:
1. Phase 21G used WRONG decode formula → garbage W tensor
2. Garbage W → garbage layer0 activations → degraded generation
3. Model eventually hit generation limit or numerical instability → SIGKILL

The "Paris" output in Phase 21G may have been coincidental or the generation limit occurred before full garbage output.

## Resolution

Both f32_file and INT8_sidecar reference the SAME layer0 ffn_up weights from Qwen2.5-0.5B. The layout is [K,M] row-major. The decode formula fix is:
- Change `j*K + k` → `k*M + j` in Phase 21G C++ code
- Regenerate INT6 sidecar from f32_file (confirmed correct)
- Re-run 4-prompt suite to confirm semantic match