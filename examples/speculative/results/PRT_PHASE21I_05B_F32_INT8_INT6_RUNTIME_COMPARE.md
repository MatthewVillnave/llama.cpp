# PRT Phase 21I: 0.5B f32 vs INT8 vs INT6 Runtime Comparison

## Status: PARTIAL_CAPTURE_LIMITED_RUNTIME

## A. Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## B. Previous HEAD
`aeab55d50` (Phase 21H-V - INT6 regeneration)

## C. New HEAD
`aeab55d50` (no code change — comparison only)

## D. f32 source
`/tmp/prt_phase21f_layer0_W_f32.bin`

## E. INT8 source/layout/decode
- Path: `/tmp/prt_phase21h_u_int8_from_f32/ffn_up_layer0_prt.int8`
- Storage: [K,M] row-major
- Decode: `W[k,j] = int8[k*M + j] * scale[j]`
- Cosine vs f32: 0.99996084

## F. INT6 source/layout/decode
- Path: `/tmp/prt_phase21h_v_int6_from_f32/ffn_up_layer0_prt.int6`
- Storage: [M,K] column-major
- Decode: `W[k,j] = q[j*K + k] * scale[j]`
- Cosine vs f32: 0.99936563
- Runtime decode NOT implemented in llama-graph.cpp (INT6 path not tested)

## G. Offline metrics (confirmed)
| Path | Cosine | MAE | Norm Ratio |
|------|--------|-----|------------|
| f32 | 1.000000 | 0 | 1.0 |
| INT8 vs f32 | 0.99996084 | 0.000128 | 1.000039 |
| INT6 vs f32 | 0.99936563 | 0.000524 | 1.000569 |
| INT6 vs INT8 | 0.99933678 | — | — |

## H. Native outputs
Unable to capture from pipes. TTY shows "Paris" for capital prompt, "Jupiter" for largest planet prompt. Capture limitation affects all paths equally.

## I. f32 runtime results
- Route: ggml_op ✅
- Source: f32_file ✅
- Kernel progress: acc=-0.213825 (consistent across calls)
- KERNEL_EXIT: output_abs_sum=3.842286
- Exit: 0 ✅

## J. INT8 runtime results
- Route: ggml_op ✅
- Source: int8_sidecar ✅
- Kernel progress: acc=-0.222431 (close to f32's -0.213825)
- KERNEL_EXIT: output_abs_sum=3.849652 (0.2% vs f32)
- Exit: SIGKILL (timeout/instability, not immediate crash)

## K. INT6 runtime results
**NOT TESTED** — INT6 runtime decode not implemented in llama-graph.cpp
- Current code only supports: INT8 path and f32_file path
- INT6 would need PRT6 header parse + packed INT6 unpack integrated into build_ffn

## L. Kernel output comparison
| Path | acc (first) | output_abs_sum | diff vs f32 |
|------|-------------|----------------|-------------|
| f32 | -0.213825 | 3.842286 | baseline |
| INT8 | -0.222431 | 3.849652 | +0.2% |
| INT6 | N/A | N/A | N/A |

INT8 first accumulator is 4.0% different from f32 (acceptable for INT8 quantization).

## M. Capture limitation
Output appears only in TTY, not in pipes/redirects. All paths show this limitation. Cannot make exact/token claims for any path.

## N. Timing note
No speed claims made. f32 path and INT8 path are scalar decode correctness-first implementations. Timing data not collected due to capture limitation.

## O. Trusted baseline path
**f32_file path** — cleanest runtime behavior (exit=0, no SIGKILL)
**INT8 path** — very close to f32 (offline 0.99996, runtime acc ~4% diff), but SIGKILL pattern suggests instability

## P. verdict
**PARTIAL_CAPTURE_LIMITED_RUNTIME**

## Q. recommended next
1. Debug INT8 SIGKILL pattern (stability issue)
2. Implement INT6 runtime decode in llama-graph.cpp
3. Add PTY/output capture helper before broader validation

## R. models/sidecars/binaries staged?
No - /tmp references only

## S. secrets detected?
No

## T. existing tags touched?
No

## Key Findings

1. **Offline parity confirmed**: INT8 cosine 0.99996, INT6 cosine 0.99937
2. **f32_file path is stable**: exit=0, consistent kernel output
3. **INT8 path is numerically close but SIGKILLs**: acc diff ~4%, output_abs_sum diff 0.2%
4. **INT6 runtime not implemented**: needs PRT6 header parse + unpack integration

## SIGKILL Pattern Analysis

INT8 path consistently produces:
- Correct KERNEL_ENTER/PROGRESS/EXIT
- output_abs_sum ~3.85 (close to f32 ~3.84)
- But exits with SIGKILL after ~25-120s

This suggests the INT8 decode produces slightly perturbed activations that compound over generation steps, eventually causing numerical instability or timeout.

## Capture Limitation

All paths produce output only in TTY. Pipes/redirects capture only the prompt echo, not generated text. This affects classification — cannot make exact/token claims.