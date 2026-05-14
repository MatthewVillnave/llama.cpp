# PRT Phase 21H-U: Regenerate INT8 from f32

## Status: PASS_REGEN_INT8_OFFLINE_PARITY

## A. Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## B. Previous HEAD
`944d5944a` (Phase 21H-T - INT8 decode formula fix)

## C. New HEAD
`e8d61f2a` (reverted - see findings)

## D. f32 reference path
`/tmp/prt_phase21f_layer0_W_f32.bin`

## E. regenerated INT8 path
`/tmp/prt_phase21h_u_int8_from_f32/ffn_up_layer0_prt.int8`

## F. quantization formula
Per-column scale: scale[j] = max_abs(W[:,j]) / 127

## G. storage layout
**[K,M] row-major**: int8[k*M + j] = round(W[k,j] / scale[j])

## H. offline cosine
**0.99996084** (near-perfect!)
- Was: column-major storage gave 0.000054
- Fixed by using row-major storage

## I. offline MAE
0.00012795

## J. norm ratio
1.000039

## K. native run result
"The capital of France is Paris." via PTY (capture in pipes fails)

## L. PRT-v2 regenerated INT8 result
- Route: ggml_op (layer 0)
- Sidecar: loads correctly with scales
- Kernel: executes successfully
- output_abs_sum_first4: ~3.85 (close to f32's 3.84)

## M. kernel logs
```
[PRT_V2_SIDECAR] layer=0 source=int8_sidecar path=...K=896 M=4864
[PRT_V2_DECODE] decoded_to=f32 K_M_row_major formula=int8[k*M+j]*scale[j]
[PRT_V2_KERNEL_EXIT] done=1 output_abs_sum_first4=3.849652
```

## N. 4-prompt result
Partially tested - kernel executes but output capture unreliable.

## O. capture limitation
Output text appears only in TTY, not in pipes. Native produces "Paris", INT8 path produces similar output sums (~3.85).

## P. verdict
**PASS_REGEN_INT8_OFFLINE_PARITY**

## Q. recommended next
Phase 21H-V: Regenerate INT6 from same f32 source with same row-major storage approach.

## R. models/sidecars/binaries staged?
No - /tmp references only

## S. secrets detected?
No

## T. existing tags touched?
No

## Key Finding: Storage Layout Fix

The critical fix was using **row-major storage** in the quantization:

| Storage | Formula | Cosine |
|----------|---------|--------|
| Column-major (wrong) | int8[j*K+k] | 0.000054 |
| Row-major (correct) | int8[k*M+j] | 0.999961 |

The C++ code expects row-major storage `[K,M]`. My initial regeneration used column-major storage, which caused the mismatch.

## Files Modified
- src/llama-graph.cpp: Changed INT8 sidecar directory path
- examples/speculative/results/*: Phase reports