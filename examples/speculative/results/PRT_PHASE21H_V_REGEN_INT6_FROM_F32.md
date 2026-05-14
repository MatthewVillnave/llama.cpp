# PRT Phase 21H-V: Regenerate INT6 from f32

## Status: PASS_REGEN_INT6_OFFLINE_PARITY

## A. Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## B. Previous HEAD
`1c9043176` (Phase 21H-U - INT8 regeneration)

## C. New HEAD
`1c9043176` (no code change — regeneration only)

## D. f32 reference path
`/tmp/prt_phase21f_layer0_W_f32.bin`

## E. regenerated INT6 path
`/tmp/prt_phase21h_v_int6_from_f32/ffn_up_layer0_prt.int6`

## F. INT6 format/header
PRT6 format (Phase 19S harness compatible):
- Magic: 0x36545250 (little-endian 'PRT6')
- Version: 1
- M=4864, K=896
- Header: 16 bytes
- Packed payload: 3268608 bytes (M*K*6/8)
- Scales: 19456 bytes (M*4)
- Total: 3288080 bytes

## G. storage layout
**[M,K] column-major** (flat index j*K + k)

This matches Phase 19S harness decode formula:
`W[k,j] = q_flat[j*K + k] * scales[j]`

## H. quantization formula
- QMAX = 31 (signed INT6 range -31..+31)
- Per-column scale: scale[j] = max_abs(f32[:,j]) / 31
- q = round(f32[k,j] / scale[j]), clamped to [-31,+31]
- Storage: q_flat[j*K + k] = q[k,j]

## I. offline cosine
**0.99936563** (near-perfect)

## J. offline MAE
0.00052367

## K. norm ratio
1.000569

## L. INT6 vs INT8 comparison
- Cosine: **0.99933678**
- Both sidecars reference same f32 source
- INT6 uses QMAX=31 (6-bit), INT8 uses QMAX=127 (8-bit)
- Small quantization error difference expected

## M. runtime result
Not run in this phase (offline only per protocol).

## N. 4-prompt result
Not run (offline only).

## O. capture limitation
Output capture remains TTY-limited.

## P. verdict
**PASS_REGEN_INT6_OFFLINE_PARITY**

## Q. recommended next
Phase 21I: Compare PRT-v2 f32, INT8, and INT6 paths at runtime.

## R. models/sidecars/binaries staged?
No - /tmp references only

## S. secrets detected?
No

## T. existing tags touched?
No

## Key Findings

1. **INT6 storage is [M,K] column-major** — different from INT8's [K,M] row-major
2. **Both formats are internally consistent** — INT6 decodes back to f32 with cosine 0.9994
3. **Phase 19S harness compatibility confirmed** — same PRT6 header format

## Storage Comparison

| Sidecar | Storage | Decode Formula | Cosine vs f32 |
|---------|---------|---------------|---------------|
| INT8 (21H-U) | [K,M] row-major | int8[k*M+j]*scale[j] | 0.999961 |
| INT6 (21H-V) | [M,K] column-major | q[j*K+k]*scale[j] | 0.999366 |

## Next Steps

1. Phase 21I: Run all three paths (f32, INT8, INT6) side-by-side
2. Verify runtime decode formula matches offline unpack
3. Confirm both INT8 and INT6 paths produce correct generation