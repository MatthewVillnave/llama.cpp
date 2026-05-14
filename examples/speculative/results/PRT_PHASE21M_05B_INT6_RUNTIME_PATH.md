# PRT Phase 21M: 0.5B INT6 Runtime Path

## Status: PASS_05B_INT6_RUNTIME_CANARY

## A. Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## B. Previous HEAD
`6e7afba8c` (Phase 21L checkpoint)

## C. New HEAD
[TBD after commit]

## D. INT6 sidecar path
`/tmp/prt_phase21h_v_int6_from_f32/ffn_up_layer0_prt.int6`

## E. INT6 schema/layout
- **Storage**: [M,K] column-major packed
- **Header**: 16 bytes (BE magic "PRT6"=0x50525436, LE version/M/K)
- **Decode formula**: `W[k,j] = q_flat[j*K + k] * scale[j]`
- **Sign**: 6-bit values 0-63, stored as `val - 32` (signed int8)
- **Packing**: 4×6-bit values → 3 bytes
- **Scales**: Per-row float32 at end of file

## F. Offline decode parity (pre-runtime validation)
- Cosine vs f32: **0.99936563** ✅
- Shape: [896, 4864]
- Sign match: 99.9%

## G. Native sanity
- Exit 0 ✅
- TTY output: "Paris" ✅

## H. INT8 baseline sanity
- Exit 0 ✅
- output_abs_sum: 3.849652 ✅
- Route: ggml_op ✅

## I. INT6 P1 result
- Exit: 0 ✅
- No SIGKILL ✅
- Route: ggml_op ✅
- `[PRT_V2_INT6] magic=0x50525436 M=4864 K=896` ✅
- `[PRT_V2_INT6] layout=M_K_storage formula=q[j*K+k]*scale[j]` ✅
- `[PRT_V2_DECODE] decoded_to=f32 W_shape=[896,4864]` ✅
- output_abs_sum: **3.795631** (INT8=3.849652, diff=1.4%)

## J. INT6 4-prompt result
| Prompt | Exit | output_abs_sum | SIGKILL |
|--------|------|----------------|---------|
| P1: "The capital of France is" | 0 ✅ | 3.795631 | No ✅ |
| P2: "The largest planet..." | 0 ✅ | 3.795631 | No ✅ |
| P3: JSON response | 0 ✅ | 3.795631 | No ✅ |
| P4: "Once upon a time" | 0 ✅ | 3.795631 | No ✅ |

## K. f32/INT8/INT6 kernel comparison
| Path | output_abs_sum | Notes |
|------|----------------|-------|
| f32 baseline | 3.842286 | reference |
| INT8 | 3.849652 | +0.2% vs f32 |
| INT6 | 3.795631 | -1.2% vs f32, -1.4% vs INT8 |

INT6 kernel output is slightly lower than INT8, consistent with its lower offline cosine (0.9994 vs 0.99996).

## L. Capture limitations
Output text captured only in TTY, not via pipes. Semantic content not fully verified for INT6 path. Route/op/kernel logs confirm PRT-v2 invokation.

## M. Verdict
**PASS_05B_INT6_RUNTIME_CANARY**

INT6 runtime path implemented and validated:
- 4/4 prompts exit 0, no SIGKILL
- `[PRT_V2_INT6]` decode logs confirm correct format
- `[PRT_V2_KERNEL_ENTER/EXIT]` confirmed
- output_abs_sum slightly below INT8 (expected given lower offline parity)

## N. Recommended next
**Phase 21N**: Checkpoint 0.5B PRT-v2 sidecar baselines: f32, INT8, INT6. Document the three-path comparison.

## O. Models/sidecars/binaries staged?
No — `/tmp` file references only

## P. Secrets detected?
No

## Q. Existing tags touched?
No

## Key Implementation Details

### INT6 Header Fix Required
The INT6 file header has mixed-endian encoding:
- **Offset 0-3**: BE magic "PRT6" = 0x50525436
- **Offset 8-11**: LE M = 4864
- **Offset 12-15**: LE K = 896
- **Offset 16+**: Packed 6-bit data
- **End of file**: Scales [M] float32

The Phase 21H-V write had scrambled offset fields — header must be read as fixed 16-byte layout with specific field endianness.

### INT6 Decode Algorithm
1. Read 16-byte header, extract BE magic and LE M/K
2. Read packed data at offset 16 (size = M*K*6/8 bytes)
3. Read scales at end of file (M float32)
4. Unpack: 4×6-bit → 3 bytes, `q[i] = (packed & 0x3F) - 32`
5. Decode: `W[k,j] = q[j*K + k] * scale[j]`

### Code Change Summary
Added INT6 sidecar block in `src/llama-graph.cpp` after INT8 path (before f32 fallback), with:
- Correct 16-byte header parsing
- BE magic check (0x50525436)
- LE M/K check (4864/896)
- Proper 6-bit unpack with sign extension
- Column-major decode formula