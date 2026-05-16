# PRT Phase 22E: Route INT8 through ggml PRT op

## Branch
experimental/prt-phase19a-alt-sidecar-backed

## Previous HEAD
a126d98b6

## New HEAD
b2c8d9e01 (Phase 22E implementation)

## modifications

### prt_graph_replace.h (INT8 → ggml-native op path)
- Added INT8 decoded-f32 path that uses ggml_prt_ffn_up()
- Decodes INT8 using W[k,j] = int8[k*M+j] * scale[j]
- Routes through ops.cpp backend (same as f32 path)

### llama-graph.cpp (INT8 loader fix)
- After INT8 decode, sets g_prt_sidecar_data[layer] = decoded_f32
- Sets g_prt_sidecar_format[layer] = 0 (marks as f32 for ggml-native path)
- Enables ggml-native op path for decoded INT8 weights

## Test status

LIMITATION: No valid INT8 sidecar file found
- Expected INT8 file: /tmp/prt_phase21h_u_int8_from_f32/ffn_up_layer0_prt.int8
- Required size: 4,377,600 bytes (K*M + M*4 = 896*4864 + 4864*4)
- Actual size: 17,432,576 bytes (f32 size, mislabeled as .int8)
- Loader rejects due to size mismatch

The f32 path (via --prt-sidecar-dir phase22c_r_f32_layer0) was verified in Phase 22D with similar implementation.

## Verdict
BLOCKED_NO_VALID_INT8_SIDECAR_FILE

The ggml-native op path implementation is complete but blocked by the INT8 file being mis-sized (17MB vs expected 4.4MB).

## Recommended next
Phase 22F: Create valid INT8 sidecar or use proper INT8 loader path with correctly sized file.
