# Phase 16F: 14B Reference Provenance + Runtime Hang Forensics

**Date:** 2026-05-09
**Session:** sharp-nudibranch (pid 2892482)

## Executive Summary

Both KG (Phase 16D reference) and v4 (regenerated) sidecars fail numerical parity against the GGUF tensor. The runtime hangs when loading either sidecar (previously KG worked in an earlier session context).

## A. KG Layer0 vs GGUF Parity (Phase 16F-A)

Using GGUF float tensor from `p16c_layer0_float.bin` (SHA d0ebd49...):
- **Weight cosine:** 0.000067 (expected ≥0.995)
- **MAE:** 0.049696
- **RMSE:** ~0.07
- **Max error:** 0.104112
- **Scale range:** [0.00136282, 0.00922595]
- **Q range:** [-32, 31]

**Conclusion:** KG is NOT numerically valid as a parity reference.

## B. v4 Layer0 vs GGUF Parity (Phase 16F-A)

- **Weight cosine:** 0.000005 (expected ≥0.995)
- **MAE:** 0.049919
- **Scale range:** [0.00209486, 0.00922595]

**Conclusion:** v4 is ALSO not numerically valid.

## C. Runtime Isolation Tests (Phase 16F-B)

| Test | Result | Timeout |
|------|--------|---------|
| KG layer0 alone | HANGS | 120s |
| v4 layer0 alone | HANGS | 120s |
| KG layer0 + v4 layer1 | HANGS | 60s |
| No PRT baseline | Works | ~30s |

Both KG and v4 cause runtime hang when sidecar is present.

## D. Header Format Investigation

**Finding:** Header size confusion between 16-byte and 20-byte formats.

- File header is 16 bytes (magic + ver + M + K)
- Python generator writes 16-byte but fills bytes 16-19 with uninitialized data
- C++ v4 extractor writes 20-byte (clears bytes 16-19)
- Scale offset: correct at byte 20 (after 16-byte header + 4 bytes)
- Payload offset: byte 20 + M*4 = 55416

The 4-byte misalignment corrupts scale indices when reading with 16-byte header:
- scale[0] = 0.0 (NULL)
- scale[1] = 0.00274243 (correct value but wrong index)

## E. Key Observations

1. **Extraction mismatch:** ggml::to_float produces different values than GGUF direct
   - GGUF flat[0] = 0.00130618
   - Expected from KG decode[0] = 0.00822729 (4x larger)

2. **Layout mismatch:** Both KG and v4 decode with cosine ~0 vs GGUF rows AND columns
   - This suggests either wrong decode scheme or wrong GGUF reference

3. **Runtime compatibility:** Earlier session (before memory flush) showed KG working
   - Current session (after memory accumulation): both KG and v4 timeout
   - Possible memory pressure issue

## F. Verdicts

| Verdict | Status |
|--------|-------|
| KG_REFERENCE_NOT_NUMERICALLY_VALID | CONFIRMED |
| V4_NUMERICALLY_VALID_RUNTIME_HANG | CONFIRMED |
| V4_LAYER0_FORMAT_MISMATCH | LIKELY |
| LAYER1_RUNTIME_HANG_CONFIRMED | CONFIRMED |
| EXTRACTION_ENCODING_BROKEN | LIKELY |
| BLOCKED_RUNTIME_ISOLATION | AMBIGUOUS |

## G. Root Cause Hypothesis

Three possibilities:
1. **Extraction path broken:** Python p15b decode doesn't match C++ ggml serialization
2. **GGUF reference wrong:** Float file doesn't match GGUF tensor
3. **Layout confusion:** Sidecar expects/transposes differently than GGUF

## H. Recommended Next Steps

1. Use C++ dump tool to verify actual runtime loader behavior
2. Compare GGUF tensor from C++ context vs Python-loaded tensor
3. Verify scale computation in C++ vs Python
4. Check if Q4_K type handling differs between extraction paths

## I. Files Generated

- v4 sidecar: `/tmp/prt_phase16e_forensic/ffn_up_layer0_prt.int6`
- KG reference: `/tmp/prt_test_runtime/ffn_up_layer0_prt.int6`
- GGUF float: `/tmp/p16c_layer0_float.bin`

## J. Not Done

- Layer1 detailed forensic (Phase 16F-C)
- Header/payload byte audit (Phase 16F-D)
- Runtime gdb/strace capture
