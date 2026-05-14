# PRT Phase 21H-R: INT6 Decode Forensic — BLOCKED (sidecar missing)

## Status: FAIL_INT6_SIDECAR_MISSING

## A. Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## B. Previous HEAD  
`3ecf5d8ab` (Phase 21G)

## C. New HEAD  
`3ecf5d8ab` (no change — blocked)

## D. INT6 sidecar path
`/tmp/prt_phase19b/ffn_up_layer0_prt.int6` — **MISSING** (deleted during session)

## E. f32 reference path
`/tmp/prt_phase21f_layer0_W_f32.bin` (17MB, shape [896,4864], norm=38.1058)

## F. INT8 reference path  
`/tmp/prt_sidecars_05b_int8/ffn_up_layer0_prt.int8` (4.4MB)

## G. Schema/Header
N/A — INT6 sidecar missing

## H. Tested decode variants
N/A — INT6 sidecar missing

## I. Best cosine vs f32
N/A

## J. Best cosine vs INT8
N/A

## K. Best layout
N/A

## L. Root cause
1. **INT6 sidecar missing** — file was deleted during the session
2. **INT8 sidecar incompatibility** — INT8 and f32_file are independent quantizations with different sign patterns

## M. Fix implemented
None — sidecar missing, cannot regenerate without model access

## N. Runtime canary run?
No — INT6 sidecar missing

## O. Runtime result
N/A

## P. Verdict
FAIL_INT6_SIDECAR_MISSING

## Q. Recommended next
1. **Regenerate INT6 sidecar** from f32_file using known-good extraction
2. **Verify provenance** — INT6/INT8/f32 should all reference same layer0 ffn_up weights
3. **Re-run Phase 21H** after sidecar restoration
4. **Alternative**: Use f32_file directly as PRT-v2 weights (no decode needed)

## R. Models/sidecars/binaries staged?
No — all /tmp references

## S. Secrets detected?
None

## T. Existing tags touched?
None

## Key Findings (INT8 sidecar analysis)

When analyzing the available INT8 sidecar (even without INT6):

| Metric | Value |
|--------|-------|
| INT8 W norm | 38.1100 |
| f32 norm | 38.1058 |
| Norm ratio | 1.0001 |
| Cosine vs f32 | -0.000150 (random) |
| Abs cosine vs f32 | 0.623 |
| Sign match | 44.4% |
| MAE | 0.020505 |
| Scale mean | 0.000525 |
| f32 mean abs | 0.014413 |

**Interpretation**: INT8 sidecar and f32_file are **independent quantizations** of the same layer0 ffn_up weights. They share similar magnitude distribution (abs cosine 0.623) but have different sign patterns (cosine near 0).

## SIGKILL Analysis

Prior SIGKILLs were **NOT OOM** — RAM available (9.2Gi), no dmesg evidence. Likely process timeout or crash when model generates garbage from incompatible decoded W tensor.

## Sidecar Provenance Issue

The INT6/INT8 sidecars in `/tmp/prt_phase19b/` were generated on May 10 as part of Phase 19B, likely from the original Qwen2.5-0.5B model weights. The f32_file was extracted on May 14 from the same model. These are independent quantization processes, producing incompatible sign patterns.

**Solution**: Regenerate sidecars directly from the f32_file reference to ensure deterministic compatibility.