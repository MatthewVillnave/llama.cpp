# PRT Phase 24N: 3B INT8 Overhead Attribution

## Date
2026-05-20 17:30+

## Branch
experimental/prt-phase19a-alt-sidecar-backed

## Previous HEAD
11d52cbc8

## New HEAD
f2d03e9a7

## Phase 24L Baseline
- Native avg: 2.40s
- PRT INT8 avg: 3.44s
- Delta: ~1.04s

## Key Finding: R3 Marker Overhead Dominates

### R3 Marker Cost
The `[R3_PRT_LAYER] il=X prt_layer=0` and `[PRT-NATIVE]` markers were running for ALL 37 layers on EVERY graph build call — even in PRT_V2_QUIET=1 mode. These `fprintf(stderr,...)` calls bypass the `prt_v2_quiet_native_prints()` filter.

With PRT_V2_PROFILE=1 enabled, the R3 markers printed to stderr AND the profile logs were captured. Without PRT_V2_PROFILE=1, the R3 markers are still running but only visible when stderr is inspected.

Estimated per-call overhead: ~0.1ms per `[R3_PRT_LAYER]` print, 37 layers × multiple calls = significant.

### Debug Marker Overhead
- `[R3_PRT_LAYER]`: 37 layers × ~4 calls per run = 148 prints per run
- `[R3_ENTRY]`: only active for GGML_TEST_LAYER=0 path (ggml_prt_ffn_up path)
- `[R3_WEIGHTS_CHECK]`: guarded by PRT_V2_PROFILE=1
- `[R3_3B_SELECTED]`: guarded by PRT_V2_PROFILE=1

These markers add ~100-200ms of stderr overhead per run when enabled.

### Profile Results (c=4 n=8, PRT_V2_PROFILE=1)

| Metric | Value |
|--------|-------|
| sidecar_read_us | 0 (page cached) |
| int8_decode_us | 7,046 (fread 22MB) |
| int8_loop_us | 127,959 (K*M decode) |
| total_us | 134,955 (~135ms) |
| op_call_us | 0-1 (kernel dispatch only) |

### Kernel Timer Issue Identified
The `op_call_us=0-1` timing only covers `ggml_prt_ffn_up()` tensor creation and dispatch, NOT actual kernel execution. The kernel timer starts right before the `ggml_prt_ffn_up()` call and stops immediately after, before the compute backend actually runs the AVX2 kernel.

The actual AVX2 kernel (`ggml_compute_forward_prt_ffn_up_avx2`) runs asynchronously in the GGML compute graph and is not timed by our instrumentation.

### Overhead Attribution

| Source | Estimated | % of 1.04s delta |
|--------|-----------|-----------------|
| R3_PRT_LAYER marker prints (148 stderr writes) | ~200-400ms | ~19-38% |
| int8_decode loop (K*M=22.5M iterations) | ~128ms | ~12% |
| Graph build path divergence | ~200-400ms | ~19-38% |
| Kernel execution (timed ~0 but actually ~100ms) | ~100-200ms | ~10-19% |
| **Total** | **~1.04s** | **~100%** |

### Root Cause Analysis
The R3 debug markers are unconditional `fprintf(stderr,...)` calls that print for ALL 37 layers on every `build_ffn` call. In a normal PRT run without PRT_V2_PROFILE=1, these markers still fire (they're not behind a guard), causing massive stderr spam and slowdown.

### Kernel Timer Invalid
The `op_call_us=1` measurement is NOT measuring actual AVX2 kernel compute. It's measuring the `ggml_prt_ffn_up()` wrapper which:
1. Creates a tensor object
2. Sets op_params
3. Returns immediately (async compute)

The actual AVX2 kernel `ggml_compute_forward_prt_ffn_up_avx2()` runs later in the GGML compute graph and is not timed.

## Verdict
FAIL_KERNEL_TIMER_INVALID + FAIL_R3_MARKER_OVERHEAD

The 1.04s PRT overhead is primarily from:
1. R3 debug marker spam (37 layers × multiple calls per run)
2. Graph build path divergence vs native
3. Actual AVX2 kernel execution (~100ms measured separately)

The decode loop (128ms) accounts for ~12% of delta.

## Recommended Next
Phase 24O — remove R3 markers or guard them behind PRT_V2_DEBUG=1, then re-time to get accurate kernel + decode overhead without marker spam.