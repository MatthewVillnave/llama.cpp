# PRT Phase 19K Rerun — INT6→F32 Predecode Runtime Validation

## Verdict

**PARTIAL_AVX2_HIT_BUT_SLOW** — AVX2 path activates but provides minimal speedup over scalar INT8

## Machine State

- RAM: 11GB available ✅
- Swap: full but inactive
- Load: healthy
- Native 0.5B: ~71-95 t/s ✅

## Summary

| Mode                | Generation t/s | vs Native | Notes |
|--------------------|---------------|----------|-------|
| Native (baseline)   | 71.3          | 1.0x    | No PRT |
| INT8 scalar        | 19.0          | 3.65x    | 24/24 loaded |
| INT8+predecode-f32 | 19.7          | 3.62x    | 24/24 predecoded |

## Root Cause Analysis

1. **Sidecar format mismatch**: 0.5B sidecars in `/tmp/prt_sidecars_05b_int8/` are `.int8` format (not `.int6`)
2. **INT6 path blocked**: `--prt-sidecar-format int6` looks for `.int6` files, finds none → 0/24 loaded
3. **INT8 path works**: `--prt-sidecar-format int8` loads 24/24 sidecars correctly
4. **Predecode added**: Added conditional to INT8 loader path to call `llama_set_prt_sidecar_int6_predecode_f32()`
5. **AVX2 activates**: Predecode function sets format=0, logs confirm f32_avx2 path

## Predecode Verification

- Flag recognized: `[PRT-PREDECODE] enabled via --prt-predecode-f32` ✅
- 24/24 layers predecoded: `[PRT-PREDECODE] layer=N M=4864 N=896 f32_bytes=17432576 format=f32_avx2` ✅
- Total RAM: ~418MB (17.4MB × 24 layers) ✅

## Code Changes

1. Fixed `--prt-predecode-f32` arg callback (no-value flag pattern)
2. Added predecode conditional to INT8 loader (line 661-667)
3. Build verified ✅

## Why No Speedup?

The AVX2 path uses float32 multiplication, but int8 scalar also uses a fast path. With M*K small (4.3M elements):
- Float32: one FP32 mul per element
- INT8 scalar: int8 multiply
- Both bounded by memory bandwidth, not compute

## Recommended Next

1. Phase 19L: Test on larger models (7B) where FMA throughput matters more
2. Alternative: Keep int8 format for storage, evaluate in-place FMA without predecoding
3. Design: INT8 resident mode to avoid float32 memory bloat

## Files Changed

- common/arg.cpp: Fixed --prt-predecode-f32 flag callback
- tools/cli/cli.cpp: Added predecode conditional to INT8 loader (lines 661-667)

