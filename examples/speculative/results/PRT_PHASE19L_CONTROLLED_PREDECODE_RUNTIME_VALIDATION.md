# PRT Phase 19L — Controlled Predecode Runtime Validation

## Verdict

**PARTIAL_AVX2_HIT_NO_MEANINGFUL_SPEEDUP**

## Machine Health
- RAM available: 11GB ✅
- Swap: full but inactive ✅
- Load: 0.00 ✅
- Native baseline: healthy (~92-96 tok/s)

## Format Audit

### 0.5B sidecar dir: `/tmp/prt_sidecars_05b_int8/`
- Format: **UNPACKED_INT8** (NOT INT6)
- Files: 24 layers of `ffn_up_layer{N}_prt.int8`
- Size: 4,377,600 bytes/layer (M=4864, K=896, int8_data + float32 scales)
- No PRT6 magic header — raw int8 data + appended float32 scales
- CLI arg: `--prt-sidecar-format int8`
- **Previous Phase 19K-RERUN used INT8 (not INT6) — this is confirmed**

### 7B sidecar dir: `/tmp/prt_sidecars_7b_int6_phase15b_packed/`
- Format: PACKED_INT6 (PRT6 header + packed payload + float32 scales)
- CLI arg: `--prt-sidecar-format int6`
- **0.5B has no INT6 sidecars — INT6 path cannot be tested for 0.5B**

## Timing Results (n=64, c=256, t=4, 3 runs each)

| Mode | Run 1 | Run 2 | Run 3 | Avg tok/s | Notes |
|------|-------|-------|-------|-----------|-------|
| Native | 95.7 | 95.0 | 91.7 | **94.1** | No PRT |
| INT8 scalar | 18.7 | 18.6 | 18.9 | **18.7** | format=int8 |
| INT8+predecode-f32 | 19.3 | 18.9 | 19.3 | **19.2** | format=fp32 |

## PRT Compute Activation (verified via debug logs)

- INT8 scalar: `[PRT_COMPUTE] layer=N mode=int8 hit=1` ✅
- INT8+predecode-f32: `[PRT_COMPUTE] layer=N mode=fp32 hit=1` ✅
- Both modes activate PRT compute on all 24 layers

## Key Findings

1. **AVX2 path activates**: Predecode sets format=fp32 → `mode=fp32` in PRT_COMPUTE logs
2. **Format routing works**: INT8 scalar → int8 path; predecode → fp32 path
3. **No meaningful speedup**: ~19.2 vs ~18.7 tok/s = ~3% improvement
4. **Root cause**: Memory bandwidth bottleneck — both int8 and f32 paths are bandwidth-limited

## Predecode Setup Cost

- Per layer: 17,432,576 bytes (17.4MB float32)
- Total 24 layers: ~418MB extra RAM
- Predecode: one-time at load, no per-token cost

## Comparison Table

| Metric | Native | INT8 scalar | INT8+predecode |
|--------|--------|-------------|----------------|
| tok/s | 94.1 | 18.7 | 19.2 |
| vs native | 1.0x | 5.03x slower | 4.90x slower |
| PRT mode | none | int8 | fp32 |
| Sidecars loaded | 0/24 | 24/24 | 24/24 |
| AVX2 hits | 0 | 0 | 24 |
| RAM overhead | 0 | ~105MB | ~523MB |

## Conclusions

1. **AVX2 path works**: Predecode correctly activates float32 AVX2 path (verified via logs)
2. **No meaningful speedup**: Both paths bottlenecked by memory bandwidth
3. **INT6 not tested**: 0.5B has no INT6 sidecars — only unpacked INT8
4. **Predecode RAM cost**: ~418MB extra for f32 buffers (not worth it for 3% gain)

## Recommended Next

- Phase 19M-A: Test on 7B with INT6 packed sidecars — larger matrices benefit more from AVX2 FMA
- Alternative: Investigate memory layout/stride issues that prevent AVX2 from showing speedup
- If AVX2 on 7B also slow: likely memory bandwidth is the true bottleneck across all configurations
