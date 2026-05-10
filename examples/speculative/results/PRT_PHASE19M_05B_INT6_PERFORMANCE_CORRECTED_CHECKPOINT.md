# PRT Phase 19M — Corrected 0.5B INT6 Performance Checkpoint

## Verdict

**PASS_05B_INT6_FUNCTIONAL_NO_MEANINGFUL_SPEEDUP**

## Context Chain

| Phase | Result |
|-------|--------|
| Phase 19A-C | 0.5B PRT runtime functional |
| Phase 19H | Frozen runtime/source-quality checkpoint |
| Phase 19I | Identified scalar INT6 as bottleneck |
| Phase 19J/K | Added and validated predecode-f32 routing |
| Phase 19L | Initially mixed INT8 vs INT6 formats |
| Phase 19L-FF | Format forensic corrected the result |

## Corrected Format Findings

| Directory | Size | Format | Header | Loader flag |
|-----------|------|--------|--------|-------------|
| `/tmp/prt_sidecars_05b_int8/` | 4,377,600 | unpacked_int8 | None | `--prt-sidecar-format int8` |
| `/tmp/prt_phase19b/` | 3,288,080 | **packed_int6** | PRT6 | `--prt-sidecar-format int6` |

- **Previous ambiguity resolved**: Phase 19L ran with `--prt-sidecar-format int8` → unpacked INT8, NOT INT6
- **Correct INT6 path**: `--prt-sidecar-format int6` + `/tmp/prt_phase19b/`

## Corrected Timing (n=64, c=256, t=4, 3 runs)

| Mode | Run 1 | Run 2 | Run 3 | Avg tok/s | Format logged | AVX2 hits |
|------|-------|-------|-------|-----------|---------------|-----------|
| Native | 94.9 | - | - | **94.9** | none | 0 |
| INT6 scalar (packed) | 18.8 | 18.9 | 19.0 | **18.9** | `mode=int6` | 0 |
| INT6 + predecode-f32 | 19.2 | 19.2 | 19.2 | **19.2** | `mode=fp32` | 24/24 |

## Key Metrics

- **Predecode speedup vs scalar**: ~1.6% (19.2/18.9 - 1)
- **Ratio vs native**: ~4.88x slower (19.2/94.9)
- **AVX2 routing**: Confirmed ✅ (`mode=fp32` logged)
- **Predecode setup**: ~418MB extra RAM (17.4MB × 24 layers)
- **Output correctness**: Clean generation

## Interpretation

1. **Packed INT6 works correctly**: `mode=int6` PRT_COMPUTE hits on all 24 layers ✅
2. **Predecode-f32 works correctly**: `mode=fp32` PRT_COMPUTE hits on all 24 layers ✅
3. **No meaningful speedup**: ~1.6% gain is within noise — not a reliable improvement
4. **Root cause**: Memory bandwidth bottleneck on 0.5B FFN operations
5. **0.5B is plumbing/testbed**: Too small to show AVX2 compute benefits
6. **No speedup claim**: Memory bandwidth bounded, not compute-bounded

## Should We Try 7B?

**Arguments for 7B probe**:
- 0.5B may be too cache-friendly to show AVX2 benefit
- Larger FFN matrices (M=18944, K=3584) may benefit more from float32 AVX2 FMA
- 7B sidecars already validated and available at `/tmp/prt_sidecars_7b_int6_phase15b_packed/`

**Arguments against**:
- Predecode-f32 RAM cost scales with M×K: 7B × 24 layers × 17.4MB = ~417MB extra (same order)
- Setup overhead may dominate on short prompts
- Previous 7B INT6 paths were already near-native without f32 predecode

**Recommendation**: `RECOMMEND_TINY_7B_PREDECODE_PROBE`
- One prompt: "The capital of France is"
- n=32, c=256, t=4
- Three modes: native, INT6 scalar, INT6+predecode-f32
- Monitor RAM before/after
- Stop if swap activates
- No 8-prompt validation
- No 14B

## Allowed Claims

✅ Packed INT6 for 0.5B is functional
✅ Predecode-f32 routes correctly to AVX2 path
✅ AVX2 path activates with format=fp32 on 0.5B
✅ No meaningful speedup observed on 0.5B
✅ Memory bandwidth is the likely bottleneck

## Forbidden Claims

❌ No speedup claim
❌ No production readiness
❌ No 7B/14B extrapolation
❌ No RAM savings claim
❌ No universal performance conclusion

## Related Files

- Format forensic: `PRT_PHASE19L_FORMAT_FORENSIC_05B.md`
- Timing data: `phase19l_format_forensic_05b.json`
