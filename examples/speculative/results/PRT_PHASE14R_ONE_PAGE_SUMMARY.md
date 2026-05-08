# PRT Phase 14 — One-Page Summary

## Result

Packed INT8 sidecars recovered near-native CPU throughput while preserving tested output quality on Qwen2.5-3B and Qwen2.5-7B.

## Why It Matters

Float32 PRT sidecars preserved quality but were too slow. INT8 sidecars fixed the memory-traffic bottleneck — the active replacement path now has a favorable quality-throughput trade-off on the measured CPU setup.

## Phase 13 vs Phase 14

| | Phase 13 (float32) | Phase 14 (INT8) |
|---|---|---|
| Output quality | ✅ Correct | ✅ Correct |
| Throughput vs native | ❌ Speed-negative | ✅ Near-native |
| Sidecar size per layer | ~18 MB float32 | ~4.3 MB int8 + scales |
| Scaling to 7B | Not tested at scale | ✅ 0.993× avg on 7B |

## Best Numbers

**Qwen2.5-3B:**
- Native avg: 21.0 t/s
- INT8 PRT avg: 21.0 t/s
- Ratio: **1.000× avg** / 1.002× median (10 independent runs)
- Quality: 8/8 semantic matches, 0 degradations
- Sidecars: 36/36 loaded

**Qwen2.5-7B:**
- Native avg: 8.75 t/s | INT8 PRT avg: 8.69 t/s
- Ratio: **0.993× avg** / 0.989× median (8-prompt suite)
- Quality: 8/8 exact or semantic matches, 6/8 exact, 0 degradations
- Sidecars: 28/28 loaded
- Every individual prompt ratio ≥ 0.976

**Qwen2.5-0.5B:**
- INT8 PRT: ~1.81× faster than float32 PRT (8/8 exact matches)

## Checkpoints

- `PRT_PHASE14G_3B_INT8_REPEATABILITY_CHECKPOINT` — 3B native parity
- `PRT_PHASE14Q_7B_INT8_FULL_VALIDATION_CHECKPOINT` — 7B full validation

## Claim Boundary

This is a measured CPU research checkpoint. Not production-ready. Not a universal speedup. Not a GPU comparison. Not generalizable beyond this setup.

## Next

1. Longer-context / larger-n stability
2. INT4 sidecar prototype
3. Native ggml/backend integration
4. Optimization toward native-beating throughput
5. Public writeup / X thread package

**Branch:** `experimental/prt-phase14a-packed-sidecars`  
**Commit:** `f9583337d`  
**Phase:** 14R (lab writeup package)
