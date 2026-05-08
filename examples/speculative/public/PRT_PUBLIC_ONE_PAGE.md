# PRT Phase 14 — Public One-Page Summary

## Result

Packed INT8 sidecars recovered near-native CPU throughput while preserving tested output quality on Qwen2.5-3B and Qwen2.5-7B (Q4_K_M).

## Phase 13 vs Phase 14

| | Phase 13 (float32) | Phase 14 (INT8) |
|---|---|---|
| Output quality | ✅ Preserved | ✅ Preserved |
| Throughput vs native | ❌ Too slow | ✅ Near-native |
| Memory footprint | ~18 MB/layer | ~4.3 MB/layer |
| Scaling to 7B | Not tested | ✅ 0.993× avg on 7B |

## Best Numbers

**Qwen2.5-3B:**
- Native avg: 21.0 t/s
- INT8 PRT avg: 21.0 t/s
- Ratio: **1.000×** (10 independent runs)
- Quality: 8/8 semantic matches, 0 degradations

**Qwen2.5-7B:**
- Native avg: 8.75 t/s | INT8 PRT avg: 8.69 t/s
- Ratio: **0.993× avg** / 0.989× median
- Quality: 8/8 exact or semantic matches, 0 degradations

**Qwen2.5-0.5B:**
- INT8 PRT ~1.81× faster than float32 PRT (8/8 exact matches)

## Checkpoints

- `PRT_PHASE14G_3B_INT8_REPEATABILITY_CHECKPOINT` — 3B native parity
- `PRT_PHASE14Q_7B_INT8_FULL_VALIDATION_CHECKPOINT` — 7B full validation

Both on `experimental/prt-phase14a-packed-sidecars` branch.

## Approved Public Claim

> "Packed INT8 sidecars let PRT preserve tested output quality while recovering near-native llama.cpp throughput on Qwen2.5-3B and 7B in my measured CPU setup."

## Important Caveat

This is a measured CPU research checkpoint only. Not production-ready. Not a universal speedup. Not a GPU comparison. Not generalizable beyond this setup.

## Next

Longer context, INT4 sidecars, native backend integration, broader validation.
