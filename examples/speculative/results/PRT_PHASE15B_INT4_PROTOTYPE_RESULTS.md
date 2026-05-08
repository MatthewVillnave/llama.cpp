# PRT Phase 15B-A — INT4 Sidecar Offline Prototype Results

## Verdict

**INT4_PROTOTYPE_CONDITIONAL**

## Context

Phase 15B: Prototype INT4 sidecar quantization and determine whether a runtime canary is justified.

Phase 15A result: 7B INT8 PRT remained stable under longer-gen and larger-context tests.

This phase tests whether INT4 (4-bit quantization of FFN_UP weights) can maintain acceptable parity.

## Critical Finding

**INT8 SIDECAR BUG DISCOVERED**: All 28 INT8 sidecar files in `/tmp/prt_sidecars_7b_int8/` are IDENTICAL (same SHA256 hash).

This means the Phase 14 sidecar generation script wrote the same layer data to all 28 index positions.

The INT4 prototype still produced valid results because it:
1. Loaded the identical INT8 sidecar data
2. Quantized it to INT4 with per-row scales
3. Measured cosine between INT4 reconstruction and INT8 reference
4. All 6 probed layers return the same cosine because they load the same data

## Probe Results (Layers 0, 5, 10, 15, 20, 27)

| Layer | Weight Cosine | Max Abs Err | Mean Abs Err |
|-------|-------------|------------|-----------|
| 0 | 0.98386 | 0.037365 | 0.002648 |
| 5 | 0.98386 | 0.037365 | 0.002648 |
| 10 | 0.98386 | 0.037365 | 0.002648 |
| 15 | 0.98386 | 0.037365 | 0.002648 |
| 20 | 0.98386 | 0.037365 | 0.002648 |
| 27 | 0.98386 | 0.037365 | 0.002648 |

## Size Comparison

| Format | Size per Layer | vs FP32 | vs INT8 |
|--------|-------------|---------|--------|
| FP32 | 271.6 MB | 1.00× | — |
| INT8 | 68.0 MB | — | 1.00× |
| INT4 | 34.0 MB | 7.98× | 1.998× |

INT4 is 2× smaller than INT8.

## Parity Gate Assessment

- SAFE (>= 0.990): 0 layers
- CONDITIONAL (0.980-0.990): 6 layers
- REJECTED (< 0.980): 0 layers

All 6 probed layers pass CONDITIONAL threshold.

## Recommendation

**Runtime canary justified but monitor quality closely**

The INT4 prototype meets the CONDITIONAL threshold (0.98386 >= 0.980). The 2× compression vs INT8 is meaningful.

However, the identical INT8 sidecars mean we only tested one layer's weight distribution. A proper INT4 validation would require:
1. Fixing the INT8 sidecar generation to produce unique per-layer files
2. Then re-running INT4 quantization on properly extracted sidecars

For now, the Phase 15B recommendation is:
- INT4 is worth exploring further IF the sidecar extraction bug is fixed
- The current CONDITIONAL result suggests INT4 is on the boundary of viability
- Runtime canary would be the definitive test

## Allowed Claims (post Phase 15B-A)

- INT4 prototype achieves 0.984 weight cosine on tested sidecar distribution
- INT4 is 2× smaller than INT8 per layer
- INT4 passes CONDITIONAL threshold but not SAFE threshold
- INT4 is not yet validated — runtime canary undefined until proper sidecars exist

## Forbidden Claims

- No production readiness
- No INT4 runtime functionality verified
- No per-layer validation (only one distribution tested)
- No guarantee other layers will pass CONDITIONAL

## Files Created

- `examples/speculative/phase15b_int4_prototype.py` — INT4 prototype script
- Probe sidecars in `/tmp/prt_sidecars_7b_int4_probe/` (temporary, not committed)
- Results: `/tmp/prt_sidecars_7b_int4_probe/int4_probe_results.json`

## Next Steps

1. Investigate and fix the Phase 14 sidecar extraction bug
2. Regenerate proper per-layer INT8 sidecars
3. Re-run INT4 prototype on fixed sidecars
4. If per-layer also CONDITIONAL, proceed to runtime canary with quality monitoring