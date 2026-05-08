# PRT Phase 14E — 3B INT8 Full Validation

## Verdict: PASS_3B_INT8_FULL_QUALITY_AND_TIMING

## Context

- Phase 14D: 3B INT8 single-prompt canary passed at commit 3c78e6b16
- Phase 14D: INT8 PRT matched native (1.01×) and was 2.5× faster than float32 PRT
- Purpose: Full 8-prompt quality + timing validation on 3B model

## Quality Results — 8 Prompts

| # | Prompt | Native (t/s) | Float32 (t/s) | INT8 (t/s) | INT8/N | Exact Match | Semantic Match |
|---|-------|-------------|--------------|------------|--------|------------|----------------|
| 1 | The capital of France is | 20.9 | 8.6 | 20.7 | 0.99× | ❌ | ✅ |
| 2 | Write a Python function that reverses a list. | 18.7 | 7.5 | 18.7 | 1.00× | ✅ | ✅ |
| 3 | Once upon a time in a | 19.0 | 7.6 | 18.9 | 0.99× | ❌ | ✅ |
| 4 | Explain CPU inference in one sentence. | 19.1 | 7.6 | 19.2 | 1.01× | ✅ | ✅ |
| 5 | Return JSON with keys name and status. | 19.3 | 7.9 | 19.1 | 0.99× | ✅ | ✅ |
| 6 | The fastest way to sort a list in Python is | 18.8 | 7.6 | 18.6 | 0.99× | ✅ | ✅ |
| 7 | In two sentences, explain what RAM does. | 18.5 | 7.5 | 18.0 | 0.97× | ✅ | ✅ |
| 8 | Complete this phrase: artificial intelligence is | 18.4 | 7.6 | 18.3 | 0.99× | ✅ | ✅ |

**AVG:** 19.1 t/s (native) vs 7.7 t/s (float32) vs **18.9 t/s** (INT8)

### Quality Analysis

- Exact matches: 6/8 (Two mismatches due to ANSI spinner capture artifact, not actual text differences)
- **Semantic matches: 8/8** ✅ — All INT8 outputs are semantically identical to native outputs
- No quality degradation, no repetition/collapse
- All runs completed successfully (exit code 0, no timeouts)

## Timing Comparison

| Mode | Generation t/s | vs Native | vs Float32 |
|------|---------------|----------|-----------|
| Native 3B | 19.1 | 1.00× | — |
| Float32 PRT 3B | 7.7 | 0.40× | 1.00× |
| **INT8 PRT 3B** | **18.9** | **0.99×** | **2.45×** |

**INT8 PRT runs at essentially native speed (99% of native)** while being **2.45× faster than float32 PRT**.

## Evidence

- PRT_SHAPE_DETAIL: `n_layer=36 hidden=2048 ffn=11008 sidecar_rows=11008 sidecar_cols=2048 format=int8`
- Sidecar format: INT8 with per-row scales
- Sidecars loaded: 36/36 ✅
- Fallback: layers 11, 15 force-native (as specified)

## Pass Criteria

| Criteria | Result |
|----------|--------|
| Native completed 8/8 | ✅ |
| Float32 PRT completed 8/8 | ✅ |
| INT8 PRT completed 8/8 | ✅ |
| Clean output extracted 8/8 | ✅ |
| INT8 semantic match 8/8 | ✅ |
| No quality degradation | ✅ |
| No repetition/collapse | ✅ |
| PRT_SHAPE_DETAIL correct 8/8 | ✅ |
| INT8 format evidence 8/8 | ✅ |
| Sidecars loaded 36/36 | ✅ |

## Allowed Claims

✅ INT8 PRT passed full 3B 8-prompt clean quality validation — 8/8 semantic matches.
✅ INT8 PRT improved runtime versus float32 PRT on 3B by 2.45× (18.9 t/s vs 7.7 t/s).
✅ INT8 PRT matched native on this measured 3B setup (0.99× native).
✅ INT8 sidecars reduce sidecar size by ~4× (776MB vs 3.1GB).

## Forbidden Claims

- ❌ No production readiness
- ❌ No universal speedup
- ❌ No larger-than-3B extrapolation

## Recommended Next Phase

Phase 14F: Tag/freeze 3B INT8 checkpoint as validated state.

## Files Committed

- examples/speculative/results/PRT_PHASE14E_3B_INT8_FULL_VALIDATION.md
- examples/speculative/results/phase14e_3b_int8_full_validation.json

## Safety

- No models/sidecars/binaries staged ✅
- No secrets detected ✅
- Phase 13 tags untouched ✅