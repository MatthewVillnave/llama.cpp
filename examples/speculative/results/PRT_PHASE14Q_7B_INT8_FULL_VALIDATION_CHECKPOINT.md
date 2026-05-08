# PRT Phase 14Q — 7B INT8 Full Validation Checkpoint

## Verdict

**PASS_7B_INT8_FULL_VALIDATION** ✅

## Branch

`experimental/prt-phase14a-packed-sidecars`

## Commit

`a05915e8f`

## What this checkpoint proves

- Qwen2.5-7B INT8 PRT completed the full 8-prompt validation suite.
- Native completed 8/8.
- INT8 PRT completed 8/8.
- Clean outputs were extracted 8/8.
- Exact matches were 6/8.
- Semantic matches covered the remaining 2/8 (prompts 2 — Python code slicing approach).
- Quality degradations were 0.
- JSON prompt passed (prompt 5).
- Code prompts were plausible (prompts 2 and 6 — Timsort).
- PRT_SHAPE_DETAIL confirmed:
  - n_layer=28
  - hidden=3584
  - ffn=18944
  - format=int8
- INT8 sidecars loaded 28/28 on all runs.
- Fallback was limited to force-native layers 11 and 15.
- INT8 PRT retained near-native throughput across the 8-prompt suite:
  - native avg/median: 8.75 / 8.60 t/s
  - INT8 avg/median: 8.69 / 8.55 t/s
  - avg ratio: 0.993
  - median ratio: 0.989
  - every prompt ratio ≥ 0.976

## What this checkpoint does NOT prove

- No universal speedup claim.
- No production-readiness claim.
- No larger-than-7B generalization claim.
- No GPU comparison.
- No guarantee across all prompts/tasks.
- No exact equivalence beyond tested prompts.
- No claim outside this measured CPU setup.

## 8-prompt result table

| # | Prompt | Match | t/s ratio |
|---|--------|-------|-----------|
| 1 | "The capital of France is" | exact | 0.990 |
| 2 | "Write a Python function that reverses a list." | semantic, slicing approach | 1.012 |
| 3 | "Once upon a time in a" | exact | 1.000 |
| 4 | "Explain CPU inference in one sentence." | exact | 0.989 |
| 5 | "Return JSON with keys name and status." | exact, valid JSON | 1.000 |
| 6 | "The fastest way to sort a list in Python is" | exact, Timsort | 0.988 |
| 7 | "In two sentences, explain what RAM does." | exact | 0.976 |
| 8 | "Complete this phrase: artificial intelligence is" | exact | 0.988 |

## 3B / 7B Symmetry

This checkpoint makes the 7B result symmetric with the 3B result:

| Property | 3B INT8 | 7B INT8 |
|----------|---------|---------|
| Prompt suite | 8 prompts | 8 prompts |
| Exact matches | 6/8 | 6/8 |
| Semantic matches | 2/8 | 2/8 |
| Quality degradations | 0 | 0 |
| Sidecar coverage | 36/36 | 28/28 |
| INT8/native avg ratio | 1.000× | 0.993× |
| Fallback | layers 11, 15 only | layers 11, 15 only |
| Checkpoint tagged | ✅ `PRT_PHASE14G_3B_INT8_REPEATABILITY_CHECKPOINT` | ✅ `PRT_PHASE14Q_7B_INT8_FULL_VALIDATION_CHECKPOINT` |

Both model sizes have now been validated through the same structured suite, with the same quality profile, and both have tagged checkpoints.

## Phase 14 series summary

| Phase | Commit | Result |
|-------|--------|--------|
| 14A | a9c198950 | Design packed sidecar prototype |
| 14B | b5c26119e | INT8 runtime canary (blocked: swap) |
| 14B-R1 | b5c26119e | INT8 runtime canary rerun — PASS (94.3 vs 50.1 t/s) |
| 14C | 104863026 | 0.5B INT8 sidecar runtime (8/8 exact, 1.81× faster) |
| 14D | 3c78e6b16 | 3B INT8 single prompt canary |
| 14E | 82726efd7 | 3B INT8 sidecar runtime validation |
| 14F | c751e3e11 | **3B INT8 checkpoint** |
| 14G | cd7068c98 | 3B INT8 repeatability (10 runs) |
| 14G-tag | 0978d957e | **3B INT8 repeatability checkpoint** |
| 14H | 54a9bdf5c | 7B INT8 feasibility plan |
| 14J | 13ff1d6c7 | Diagnose 7B INT8 parity failure |
| 14K | 257a1b18c | 7B INT8 single prompt canary |
| 14L | 772654ade | 7B INT8 4-prompt validation |
| 14M | b6639a60e | 7B INT8 repeatability (10 runs) |
| 14N | 535f82f60 | **7B INT8 repeatability checkpoint** |
| 14O | 4abf5a12e | Next direction decision |
| 14P | a05915e8f | Full 8-prompt 7B INT8 validation |
| **14Q** | **a05915e8f** | **7B INT8 full validation checkpoint** |

## Allowed claims

- Qwen2.5-7B INT8 PRT passed full 8-prompt validation with 8/8 exact or semantic matches and 0 quality degradations on this measured CPU setup.
- Qwen2.5-7B INT8 PRT retained 0.993× average and 0.989× median native throughput across the measured 8-prompt suite.
- Qwen2.5-7B INT8 sidecars loaded 28/28 and fallback was limited to force-native layers 11 and 15.
- The 7B INT8 result is symmetric with the 3B INT8 result for the shared 8-prompt validation suite.

## Forbidden claims

- ❌ Do not claim universal speedup.
- ❌ Do not claim production readiness.
- ❌ Do not claim larger-than-7B success.
- ❌ Do not claim GPU comparison.
- ❌ Do not claim exact equivalence across all prompts/tasks.
- ❌ Do not claim results outside this measured CPU setup.

## Recommended next phase

**Phase 14R**: Package Phase 14 results into a claims-disciplined lab writeup.

The full 14-series validation (0.5B, 3B, 7B — float32 path, INT8 path, sidecar loading, repeatability, 8-prompt suites) is complete and frozen. The next logical step before starting new architecture work is to document the result with strict claims discipline, producing a writeup suitable for lab records or external communication.