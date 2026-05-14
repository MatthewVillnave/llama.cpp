# Phase 10D Guard Amortization Results — CORRECTED

## Status: INCONCLUSIVE — ANALYTICAL PROJECTION ONLY

**The guard-amortization C++ test did not complete (process killed).**
**All timing below is analytical projection from Phase 10B measured data.**

## Corrected Speed Math

| Path | Per-hit | Source |
|------|---------|--------|
| Float baseline | 1.38 ms | Phase 10B measured |
| PRT-only | 1.20 ms | Phase 10B measured |
| Every-hit guard | 2.64 ms | Phase 10B measured |

**Correct comparisons:**
- PRT-only vs float baseline: 1.38 / 1.20 = **1.15x** (PRT is faster)
- PRT-only vs every-hit guard: 2.64 / 1.20 = **2.20x** (PRT is faster)
- every-hit guard vs baseline: 1.38 / 2.64 = **0.52x** (guard is SLOWER than baseline!)

## Projected Mode Analysis

Based on Phase 10B measured timings:

| Mode | Per-hit (ms) | vs Baseline | vs Every-hit Guard |
|------|-------------|------------|--------------------|
| ModeE_prt_only | 1.20 | **1.15x** | **2.20x** |
| ModeB_warmup_1 | 1.23 | 1.12x | 2.15x |
| ModeC_periodic_16 | 1.18 | 1.17x | 2.24x |
| ModeD_layer_guards | ~1.10 | ~1.25x | ~2.40x |
| ModeA_every_hit | 2.64 | 0.52x | 1.00x |
| baseline_float | 1.38 | 1.00x | 0.52x |

## Allowed Claims

✅ every-hit guard is too expensive (0.52x — slower than baseline!)
✅ PRT-only is fastest projected path (1.15x vs baseline)
✅ guard amortization is likely required to achieve any speedup
✅ Phase 10B measured ffn_up PRT path was 1.15x faster than float

## Forbidden Claims

❌ Phase 10D modes were tested in C++ (process killed before completion)
❌ Mode E generation quality passed (end-to-end not tested)
❌ accept rate stayed stable (not measured)
❌ broad benchmark is justified
❌ PRT-only end-to-end speedup is proven

## Verdict: INCONCLUSIVE — ANALYTICAL PROJECTION

Guard amortization logic is sound, but actual C++ test did not run to completion.
Next step: Phase 10E narrow canary with actual end-to-end generation.