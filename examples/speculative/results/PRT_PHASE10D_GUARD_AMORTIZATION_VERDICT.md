# PRT_PHASE10D_GUARD_AMORTIZATION_VERDICT

## Status: INCONCLUSIVE — ANALYTICAL PROJECTION ONLY

**NOTE:** The guard-amortization test did NOT complete. The naive C++ matmul
was too slow (Process was killed after several minutes). All timing results
below are analytical projections based on Phase 10B measured timings.

## Measured Data (Phase 10B)

| Path | Per-hit | Source |
|------|---------|--------|
| Float baseline | 1.38 ms | Phase 10B measured |
| PRT-only | 1.20 ms | Phase 10B measured |
| Every-hit guard | 2.64 ms | Phase 10B measured |

## Correct Speed Calculations

| Comparison | Calculation | Result |
|-----------|-------------|--------|
| PRT-only vs float baseline | 1.38 / 1.20 | **1.15x** |
| PRT-only vs every-hit guard | 2.64 / 1.20 | **2.20x** |

**Previous claim of "2.08x vs baseline" was INCORRECT. Fixed now.**

## Allowed Claims

✅ every-hit guard is too expensive (2.64ms vs 1.38ms baseline)
✅ PRT-only remains the fastest projected path (1.15x vs baseline)
✅ guard amortization is required to beat float baseline
✅ Phase 10B measured ffn_up PRT path was 1.15x faster than float

## Forbidden Claims (DO NOT MAKE)

❌ Phase 10D modes were tested
❌ Mode E generation quality passed
❌ accept rate stayed stable
❌ broad benchmark is justified
❌ PRT-only end-to-end speedup is proven

## Next Valid Step: Phase 10E

Narrow PRT-only end-to-end canary:
- temp 0, seed 42, short prompt
- baseline run first, PRT-only run second
- ffn_up only, fragile layers avoided
- Measure: output coherence, accept rate, latency, tokens/sec
- Do NOT claim speedup until wall-clock improves