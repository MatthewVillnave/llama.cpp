# Phase 10D PRT-Only Canary

## Mode E: PRT-Only (No Guard)

| Metric | Value |
|--------|-------|
| Per-hit time | 1185492.9 μs |
| Speedup vs every-hit guard | 0.000x |
| Speedup vs float baseline | 0.000x |
| Guard checks | 0 (no guard) |
| Local cosine | N/A (no comparison) |
| End-to-end generation | NOT TESTED |

## Canary Status

**Local matmul test:** PASS (no crashes)

**End-to-end generation:** NOT TESTED in this phase.

Mode E is the fastest option but has no safety net.
Next step: run Phase 10E end-to-end generation with Mode E.
If output quality degrades or acceptance collapses, fall back to Mode C or D.
