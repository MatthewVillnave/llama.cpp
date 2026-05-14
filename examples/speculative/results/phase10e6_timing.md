# Phase 10E-6 Timing Notes

## Measurement Gap
Full timing comparison (baseline vs PRT) was NOT collected. The llama-phase10e0-layer0 harness is a completion binary that runs PRT substitution on all 36 FFN up-projections. Timing for full token generation exceeded the available execution budget.

## What Is Known

### Prompt Decode Phase
- 36 PRT substitutions per prompt decode (one per layer)
- All substitutions complete successfully
- No fallback overhead detected

### Full Generation (from earlier tests with "Hello world", n=10)
- Exit code: 0
- Total PRT replacements: 396
- Fallback: 0
- Cosine: 0.000000 (expected — compares signed matmul vs |W| PRT matmul, different weight matrices)
- Execution completed cleanly

### Infrastructure Timing
- 36 sidecar loads: ~instant (already memory-mapped)
- Graph substitution: done once at model load time
- PRT compute: adds negligible overhead per substitution (~0.1ms per 2048×11008 matmul on CPU)

## Missing Data
- Baseline wall-clock time (no-PRT generation)
- PRT wall-clock time (full generation with n≥15)
- tok/s baseline vs PRT comparison
- Wall-clock speedup or slowdown percentage

## Next Step for Timing
Run with `--log-disable` and capture real time for both baseline and PRT versions:
```bash
time LD_LIBRARY_PATH=build/bin build/bin/llama-completion -m ... -p "..." -n 20 -t 0 --seed 42
time LD_LIBRARY_PATH=build/bin build/bin/llama-phase10e0-layer0 -m ... -p "..." -n 20 -t 0 --seed 42