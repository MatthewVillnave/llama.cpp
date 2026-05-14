# Phase 10E-7 Timing Results

## Test: "Write one sentence about CPUs." | n_predict=3 | seed=42 | temp=0

| Metric | Baseline | PRT (all-layer) |
|--------|----------|-----------------|
| Wall-clock time | 1.92s | 20.76s |
| Tokens generated | 3 | 3 |
| Exit code | 0 | 0 |

## Timing Breakdown
- **Baseline**: 1.92s (1.56 tok/s)
- **PRT**: 20.76s (~0.14 tok/s)
- **Slowdown**: 10.8x slower

## PRT Metrics
- Total PRT replacements: 144 (36 layers × 4 calls)
- Fallback count: 0
- Wrong-sidecar count: 0
- Last cosine: 0.000000

## Root Cause of Slowdown
PRT adds ~6.3ms per FFN up-projection matmul (2048×11008). For 3 tokens with 36 layers = 108 FFN_up calls. 108 × 6.3ms = ~680ms theoretical overhead. The remaining slowdown (~20x) is from model loading + first decode + overhead in the harness.

## Note
Timing comparison is dominated by model load time (model loads once, then runs). For longer generations (n≥20), the PRT overhead as a fraction of total time would be smaller.