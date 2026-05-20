# PRT Phase 24L: 3B INT8 Timing Smoke

## Date
2026-05-20 16:17+ (updated 17:08)

## Branch
experimental/prt-phase19a-alt-sidecar-backed

## HEAD
7a028cebf (pre-restart)

## Timing Results (3 Native + 3 PRT, interleaved)

### Native Runs
| Run | Real  | User  | Sys   |
|-----|-------|-------|-------|
| 1   | 2.35s | 2.05s | 0.30s |
| 2   | 2.45s | 2.08s | 0.37s |
| 3   | 2.39s | 2.03s | 0.35s |
| Avg | 2.40s |       |       |

### PRT INT8 Runs
| Run | Real  | User  | Sys   |
|-----|-------|-------|-------|
| 1   | 3.45s | 2.81s | 0.64s |
| 2   | 3.45s | 2.78s | 0.66s |
| 3   | 3.43s | 2.76s | 0.66s |
| Avg | 3.44s |       |       |

### Timing Ratio
- **Native avg: 2.40s**
- **PRT avg: 3.44s**
- **Ratio: PRT 1.43x SLOWER** (consistent across all 3 runs)

### Variance
- Very low in both paths (within ~0.1s)
- PRT is consistently slower by ~1.0s absolute

### Interpretation
- PRT custom-op overhead is real and consistent at c=4 n=8
- This is layer0-only, c=4 small-context smoke
- Custom-op decode overhead ~1.0s dominates at tiny token count
- Larger context (c=32+) may show different ratio

## Verdict
PARTIAL_PRT_SLOWER_IN_NARROW_SMOKE - PRT slower at c=4

## Allowed Claims
- 3B layer0 PRT INT8 timing smoke captured
- Narrow smoke shows PRT ~1.43x slower than native at c=4 n=8

## Forbidden Claims
- No production speedup
- No broad performance claims
- No 7B timing
- No multi-layer claim

## Recommended Next
Phase 24M — profile custom-op overhead: sidecar decode/load, graph overhead vs native fallback