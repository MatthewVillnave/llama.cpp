# PRT Phase 12E: L12+L15 vs L11+L15 Policy Comparison

## Speed Comparison
| Metric | L12+L15 (current) | L11+L15 (candidate) |
|--------|-------------------|--------------------|
| Average speedup | 1.7869x | 1.8216x |
| Median speedup | 1.7942x | 1.8030x |
| Min speedup | 1.5104x | 1.6489x |
| Max speedup | 1.8372x | 2.0551x |
| Delta | — | +0.0347 (+1.94%) |

## Quality Comparison
| Metric | L12+L15 | L11+L15 |
|--------|---------|---------|
| Token-0 match rate | 20/22 (90.9%) | 20/22 (90.9%) |
| Severe corruption | 8/24 | 9/24 |
| JSON validity (p13-16) | 0/4 | 0/4 |

## Counter Cleanliness
- **callback_overwrites**: L12=0/24 ✓, L11=0/24 ✓ (PASS — both clean)
- **native_fallback_calls**: L12=22/22 with 16 calls each (expected warmup), L11=same (PASS)

## Recommendation
**CAUTION** — L11+L15 faster by 1.94% but more corruption (9 vs 8)
