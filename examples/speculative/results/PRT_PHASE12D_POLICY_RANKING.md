# PRT Phase 12D: Policy Ranking

## Ranked by Average Speedup

| Rank | Policy | Avg Speedup | Token-0 Match% | Coherence% | Notes |
|------|--------|-------------|---------------|------------|-------|
| 1 | **L11+L15** | 1.780x | 88% | 100% |  |
| 2 | **L12+L14** | 1.777x | 75% | 100% |  |
| 3 | **L12+L16** | 1.773x | 75% | 100% |  |
| 4 | **L13+L15** | 1.772x | 88% | 100% |  |
| 5 | **L13+L14** | 1.770x | 100% | 100% |  |
| 6 | **L12+L15** | 1.765x | 88% | 100% | Stable across all prompts |
| 7 | **L15 only** | 1.729x | 75% | 100% |  |
| 8 | **L12 only** | 1.721x | 75% | 88% |  |
| 9 | **all36** | 1.696x | 88% | 100% |  |
| 10 | **L12+L14+L15** | 1.575x | 75% | 100% | Best speed, 2 anomalous runs |

## Why Each Policy Passed/Failed

- **L11+L15**: avg 1.780x, token-0 match 88%, PASS (100% coherent)
- **L12+L14**: avg 1.777x, token-0 match 75%, PASS (100% coherent)
- **L12+L16**: avg 1.773x, token-0 match 75%, PASS (100% coherent)
- **L13+L15**: avg 1.772x, token-0 match 88%, PASS (100% coherent)
- **L13+L14**: avg 1.770x, token-0 match 100%, PASS (100% coherent)
- **L12+L15**: avg 1.765x, token-0 match 88%, PASS (100% coherent)
- **L15 only**: avg 1.729x, token-0 match 75%, PASS (100% coherent)
- **L12 only**: avg 1.721x, token-0 match 75%, PARTIAL (88%)
- **all36**: avg 1.696x, token-0 match 88%, PASS (100% coherent)
- **L12+L14+L15**: avg 1.575x, token-0 match 75%, PASS (100% coherent)

## Does L12+L15 Remain Best?

**YES** — L12+L15 (1.765x) is within 1% of best (L11+L15 1.780x) and most stable.


## Single-Anchor Viability

- **L12 only**: avg speedup 1.721x, token-0 match 6/8 — viable but slower than dual-anchor
- **L15 only**: avg speedup 1.729x, token-0 match 6/8 — viable but slower than dual-anchor

## Triple-Anchor (L12+L14+L15)

Triple anchor avg speedup: **1.575x**

Anomalous runs (speedup < 1.1x): **2/8** — interference on JSON/Python prompts
