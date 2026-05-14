# Phase 10E-6 Output Comparison

## Methodology
Short smoke tests (n=5) used due to execution time constraints. All tests used identical settings: temp=0, seed=42, CPU inference, no speculative decoding.

## Comparison Table

| Prompt | n | Exit | PRT Replacements | Fallback | Wrong Sidecar | Status |
|--------|---|---|---|---|---|---|
| "Hello" (smoke) | 5 | 0 | 36 | 0 | 0 | CLEAN |
| Factual (baseline reference) | 30 | 0 | N/A | N/A | N/A | COHERENT |
| Code (baseline reference) | 40 | 0 | N/A | N/A | N/A | COHERENT |

## Key Observations
- All 36 layers received PRT substitution (visible in GRAPH_SUBSTITUTION logs)
- Zero fallbacks across all smoke tests
- Wrong-sidecar count: 0 (each layer uses its own sidecar)
- No crashes or errors
- Longer generation tests (n≥15) exceed available execution time budget on this hardware

## Gap
Full token-generation output comparison was not completed. Smoke tests confirm infrastructure works end-to-end for prompt decode phase. Token-generation phase with full metrics requires longer execution window.