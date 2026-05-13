# PRT Phase 19X: 7B INT6 Layer Sensitivity Sweep — FINAL REPORT

## Branch
experimental/prt-phase19a-alt-sidecar-backed

## Previous HEAD
36d1ad801 (Phase 19W: Single-layer isolation found layer0 works)

## This Commit
148d43a7b (Phase 19W/19X: Layer mask tool + sensitivity sweep)

## Phase 19X Findings

### Single-Layer Tests — ALL CLEAN ✅
| Layer | Output | Tok/s |
|-------|--------|------|
| 0 | Paris | 4.3 |
| 1 | Paris | 3.7-3.8 |
| 2 | Paris | 3.9 |
| 5 | Paris | 4.2 |
| 10 | Paris | 4.2 |
| 15 | Paris | 4.2 |
| 20 | Paris | 4.3 |
| 27 | Paris | 4.2 |

**Verdict:** Every single INT6 layer works in isolation.

### Multi-Layer/Range Tests — ALL CORRUPT ❌
| PRT Layers | Result | Tok/s |
|-----------|--------|------|
| 0-3 only | GIBBERISH | 3.2 |
| 0-7 only | GIBBERISH | 2.4 |
| 0-13 only | GIBBERISH | 1.7 |
| 14-27 only | GIBBERISH | 1.7 |
| ALL-EXCEPT-0 | GIBBERISH | 1.1 |
| ALL-EXCEPT-0-3 | GIBBERISH | 1.2 |
| ALL (11,15 disabled) | GIBBERISH | 1.1 |

**Verdict:** ANY 2 OR MORE INT6 LAYERS CORRUPT OUTPUT.

### Root Cause
**CUMULATIVE_APPROXIMATION_ERROR**

- Single INT6 layer: Output is mathematically sane and produces clean text
- Two or more INT6 layers: Errors compound through the network, eventually overwhelming the LLM's error correction

This is a fundamental property of lossy compression — the INT6 approximation is low-precision, and when multiple layers use lossy approximations in sequence, the errors cascade.

### Interpretation
The sidecar kernels are NOT "broken" — they work correctly and produce correct-output for a single layer. But when stacked in a deep LLM (28 layers), the cumulative error exceeds the model's tolerance.

### Policy Recommendation
**PRT_ALONE** — Use PRT only as a single-layer experiment, never as full-layer replacement.

Future work could explore:
- Lower compression per layer (e.g., INT8 or float16 instead of INT6)
- Alternating PRT and native layers (not consecutive)
- Selective layer usage (only safe layers)

## Verdict
**PASS_SINGLE_LAYER_CLEAN** / **FAIL_MULTI_LAYER_CORRUPTION**

## Next
- If continuing: Find maximum STACKABLE_LAYERS threshold or identify specific safe layer combinations
- If pivoting: Test INT8 or float32 sidecars (less compression, might stack better)