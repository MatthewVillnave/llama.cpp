# Phase 10E-6 Acceptance Metrics

## Note
Acceptance metrics (n_accept, accept rate, baseline vs PRT comparison) could NOT be collected for full token generation due to execution time constraints. The llama-phase10e0-layer0 binary is a completion harness, not a speculative decoding harness, so speculative accept/reject metrics are not natively tracked.

## What Was Measured

### Smoke Tests (n=5)
- Exit code: 0 for all tests
- PRT replacements: 36 (prompt decode, all 36 layers)
- Fallback count: 0
- Wrong-sidecar count: 0

### Infrastructure Confirmation (from earlier longer runs)
- Total PRT replacements (full generation): 756 (36 layers × 21 activations)
- Fallback: 0
- All 36 layers confirmed active via GRAPH_SUBSTITUTION logs

## What Could Not Be Measured
- Baseline tok/s (would require running llama-completion without PRT)
- PRT tok/s (would require full generation completion)
- Accept rate (not speculative mode — this is pure completion with PRT substitution)
- Wall-clock speedup

## Required Data for Full Acceptance Metrics
To complete this section, would need:
1. Baseline run: `llama-completion` without PRT, full generation timing
2. PRT run: `llama-phase10e0-layer0` with all 36 layers, full generation timing
3. Speculative mode: both runs with `-sp` flag for accept rate tracking