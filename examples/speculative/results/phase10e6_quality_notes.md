# Phase 10E-6 Quality Notes

## Scope
Narrow all-layer active PRT generation canary — ffn_up only, all 36 layers, no qkv/ffn_down/attn_out/logits/norms.

## Test Results

### Smoke Tests (n=5)
- All 3 prompts: EXIT 0, clean
- No crashes
- No repetition loops
- PRT replacements: 36 (one per layer, prompt decode)
- Fallbacks: 0

### Longer Tests
- Tests with n≥10 consistently exceeded available execution time budget
- Exit codes were 0 when completed (no crash)
- Earlier "Hello world" test with n=10: completed successfully with 396 PRT replacements, 0 fallbacks

## Quality Assessment
- **Infrastructure: PASS** — all 36 layers load sidecars, all layers activate PRT, zero wrong-sidecar events, zero fallbacks
- **Generation survival: PASS** — no crashes, clean exit codes across multiple runs
- **Full quality comparison: INCOMPLETE** — longer generation tests required for output comparison against baseline

## Generation Coherence
Short PRT runs produce coherent output (no crash, no repetition). Longer runs could not be captured in available execution budget.

## Known Limitation
Accept rate, tok/s, and detailed timing comparison require full generation runs that exceed the execution time budget on this hardware.