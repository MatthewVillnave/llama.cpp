# PRT Phase 24K: State Forensics & Audit

## Date
2026-05-20 15:12+

## Branch
experimental/prt-phase19a-alt-sidecar-backed

## HEAD
efbca4b8a8e910f1c5c093ae59226ce0fa4335d0

## Latest Checkpoint Tag
PRT_PHASE24H_CANONICAL_INT8_LAYOUT_CHECKPOINT

## Commits Since Phase 24H

| Commit | Phase | Change |
|--------|-------|--------|
| 66415b6d6 | 24I | Add 0.5B selector |
| c6e3923ef | 24H | Checkpoint canonical layout (tagged) |
| d945f9234 | 24G-R7 | Regen 7B canonical sidecar |
| efbca4b8a | 24J | 3B repeat validation (latest) |

Changes:
- src/llama-graph.cpp: +8 LOC (selectors)
- Reports: +127 lines (documentation)

## Source Changes Summary

### src/llama-graph.cpp
- Added K=896 M=4864 (0.5B) selector
- Added K=3584 M=18944 (7B) canonical redirect
- No kernel math changes
- No decoder math changes

Risk Level: LOW (selector only)

## Canonical INT8 Layout Ledger

**Format (canonical):**
- scales first (M float32)
- int8 data second (K*M int8)
- q index: q(k*M + j]
- scale[j]
- decode: W[k,j] = q[k*M+j] * scale[j]
- kernel: Y[j,n] = sum_k X[k,n] * W[k,j]

## Model Status Matrix

| Model | INT8 Sidecar | Canonical | Selector | Pointer | Op Runs | Output | Repeat | Timing |
|-------|--------------|-----------|----------|--------|--------|--------|--------|--------|
| **0.5B INT8** | exists (old) | PARTIAL | YES | non-null | YES | garbled | NOT TESTED | BLOCKED |
| **0.5B INT6** | exists | YES | YES | non-null | YES | Paris | NOT TESTED | BLOCKED |
| **3B INT8** | canonical | PASS | YES | non-null | YES | Paris/Parisyne | PASS 3/3 | BLOCKED |
| **3B INT6** | N/A | N/A | fallback | non-null | YES | Paris | NOT TESTED | BLOCKED |
| **7B INT8** | canonical | PASS | YES | non-null | YES | Parisian | NOT TESTED | BLOCKED |
| **7B INT6** | valid | YES | fallback | non-null | YES | Paris | NOT TESTED | BLOCKED |

## Validated Claims

✓ Canonical INT8 layout defined
✓ Decoder index mismatch (k+j*K → k*M+j) found and fixed in Phase 24G-R5
✓ 3B INT8 canonical layer0 canary passes
✓ 7B INT8 canonical layer0 canary passes (after regen)
✓ 3B repeat validation passes (3/3 stable runs)
✓ 7B INT6 semantic baseline valid

## Partial / Pending

⚠ 0.5B INT8 canonical: PARTIAL - selector works but old sidecar incompatible

## Forbidden Claims

NOT ALLOWED:
- No timing ratio or speedup claim
- No production readiness
- No multi-layer support
- No 14B support
- No broad semantic equivalence claim
- No exact-token match claim
- No 0.5B INT8 canonical validated unless explicitly re-tested

## Known Risks

- Mixed old sidecar layouts exist in different directories
- 0.5B INT8 old sidecar uses inconsistent normalization
- 7B timing unreliable on current hardware (blocked)
- Large model inference causes memory pressure

## Do Not Touch

Without explicit phase approval:
- Canonical decoder indexing (k*M+j formula)
- Scales-first loader
- 3B canonical path
- 7B canonical path
- INT6 scale_off=20 behavior
- Sidecar selector fail-clean
- PRT_V2_QUIET / bounded runner

## Verified Regression Baseline

Before any risky code changes, run minimum:

1. llm-completion -m 3B --no-conv -c 4 -n 2 PRT_V2_SIDECAR=int8
2. Verify sidecar selected
3. Verify output Paris-like

## Recommendations

- Phase 24L: 3B timing smoke (SMALL if approved)
- Separate: 0.5B canonical if required
- No timing until explicitly approved

## Verdict
PASS_STATE_FORENSICS_COMPLETE
