# Phase 30F — Active Sidecar Residency Rerun

## Classification: ADDITIVE_OVERHEAD_CONFIRMED_ACTIVE

## Context
- Follows Phase 30E flag/pager init fix (commit a9a50e4b7 on experimental/prt-phase19a-alt-sidecar-backed)
- Prompt: phase30d_request (delegated from Matt, 2026-05-28)

## What was tested
16 test runs across Q4, Q2, Q3 baselines vs active sidecars:
- A: Baseline (no sidecars) — Q4, Q2, Q3
- B: Q4 + layer0 sidecars (true injection) — attn_out, ffn_up, ffn_down
- C: Q2 + layer0 sidecars (true injection) — attn_out, ffn_up, ffn_down
- D: Q3 + layer0 sidecars (true injection) — attn_out, ffn_up, ffn_down
- E: Budget controls (Q4 + attn_out, budget=0/1/8/512)

## Results table

| Test | RSS KB | Δ vs Baseline | Activated | Influenced |
|------|--------|--------------|-----------|------------|
| A_Q4_baseline | 967,332 | — | — | — |
| A_Q2_baseline | 1,001,780 | — | — | — |
| A_Q3_baseline | 1,062,548 | — | — | — |
| B_Q4_plus_attn_out | 1,022,780 | +55,448 (+5.7%) | 1 | 0 |
| B_Q4_plus_ffn_up | 1,245,052 | +277,720 (+28.7%) | 1 | 1 |
| B_Q4_plus_ffn_down | 1,244,948 | +277,616 (+28.7%) | 1 | 1 |
| C_Q2_plus_attn_out | 1,057,216 | +55,436 (+5.5%) | 1 | 0 |
| C_Q2_plus_ffn_up | 1,279,376 | +277,596 (+27.7%) | 1 | 1 |
| C_Q2_plus_ffn_down | 1,279,300 | +277,520 (+27.7%) | 1 | 1 |
| D_Q3_plus_attn_out | 1,118,228 | +55,680 (+5.2%) | 1 | 0 |
| D_Q3_plus_ffn_up | 1,340,332 | +277,784 (+26.1%) | 1 | 1 |
| D_Q3_plus_ffn_down | 1,340,476 | +277,928 (+26.2%) | 1 | 1 |
| E_Q4_attn_budget0 | 967,444 | +112 (+0.0%) | 0 | 0 |
| E_Q4_attn_budget1 | 967,444 | +112 (+0.0%) | 0 | 0 |
| E_Q4_attn_budget8 | 1,022,876 | +55,544 (+5.7%) | 1 | 0 |
| E_Q4_attn_budget512 | 1,022,764 | +55,432 (+5.7%) | 1 | 0 |

All baselines: Q4=947 MB, Q2=978 MB, Q3=1,038 MB.

## Key Findings

1. **Active sidecar overhead is additive and substantial**: ffn_up/ffn_down sidecars add ~277 MB (+27-29%) to any base. attn_out adds ~55 MB (+5.5%).

2. **Sidecar overhead is base-agnostic**: The ~277 MB FFN overhead does not scale with the base model. Q4+ffn_up ≈ Q2+ffn_up ≈ Q3+ffn_up (all ~1.24-1.28 GB). This suggests the overhead is the full residual tensor size regardless of base quant precision.

3. **attn_out is lightweight but non-contributing**: attn_out activates but produces 0 influenced outputs (no math change). The ~55 MB may be loading but not participating.

4. **Budget filter works**: budget=0 and budget=1 both correctly block injection (0 activations, budget_rejects=75). budget=8 and above allow activation.

5. **Q2/Q3 sidecar RSS always higher than Q4 baseline**: Q4_baseline=967 MB. Q2+ffn_up=1,279 MB. Q3+ffn_up=1,340 MB. Active sidecars do not beat Q4 baseline on this setup.

6. **true_injection flag works**: [PRT-TRUE-INJECTION] confirmed in telemetry for all active injection tests.

## Classification Rationale
ADDITIVE_OVERHEAD_CONFIRMED_ACTIVE: All sidecar activations succeed, true injection fires, but RSS increases on all bases. The path is live and working, not blocked. Overhead is large (+27%) for FFN-type sidecars and fails to achieve memory parity vs Q4 baseline.

## What this does NOT claim
- No quality claim
- No correctness claim
- No speedup claim
- No Q2-to-Q4 recovery claim
- Not a global SDI verdict
- Not production readiness

## Next Steps (from Phase 30F spec)
- Phase 30G: Budget-optimal policy search
- Phase 30H: Quality delta measurement under active sidecar injection
- Revisit family-filter bug (ffn_up/ffn_down load unwanted sidecar entries)

## Artifacts
- phase30f_active_residency_harness.py
- phase30f_active_sidecar_residency_rerun.json
