# Phase 28BR-K — Token / Logit Delta Numerical Capture

## Verdict: PASS

True injection produces a **repeatable numerical token ID delta** vs all other modes.

## Setup
- **Model:** Qwen2.5-0.5B-Instruct-Q4_K_M.gguf
- **Prompt:** "Hi"
- **n_predict:** 1
- **Temperature:** 0
- **seed:** 42 (fixed)
- **Instrumentation:** `common/sampling.c` — added `[TOKEN] id=%d` printf after `llama_sampling_sample()`
- **Branch:** experimental/prt-phase19a-alt-sidecar-backed

## Token ID Results

| Mode | Token ID | sidecar_math_influenced | Notes |
|------|----------|-------------------------|-------|
| A (baseline) | **9707** | 0 | clean native |
| B (observe) | **9707** | 0 | pager active, no injection |
| C (shadow) | **9707** | 0 | apply ran, no injection |
| D (true injection) | **374** | 1 | **graph mutated, token changed** |
| E (wrong target) | **9707** | 0 | layer=1, injection_skipped=1 |
| F (budget=0) | **9707** | 0 | budget_rejects=5, no activation |

## Token Delta Summary

```
A vs D:  9707 → 374   DELTA ⚡
B vs D:  9707 → 374   DELTA ⚡
C vs D:  9707 → 374   DELTA ⚡
E vs D:  9707 → 374   DELTA ⚡
A vs B:  9707 = 9707  SAME
A vs C:  9707 = 9707  SAME
E vs A:  9707 = 9707  SAME  (control confirms isolation)
```

**True injection changed the token from 9707 to 374. All other modes produced token 9707.**

## Control Results

### E — Wrong target (layer=1)
```
injection_skipped=1 reason=injection_skipped_wrong_layer
sidecar_math_influenced_output=0
Token: 9707 (same as baseline)
```
✅ As expected — wrong layer, no injection, token matches baseline.

### F — Budget=0
```
budget_rejects=5 activation_successes=0 resident_bytes=0
Token: 9707 (same as baseline)
```
✅ As expected — all activations rejected, no sidecar influence, token matches baseline.

## Proven

- True injection produces **repeatable numerical token ID delta** (9707→374)
- Delta is consistent across runs
- Baseline/observe/shadow/wrong-target/budget-zero all produce token 9707 (isolated)
- Token capture via `[TOKEN] id=%d` printf works cleanly

## Not Proven

- Output correctness or quality parity
- Logit numerical values (printf only captures token ID, not logit)
- Speedup or latency improvement
- Q2→Q4 recovery or accuracy restoration
- Long generation stability
- FFN or non-square tensor support
- Multi-layer or multi-family injection
- 30B or larger model feasibility
- Production readiness
- Exact numerical logits

## Files Changed

- `common/sampling.c` — added token ID printf (1 line)
- No other source changes

## Next Phase Recommendation

Phase 28BR-L: Capture logits alongside token IDs. Add `[LOGITS] top_token=%d top_logit=%.4f` to capture the chosen token's logit value. This enables logit delta measurement, not just token ID delta.
