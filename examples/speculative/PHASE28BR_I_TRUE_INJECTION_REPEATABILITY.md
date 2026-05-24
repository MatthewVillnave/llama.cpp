# Phase 28BR-I: True Injection Repeatability

## Verdict: PASS

True injection (Test D) repeats consistently across 3 runs. All gating sanity checks (E, F, G) behave as expected.

## Test Results

| Test | Run | injection_attempts | injection_successes | injection_skipped | sidecar_math_influenced | Notes |
|------|-----|--------------------|---------------------|-------------------|------------------------|-------|
| A    | 1   | 0                  | 0                   | 0                 | 0                      | baseline, no PRT activity |
| A    | 2   | 0                  | 0                   | 0                 | 0                      | baseline |
| A    | 3   | 0                  | 0                   | 0                 | 0                      | baseline |
| B    | 1   | 0                  | 0                   | 0                 | 0                      | observe-only, no injection |
| B    | 2   | 0                  | 0                   | 0                 | 0                      | observe-only |
| B    | 3   | 0                  | 0                   | 0                 | 0                      | observe-only |
| C    | 1   | 0                  | 0                   | 0                 | 0                      | shadow apply, no injection |
| C    | 2   | 0                  | 0                   | 0                 | 0                      | shadow apply |
| C    | 3   | 0                  | 0                   | 0                 | 0                      | shadow apply |
| D    | 1   | 1                  | 1                   | 0                 | 1                      | **true injection SUCCESS** |
| D    | 2   | 1                  | 1                   | 0                 | 1                      | **true injection SUCCESS** |
| D    | 3   | 1                  | 1                   | 0                 | 1                      | **true injection SUCCESS** |
| E    | 1   | 0                  | 0                   | 1                 | 0                      | wrong layer (layer=1, target=0) |
| F    | 1   | 0                  | 0                   | 0                 | 0                      | budget=0 → all activation rejected |
| G    | 1   | —                  | —                   | —                 | —                      | missing manifest → deterministic fail |

## Key Findings

### Test D — True Injection (3/3 SUCCESS)
All three runs produced identical results:
- `injection_attempts=1, injection_successes=1, injection_failures=0`
- `contribution_finite_before_injection=1`
- `sidecar_math_influenced_output=1` (verified via PRT-PAGER-COUNTERS)
- `R=[896,896] X=[896,30] out=[896,30]` — shape consistent across all runs

### Test E — Wrong Target
Layer mismatch correctly skips injection:
- `reason=injection_skipped_wrong_layer`
- `injection_skipped=1`
- No crash, no sidecar influence

### Test F — Budget=0
Zero budget correctly rejects all activation:
- `budget_rejects=3` (attn_out + ffn_up + ffn_down blocked)
- `activation_successes=0`
- `resident_bytes=0`

### Test G — Missing Manifest
Deterministic failure with clear error:
- `manifest not found: /tmp/nonexistent/manifest.json`

## Counter Summary (Test D)

From PRT-PAGER-COUNTERS across D runs:
- `activation_attempts=1, activation_successes=1` — layer 0 activated
- `non_null_views=2, null_views=1` — attn_out decoded
- `resident_bytes=5240752, peak_resident_bytes=5240752` — stable
- `trit_validated=4, checksum_ok=4` — all .trit files valid
- `sidecar_math_influenced=1` — confirmed via counter

## Flags Used

```
--prt-mode 5700
--enable-prt-sidecar-pager
--prt-sidecar-dir /tmp/phase28bo_layer0_multi_family
--prt-sidecar-manifest /tmp/phase28bo_layer0_multi_family/manifest.json
--prt-sidecar-budget-mb 64
--prt-sidecar-apply
--prt-sidecar-apply-layer 0
--prt-sidecar-apply-family attn_out
--prt-sidecar-true-injection
```

## Phase History

- Phase 28BR-H (MethodA in-graph): PASS
- Phase 28BR-I (Repeatability): PASS ← this phase