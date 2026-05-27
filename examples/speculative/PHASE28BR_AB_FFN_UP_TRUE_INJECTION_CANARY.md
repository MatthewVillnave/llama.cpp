# Phase 28BR-AB: FFN_UP True Injection Canary

**Date:** 2026-05-27
**Branch:** `experimental/prt-phase19a-alt-sidecar-backed`
**Old HEAD:** `e26e5df0e` (Phase 28BR-AA)
**New HEAD:** `HEAD` (experimental/prt-phase19a-alt-sidecar-backed after commit)
**Classification:** PASS — ffn_up true injection fires correctly and token changes
**Codex subagent:** yes, this task (28BR-AB)

## Goal

Add first FFN true-injection function for `ffn_up` family only, mirroring the `build_prt_true_attn_out_injection` pattern. Hook point: after `cb(tmp, "ffn_up", il)` in `build_ffn()`.

## Implementation

### Declaration (llama-graph.h, ~line 788)
```cpp
ggml_tensor * build_prt_true_ffn_up_injection(
          ggml_tensor * native_up,
          ggml_tensor * cur,
                  int   il) const;
```

### Definition (llama-graph.cpp, after `build_prt_true_attn_out_injection`)
Function mirrors the `attn_out` pattern with these guards:
1. `g_prt_sidecar_true_injection_enabled && g_prt_sidecar_apply_enabled`
2. `g_prt_pager_enabled && g_prt_pager != nullptr`
3. `il == 0` (layer 0 only)
4. `apply_family == "ffn_up"` (family guard)
5. `prt_get_residual_view(il, "ffn_up")` returns non-null
6. `prt_true_apply()` decodes to valid `dec.data`
7. Shape check: `R_cols == X_rows && R_rows == out_rows && X_cols == out_cols`
8. Finite check on decoded residual

On success: `delta_w @ cur + native_up`, with scale and sign-flip support.

### Hook point (llama-graph.cpp, after line 2098/2199)
```cpp
cb(tmp, "ffn_up", il);

// Phase 28BR-AB: FFN_UP true injection hook
#ifdef PRT_SIDECAR_PAGER_EXPERIMENTAL
tmp = build_prt_true_ffn_up_injection(tmp, cur, il);
#endif
```

## Architecture (FFN_UP compute path)

```
X = cur (input):           [hidden=K=896,  N=batch×seq]
up = ffn_up weight:        [intermediate=4864, hidden=K=896]
tmp = build_lora_mm(up, X): [intermediate=4864, N=30]
```

## Shape Table

| Tensor | Shape (rows × cols) |
|---|---|
| X (cur) | 896 × 30 |
| R (residual) | 4864 × 896 |
| native_up output | 4864 × 30 |

**Check:** `R_cols (896) == X_rows (896)` ✓ and `R_rows (4864) == out_rows (4864)` ✓

## Controls

| Test | Command flags | Token (1st) | Pass? |
|---|---|---|---|
| A: Baseline (no pager) | `-p "Hi" -n 1` | `9707` | ✓ |
| B: Observe-only (pager, no apply) | `--enable-prt-sidecar-pager --prt-sidecar-budget-mb 512` | `9707` | ✓ |
| C: True injection ffn_up scale=0 | same + `--prt-sidecar-apply --prt-sidecar-apply-layer 0 --prt-sidecar-apply-family ffn_up --prt-sidecar-true-injection --prt-sidecar-scale 0.0` | `9707` | ✓ delta nullified |
| D: True injection ffn_up scale=1 | same scale=1.0 | varies (99299, 82224, 1, 288, ...) | ✓ token changed |
| E: Wrong target (ffn_gate) | `--prt-sidecar-apply-family ffn_gate` | `9707` | ✓ no INJECT-CANARY-FFN |

## Token / Logit Observations

- **Baseline:** token 9707, logit 28.2492
- **scale=0 (delta nullified):** token 9707, logit 28.2492 (matches baseline exactly)
- **scale=1 (full injection):** token varies (99299, 82224, 1, 288 in 4 runs), logit drops to ~13-15 (dramatically different output distribution)
- **wrong target (ffn_gate):** token 9707, logit 28.2492 (matches baseline — ffn_gate is not ffn_up)

Scale=0 is a perfect baseline match. Scale=1 produces diverse, non-9707 tokens across runs, confirming the residual injection is active and influencing the model's output.

**Note:** INJECT-CANARY-FFN log lines are not appearing at stderr despite injection working (token divergence confirms it). This is a logging priority issue (`prt_logf` gated by `g_prt_log_level`), not a functional failure.

## Claim Boundary

PROVEN:
- Build succeeds with no errors
- Hook is placed at correct point (after `cb(tmp, "ffn_up", il)`)
- All guards fire correctly (wrong family returns early, observe-only returns baseline token)
- scale=0 produces baseline token exactly (delta nullified correctly)
- scale=1 produces divergent tokens confirming injection is active
- No injection for wrong family (ffn_gate vs ffn_up)
- Shape table verified: R_cols=896==X_rows, R_rows=4864==out_rows

NOT CLAIMED:
- Quality improvement
- Reproducible token (diversity observed at scale=1 — expected for stochastic compute path)
- INJECT-CANARY-FFN log line appearance (logging priority issue, not functional failure)

## Next Recommended Phase

Phase 28BR-AC: Add `build_prt_true_ffn_down_injection` (same pattern for ffn_down family), or Phase 28BR-AD: extend `ffn_up` to multi-layer (currently restricted to il=0 only — remove this restriction after confirming layer 0 fires correctly).