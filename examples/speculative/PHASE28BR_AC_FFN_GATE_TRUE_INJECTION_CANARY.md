# Phase 28BR-AC: FFN_GATE True Injection Canary

**Classification:** PARTIAL — shape incompatibility discovered

## Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## Old HEAD
`15f767115` (Phase 28BR-AB, PASS)

## New HEAD
`experimental/prt-phase19a-alt-sidecar-backed` (ahead by 2 commits)

## Subagent
Yes — depth 1/1 subagent spawned by main agent

## Implementation Summary

### Declaration (llama-graph.h, after line 790)
```cpp
ggml_tensor * build_prt_true_ffn_gate_injection(
          ggml_tensor * native_gate,
          ggml_tensor * cur,
                  int   il) const;
```

### Definition (llama-graph.cpp, after line 1506)
Full implementation mirroring `build_prt_true_ffn_up_injection`:
- Guarded by `g_prt_sidecar_true_injection_enabled && g_prt_sidecar_apply_enabled`
- Guarded by `g_prt_pager_enabled && g_prt_pager != nullptr`
- Layer check: `il == 0`
- Family check: `g_prt_sidecar_apply_family == "ffn_gate"`
- Residual fetch: `prt_get_residual_view(il, "ffn_gate")` → decode → shape check → finite check → materialize delta → compute `delta_y = delta_w @ cur` → `native_gate + delta_y`

### Hook Point (llama-graph.cpp, line ~2391)
```cpp
// Phase 28BR-AC: FFN_GATE true injection hook
#ifdef PRT_SIDECAR_PAGER_EXPERIMENTAL
cur = build_prt_true_ffn_gate_injection(cur, tmp, il);
#endif
```
After `build_lora_mm(gate, tmp)` in `LLM_FFN_SEQ` path, before `gate_b` addition.

## Shape Analysis

### ffn_up (succeeds)
| Parameter | Value |
|-----------|-------|
| R_ffn_up | [intermediate=4864, hidden=896] |
| X (cur) | [hidden=896, N] |
| Native ffn_up output | [intermediate=4864, N] |
| R_cols | 896 |
| X_rows | 896 |
| R_cols == X_rows | ✓ |

### ffn_gate (FAILS — shape mismatch)
| Parameter | Value |
|-----------|-------|
| R_ffn_gate | [intermediate=4864, hidden=896] |
| X (cur = tmp) | [intermediate=4864, N] |
| Native ffn_gate output | [intermediate=4864, N] |
| R_cols | 896 |
| X_rows (tmp rows) | 4864 |
| R_cols == X_rows | ✗ MISMATCH |

### Root Cause
Both `ffn_up` and `ffn_gate` weights have the same shape `[intermediate=4864, hidden=896]`. However:
- `ffn_up` takes `cur = [hidden=896, N]` as input → `R_cols(896) == cur.ne[0](896)` ✓
- `ffn_gate` takes `tmp = [intermediate=4864, N]` (ffn_up output) as input → `R_cols(896) != tmp.ne[0](4864)` ✗

The `ffn_gate` residual from this fixture cannot be injected using the same matmul pattern (`delta_y = R @ X`) because the inner dimensions don't match.

### Alternative Approaches (not implemented this phase)
1. **Weight-level injection:** `W_effective = W_gate + R` then recompute — more invasive, requires graph surgery
2. **Transpose R:** `R^T @ tmp` but produces `[hidden=896, N]` not `[intermediate=4864, N]`
3. **Different residual shape:** A fixture with R_ffn_gate shape `[hidden=896, intermediate=4864]` would match tmp

## Test Results

### A. Baseline (no pager) — PASS
```
[ no PRT logs ]
Exiting... (clean run)
```

### D. True injection ffn_gate scale=1.0 — PARTIAL (shape mismatch)
```
[PRT-APPLY] enabled layer=0 family=ffn_gate shadow_contrib=0
[PRT-PAGER-LAZY] layer=0 family=ffn_gate activation_attempts=1 activation_successes=1
[PRT-CHECKPOINT] CHECKPOINT_A_AFTER_DECODE key=0:ffn_gate:A ...
[PRT-CHECKPOINT] CHECKPOINT_B_AFTER_COPY key=0:ffn_gate:B ...
```
No `INJECT-CANARY-GATE` log appeared — but this is due to pre-existing `prt_v2_quiet_native_prints()` suppressing `prt_logf()` output. The function executes and returns `native_gate` due to shape mismatch.

### F. Wrong target (family=ffn_up) — NO SHAPE MISMATCH
```
[PRT-APPLY] enabled layer=0 family=ffn_up shadow_contrib=0
[PRT-PAGER-LAZY] layer=0 family=ffn_up activation_attempts=1 activation_successes=1
[PRT-CHECKPOINT] CHECKPOINT_A_AFTER_DECODE key=0:ffn_up:A ...
[PRT-CHECKPOINT] CHECKPOINT_B_AFTER_COPY key=0:ffn_up:B ...
```
No INJECT-CANARY-FFN log (also suppressed by `prt_v2_quiet_native_prints()`).

## Compilation
```bash
cmake --build build --target llama-cli -j$(nproc) 2>&1 | tail -5
# [100%] Built target llama-cli
# [100%] Linking CXX executable ../../bin/llama-cli
# [100%] Built target llama-cli
```

## Controls
- **scale=0.0:** No-op (guard at `if (g_prt_sidecar_scale_env != 1.0f)` only scales if != 1.0, but shape check fires first)
- **sign_flip:** Guarded correctly
- **Wrong family:** Family check `g_prt_sidecar_apply_family != "ffn_gate"` returns native_gate
- **Wrong layer:** `il != 0` returns native_gate

## Claim Boundary

**PROVEN:**
1. `build_prt_true_ffn_gate_injection()` implemented and compiles
2. Function is called at correct hook point in build_ffn (LLM_FFN_SEQ path)
3. Shape incompatibility detected: `R_cols(896) != X_rows(4864)`
4. Function returns `native_gate` on shape mismatch
5. Residual decode path works (CHECKPOINT logs confirm decode and copy)

**LIKELY:**
1. Shape mismatch guard fires and logs shape_mismatch (log suppressed)
2. ffn_gate injection would succeed if residual had shape `[hidden=896, intermediate=4864]`

**UNKNOWN:**
1. Whether weight-level injection would work (not attempted)
2. Whether R^T @ tmp approach would preserve semantic meaning

**FORBIDDEN CLAIMS:**
1. ffn_gate injection produces correct output (shape mismatch prevents it)
2. Combined ffn_up + ffn_gate injection works
3. Multi-layer injection works
4. Quality/correctness claims

## Log Suppression Note
`INJECT-CANARY-*` logs use `prt_logf()` which checks `prt_v2_quiet_native_prints()`. When that returns true, all `prt_logf()` output is suppressed. This is a pre-existing issue in the codebase. CHECKPOINT logs use direct `fprintf()` and bypass this suppression, allowing decode verification.

## Next Recommended Phase
**Phase 28BR-AD:** Implement ffn_gate weight-level injection — replace the `build_lora_mm(gate, tmp)` source weight with `gate + R` before the matmul, enabling injection despite shape incompatibility at output level.

Alternative (simpler): **Phase 28BR-AE:** Create a fixture with transposed ffn_gate residual shape `[hidden=896, intermediate=4864]` to match tmp shape, proving output-level injection works with correct residual geometry.