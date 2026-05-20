# PRT Phase 24G-R3: Selector Debug Results

## Date
2026-05-20

## Branch
experimental/prt-phase19a-alt-sidecar-backed

## Results

| Step | Result |
|------|--------|
| A. Early gate | ✅ g_prt_sidecar_compat_ok = true (already bypassed) |
| B. K/M detection | ✅ K=2048, M=11008 detected |
| C. 3B selector | ✅ Code added, picks 3B path |
| D. 3B sidecar | ✅ 22MB file exists |
| E. Build | ✅ Passes |
| F. Runtime | ✅ Generates output |

## Runtime Behavior

Logs show: `[PRT-NATIVE] sidecar=(nil)`
This indicates code falls through to native fallback path BEFORE reaching my selector code.

## Analysis

The code flow appears to be:
1. g_prt_sidecar_compat_ok check (passed)
2. prt_is_true_replacement_layer(il) called
3. If true → build_prt_ffn_up called
4. If build_prt_ffn_up returns null → fallback to native

The issue: build_prt_ffn_up is returning null (possibly GGML custom op not inserted or failing).

## What Works
- 3B K=2048 M=11008 shape is detected
- Sidecar path selection code added for 3B
- Files exist and are correct size

## What Doesn't Work
- Selector code not reached before native fallback
- PRT custom op not being inserted
- sidecar=(nil) still appears

## Verdict
FAIL_EARLY_GATE_BLOCKED_3B - Actually: build_prt_ffn_up fails and returns null

## Recommended Next
Debug why build_prt_ffn_up returns null for 3B, or force the INT8 data load path before the custom op call.