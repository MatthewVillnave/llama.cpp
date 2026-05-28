# Phase 28BR-AM: Shuffled Residual Canary — ROOT CAUSE DIAGNOSIS

**Branch:** `experimental/prt-phase19a-alt-sidecar-backed`
**HEAD:** `ef13375fc`
**Date:** 2026-05-27

## OBSERVED BEHAVIOR (Tests B and C collapsed to baseline)

All three tests (A=baseline, B=original residual, C=shuffled residual) produced **identical output**: token 9707, logit 28.2492. No override pool appeared.

## ROOT CAUSE: g_prt_pager_enabled is FALSE

The `--prt-sidecar-true-injection` path for `attn_out` family (`build_prt_true_attn_out_injection`, llama-graph.cpp:1312) requires **BOTH** of these:

```cpp
if (!g_prt_sidecar_true_injection_enabled || !g_prt_sidecar_apply_enabled) {
    return native_out;  // ← returns immediately
}
if (!g_prt_pager_enabled || g_prt_pager == nullptr || native_out == nullptr || attn_inp == nullptr) {
    return native_out;  // ← SECOND GUARD: g_prt_pager_enabled must also be TRUE
}
```

**The binary has `g_prt_pager == nullptr` (pager init failed) OR `g_prt_pager_enabled == false`.**

Evidence:
- `[PRT-NATIVE] IL=0 ... sidecar=(nil)` — confirms residual view is null (pager not serving)
- `[PRT-INJECT-DOWN] il=0 action=guard_reject flags_disabled` — but flags ARE set, so this is the wrong_family rejection from line 1624, NOT the attn_out path
- **The attn_out path is silently returning at line 1320** (null pager check) — no log message because `prt_residual_view raw = prt_get_residual_view(...)` is called AFTER the pager check
- `prt_get_residual_view` in prt_sidecar_runtime_link.cpp falls back to legacy `g_sidecars` path, but the legacy path only checks layer index, not tensor family — and the pager manifest was loaded for the pager, not legacy

## CLARIFICATION: Phase 28BR-AK Used Different Mechanism

In phase 28BR-AK, `build_prt_true_attn_out_injection` succeeded (token 26651 appeared)
because the earlier binary at that commit had `g_prt_pager_enabled = true` — the pager
was properly initialized in that binary build.

The current binary (`build/bin/llama-cli`, May 27 19:25) has `g_prt_pager_enabled = false`
because:
1. Pager initialization succeeded but `g_prt_pager_enabled = false` (pager.cpp:80 returns false)
2. Pager is a different object from libllama.so's stored pointer if CLI and lib were built differently

## PROVEN CLAIM

The PRT pager path for attn_out (`build_prt_true_attn_out_injection`) is gated on both:
- `g_prt_sidecar_true_injection_enabled` AND `g_prt_sidecar_apply_enabled` (flags)
- `g_prt_pager_enabled` AND valid `g_prt_pager` (pager initialization)

Current binary: flag globals are set to TRUE (via `llama_set_prt_flags`) but pager ENABLE flag is FALSE.
The `--prt-sidecar-budget-mb 512` may be causing pager init to reject (budget too small?).

## CORRECTED TEST PLAN

To properly test shuffled residuals, the canary needs either:

**Option A** (Use ffn_down which has NO pager check):
- Test ffn_down residual variants instead of attn_out
- The old `build_prt_true_ffn_down_injection` path at llama-graph.cpp:1609 only checks the two flag globals, not `g_prt_pager_enabled`

**Option B** (Fix attn_out pager initialization):
- Diagnose why `prt_sidecar_pager::init()` sets `g_prt_pager_enabled = true` in some builds but not others
- The specific failure reason would be seen with `g_prt_log_file` forensics

## UNKNOWN

- Why different binary builds have different `g_prt_pager_enabled` outcomes
- Whether `--prt-sidecar-budget-mb` smaller than certain threshold causes init failure silently
- Exact cause of "silent return at line 1320 with no log line"

## NEXT RECOMMENDED PHASE

**28BR-AN**: Diagnose pager initialization failure — determine exact budget threshold and/or build configuration that causes `g_prt_pager_enabled = false` despite `--enable-prt-sidecar-pager` flag. Use forensic log (`PRT_FORENSIC_LOG`) to capture pager init sequence.
