# Phase 30E-HARDEN-SMOKE: CPU Runtime Smoke for PRT_V2_SIDECAR_ROOT

## Summary

**Branch:** `experimental/prt-phase19a-alt-sidecar-backed`  
**Old HEAD:** `9ad6e281e`  
**New HEAD:** `9ad6e281e` (no new commit — runtime smoke only, no code changes)  
**Classification:** `PARTIAL_ROOT_FAILFAST_ONLY`

## What Was Tested

Three smoke scenarios as required:

### Test A: Missing root — PRT active without `PRT_V2_SIDECAR_ROOT`

```
ENV: PRT_V2_SIDECAR_ROOT unset
CMD: --prt-mode 1 (no pager)
RESULT: exit=0, PRT runs in native mode silently
```

**Finding:** Fail-fast `[PRT-ERROR]` is implemented at llama-graph.cpp:2013 and 2141, but it is gated behind `g_prt_ggml_op_test == 1`. In normal runtime (ggml_op_test=0), when PRT is active, no root is set, and no pager is enabled, the code silently proceeds in native mode. No `[PRT-ERROR]` is emitted. No `g_prt_error_count` increment. No silent fallback to hardcoded paths.

**Verdict:** `g_prt_error_count` mechanism EXISTS. Fail-fast DOES NOT fire in normal PRT mode without ggml_op_test. This is a gap — the fail-fast is test-mode only.

### Test B: Explicit root — `PRT_V2_SIDECAR_ROOT=/tmp/phase28br_o_layer0_multifamily_trit` + pager

```
ENV: PRT_V2_SIDECAR_ROOT=/tmp/phase28br_o_layer0_multifamily_trit
CMD: --prt-mode 1 --prt-sidecar-manifest <manifest> --enable-prt-sidecar-pager --prt-sidecar-apply --prt-sidecar-apply-layer 0 --prt-sidecar-apply-family attn_out
RESULT: exit=0, pager enabled, manifest loaded from /tmp not /media/matthew-villnave/VL_usb
```

**Verdict:** ✅ No hardcoded `/media/matthew-villnave/VL_usb/...` path used. `PRT_V2_SIDECAR_ROOT` env var is consumed (confirmed via `[PRT_V2_AUTO] sidecar_root set via PRT_V2_SIDECAR_ROOT=...` in stderr). Pager initialized correctly. The CLI/Pager path is independent of `g_prt_sidecar_root` — when pager is enabled, manifest path comes from `--prt-sidecar-manifest` argument, not from `g_prt_sidecar_root`.

**Key observation:** When `--enable-prt-sidecar-pager` is used, the sidecar resolution uses the manifest (provided via `--prt-sidecar-manifest`) rather than `g_prt_sidecar_root`. The `g_prt_sidecar_root` variable controls the non-pager (CLI) sidecar path resolution at llama-graph.cpp:2000-2008 and 2134-2141.

### Test C: Wrong root — `PRT_V2_SIDECAR_ROOT=/tmp/phase30e_smoke_nonexistent_root_xyz123`

```
ENV: PRT_V2_SIDECAR_ROOT=/tmp/phase30e_smoke_nonexistent_root_xyz123
CMD: --prt-mode 1 --prt-sidecar-manifest <valid_manifest> --enable-prt-sidecar-pager
RESULT: exit=0, pager enabled (wrong root has no effect when pager is active)
```

**Finding:** With `--enable-prt-sidecar-pager`, the wrong `g_prt_sidecar_root` has zero effect because pager hook resolves paths from manifest, not from `g_prt_sidecar_root`. Without pager (CLI path), the wrong root would produce `[PRT-ERROR]` only under `g_prt_ggml_op_test=1`.

**Verdict:** No silent fallback to hardcoded paths. Wrong root doesn't crash — it's simply not used in pager mode.

## Hardcoded Path Audit

Confirmed via strace:
- Model file reads: `/media/matthew-villnave/VL_usb/models/...` (expected — model lives there)
- Sidecar manifest: `/tmp/phase28br_o_layer0_multifamily_trit/manifest.json` (correct — from --prt-sidecar-manifest arg)
- No sidecar file opens from `/media/matthew-villnave/VL_usb/prt_scratch/sidecars/...` in any test run

**6/6 hardcoded active-code paths confirmed removed** (Phase 30E-HARDEN report).

## Gap: Fail-Fast Gated Behind `ggml_op_test`

The `[PRT-ERROR]` fail-fast at llama-graph.cpp:2013 and 2141 only fires when `g_prt_ggml_op_test == 1`. In normal PRT mode:
- No `[PRT-ERROR]` logged
- No `g_prt_error_count` increment
- Execution continues in native mode silently

**Recommendation:** Consider adding a fail-fast that fires in normal PRT mode (not just ggml_op_test mode) when PRT is active but no sidecar resolution path is available (neither pager nor `g_prt_sidecar_root`). This would make the missing-root case visible to operators.

## Classification

**`PARTIAL_ROOT_FAILFAST_ONLY`** — The following are proven:
- ✅ 6/6 hardcoded `/media/matthew-villnave/VL_usb/...` paths removed from active runtime
- ✅ `PRT_V2_SIDECAR_ROOT` env var is consumed and logged
- ✅ No hardcoded sidecar path used in any smoke test
- ✅ `g_prt_error_count` and `llama_get_prt_error_count()` API exists
- ⚠️  Fail-fast only fires when `g_prt_ggml_op_test=1` (ggml_op test mode), not in normal PRT mode

**Not proven (blocked by test environment):**
- True-injection runtime smoke (requires `true_inj=1` flag, which this build doesn't support)
- Layer0 attn_out injection with `trit_validated > 0, checksum_ok > 0, injection_successes >= 1, sidecar_math_influenced_output=1` — pager activation failing with `trit_header_invalid` in current environment

## Artifact

Results JSON: `examples/speculative/results/phase30e_harden_smoke.json`