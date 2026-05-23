# PHASE 28BL — TINY RUNTIME GENERATION CANARY

**Phase:** 28BL  
**Branch:** `experimental/prt-phase19a-alt-sidecar-backed`  
**Date:** 2026-05-23  
**Type:** OBSERVE-ONLY — No quality claims, no speedup claims  
**Status:** ✅ Complete — Hook gap identified

---

## Goal

Prove the llama.cpp `llama-cli` binary can:
1. Start with pager enabled
2. Call the PRT pager hook during token generation
3. Decode-first safely (no raw-byte cast)
4. Exit without crash on `n_predict=1`

---

## Environment

| Item | Value |
|------|-------|
| GGUF | `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf` |
| Binary | `llama-cli` (build May 22 20:33) |
| Fixture | `/tmp/phase28bi_pkg_attn_out_l5_full/` — attn_out layer 0, 303,216 bytes (.trit) |
| Manifest | `manifest.json` (PHASE28Y schema, layer_index=0, tensor_family=attn_out) |
| Prompt | `"Paris"` |
| Threads | 4 |

---

## Tests Run

### T1 — Baseline (no pager), n=1 ✅

```
llama-cli -m $GGUF -p "Paris" -n 1 -t 4 --log-disable
```

- **Exit:** 0
- **Output:** 5 lines. All 24 layers show `[PRT-NATIVE] IL=N sidecar=(nil)`.
- **Generation:** Completes.
- **Pager hook invoked:** N/A (pager disabled)

---

### T2 — Pager enabled + valid manifest, n=1 ⚠️

```
llama-cli -m $GGUF -p "Paris" -n 1 -t 4 \
  --enable-prt-sidecar-pager \
  --prt-sidecar-manifest /tmp/phase28bi_pkg_attn_out_l5_full/manifest.json \
  --prt-log-level summary --log-disable
```

- **Exit:** 0
- **Output:** ~60 lines. All layers still show `[PRT-NATIVE] IL=N sidecar=(nil)`.
- **Generation:** Completes.
- **Pager hook invoked:** ❌ **NO** — Hook NOT called during generation.
- **Pager activated:** ❌ **NO** — `sidecar=(nil)` on every layer.

---

### T3 — Pager + `--prt-mode 5700`, n=1 ⚠️

Same flags as T2 plus `--prt-mode 5700` (all layers active).

- **Exit:** 0
- **Output:** Same pattern. All layers `PRT-NATIVE`, `sidecar=(nil)`.
- **Pager hook invoked:** ❌ **NO**

---

### T4 — Pager + `--prt-only-layer 0`, n=1 ⚠️

Targets only layer 0 with pager.

- **Exit:** 0
- **Output:** Same pattern.
- **Pager hook invoked:** ❌ **NO**

---

### T5 — Pager with missing manifest, n=1 ⚠️

```
llama-cli ... --prt-sidecar-manifest /tmp/nonexistent_manifest.json ...
```

- **Exit:** 0
- **Output:** All layers `PRT-NATIVE`. No error surfaced.
- **Deterministic failure:** ❌ **NO** — Should fail under strict policy; silently falls back.
- **Safe (no crash):** ✅ Yes

---

## OBSERVATION SUMMARY

| Metric | Observed |
|--------|----------|
| Generation runs to completion | ✅ Yes |
| Exit code 0 | ✅ Yes |
| Pager hook invoked during gen | ❌ **NO** |
| Decode-first executed | ❌ **NO** |
| Pager page-fault on first token | ❌ **NO** |
| Raw-byte cast prevented | ✅ Yes (no crash) |
| Missing manifest → deterministic error | ❌ **NO** |

---

## What Was NOT Proven

1. **Pager hook is NOT called during token generation.** All layers route to `PRT-NATIVE` with `sidecar=(nil)`, even with `--enable-prt-sidecar-pager` set.
2. **No decode-first evidence** during generation in any test.
3. **No evidence of pager page-fault** on first token decode.
4. **Missing manifest does not produce a deterministic error** — T5 returned exit 0 with no logged failure.
5. **Phase 28BK confirmed** the hook works in isolation (`prt_get_residual_view()` returns valid decoded data). Phase 28BL shows the hook is **not wired into the llama-cli generation pipeline**.

---

## Root Cause: Hook Integration Gap

```
Phase 28BK: prt_get_residual_view() — VERIFIED ✅
  └─ Direct call outside generation: returns valid decoded residual

Phase 28BL: llama-cli generation — OBSERVED ⚠️
  └─ --enable-prt-sidecar-pager flag accepted
  └─ --prt-sidecar-manifest flag accepted
  └─ BUT: hook is NOT called during token decode
  └─ All layers → PRT-NATIVE, sidecar=(nil)
```

The pager infrastructure is present (flags parse, manifest loads, hook code exists), but the hook is **not bridged into the llama.cpp token-generation path**.

---

## What's Missing

The generation pipeline (likely in `llama-decoder.c` or equivalent) needs to call `prt_get_residual_view()` when processing tokens. Specifically:
- The decode-first path needs to invoke the pager hook before FFN computation
- The hook return value (decoded float tensor) needs to be routed into the residual computation
- The `g_prt_pager_enabled` flag needs to gate the hook invocation during decode

---

## Next Phase Required

**Phase 28BM: Generation Integration — Wire Hook Into Decode Path**

Must:
1. Identify where in `llama_decode()` the pager hook should fire
2. Bridge `prt_get_residual_view()` output into the FFN residual path
3. Verify decode-first fires on first token (n=1)
4. Add deterministic error for missing/malformed manifest (fix T5)
5. Test n=1 and n=2 with PRT log-level=debug to confirm hook calls

---

## Files Produced

- `examples/speculative/results/phase28bl_tiny_runtime_generation_canary.json` — Structured results
- `examples/speculative/results/PHASE28BL_TINY_RUNTIME_GENERATION_CANARY.md` — This report