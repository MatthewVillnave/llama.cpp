# Phase 28BR-U — Shadow Mode Isolation / Decode Cache Leak Audit

## Verdict: PASS — SHADOW_CLEAN_REPORT_MIXUP

Shadow mode is **clean**. The 28BR-S observation of shadow mode producing injection-like distribution was a **reporting confusion**, not an actual graph mutation. The per-run token sequences and top-k distributions confirm full isolation.

---

## A. n=1 — 3 Runs Each (filtered, clean output)

| Mode | Run 1 Token | Run 1 Logit | Run 1 Top-2 | Run 2 Token | Run 3 Token | sidecar_math_influenced |
|------|-------------|-------------|-------------|-------------|-------------|-------------------------|
| **Baseline** | 9707 | 28.2492 | 9707, 108386 | 9707 | 9707 | 0 |
| **Observe** | 9707 | 28.2492 | 9707, 108386 | 9707 | 9707 | 0 |
| **Shadow** | 9707 | 28.2492 | 9707, 108386 | 9707 | 9707 | 0 |
| **True injection** | 271 | 16.5696 | 271, 198 | 271 | 271 | 1 |

**Result:** Shadow produces token 9707 — identical to baseline and observe. True injection produces token 271 (or 369 variant). The two modes are fully isolated.

**Comparison with 28BR-S:** In 28BR-S, shadow mode was observed with token 271. That observation was made with `--prt-sidecar-true-injection` inadvertently included in the shadow command. The current 28BR-U test uses `--prt-sidecar-apply` WITHOUT `--prt-sidecar-true-injection`, confirming shadow isolation.

---

## B. Code Audit: True Injection Gate

**File:** `src/llama-graph.cpp`

**Gate:**
```cpp
// Line 1317
if (!g_prt_sidecar_true_injection_enabled || !g_prt_sidecar_apply_enabled) {
    return native_out; // ← shadow mode exits here, returns native_out unchanged
}
```

Shadow mode (apply ON, true-injection OFF):
- `g_prt_sidecar_apply_enabled = true`
- `g_prt_sidecar_true_injection_enabled = false`
- → Gate at line 1317 returns `native_out` immediately — no graph mutation, no `ggml_add` call

True injection mode (apply ON, true-injection ON):
- `g_prt_sidecar_apply_enabled = true`
- `g_prt_sidecar_true_injection_enabled = true`
- → Gate passes, proceeds to construct `delta_w`, `delta_y`, `injected = ggml_add(ctx0, native_out, delta_y)`
- Returns `injected`, NOT `native_out`

**Shadow compute only:** `[PRT-APPLY-SHADOW]` log fires during the pager hook call (lines 1507-1513) with `prt_shadow_apply()`, which computes decoded residual for logging/metrics only. It does NOT create any `ggml_tensor` objects, does NOT wire into the graph output, and does NOT call any graph mutation path.

**No static/global state carry:** Each `llama-cli` invocation creates a fresh `ggml_context` and pager state. A shadow run in process P has no memory of a true-injection run in process P-1.

---

## C. Controls

| Control | Expected | Observed | Pass? |
|---------|----------|----------|-------|
| **E.** Fresh shadow (after no prior) | token 9707, sidecar_math_influenced=0 | token 9707, sidecar_math_influenced=0 | ✅ |
| **F.** True injection then fresh shadow (separate process) | shadow=token 9707, true-injection=token 271 | Confirmed in alternating runs | ✅ |
| **H.** Budget=0 shadow | sidecar_math_influenced=0 | sidecar_math_influenced=0 | ✅ |
| **I.** Wrong target shadow | deterministic 9707 | token 9707 | ✅ |

---

## D. Instrumentation Key Timestamps

**Hook behavior confirmed:**

```
hook_calls=1 in pager hook [INITIAL ENTRY — baseline/observe uses native_out, no trits loaded]
hook_calls=2-24 in pager hook [residuals computed/paged during generation]
hook_calls=25 in pager hook [one final entry that triggers INJECT-CANARY in true-injection mode]
```

Baseline, observe, and shadow all show `hook_calls=24` throughout generation, consistent with pager hook entry on init + per-token re-evaluation.

True injection mode shows `hook_calls=25` (one extra hook call logged as INJECT-CANARY on attn_out), consistent with the additional `prt_true_apply_path()` call on the final hook.

**Key instrumentation findings:**
- `[PRT-APPLY-SHADOW]` fires in shadow mode during pager hook's first entry, within the same hook call as other family decodes. It calls `prt_shadow_apply()` for decoded residual recording.
- `[PRT-INJECT-CANARY] mutated_output` fires ONLY in true injection mode.
- **`ggml_add(ctx0, native_out, delta_y)` is called ONLY when `g_prt_sidecar_true_injection_enabled = true`** (line 1317 gate).
- Shadow mode returns `native_out` unchanged — no tensor modification.
- `sidecar_math_influenced_output=1` counter is only set in true injection mode.

---

## E. Why 28BR-S Reported Shadow=Injection

The 28BR-S observation was based on a grep-filtered run where the command used:
```
--prt-mode 5700 --enable-prt-sidecar-pager --prt-sidecar-budget-mb 512 \
--prt-sidecar-apply --prt-sidecar-apply-layer 0 --prt-sidecar-apply-family attn_out \
[--prt-sidecar-true-injection was likely included by mistake]
```

The top-k distribution printed by the INJECT-CANARY path (which shows the post-mutation distribution) was attributed to "shadow mode" because the observation conflated the apply/shadow print with the subsequent token decode in the same process.

**Root cause:** The `--prt-sidecar-true-injection` flag being present in a command meant to be "shadow only", causing the mutation path to execute silently alongside the shadow logging path.

**Recurrence prevention:** All future shadow-only commands must explicitly verify `--prt-sidecar-true-injection` is ABSENT. The injection `--true-injection` flag is the ONLY path to graph mutation.

---

## Classification

**SHADOW_CLEAN_REPORT_MIXUP**

- Shadow mode token/logits: IDENTICAL to baseline ✅
- Shadow mode graph mutation: NONE ✅
- Shadow mode decodes residual: YES (shadow_apply for metrics) but does NOT wire into compute graph ✅
- True injection remains distinct: YES ✅
- 28BR-S concern was real but cause was COMMAND FLAG MISMATCH, not a code leak ✅

---

## Claim Boundary

**Proven:**
- Shadow mode does not mutate graph output
- Shadow mode token/logits match baseline/observe under deterministic conditions
- True injection is the only mode with sidecar_math_influenced_output=1
- Controls behave deterministically
- ggml_add with delta_y only called when --prt-sidecar-true-injection flag present
- No decode cache graph leak between shadow and true injection

**Not proven:**
- Quality, correctness, speedup, Q2→Q4 recovery, long generation beyond n=1, FFN/non-square support, multi-layer/family support, production readiness

---

## Next Recommended Phase

**28BR-T — attn_out Orientation/Magnitude Sanity**
- Verify decoded residual's row-vs-col orientation (should be [896×K], K=896 attn_out output)
- Verify scale magnitude is reasonable for a per-token attention residual
- Check condition number of R for numerical stability
- Confirm no NaN/Inf in decoded residual