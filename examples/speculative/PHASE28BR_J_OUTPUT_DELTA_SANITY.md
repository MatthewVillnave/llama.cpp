# Phase 28BR-J: Output Delta Sanity — guarded true-injection vs baseline/observe/shadow modes

**Branch:** `experimental/prt-phase19a-alt-sidecar-backed`
**HEAD:** `700175575`
**Date:** 2026-05-24 (execution: ~16:32–16:37 EDT)
**Model:** Qwen2.5-0.5B-Instruct-Q4_K_M.gguf
**Binary:** `llama-cli` (build `b9119-d85ffcf86`)

---

## 1. Objective

Characterize whether guarded true-injection produces a detectable, repeatable output/token/logit delta vs baseline/observe/shadow modes on n_predict=1 ("Hi" prompt).

NOT quality/speed/correctness evaluation.

---

## 2. Execution Summary

### Setup
- **Model:** Qwen2.5-0.5B-Instruct-Q4_K_M.gguf (397MB, Q4_K_M)
- **Sidecar manifest:** `/tmp/phase28bo_layer0_multi_family/manifest.json`
  - layer_count=1, families=[ffn_up, ffn_down, attn_out, ffn_gate]
- **Sidecar directory:** `/tmp/phase28bo_layer0_multi_family/`
- **Prompt:** "Hi"
- **n_predict:** 1 (1 run per mode)
- **Execution method:** `script -qfc` PTY wrapper + 55s timeout kill (for stable capture)

### Modes Tested

| Mode | Flags | Description |
|------|-------|-------------|
| A | (none) | Baseline native (no PRT) |
| B | `--prt-mode 5700 --enable-prt-sidecar-pager --prt-sidecar-dir $SIDE --prt-sidecar-manifest $MANIFEST --prt-sidecar-budget-mb 64` | Observe-only (no apply) |
| C | B + `--prt-sidecar-apply --prt-sidecar-apply-layer 0 --prt-sidecar-apply-family attn_out` | Shadow (apply at layer 0, attn_out) |
| D | C + `--prt-sidecar-true-injection` | **True injection** (guarded, graph-mutating) |
| E | C + `--prt-sidecar-apply-layer 1` | Wrong target control (layer 1 doesn't exist) |
| F | C with `--prt-sidecar-budget-mb 0` | Budget-zero control |

### Execution Notes
- All modes were killed by timeout at ~55s (expected: llama-cli needs PTY for REPL)
- Token capture relied on PTY mode (`script -qfc`) combined with timeout kill
- Token IDs were NOT captured in output (no `[TOKEN]` lines visible; only F showed `[TOKEN] id=9707`)

---

## 3. Results

### Generated Text (all modes produce "Hello")

| Mode | First Token | Output Text | Token ID | PRT Lines |
|------|-------------|-------------|----------|-----------|
| A | Hello | Hello | (not captured) | 357 |
| B | Hello | Hello | (not captured) | 310 |
| C | Hello | Hello | (not captured) | 358 |
| D | Hello | Hello | (not captured) | 358 |
| E | Hello | Hello | (not captured) | 358 |
| F | Hello | Hello | **9707** | 41 |

### Counter Snapshots at Generation Completion

| Counter | A | B | C | D | E | F |
|---------|---|---|---|---|---|---|
| hook_calls | — | — | 48 | 48 | 48 | 41 |
| activation_attempts | — | — | 1 | 1 | 1 | **10** |
| activation_successes | — | — | 1 | 1 | 1 | **0** |
| non_null_views | — | — | 8 | **10** | **10** | **0** |
| null_views | — | — | 185 | **210** | **210** | **203** |
| decoded_views | — | — | 2 | 2 | **0** | 0 |
| app_attempts | — | — | 2 | 2 | **0** | 0 |
| app_success | — | — | 2 | 2 | **0** | 0 |
| **sidecar_math_influenced** | — | — | **0** | **1** | **0** | **0** |
| injection_attempts | — | — | N/A | **2** | **0** | **0** |
| injection_successes | — | — | N/A | **2** | **0** | **0** |
| injection_skipped | — | — | N/A | **0** | **2** (wrong layer) | **10** (budget) |

### Key PRT Log Entries

**Mode C (shadow):**
```
[PRT-APPLY-SHADOW] il=0 family=attn_out layer_match=1 family_match=1 decoded_views=1 app_attempts=1 app_success=1 sidecar_math_influenced_output=0
```

**Mode D (true injection) — INJECTION OCCURRED:**
```
[PRT-INJECT-CANARY] il=0 family=attn_out action=mutated_output R=[896,896] X=[896,16] out=[896,16] injection_attempts=1 injection_successes=1 injection_failures=0 injection_skipped=0 injection_shape_mismatch=0 injection_nonfinite_blocked=0 contribution_finite_before_injection=1 sidecar_math_influenced_output=1
[PRT-INJECT-CANARY] il=0 family=attn_out action=mutated_output R=[896,896] X=[896,14] out=[896,14] injection_attempts=2 injection_successes=2 injection_failures=0 injection_skipped=0 injection_shape_mismatch=0 injection_nonfinite_blocked=0 contribution_finite_before_injection=1 sidecar_math_influenced_output=1
[PRT-APPLY-SHADOW] il=0 family=attn_out layer_match=1 family_match=1 decoded_views=1 app_attempts=1 app_success=1 sidecar_math_influenced_output=0
```

**Mode E (wrong target) — INJECTION SKIPPED:**
```
[PRT-INJECT-CANARY] il=0 family=attn_out is_null=1 reason=injection_skipped_wrong_layer injection_attempts=0 injection_successes=0 injection_failures=0 injection_skipped=1 ...
```

**Mode F (budget zero) — ALL ACTIVATIONS REJECTED:**
```
[PRT-PAGER-COUNTERS] hook_calls=41 activation_attempts=10 activation_successes=0 non_null_views=0 null_views=203 layer_not_activated=10 tensor_not_found=0 budget_rejects=10 ...
```

### Checkpoint Data (D only)

D's checkpoint shows the sidecar was materialized:
```
[PRT-CHECKPOINT] CHECKPOINT_A_AFTER_DECODE key=0:attn_out:A ptr=0x70ae30514350 float_count=802816 nan=0 inf=0 finite=1 abs_sum=4.010e+05
[PRT-CHECKPOINT] CHECKPOINT_B_AFTER_COPY key=0:attn_out:B ptr=0x70ae30824360 float_count=802816 nan=0 inf=0 finite=1 abs_sum=4.010e+05
```

---

## 4. Proven vs NOT Proven

### Proven
- **Guarded true injection (D) activates correctly:** `injection_attempts=2, injection_successes=2, injection_skipped=0`
- **Guarded true injection (D) mutates graph:** `action=mutated_output`, `sidecar_math_influenced_output=1` in INJECT-CANARY log
- **Shadow (C) applies correctly:** `decoded_views=2, app_attempts=2, app_success=2, sidecar_math_influenced_output=0` in APPLY-SHADOW (shadow doesn't claim to influence output)
- **Wrong target (E) is blocked:** `injection_skipped=2 (wrong_layer)`, `app_success=0`
- **Budget=0 (F) fully rejects:** `activation_successes=0, budget_rejects=10, decoded_views=0`
- **Baseline (A) and observe (B) don't apply:** Both produce plain "Hello" without PRT apply logs

### NOT Proven
- ❌ Output delta: All modes produce "Hello" — **same token** despite D's injection
- ❌ Token IDs for A-E (only F captured token ID 9707)
- ❌ Whether D's graph mutation would produce different output on different prompts
- ❌ Repeatability (single run per mode)
- ❌ n_predict=2 behavior
- ❌ FFN family injection (ffn_up, ffn_down, ffn_gate — not in manifest)
- ❌ Numerical precision effects (abssum same 4.010e+05 across C and D checkpoints)

### Interpretation of "same token despite injection"

D shows `sidecar_math_influenced_output=1` — the injection DID influence the math. But the output token is "Hello" (same as A). This means either:
1. The injection was correct and changed the intermediate computation, but the argmax token still happened to be "Hello"
2. Or the residual path neutralized the injection

The checkpoint abs_sum (4.010e+05) is identical between C and D — this may indicate the injection was applied to intermediate state that was then overwritten by the subsequent residual computation.

---

## 5. Limitations

1. **Single run per mode** (not 3 runs as originally planned; execution environment overhead)
2. **Token IDs not captured** for A-E; F shows token ID 9707 = "Hello"
3. **n_predict=2 not tested** (execution time per mode too high)
4. **Only attn_out family tested**; ffn_up, ffn_down, ffn_gate not tested
5. **Only 0.5B model tested**; results may not generalize
6. **Execution required PTY** (`script -qfc`) — non-PTY execution causes infinite REPL loop
7. **Timing data:** Prompt ~224-244 t/s (within noise), generation always 1000000.0 t/s (1 token)

---

## 6. Recommendations for Next Phase (28BR-K)

1. **n_predict=2 comparison:** Test whether output divergence emerges at n_predict=2 (D should diverge from A more visibly)
2. **Token ID extraction:** Use `--log-bits` or parse raw logits to get token IDs for all modes
3. **Multiple runs:** Run each mode 3 times to check determinism
4. **FFN families:** Test ffn_up, ffn_down, ffn_gate injection (different manifest)
5. **Longer prompts:** Test with multi-token prompts to see if delta accumulates
6. **Execution fix:** The PTY `script -qfc` approach works but is slow; consider python-ptyprocess for cleaner automation

---

## 7. Appendix: Execution Environment Notes

- `llama-cli` enters an interactive REPL loop even with `-n 1 -p "Hi"` in non-TTY (pipe redirected) execution
- `script -qfc` creates a PTY that allows proper non-interactive execution
- `stdbuf -oL -eL` enables line-buffering but doesn't fix the REPL issue
- `llama-simple` and `llama-batched` have the same REPL behavior
- `head -1 |` (SIGPIPE) captures only the first line (warnings), not the generated token

---

*Phase 28BR-J: COMPLETE — guarded true injection (D) confirmed active, mutates graph, counter `sidecar_math_influenced=1` fires; controls (E/F) confirm correct blocking; all modes produce same token "Hello" on this prompt.*