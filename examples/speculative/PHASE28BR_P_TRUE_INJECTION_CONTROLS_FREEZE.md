# Phase 28BR-P: True Injection Controls Freeze

**Date:** 2026-05-27
**Branch:** `experimental/prt-phase19a-alt-sidecar-backed`
**Start HEAD:** `57f803391`
**Classification:** PASS
**Codex subagent:** used (`019e69dd-503a-7f43-8940-d2c73bfa3dbd`)

## Goal

Freeze a narrow, deterministic control matrix around the Phase 28BR-O fixture:

- baseline native path must not report sidecar influence
- observe-only pager must load the fixture without injection
- shadow-only apply must decode but not mutate compute
- true injection must mutate only for the valid target
- wrong target must not inject
- missing manifest must fail deterministically before generation

No quality evals, speed evals, broad benchmarks, model files, generated sidecar data, or large logs were staged.

## Runner Fix

The old `n_predict=1` runner could keep emitting spinner/log noise and hit timeout after the relevant token/canary already appeared. The stable one-token control runner is:

```bash
timeout 90 ./build/bin/llama-cli \
  -m /home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf \
  -p "Hi" -n 1 -t 4 \
  --no-conversation --single-turn --no-display-prompt \
  --top-k 10 --temp 0
```

For pager controls, add:

```bash
--enable-prt-sidecar-pager \
--prt-mode 5700 \
--prt-sidecar-manifest /tmp/phase28br_o_layer0_multifamily_trit/manifest.json \
--prt-sidecar-dir /tmp/phase28br_o_layer0_multifamily_trit/ \
--prt-sidecar-budget-mb 512
```

## Matrix

| ID | Mode | Expected | Observed | Verdict |
|---|---|---|---|---|
| A | Baseline, no pager | exit 0, native token/logit, no sidecar influence | exit 0, token `9707`, logit `22.5994`, no PRT pager/apply/inject lines | PASS |
| B | Observe-only pager | exit 0, pager activation, no apply/inject | exit 0, pager enabled, layer 0 all four families loaded, token `9707` | PASS |
| C | Shadow-only apply | exit 0, decode/apply counters, no compute mutation | exit 0, `PRT-APPLY-SHADOW`, `decoded_views=1`, `app_success=1`, `sidecar_math_influenced_output=0`, token `9707` | PASS |
| D | True injection valid target | exit 0, `mutated_output`, sidecar influence true | exit 0, `action=mutated_output`, `injection_successes=1`, `sidecar_math_influenced_output=1`, token `271` | PASS |
| E | Wrong target layer | exit 0, injection skipped, no sidecar influence | exit 0, `reason=injection_skipped_wrong_layer`, `injection_successes=0`, `sidecar_math_influenced_output=0`, token `9707` | PASS |
| F | Missing manifest | deterministic hard fail before generation | exit 1, `common_init_result: PRT sidecar pager manifest not found` | PASS |

## Key Evidence

Baseline:

```text
[TOKEN] id=9707 logit=22.5994 top_ids=9707,108386 top_logits=22.5994,-inf
```

Observe-only:

```text
[PRT-PAGER] enabled via --enable-prt-sidecar-pager manifest=/tmp/phase28br_o_layer0_multifamily_trit/manifest.json
[PRT-PAGER-LAZY] layer=0 family=ffn_up ... activation_ok=1 ... resident_bytes=5240752 ...
[PRT-PAGER-LAZY] layer=0 family=ffn_down ... size=1645760 ...
[PRT-PAGER-LAZY] layer=0 family=attn_out ... size=303216 ...
[PRT-PAGER-LAZY] layer=0 family=ffn_gate ... size=1645888 ...
[TOKEN] id=9707 logit=22.5994 top_ids=9707,108386 top_logits=22.5994,-inf
```

Shadow-only:

```text
[PRT-APPLY-SHADOW] il=0 family=attn_out layer_match=1 family_match=1 decoded_views=1 app_attempts=1 app_success=1 sidecar_math_influenced_output=0
[TOKEN] id=9707 logit=22.5994 top_ids=9707,108386 top_logits=22.5994,-inf
```

True injection:

```text
[PRT-INJECT-CANARY] il=0 family=attn_out action=mutated_output R=[896,896] X=[896,30] out=[896,30] injection_attempts=1 injection_successes=1 injection_failures=0 injection_skipped=0 injection_shape_mismatch=0 injection_nonfinite_blocked=0 contribution_finite_before_injection=1 sidecar_math_influenced_output=1
[TOKEN] id=271 logit=13.2557 top_ids=271,198,220,311,11,374,320,646,13 top_logits=13.2557,-inf,-inf,-inf,-inf,-inf,-inf,-inf,-inf
```

Wrong target:

```text
[PRT-INJECT-CANARY] il=0 family=attn_out is_null=1 reason=injection_skipped_wrong_layer injection_attempts=0 injection_successes=0 injection_failures=0 injection_skipped=1 injection_shape_mismatch=0 injection_nonfinite_blocked=0 contribution_finite_before_injection=0 sidecar_math_influenced_output=0
[TOKEN] id=9707 logit=22.5994 top_ids=9707,108386 top_logits=22.5994,-inf
```

Missing manifest:

```text
common_init_result: PRT sidecar pager manifest not found: /tmp/phase28brp_missing_manifest_DOES_NOT_EXIST.json
```

## Claim Boundary

PROVEN:

- 28BR-O fixture works in observe-only, shadow-only, and true-injection modes.
- Shadow-only decodes the sidecar but does not claim output influence.
- True injection mutates graph math only for the valid target.
- Wrong layer blocks true injection.
- Missing manifest fails deterministically before generation.
- The stable narrow control runner exits cleanly with `--no-conversation --single-turn --no-display-prompt`.

NOT CLAIMED:

- Quality improvement.
- Speed improvement.
- Broad benchmark result.
- Generalization beyond the layer-0 fixture.
- Token 374 reproduction.

## Artifacts

Runtime logs were kept in `/tmp/phase28brp_*` only and were not staged.

Committed artifacts:

- `examples/speculative/PHASE28BR_P_TRUE_INJECTION_CONTROLS_FREEZE.md`
- `examples/speculative/results/phase28br_p_true_injection_controls_freeze.json`

Next recommended phase: Phase 28BR-Q should either codify this matrix as a lightweight automated harness or move to the next narrow correctness question now that true-injection controls are frozen.
