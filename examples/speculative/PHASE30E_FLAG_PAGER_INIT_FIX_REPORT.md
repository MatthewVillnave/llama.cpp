# Phase 30E - PRT Flag / Pager Init Call Path

## Scope

Narrow fix only. No quality or speed benchmark was run.

## Files Changed

- `common/arg.cpp`
- `tools/cli/cli.cpp`

## Fix

- Added fail-fast CLI validation:
  - `--prt-sidecar-apply` now requires `--enable-prt-sidecar-pager`.
  - `--prt-sidecar-true-injection` now requires `--prt-sidecar-apply`.
  - Pager/apply/true-injection paths require `--prt-sidecar-manifest`.
- Moved PRT flag propagation out of the pager-init conditional and before model load.
- Moved pager init before model load, so graph construction/decode does not observe default flags first.
- Kept actual sidecar view/application guarded in graph code by pager enabled + pager instance + requested apply/true-injection flags.
- Added explicit CLI logs:
  - `[PRT-FLAGS-REQUESTED]`
  - `[PRT-PAGER] enabled`
  - `[PRT-APPLY] enabled`
  - `[PRT-TRUE-INJECTION] enabled`
- Confirmed `llama_set_prt_flags()` exists in `include/llama.h` and `src/llama.cpp`; `[PRT-FLAGS-SET]` is emitted from the library function.

## Build

`cmake --build build --target llama-cli -j2`

Result: PASS. Existing warnings only.

## Validation Logs

Raw logs are under `/tmp/phase30e_validation/`.

### A. Invalid Apply Without Pager

Command class: `--prt-sidecar-apply` without `--enable-prt-sidecar-pager`.

Result: PASS.

Evidence:

```text
error: --prt-sidecar-apply requires --enable-prt-sidecar-pager
```

### B. Pager Only

Command class: pager + manifest only.

Result: PASS.

Evidence:

```text
[PRT-FLAGS-REQUESTED] apply=0 true_inj=0 layer=-1 family= shadow=0 scale=1.00
[PRT-FLAGS-SET] apply=0 true_inj=0 layer=-1 family_len=0 scale=1.00 sign_flip=0
[PRT-PAGER] enabled manifest=/tmp/phase28br_o_layer0_multifamily_trit/manifest.json root=/tmp/phase28br_o_layer0_multifamily_trit budget_mb=64
```

No `[PRT-APPLY] enabled` log appeared.

### C. Apply Shadow / Inert

Command class: pager + manifest + apply + layer 0 + `attn_out` + shadow contribution.

Result: PASS with a precision note: generated text matched baseline; full stdout was not byte-identical because llama-cli prints spinner/perf telemetry.

Evidence:

```text
[PRT-FLAGS-REQUESTED] apply=1 true_inj=0 layer=0 family=attn_out shadow=1 scale=1.00
[PRT-FLAGS-SET] apply=1 true_inj=0 layer=0 family_len=8 scale=1.00 sign_flip=0
[PRT-PAGER] enabled manifest=/tmp/phase28br_o_layer0_multifamily_trit/manifest.json root=/tmp/phase28br_o_layer0_multifamily_trit budget_mb=64
[PRT-APPLY] enabled layer=0 family=attn_out shadow_contrib=1
[PRT-APPLY-SHADOW] il=0 family=attn_out layer_match=1 family_match=1 decoded_views=1 app_attempts=1 app_success=1 sidecar_math_influenced_output=0
[PRT-CONTRIB-SHADOW] il=0 family=attn_out X_synthetic=I_KK R=[896x896] Y=[896x896] abs_sum=4.009560e+05 max_abs=1.000000e+00 nan=0 inf=0 finite=1
```

### D. True Injection

Command class: pager + manifest + apply + true injection + layer 0 + `attn_out`.

Result: PASS.

Evidence:

```text
[PRT-FLAGS-REQUESTED] apply=1 true_inj=1 layer=0 family=attn_out shadow=0 scale=1.00
[PRT-FLAGS-SET] apply=1 true_inj=1 layer=0 family_len=8 scale=1.00 sign_flip=0
[PRT-PAGER] enabled manifest=/tmp/phase28br_o_layer0_multifamily_trit/manifest.json root=/tmp/phase28br_o_layer0_multifamily_trit budget_mb=64
[PRT-APPLY] enabled layer=0 family=attn_out shadow_contrib=0
[PRT-TRUE-INJECTION] enabled layer=0 family=attn_out scale=1.00 sign_flip=0
[PRT-INJECT-CANARY] il=0 family=attn_out action=mutated_output R=[896,896] X=[896,1] out=[896,1] injection_attempts=1 injection_successes=1 injection_failures=0 injection_skipped=0 injection_shape_mismatch=0 injection_nonfinite_blocked=0 contribution_finite_before_injection=1 sidecar_math_influenced_output=1
```

Generated output differed from the inert baseline, consistent with the controlled mutation.

### E. Wrong Target Control

Command class: pager + manifest + apply + true injection + layer 0 + `wrong_family`.

Result: PASS.

Evidence:

```text
[PRT-FLAGS-REQUESTED] apply=1 true_inj=1 layer=0 family=wrong_family shadow=0 scale=1.00
[PRT-FLAGS-SET] apply=1 true_inj=1 layer=0 family_len=12 scale=1.00 sign_flip=0
[PRT-PAGER] enabled manifest=/tmp/phase28br_o_layer0_multifamily_trit/manifest.json root=/tmp/phase28br_o_layer0_multifamily_trit budget_mb=64
[PRT-APPLY] enabled layer=0 family=wrong_family shadow_contrib=0
[PRT-TRUE-INJECTION] enabled layer=0 family=wrong_family scale=1.00 sign_flip=0
[PRT-INJECT-DOWN] il=0 action=guard_reject wrong_family
sidecar_math_influenced=0
```

Generated text matched the inert baseline.

## Classification

PROVEN:

- The CLI now fails fast for invalid apply/pager/manifest combinations.
- `llama_set_prt_flags()` exists and is called before model load.
- Pager init occurs before model load when pager is requested.
- Apply-shadow path can resolve the layer 0 `attn_out` sidecar and stays inert.
- True injection can mutate layer 0 `attn_out` and increments injection success counters.
- Wrong-family control guard rejects cleanly and does not influence output.

LIKELY:

- This fixes the Phase 30D-R flag propagation regression for the active `llama-cli` path on this branch.

UNKNOWN:

- Whether this remains correct for all model sizes, all manifest schemas, and all sidecar families.
- Whether true injection preserves useful model behavior.

FORBIDDEN CLAIMS:

- No speedup claim.
- No quality claim.
- No global SDI/PRT success claim.
- No residency win claim.
