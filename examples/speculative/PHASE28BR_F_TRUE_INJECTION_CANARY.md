# Phase 28BR-F: True Injection Canary

**Branch:** `experimental/prt-phase19a-alt-sidecar-backed`
**Base HEAD:** `5178be31b` (Phase 28BR-E: fix pager global ODR wiring)
**Timestamp:** `2026-05-24T11:25 EDT`
**Verdict:** `PARTIAL_BLOCKED_ON_GRAPH_CONSTANT_MATERIALIZATION`

---

## Goal

Add a narrow, target-gated true residual injection canary after Phase 28BR-E proved the pager, decode/cache, and contribution-shadow paths were finite.

Target: `layer=0`, `family=attn_out`.

Boundary:

- No clamping.
- No NaN zeroing.
- No fake injection.
- No quality claim.
- No speed claim.
- No production-readiness claim.
- Default and observe-only paths must remain safe.

## Implementation Summary

- Added `--prt-sidecar-true-injection`, disabled by default.
- Wired the flag through `common_params` and CLI global state.
- Added guarded injection counters:
  - `injection_attempts`
  - `injection_successes`
  - `injection_failures`
  - `injection_skipped`
  - `injection_shape_mismatch`
  - `injection_nonfinite_blocked`
  - compatibility aliases for older skip counters
  - `contribution_finite_before_injection`
- Added `prt_true_apply()` to decode the target sidecar and prove decoded R is finite before graph mutation.
- Added graph-side `attn_out` injection gate after native `wo` matmul:
  - target must be enabled
  - pager must be enabled
  - raw sidecar view must exist
  - decoded R must be finite
  - shape must match native attn output path
  - only then attempt `delta_y = R @ attn_out_input` and `native_out + delta_y`

## Build

```text
cmake --build build --target llama-cli -j4
```

Result: build passed. Existing warnings remain; no new fatal compiler errors.

## Canary Commands

All runs used:

```text
model=/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf
manifest=/tmp/phase28bo_layer0_multi_family/manifest.json
sidecar_dir=/tmp/phase28bo_layer0_multi_family/
prompt='Hi'
n_predict=1
```

Runs:

1. Default pager only, no apply.
2. Apply + shadow contribution, no true injection.
3. Apply + shadow contribution + `--prt-sidecar-true-injection`.

Each `llama-cli` run exited with timeout `124` because the CLI remained in prompt mode after emitting required forensic lines. Evidence was emitted before timeout.

## Control Results

### Default / Observe Only

No apply or injection counters fired:

```text
decoded_views=0 app_attempts=0 app_success=0 sidecar_math_influenced=0
```

### Apply + Shadow Contribution

Decoded R remained finite across A-D:

```text
CHECKPOINT_A_AFTER_DECODE nan=0 inf=0 finite=1
CHECKPOINT_B_AFTER_COPY nan=0 inf=0 finite=1
CHECKPOINT_C_AFTER_RETRIEVAL nan=0 inf=0 finite=1
CHECKPOINT_D_BEFORE_LOOP nan=0 inf=0 finite=1
```

Contribution shadow remained finite:

```text
[PRT-CONTRIB-SHADOW] il=0 family=attn_out X_synthetic=I_KK R=[896x896] Y=[896x896] abs_sum=4.009560e+05 max_abs=1.000000e+00 nan=0 inf=0 finite=1
```

Output mutation stayed off:

```text
sidecar_math_influenced_output=0
```

### True Injection Flag

The true injection flag reached the graph-side `attn_out` insertion gate and decoded finite R:

```text
CHECKPOINT_A_AFTER_DECODE nan=0 inf=0 finite=1
CHECKPOINT_B_AFTER_COPY nan=0 inf=0 finite=1
CHECKPOINT_C_AFTER_RETRIEVAL nan=0 inf=0 finite=1
CHECKPOINT_D_BEFORE_LOOP nan=0 inf=0 finite=1
```

The mutation attempt was blocked before graph mutation:

```json
{"event":"TRUE_INJECTION_ATTEMPT","success":0,"reason":"delta_weight_tensor_allocation_failed","injection_attempts":1,"injection_successes":0,"injection_failures":1,"sidecar_math_influenced_output":0}
```

## Classification

`BLOCKED_ON_GRAPH_CONSTANT_MATERIALIZATION`

The target, decode, finiteness, and contribution gates are working. The actual graph mutation is blocked because `ggml_new_tensor()` creates a graph tensor whose `data` pointer is null during graph construction in this path, so the runtime cannot safely `memcpy()` decoded sidecar data into a GGML tensor at that point.

The implementation correctly:

- records `injection_attempts=1`
- records `injection_failures=1`
- records `injection_successes=0`
- keeps `sidecar_math_influenced_output=0`
- avoids crashing
- avoids silently claiming success
- does not mutate model output

## Pass Criteria Status

| Criterion | Status |
|---|---|
| Injection is target-gated | PASS |
| Contribution is finite before injection | PASS |
| Controls behave deterministically | PASS |
| Observe-only/default paths remain safe | PASS |
| Output mutation recorded only as observation | PASS |
| True graph mutation succeeds | BLOCKED |
| No clamping / NaN zeroing | PASS |
| No quality/speed claims | PASS |

## Not Proven

- Output correctness.
- Quality parity.
- Speedup.
- Q2 to Q4 recovery.
- Long generation stability.
- Multi-layer or multi-family application.
- Full-model sidecar correctness.
- 30B feasibility.
- Production readiness.

## Next Narrow Step

Do not keep guessing in this path. The next step should be a small GGML constant-materialization probe that identifies the correct way to provide decoded F32 sidecar data to a graph node at build time, or explicitly selects a different safe insertion mechanism.

Artifacts:

- `/tmp/phase28br_f_default2.log`
- `/tmp/phase28br_f_apply2.log`
- `/tmp/phase28br_f_inject2.log`
- `/tmp/phase28br_f_default2.jsonl`
- `/tmp/phase28br_f_apply2.jsonl`
- `/tmp/phase28br_f_inject2.jsonl`
- `examples/speculative/PHASE28BR_F_TRUE_INJECTION_CANARY.md`
- `examples/speculative/results/phase28br_f_true_injection_canary.json`
