# Phase 28BR-O: Fixture Reconstruction / Freeze

**Date:** 2026-05-27
**Branch:** `experimental/prt-phase19a-alt-sidecar-backed`
**Old HEAD:** `94781655f`
**Classification:** PASS for fixture reconstruction and guarded true-injection canary
**Codex subagent:** used (`019e69bc-f559-7a40-892f-4688c3ad549c`) for parallel fixture archaeology; main session completed generator, validation, runtime canary, and artifacts.

## Goal

Reconstruct and freeze the exact layer-0 multi-family `.trit` fixture format needed for guarded true injection:

- `format_version: 1` manifest
- layer 0 entries for `ffn_up`, `ffn_down`, `ffn_gate`, `attn_out`
- valid `.trit` files compatible with `prt_trit_decoder::decode_bytes()`
- `covers(0, attn_out)=true`
- true injection canary fires again

No quality evals, speed evals, broad benchmarks, model files, sidecar data, giant logs, or private paths were staged.

## Generator

Added:

- `examples/speculative/phase28br_o_reconstruct_fixture.py`

The generator reuses the Phase 28BO deterministic fixture metadata:

| Family | Shape | Seed | Block | File Bytes |
|---|---:|---:|---:|---:|
| `ffn_up` | 4864x896 | 1990868534 | 32x48 | 1645888 |
| `ffn_down` | 896x4864 | 1792446221 | 32x48 | 1645760 |
| `ffn_gate` | 4864x896 | 895829913 | 32x48 | 1645888 |
| `attn_out` | 896x896 | 4244712744 | 32x48 | 303216 |

Generated temp fixture:

- `/tmp/phase28br_o_layer0_multifamily_trit/manifest.json`
- `/tmp/phase28br_o_layer0_multifamily_trit/layers/layer_000/ffn_up.trit`
- `/tmp/phase28br_o_layer0_multifamily_trit/layers/layer_000/ffn_down.trit`
- `/tmp/phase28br_o_layer0_multifamily_trit/layers/layer_000/ffn_gate.trit`
- `/tmp/phase28br_o_layer0_multifamily_trit/layers/layer_000/attn_out.trit`

Total sidecar bytes: `5240752`, matching Phase 28BO.

## Manifest Schema

The reconstructed manifest uses the Phase28Y parser path:

```json
{
  "format_name": "prt_residual_sidecar",
  "format_version": 1,
  "entries": [
    {
      "layer_index": 0,
      "tensor_family": "attn_out",
      "file_path": "layers/layer_000/attn_out.trit"
    }
  ]
}
```

The actual manifest contains all four layer-0 families with `tensor_family`, `layer_index`, relative `file_path`, rows, cols, block sizes, scale counts, byte sizes, seeds, and SHA-256 checksums. This avoids the legacy parser fallback that hardcodes `family=ffn_up`.

## Offline Validation

Command:

```bash
python3 examples/speculative/phase28br_o_reconstruct_fixture.py \
  --out-dir /tmp/phase28br_o_layer0_multifamily_trit \
  --report-json /tmp/phase28br_o_fixture_validation.json
```

Result: PASS.

| Family | Magic/CRC | Roundtrip | NaN | Inf | Decoded F32 Bytes |
|---|---:|---:|---:|---:|---:|
| `ffn_up` | valid | true | 0 | 0 | 17432576 |
| `ffn_down` | valid | true | 0 | 0 | 17432576 |
| `ffn_gate` | valid | true | 0 | 0 | 17432576 |
| `attn_out` | valid | true | 0 | 0 | 3211264 |

`attn_out` validation:

- rows: `896`
- cols: `896`
- block rows: `32`
- block cols: `48`
- scales: `532`
- payload offset: `32`
- scale offset: `301088`
- raw bytes: `303216`
- decoded F32 bytes: `3211264`

`covers(0, attn_out)=true`.

## Runtime Canary

Command:

```bash
timeout 120 ./build/bin/llama-cli \
  -m /home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf \
  -p "Hi" -n 1 -t 4 \
  --enable-prt-sidecar-pager \
  --prt-mode 5700 \
  --prt-sidecar-manifest /tmp/phase28br_o_layer0_multifamily_trit/manifest.json \
  --prt-sidecar-dir /tmp/phase28br_o_layer0_multifamily_trit/ \
  --prt-sidecar-budget-mb 512 \
  --prt-sidecar-apply \
  --prt-sidecar-apply-layer 0 \
  --prt-sidecar-apply-family attn_out \
  --prt-sidecar-true-injection \
  --top-k 10 \
  --temp 0 \
  --log-disable
```

Required line observed:

```text
[PRT-INJECT-CANARY] il=0 family=attn_out action=mutated_output R=[896,896] X=[896,30] out=[896,30] injection_attempts=1 injection_successes=1 injection_failures=0 injection_skipped=0 injection_shape_mismatch=0 injection_nonfinite_blocked=0 contribution_finite_before_injection=1 sidecar_math_influenced_output=1
```

The process hit the known `prt-mode 5700` scan timeout after the canary fired (`timeout` exit `124`). That does not block the canary result: true injection fired before timeout.

## Token / Logit Observation

Baseline:

```text
[TOKEN] id=9707 logit=22.5994 top_ids=9707,108386 top_logits=22.5994,-inf
```

True injection:

```text
[TOKEN] id=271 logit=13.2557 top_ids=271,198,220,311,11,374,320,646,13 top_logits=13.2557,-inf,-inf,-inf,-inf,-inf,-inf,-inf,-inf
```

Observation only. No quality claim.

## Hygiene

Generated sidecar data remained under `/tmp/phase28br_o_layer0_multifamily_trit/` and was not staged. Runtime logs were written under `/tmp` and were not staged. The canary log was large because `prt-mode 5700` continues scanning uncovered layers until timeout.

Allowed repo artifacts:

- `examples/speculative/phase28br_o_reconstruct_fixture.py`
- `examples/speculative/PHASE28BR_O_FIXTURE_RECONSTRUCTION_FREEZE.md`
- `examples/speculative/results/phase28br_o_fixture_reconstruction_freeze.json`

## Claim Boundary

PROVEN:

- Deterministic Phase 28BO layer-0 multi-family fixture was reconstructed in valid `.trit` format.
- Manifest uses `format_version: 1` and registers all four families without legacy `ffn_up` fallback.
- `attn_out` is registered and `covers(0, attn_out)=true`.
- All four `.trit` files validate and decode to finite values.
- Guarded true injection fired for layer 0 `attn_out`.
- Token/logit capture works for baseline and true-injection canary.

NOT CLAIMED:

- Quality improvement.
- Speed improvement.
- Broad benchmark result.
- Token 374 reproduction.
- Persistence of sidecar payloads in git.

Next recommended phase: Phase 28BR-P should freeze a narrow, non-timeout canary route or fix the `prt-mode 5700` uncovered-layer scan drain so future true-injection proof does not require accepting a timeout after success.
