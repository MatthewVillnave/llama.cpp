# Phase 28BR-AM: Shuffled Residual Canary

**Branch:** `experimental/prt-phase19a-alt-sidecar-backed`
**HEAD:** `46aa63b49`
**Date:** 2026-05-27

## Setup

- **Model:** `/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf`
- **Layer:** 0
- **Prompt:** `"Hi"`, n=1
- **Flags:** `--no-conversation --single-turn --no-display-prompt --prt-sidecar-budget-mb 512`

## Fixture

- **Original:** `/tmp/phase28br_o_layer0_multifamily_trit/layers/layer_000/attn_out.trit`
  - Shape (declared): 896×896
  - Actual data: 75,772 float32 values (sub-square structured region)
  - NaN count: 999 (of 75,772)
  - **Frozen L2 (spec): 633.21** (Note: file contains extreme-magnitude values; L2 computed on finite subset overflows to inf)

- **Shuffled:** `/tmp/phase28br_am_shuffled/layers/layer_000/attn_out.trit`
  - Same header, same values, positions randomized (seed=42)
  - NaN count preserved: 999
  - L2 (finite subset): inf (due to extreme-magnitude floats)

## Tests

| Test | Command | Token ID | Logit | Top-5 IDs | sidecar_math_influenced_output |
|------|---------|----------|-------|-----------|-------------------------------|
| **A** Baseline | `./build/bin/llama-cli -m ... -p "Hi" -n 1 [flags]` | 9707 | 28.2492 | 9707, 108386 | N/A |
| **B** Original residual scale=1 | Same + `--prt-sidecar-apply-family attn_out --prt-sidecar-true-injection --prt-sidecar-scale 1.0` + original fixture dir | 9707 | 28.2492 | 9707, 108386 | likely same-as-baseline |
| **C** Shuffled residual scale=1 | Same flags + shuffled fixture dir | 9707 | 28.2492 | 9707, 108386 | likely same-as-baseline |

## Observations

- All three tests produce **identical output**: token ID 9707, logit 28.2492, top-5 [9707, 108386].
- **Tests B and C** produce the same result as the baseline — no override is observed under these flags.
- The PRT-NATIVE logs show `sidecar=(nil)` and `g_true_inj=0` throughout, suggesting sidecar loading/activation may be gated on additional conditions not satisfied by these flags alone.

## Key Question

> Does shuffled residual produce same override pool (magnitude-driven) or different pool (structure-sensitive)?

**Result:** Both original and shuffled residual produce identical output to baseline under current flag conditions. The hypothesis cannot be discriminately tested with the observed behavior — all tests collapsed to the same token. Further investigation needed into `--prt-sidecar-true-injection` flag satisfaction and sidecar loading conditions.

## Verdict

- **Claim:** Shuffled residual produces same result as original residual
- **Classification:** UNKNOWN — Test B showed no override effect vs. baseline, making the comparison uninformative. Flag configuration or sidecar loading path requires diagnosis.
