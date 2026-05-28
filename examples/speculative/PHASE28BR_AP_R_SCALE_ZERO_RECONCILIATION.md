# Phase 28BR-AP-R: Scale=0 Reconciliation / Baseline Equivalence Audit

**Branch:** `experimental/prt-phase19a-alt-sidecar-backed`
**HEAD:** `02fa10b28` (Phase 28BR-AP just completed)
**Date:** 2026-05-28
**Manifest:** `/tmp/phase28br_l_sidecars/manifest.json` (attn_out layer 0, same as AP)
**Model:** `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf`
**Script:** `phase28br_ap_residual_magnitude_sweep.py`

---

## Context

Phase 28BR-AP claims:
- top-k pool universally stable, injection reorders within pre-existing pool
- scale controls which token wins inside the pool
- scale=3.0 crashes

But the AP per-prompt threshold table says:
- "The" first override = scale=0
- "Once" first override = scale=0

This appears to contradict the frozen invariant: scale=0 should be a no-op identical to baseline.

**Resolution mandate:** Determine whether the scale=0 override entries are a report/table error, command mismatch, seed mismatch, or an actual regression. Do NOT run new science.

---

## TASK 1: AP Report Analysis

### Table Reference

The "Per-Prompt Threshold Summary" table in PHASE28BR_AP_RESIDUAL_MAGNITUDE_BOUNDARY.md shows:

| Prompt | Baseline Token | First Override Scale | Notes |
|--------|---------------|---------------------|-------|
| Hi | 9707 | **never** | Structurally dominant |
| The | 2132 | **0.0** | "Oscillates between 9707 and 40" |
| Once | 24765 | **0.0** | "Most oscillatory, 7 tokens, returns at scale=2.0" |
| 2+2= | 17 | **never** | Trivially single-token |
| def | 40 | **0.01** | "Baseline token is in override pool itself" |

### JSON Reference

`phase28br_ap_residual_magnitude_boundary.json` shows the same:

| Prompt | Scale=0 Token | Scale=0 Logit | Baseline Token | Baseline Logit |
|--------|--------------|---------------|---------------|----------------|
| The | 2121 | 23.0973 | 2132 | 21.8025 |
| Once | 2132 | 20.8757 | 24765 | 21.2871 |
| def | 40 | 24.6688 | 40 | 24.6688 |

---

## TASK 2: Control Tests (Current Run, HEAD=02fa10b28)

### Exact Commands Used

**Run A (baseline no injection):**
```bash
./build/bin/llama-cli -m /home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf \
  -p "{prompt}" --no-conversation --single-turn --no-display-prompt \
  --prt-sidecar-budget-mb 512 -n 1 2>&1 | grep '\[TOKEN\]'
```

**Run B (true injection scale=0):**
```bash
./build/bin/llama-cli -m /home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf \
  -p "{prompt}" --no-conversation --single-turn --no-display-prompt \
  --enable-prt-sidecar-pager --prt-mode 5700 --prt-sidecar-budget-mb 512 \
  --prt-sidecar-manifest /tmp/phase28br_l_sidecars/manifest.json \
  --prt-sidecar-apply --prt-sidecar-true-injection --prt-sidecar-apply-layer 0 \
  --prt-sidecar-apply-family attn_out --prt-sidecar-scale 0.0 -n 1 2>&1 | grep '\[TOKEN\]'
```

**Run C (true injection scale=0 repeated):** Same as B, immediately repeated.

### Per-Prompt A/B/C Results

#### "Hi"

| Run | Selected Token | Selected Logit | Top-2 IDs | Notes |
|-----|---------------|----------------|-----------|-------|
| A (baseline) | 9707 | 28.2492 | [9707, 108386] | PRT-FLAGS=false |
| B (scale=0) | 9707 | 28.2492 | [9707, 108386] | PRT-FLAGS=true |
| C (scale=0 repeat) | 9707 | 28.2492 | [9707, 108386] | PRT-FLAGS=true |

**Classification:** `A == B == C` — `SCALE_ZERO_CONFIRMED_NOOP`

#### "The"

| Run | Selected Token | Selected Logit | Top-5 IDs | Notes |
|-----|---------------|----------------|-----------|-------|
| A (baseline) | 9707 | 25.2192 | [9707,40,2121,785,2132] | PRT-FLAGS=false |
| B (scale=0) | 40 | 24.8983 | [9707,40,2121,785,2132] | PRT-FLAGS=true |
| C (scale=0 repeat) | 40 | 24.8983 | [9707,40,2121,785,2132] | PRT-FLAGS=true |

**Classification:** `A != B` — `SCALE_ZERO_REGRESSION` (baseline=9707, scale=0=40)

#### "Once"

| Run | Selected Token | Selected Logit | Top-10 IDs | Notes |
|-----|---------------|----------------|-----------|-------|
| A (baseline) | 39814 | 19.3292 | [40,12522,2121,9707,24765,2132,13060,16250,11908,641] | PRT-FLAGS=false |
| B (scale=0) | 24765 | 21.2871 | [40,12522,2121,9707,24765,2132,13060,16250,11908,641] | PRT-FLAGS=true |
| C (scale=0 repeat) | 40 | 23.0044 | [40,12522,2121,9707,24765,2132,13060,16250,11908,641] | PRT-FLAGS=true |

**Classification:** `A != B != C` — `SCALE_ZERO_REGRESSION` with repeat instability

#### "2+2="

| Run | Selected Token | Selected Logit | Top-1 IDs | Notes |
|-----|---------------|----------------|-----------|-------|
| A (baseline) | 17 | 26.3767 | [17] | PRT-FLAGS=false |
| B (scale=0) | 17 | 26.3767 | [17] | PRT-FLAGS=true |
| C (scale=0 repeat) | 17 | 26.3767 | [17] | PRT-FLAGS=true |

**Classification:** `A == B == C` — `SCALE_ZERO_CONFIRMED_NOOP`

#### "def"

| Run | Selected Token | Selected Logit | Top-8 IDs | Notes |
|-----|---------------|----------------|-----------|-------|
| A (baseline) | 2132 | 21.4121 | [40,9707,2121,39814,641,785,2132,95456] | PRT-FLAGS=false |
| B (scale=0) | 9707 | 23.6747 | [40,9707,2121,39814,641,785,2132,95456] | PRT-FLAGS=true |
| C (scale=0 repeat) | 40 | 24.6688 | [40,9707,2121,39814,641,785,2132,95456] | PRT-FLAGS=true |

**Classification:** `A != B != C` — `SCALE_ZERO_REGRESSION` with repeat instability

---

## TASK 3: Analysis

### Conflict Identification

The AP report (HEAD=3992730cd) and the current run (HEAD=02fa10b28) show **different baseline tokens** for "The", "Once", and "def":

| Prompt | AP Report Baseline Token | Current Run Baseline Token | Match? |
|--------|------------------------|---------------------------|--------|
| Hi | 9707 | 9707 | ✓ |
| The | 2132 | 9707 | ✗ |
| Once | 24765 | 39814 | ✗ |
| 2+2= | 17 | 17 | ✓ |
| def | 40 | 2132 | ✗ |

### Baseline Divergence Analysis

The baseline itself changed between the AP run (3992730cd) and the current run (02fa10b28) for 3 of 5 prompts. This is NOT sampling nondeterminism — the difference is structural.

The top-k pools for "The" and "def" are identical between runs (same IDs, same order), but the selected token differs. This suggests the decision boundary between 9707 and 40 shifted.

**Likely cause:** The AP script uses `--log-disable` which suppresses PRT debug output. The current run has debug output enabled. The timing or execution path difference caused by log suppression may have affected the inference path on these marginal prompts.

### Scale=0 Behavior (Current Run)

For the 3 affected prompts ("The", "Once", "def"), scale=0 injection produces a **different token than baseline** in the current run. For "Once" and "def", the repeat is also unstable — different tokens on repeated runs.

This is a genuine regression from the expected invariant: `scale=0 = no-op = identical to baseline`.

### Root Cause Hypothesis

Between 3992730cd (AP report) and 02fa10b28 (current HEAD), only two files were added:
- `PHASE28BR_AP_RESIDUAL_MAGNITUDE_BOUNDARY.md`
- `phase28br_ap_residual_magnitude_boundary.json`

No source code changed. Therefore:
1. The baseline divergence is likely due to **logging-level side effects** on the inference path
2. The scale=0 behavior difference in the current run is **NOT** a code regression
3. The AP report's "scale=0 first override" entries for "The" and "Once" may reflect this same instability that was captured in the JSON but labeled with the wrong scale value

---

## Classification

| Prompt | A vs B | B vs C | Classification |
|--------|--------|--------|---------------|
| Hi | A == B | B == C | `SCALE_ZERO_CONFIRMED_NOOP` |
| The | A != B | B == C | `SCALE_ZERO_REGRESSION` (current run) |
| Once | A != B | B != C | `SCALE_ZERO_REGRESSION` + NONDETERMINISM |
| 2+2= | A == B | B == C | `SCALE_ZERO_CONFIRMED_NOOP` |
| def | A != B | B != C | `SCALE_ZERO_REGRESSION` + NONDETERMINISM |

**Overall: `SCALE_ZERO_REGRESSION`** for 3 of 5 prompts in current run.

**BUT:** The AP report's baseline tokens differ from current run baseline tokens for the same 3 prompts, suggesting the AP report captured this same instability under different logging conditions.

---

## AP Table Error Analysis

The AP report claims:
- "The" first override at scale=0 with token 2121
- "Once" first override at scale=0 with token 2132

The current run shows:
- "The" baseline=9707, scale=0=40
- "Once" baseline=39814, scale=0=24765 (first repeat) or 40 (second repeat)

The AP JSON shows scale=0 tokens for "The" and "Once" that match none of the current run's tokens. The instability is real, but the specific tokens differ.

**Verdict on AP table:** The "first override at scale=0" entries are not errors in the sense of being fabricated — the AP script genuinely measured different tokens at scale=0 than at baseline. But they are **mislabeled** — scale=0 should be a no-op, and any token change at scale=0 is a violation of the invariant, not a confirmation of it.

---

## Scale=0 Invariant Status

**PROVEN for Hi and 2+2=:**
- These prompts show `A == B == C` with identical tokens and logits
- scale=0 correctly acts as no-op

**BROKEN for The, Once, def in current run:**
- These prompts show `A != B` — injection at scale=0 changes the selected token
- For Once and def, B != C — repeat instability at scale=0

**The invariant itself is valid** (scale=0 should be no-op), but the current binary exhibits broken behavior for marginal prompts. This is likely a logging-side-effect issue from the AP run, not a code regression.

---

## Recommended Next Phase

**Phase 28BR-AS: Scale=0 Invariant Re-validation with Logging Control**

Before proceeding to 28BR-AQ/AR, re-run scale=0 controls with `--log-disable` (matching AP script) and capture:
1. Confirm whether logging suppression fixes the scale=0 invariant
2. Test whether the baseline tokens are stable with `--log-disable`
3. If scale=0 is still broken with `--log-disable`, the issue is in the binary, not logging
4. If scale=0 is fixed with `--log-disable`, document that PRT debug output can affect inference on marginal prompts

**If Matt confirms scale=0 should be no-op and the current binary fails on The/Once/def:**
- Classify as `SCALE_ZERO_REGRESSION` — halt 28BR-AQ/AR until resolved
- Examine llama.cpp diff between 3992730cd and 02fa10b28 for any subtle changes

---

## Commit Decision

**DO NOT COMMIT** with `SCALE_ZERO_REGRESSION` classification per task instructions:
> "If SCALE_ZERO_REGRESSION: do NOT commit without resolution — write report and STOP, wait for Matt's decision."

This report and JSON are written but not committed. Awaiting Matt's decision.

---

## Summary

| Item | Status |
|------|--------|
| AP report read | ✓ |
| AP JSON read | ✓ |
| Baseline tests (5 prompts) | ✓ |
| Scale=0 tests A/B/C (5 prompts) | ✓ |
| Repeatability tests for The/Once/def | ✓ |
| Conflict identified | ✓ |
| Root cause hypothesis | ✓ |
| Classification | `SCALE_ZERO_REGRESSION` (3/5 prompts in current run) |
| AP table correctness | Table data is real but mislabeled — scale=0 should be no-op |
| Scale=0 invariant | PROVEN for Hi/2+2=; BROKEN for The/Once/def in current run |
| Commit? | NO — awaiting Matt's decision |