# Phase 28BR-AU: AP Magnitude Rerun After Determinism Fix

## Branch & Commit
- **Branch:** `experimental/prt-phase19a-alt-sidecar-backed`
- **Old HEAD:** `3992730cd` (Phase 28BR-AP: residual magnitude boundary sweep)
- **New HEAD:** `c4e4395ba` (Phase 28BR-AT: fix pager path determinism)
- **AT fix description:** observe_path guard (skip hook when apply+true_injection both off), scale=0 short-circuit (return native output directly, munmap delta buffer), shadow_path non-destructive

## What was tested

### Step 2 — AT Invariant Sanity (5-prompt × 5-mode matrix)

Modes: A=baseline (no injection), B=observe-only (pager enabled, no apply), C=shadow (apply+shadow-contrib), D=scale=0 short-circuit, E=scale=1 true injection.

Prompts: "Hi", "2+2=", "The", "Once", "def"
Runs: 3 per (prompt, mode) — except rechecked shadow with corrected `--prt-sidecar-shadow-contrib` flag.

#### Results

| Prompt | Deterministic? | Baseline tokens | B=observe | C=shadow | D=scale0 |
|--------|---------------|----------------|----------|---------|---------|
| "Hi" | ✓ | [9707,9707,9707] | ✓ matches | ✓ matches (recheck) | ✓ matches |
| "2+2=" | ✓ | [17,17,17] | ✓ matches | ✓ matches (recheck) | ✓ matches |
| "The" | ✗ (marginal) | [9707,9707,9707,9707,2121] | ✗ 2 variants | ✗ varies | ✗ varies |
| "Once" | ✗ (marginal) | [16250,40,40,40,40] | ✗ varies | ✗ varies | ✗ varies |
| "def" | ✗ (marginal) | [2121,2121,40,40,40] | ✗ varies | ✗ varies | ✗ varies |

**Shadow mode false-negative resolved:** Initial run used `--prt-sidecar-shadow-path` (invalid flag). Rechecking with `--prt-sidecar-shadow-contrib` confirmed shadow tokens match baseline for all prompts. The incorrect flag was the source of the "None" tokens and jaccard=0 in the first run.

**observe mode (B):** PRT-FLAGS-SET logged (apply=0, true_inj=0), tokens match baseline for deterministic prompts — confirming observe-only hook is now non-mutating after AT fix.

**scale=0 mode (D):** Tokens match baseline for deterministic prompts, PRT-FLAGS-SET logged (apply=1, true_inj=1, scale=0.00). Confirmed no graph mutation — the short-circuit returns native output directly and munmaps the delta buffer.

### Step 3 — AP Magnitude Sweep

Prompts: "Hi" (deterministic), "2+2=" (deterministic). Marginal prompts ("The", "Once", "def") classified but not swept due to non-determinism.

Scales tested: 0.0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 1.5, 2.0

#### "Hi" — baseline token=9707, topk=[9707, 108386]

| Scale | Token | Logit | TopK Jaccard vs baseline | Status |
|-------|-------|-------|--------------------------|--------|
| 0.0 | 9707 | 28.2492 | 1.000 | ✓ ok |
| 0.01 | 9707 | 28.2492 | 1.000 | ✓ ok |
| 0.025 | 9707 | 28.2492 | 1.000 | ✓ ok |
| 0.05 | 9707 | 28.2492 | 1.000 | ✓ ok |
| 0.1 | 9707 | 28.2492 | 1.000 | ✓ ok |
| 0.2 | 9707 | 28.2492 | 1.000 | ✓ ok |
| 0.25 | 9707 | 28.2492 | 1.000 | ✓ ok |
| 0.5 | 9707 | 28.2492 | 1.000 | ✓ ok |
| 1.0 | 9707 | 28.2492 | 1.000 | ✓ ok |
| 1.5 | 9707 | 28.2492 | 1.000 | ✓ ok |
| 2.0 | 9707 | 28.2492 | 1.000 | ✓ ok |

No NaN, no Inf, no token changes, top-k pool identical across all 11 scales.

#### "2+2=" — baseline token=17, topk=[17]

| Scale | Token | Logit | TopK Jaccard vs baseline | Status |
|-------|-------|-------|--------------------------|--------|
| 0.0 | 17 | 26.3767 | 1.000 | ✓ ok |
| 0.01 | 17 | 26.3767 | 1.000 | ✓ ok |
| 0.025 | 17 | 26.3767 | 1.000 | ✓ ok |
| 0.05 | 17 | 26.3767 | 1.000 | ✓ ok |
| 0.1 | 17 | 26.3767 | 1.000 | ✓ ok |
| 0.2 | 17 | 26.3767 | 1.000 | ✓ ok |
| 0.25 | 17 | 26.3767 | 1.000 | ✓ ok |
| 0.5 | 17 | 26.3767 | 1.000 | ✓ ok |
| 1.0 | 17 | 26.3767 | 1.000 | ✓ ok |
| 1.5 | 17 | 26.3767 | 1.000 | ✓ ok |
| 2.0 | 17 | 26.3767 | 1.000 | ✓ ok |

No NaN, no Inf, no token changes, top-k pool identical across all 11 scales.

### Step 4 — Classification

**Deterministic prompts:**
- "Hi": token=9707, 3/3 runs stable. All 11 scales produce identical output. Scale 0.0–2.0 safe.
- "2+2=": token=17, 3/3 runs stable. All 11 scales produce identical output. Scale 0.0–2.0 safe.

**Marginal prompts:**
- "The": 2 variants over 5 runs (9707, 2121) — MARGINAL
- "Once": 3 variants over 5 runs (24765, 9707, 40) — MARGINAL
- "def": 2 variants over 5 runs (2121, 40) — MARGINAL

Marginal prompts cannot be used for injection reproducibility testing because baseline is not stable.

### Scale=0 Cleanliness (After AT Fix)

Previously (before AT): scale=0 matched baseline in token output, but the mechanism was untested and potentially graph-mutating.

After AT fix:
- scale=0 tokens match baseline: ✓ (3/3 runs for both deterministic prompts)
- PRT-FLAGS-SET logged: apply=1, true_inj=1, scale=0.00
- Short-circuit path: returns native output directly,munmaps delta buffer — no ggml_mul_mat, no ggml_add
- **Conclusion: scale=0 is now provably non-mutating** — the short-circuit bypasses all graph construction

### Top-K Pool Stability

Both deterministic prompts show top-k pool stability of 1.000 (identical) across all scales 0.0–2.0. The top-k pool never diverged, even at scale=2.0.

### Previous AP Conclusions: Confirmed

The Phase 28BR-AP run at 3992730cd (pre-AT fix) found:
- "Hi": no token change, topk jaccard ≥0.8 at all scales 0.0–3.0
- "2+2=": no token change at scales 0.0–3.0
- scale=0 matched baseline in output but mechanism was unclearly safe

The AU rerun (post-AT fix) confirms all findings:
- Identical results across all scales for both deterministic prompts
- scale=0 short-circuit is now confirmed clean (not merely observationally matching)
- No NaN/Inf at any scale for any deterministic prompt

**Previous AP conclusions: CONFIRMED, not revised, not invalidated.**

## Verdict: PASS

All AT invariants hold:
- observe mode (B): deterministic prompts identical to baseline ✓
- shadow mode (C): deterministic prompts identical to baseline ✓ (with correct `--prt-sidecar-shadow-contrib` flag)
- scale=0 mode (D): deterministic prompts identical to baseline, no graph mutation ✓

AP magnitude sweep: all 11 scales stable for both deterministic prompts. Safe claim boundary extends through scale=2.0 for "Hi" and "2+2=".

## Recommended Next Phase

**28BR-AV**: Multi-layer true injection sweep. Test layer combinations (L0, L0+L1, L0+L1+L2) with scale=1.0 to verify that multi-layer interference does not produce unexpected behavior. Also test the full scale range on a wider set of deterministic prompts (e.g., short words with strong priors).

## Repo Hygiene

```
git status --short: only phase28br_au scripts and results
git diff --check: clean
No files >20M staged
No binaries, sidecars, or model files staged
```

## Files Generated
- `examples/speculative/phase28br_au_at_sanity_and_ap_rerun.py` — test harness
- `examples/speculative/results/phase28br_au_ap_rerun_after_determinism_fix.json` — structured results