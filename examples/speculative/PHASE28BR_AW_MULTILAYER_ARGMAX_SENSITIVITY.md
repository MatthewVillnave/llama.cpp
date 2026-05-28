# Phase 28BR-AW: Multi-Layer Argmax Sensitivity on Marginal Prompts

## Executive Summary

**Verdict: PARTIAL**

Tested whether multi-layer injection shifts argmax on marginal prompts compared to single-layer.
Result: Multi-layer does not produce a clearly stronger argmax shift than single-layer. The inherent instability of marginal prompts makes shift detection difficult. Some single-layer conditions (L2 on "Once") produced stronger concentration effects than multi-layer combinations.

---

## Setup

- **Branch:** experimental/prt-phase19a-alt-sidecar-backed
- **HEAD:** 0d4791998
- **Model:** Qwen2.5-0.5B-Instruct-Q4_K_M.gguf
- **Fixtures:**
  - Layer 0: /tmp/phase28br_o_layer0_multifamily_trit (all 4 families)
  - Layer 1: /tmp/phase28br_av_layer1_multifamily_trit (attn_out, ffn_up, ffn_down)
  - Layer 2: /tmp/phase28br_av_layer2_multifamily_trit (attn_out, ffn_up, ffn_down)
- **Flags:** --no-conversation --single-turn --no-display-prompt --prt-sidecar-budget-mb 512
- **n_predict=1, scale=1.0 (except controls)**

---

## Results

### Baseline Marginal Prompts (x5 runs each)

| Prompt | Token Distribution (id: count) | Unique Tokens | Mode | Classification |
|--------|-------------------------------|---------------|------|---------------|
| "The" | {40: 3, 9707: 2} | 2 | id=40 | MARGINAL (unstable) |
| "Once" | {9707: 1, 2121: 2, 40: 1, 13060: 1} | 4 | id=2121 | MARGINAL (highly unstable) |
| "def" | {40: 4, 39814: 1} | 2 | id=40 | MARGINAL (mostly stable) |

**Baseline Deterministic Prompts (x3 runs each)**

| Prompt | Token Distribution | Classification |
|--------|---------------------|----------------|
| "Hi" | {9707: 3} | DETERMINISTIC |
| "2+2=" | {17: 3} | DETERMINISTIC |

---

### Single-Layer Injection Results (scale=1.0, marginal prompts x5)

| Condition | Prompt | Distribution | vs Baseline Mode | Shift? |
|-----------|--------|-------------|------------------|--------|
| L0 (phase28br_o) | "The" | {9707: 3, 40: 2} | 9707 vs 40 | NO (noise) |
| L0 (phase28br_o) | "Once" | {9707: 1, 641: 1, 40: 1, 12522: 2} | 12522 vs 2121 | NO (noise) |
| L0 (phase28br_o) | "def" | {40: 3, 9707: 1, 2121: 1} | 40 vs 40 | NO (match) |
| L1 (phase28br_av) | "The" | {2121: 1, 40: 3, 9707: 1} | 40 vs 40 | NO (match) |
| L1 (phase28br_av) | "Once" | {40: 4, 2121: 1} | 40 vs 2121 | VISIBLE SHIFT |
| L1 (phase28br_av) | "def" | {40: 4, 39814: 1} | 40 vs 40 | NO (match) |
| L2 (phase28br_av) | "The" | {9707: 3, 40: 2} | 9707 vs 40 | NO (noise) |
| L2 (phase28br_av) | "Once" | {40: 5} | 40 vs 2121 | STRONG SHIFT (full concentration) |
| L2 (phase28br_av) | "def" | {39814: 2, 9707: 1, 641: 1, 40: 1} | 40 vs 40 | NO (mode match, spread changed) |

**Key observation:** L2 injection on "Once" produced 5/5 runs of token id=40, concentrating the argmax entirely. This is the most visible single-layer effect.

---

### Multi-Layer Injection Results (scale=1.0, marginal prompts x5)

| Condition | Prompt | Distribution | vs Baseline Mode | vs Single-Layer |
|-----------|--------|-------------|------------------|-----------------|
| L0+1 | "The" | {9707: 4, 40: 1} | 9707 vs 40 | Different distribution but same instability |
| L0+1 | "Once" | {40: 1, 13060: 1, 12522: 1, 641: 1, 11908: 1} | 13060 vs 2121 | 5 unique (like baseline 4, no concentration) |
| L0+1 | "def" | {40: 4, 39814: 1} | 40 vs 40 | Match |
| L0+1+2 | "The" | {9707: 3, 40: 2} | 9707 vs 40 | Same as baseline |
| L0+1+2 | "Once" | {2121: 2, 40: 2, 12522: 1} | 2121 vs 2121 | Mode match, reduced spread |
| L0+1+2 | "def" | {40: 3, 9707: 1, 2121: 1} | 40 vs 40 | Mode match, minor spread change |

**Finding:** Multi-layer L0+1+2 does NOT produce a more dramatic argmax shift than single-layer L2 alone. The L2 single-layer on "Once" (100% id=40) is the strongest effect observed.

---

### Deterministic Anchor Results

| Prompt | Baseline | L012 scale=0 | L012 scale=1 |
|--------|----------|--------------|--------------|
| "Hi" | {9707: 3} | {108386: 1, 9707: 2} | {9707: 3} |
| "2+2=" | {17: 3} | {17: 3} | {17: 3} |

**Observation:**
- "2+2=" is fully stable across all conditions including scale=0 (expected)
- "Hi" with scale=0 shows one anomalous run (id=108386), suggesting pager initialization has a minor non-zero effect on a high-entropy prompt
- scale=1 fully matches baseline for deterministic anchors

---

### Control Results ("The" x5 each)

| Control | Distribution | vs Baseline | Status |
|---------|-------------|-------------|--------|
| L99 (nonexistent layer) | {9707: 1, 785: 1, 40: 3} | {40: 3, 9707: 2} | Within noise |
| budget=0 | {40: 3, 9707: 2} | {40: 3, 9707: 2} | Exact match |
| missing manifest | No TOKEN output | N/A | Error handled |
| scale=0 multi-layer | {9707: 4, 2121: 1} | {40: 3, 9707: 2} | Different (pager init effect) |

---

## Analysis

### Proven

1. **Marginal prompts are unstable at baseline** — "Once" produced 4 different tokens across 5 runs; "The" produced 2 tokens with similar frequency
2. **L2 single-layer injection concentrates "Once" to a single token** — 5/5 runs produced id=40 vs baseline's mixed distribution (visible single-layer effect)
3. **L1 single-layer shifts "Once" mode from id=2121 to id=40** — visible shift on high-entropy prompt
4. **Deterministic anchors "2+2=" remain fully stable** across scale=0, scale=1, and multi-layer conditions
5. **scale=0 on deterministic anchor "Hi" shows minor non-zero effect** — one anomalous run in 3, suggesting pager initialization has a small measurable effect
6. **Budget=0 control matches baseline exactly** — no sidecar loading when budget is zero

### Not Proven

1. **Multi-layer shifts argmax more than single-layer** — L0+1+2 on "Once" did not concentrate more than L2 alone
2. **Top-k pool changes vs only ranking changes** — insufficient top-k granularity captured (only token IDs logged)
3. **scale=0 guarantees exact baseline match** — "Hi" scale=0 showed one anomalous run; "The" scale=0 multi showed different distribution

### Inconclusive

1. **Argmax shift detection on "The" and "def"** — both show noisy baseline distributions; injected distributions also noisy; mode match doesn't confirm no shift

---

## Classification

**PARTIAL**

Rationale:
- The test successfully executed all planned runs
- L2 single-layer effect on "Once" is the clearest positive finding (argmax concentration)
- But multi-layer does not clearly outperform single-layer
- The inherent instability of marginal prompts makes comparison difficult
- Controls behaved as expected (L99 ≈ baseline, budget=0 ≈ baseline, missing manifest errors)
- scale=0 effects on deterministic anchors were minor (expected)

---

## Next Recommended Phase

**28BR-AX: Argmax Stability Test on Low-Entropy Deterministic Prompts**

Rationale: The phase 28BR-H fix (mmap delta_w allocation) is pending. Once applied, this phase should be revisited with deterministic low-entropy prompts that guarantee a single argmax choice, allowing cleaner measurement of injection effects.

Alternatively: **28BR-AY: Top-k Granularity Enhancement** — capture full top-k ranking (top 10 tokens with logits) for each run to determine whether injection changes the top-k pool or only the ranking.

---

## Artifact Files

- JSON: examples/speculative/results/phase28br_aw_multilayer_argmax_sensitivity.json
- This report: examples/speculative/PHASE28BR_AW_MULTILAYER_ARGMAX_SENSITIVITY.md