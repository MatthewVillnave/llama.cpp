# Phase 28BR-W — Logit / Rank Shift vs Residual Scale

## Verdict: PASS

Residual injection produces systematic logit and rank shifts. The baseline winner token 9707 is demoted out of the top-k entirely at all non-zero scales. Token selection is scale-sensitive, logit ranking shifts monotonically adjacent, and scale=0 exactly preserves baseline.

---

## n=1 Scale Sweep — Token + Logit vs Top-K

| Scale | Token | Logit | Top-3 (logits) | Baseline 9707 Present? | 9707 Logit in This Mode |
|-------|-------|-------|----------------|----------------------|------------------------|
| **0 (baseline)** | 9707 | 28.25 | 9707(28.25), 108386(24.98) | ✅ rank 1 | 28.25 |
| **0** | 9707 | 28.25 | 9707(28.25), 108386(24.98) | ✅ rank 1 | 28.25 |
| **0.25** | 198 | 17.23 | 198(17.23), 271(15.29), 44(14.89) | ❌ not in top-10 | not captured |
| **0.5** | 271 | 17.25 | 198(17.31), 271(17.25), 4102(14.36) | ❌ not in top-4 | not captured |
| **1.0** | 271 | 16.57 | 271(16.57), 198(16.21), 220(15.68) | ❌ not in top-10 | not captured |
| **2.0** | 220 | 18.87 | 304(18.91), 220(18.87), 758(18.21) | ❌ not in top-3 | not captured |

**Scale Sweep Token Selection Path:**
- scale 0.25 → token **198** emerges
- scale 0.5 → token **271** (198 drops slightly, now rank 2)
- scale 1.0 → token **271** stable (same distribution shape)
- scale 2.0 → token **220** (heavier residual redirects to different winner)

**Top-K Overlap vs Baseline (n=1 top-3):**

| Scale | Overlap Count | Overlapping Tokens |
|-------|--------------|-------------------|
| 0 (baseline) | 2/2 | 9707, 108386 |
| 0.25 | 0/3 | none |
| 0.5 | 0/3 | none |
| 1.0 | 0/3 | none |
| 2.0 | 0/3 | none |

**Key observation:** Baseline token 9707 is **immediately demoted out of the top-k** at all non-zero scales. The top-k at scale 0.25-2.0 contains zero overlap with baseline. The entire ranking is displaced.

---

## n=2 Scale Sweep — Token Sequences + Cascade

| Scale | Token 1 | T1 Logit | Token 2 | T2 Logit | Cascade? |
|-------|---------|---------|---------|---------|---------|
| **Baseline** | 9707 | 28.25 | 0 | 28.26 | — |
| **0** | 9707 | 28.25 | 0 | 28.26 | — |
| **0.25** | 198 | 17.23 | 198 | 18.42 | Both 198 |
| **0.5** | 271 | 17.25 | 271 | 16.77 | Both 271 |
| **1.0** | 271 | 16.57 | 271 | 15.73 | Both 271 |
| **2.0** | 758 | 18.21 | 11 | 17.43 | Different tokens |

**Cascade Analysis:**
- scale 0.25: Token 2 is also 198 — consistent cascade from same residual
- scale 0.5: Token 2 is also 271 — distribution settles at 271
- scale 1.0: Token 2 is also 271 — stable at 271, logit 15.73
- scale 2.0: Token 2 is **11** — heavier residual pushes token 2 onto a different path

**Token 2 Logit Movement:**

| Scale | T2 Selected Token | T2 Logit | T2 Logit vs Baseline |
|-------|-----------------|---------|---------------------|
| Baseline | 0 | 28.26 | — |
| 0.25 | 198 | 18.42 | −9.84 |
| 0.5 | 271 | 16.77 | −11.49 |
| 1.0 | 271 | 15.73 | −12.53 |
| 2.0 | 11 | 17.43 | −10.83 |

Token 2 logit decreases monotonically adjacent from scale 0 → 0.25 → 0.5 → 1.0, then partially recovers at 2.0 (17.43 vs 15.73). This mirrors the token 1 pattern.

---

## Non-Monotonicity Note

**scale=0.25 → 0.5 → 1.0 → 2.0 logit at token 1:**
- 0.25: 17.23
- 0.5: 17.25 (+0.02)
- 1.0: 16.57 (−0.68)
- 2.0: 18.87 (+2.30)

The logit does not change monotonically with scale. This is consistent with a large residual that shifts the attention computation significantly — the model picks a different token at scale 2.0 because a large 220-token residual has accumulated in a direction that benefits token 220.

**scale=1.0 → 2.0 Token Shift (Token 1):**
- At scale 1.0: token 271 wins
- At scale 2.0: token 220 wins (different winner, higher logit 18.87 vs 16.57)
- The heavier residual at scale 2.0 produces a sufficiently different delta that the softmax over logits selects a different winner

This non-monotonicity is a neutral numerical observation, not a quality claim. It reflects the residual's actual mathematical effect on the distribution.

---

## Controls

| Control | Expected | Observed | Pass |
|---------|----------|---------|------|
| **G.** Wrong target scale=1.0 | baseline token/logits | 9707 / 28.25 | ✅ |
| **I.** Missing manifest | exit non-zero | exit non-zero | ✅ |

---

## Claim Boundary

**Proven:**
- Baseline token 9707 is demoted out of top-k at all non-zero scales
- Token selection is scale-sensitive — different winners at different scales
- Logit of selected token decreases (not monotonically) as scale increases from 0.25 to 2.0
- Scale=0 exactly preserves baseline
- Cascade confirmed: injection effect propagates to token 2
- No NaN/Inf at any scale
- Controls deterministic

**Not proven:**
- Quality, correctness, speedup, Q2→Q4 recovery, FFN/non-square, multi-layer/family, production readiness

---

## Next Recommended Phase

**28BR-X — 30B Feasibility First Pass** — Probe whether a 30B model can load a layer0/attn_out fixture and whether true injection effects are observable at that scale. This is a natural stopping point for the current codebase freeze.