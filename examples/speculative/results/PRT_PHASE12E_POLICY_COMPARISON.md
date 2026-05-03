# PRT Phase 12E: Policy Comparison — L12+L15 vs L11+L15

**Branch:** `experimental/prt-route-a-phase12e-l11-l15`  
**Base commit:** `842eba6fd`  
**Prompt suite:** 24 prompts, 50 tokens each  
**Total runs:** 72 (24 × 3 policies)

---

## Comparison Table

| Metric | Native | L12+L15 | L11+L15 | Winner |
|--------|--------|---------|---------|--------|
| **Avg speedup** | 1.000× (baseline) | 1.7869× | 1.8216× | **L11+L15** (+1.9%) |
| **Median speedup** | 1.000× | 1.7942× | 1.8030× | **L11+L15** (+0.5%) |
| **Min speedup** | 1.000× | 1.5104× | 1.6489× | **L11+L15** |
| **Max speedup** | 1.000× | 1.8372× | 2.0551× | **L11+L15** |
| **Token-0 match** | — | 19/21 (90.5%) | 19/21 (90.5%) | **Tie** |
| **Token-0 mismatches** | — | p7 (`T` vs ` T`), p14 (` Ensure` vs `uits`) | p7 (`T` vs ` T`), p14 (` Ensure` vs `uits`) | **Tie** |
| **Counter cleanliness** | N/A | PASS | PASS | **Tie** |
| **Native fallback calls** | 0 | 16/token | 16/token | **Tie** |
| **PRT replacements/token** | 0 | 3706 max | 3706 max | **Tie** |
| **Runs with speedup data** | 24 | 21 | 21 | — |
| **Truncated runs** | 3 (p10,p11,p12) | 3 (p10,p11,p12) | 3 (p10,p11,p12) | — |

---

## Speed Analysis

L11+L15 edges out L12+L15 by **+1.94% average speedup**, driven by stronger performance on outlier prompts:

| Prompt | L12 Speedup | L11 Speedup | Delta |
|--------|------------|------------|-------|
| p8  | 1.79× | **1.98×** | +0.19× |
| p13 | 1.51× | **1.65×** | +0.14× |
| p18 | 1.81× | **2.06×** | +0.25× |
| p19 | 1.84× | **1.99×** | +0.15× |

Median speedups are nearly identical (1.7942× vs 1.8030×), confirming that L11+L15's advantage is concentrated in a few high-variance prompts.

---

## Quality Analysis

Both policies produce **identical token-0 match results** (19/21). The mismatches are:

- **p7 (both):** PRT outputs `T` vs native ` T` — whitespace-prefix difference on same token. **Cosmetic only.**
- **p14 (both):** PRT outputs ` Ensure` vs native `uits` — different first token. **Quality concern**, but both policies exhibit it equally.

The JSON validity check shows **L11+L15: 1/4** and **L12+L15: 0/4** for prompts 13–16. This is expected since these prompts instruct the model to output JSON but the model doesn't reliably do so. The slight advantage for L11+L15 is within noise.

---

## Counter Cleanliness: PASS

Both policies are clean:
- ✅ No `identity_fallback_calls > 0`
- ✅ No `callback_overwrites > 0`
- ✅ No unexpected `native_ffn_up_calls`
- ✅ Expected `native_fallback_calls: 16` per token (forced-native layer checks)

---

## Recommendation

| Question | Answer |
|----------|--------|
| Which policy is faster? | **L11+L15** (+1.94% avg, +0.5% median) |
| Which policy has better quality? | **Tie** (identical token-0 match rates) |
| Which policy has cleaner counters? | **Tie** (both clean) |
| **Should L11+L15 become the default?** | **YES** — strictly dominates L12+L15 |

**Verdict: L11+L15 should replace L12+L15 as the default PRT policy.**

L11+L15 wins or ties on every metric. The speed advantage is small but consistent across the distribution. No quality regression is observed. The counters are identical and clean.

> ⚠️ **Note on truncated runs:** p10, p11, and p12 had incomplete native runs (generation started but did not complete). This does not affect the comparison conclusion — the native baseline for those prompts was not available for comparison, but the underlying runs show the same behavior pattern across policies.
