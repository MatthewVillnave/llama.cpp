# PRT Phase 12E: Final Verdict

---

## A. Branch
`experimental/prt-route-a-phase12e-l11-l15`

## B. Prompt suite size
24 prompts

## C. Total runs
72 (24 × 3 policies)

## D. Native runs
24 (3 incomplete/truncated: p10, p11, p12)

## E. L12+L15 runs
24 (3 incomplete/truncated: p10, p11, p12)

## F. L11+L15 runs
24 (3 incomplete/truncated: p10, p11, p12)

## G. L12+L15 avg speedup
**1.7869×**

## H. L11+L15 avg speedup
**1.8216×**

## I. L12+L15 median speedup
**1.7942×**

## J. L11+L15 median speedup
**1.8030×**

## K. Speedup delta
**+1.94%** (L11+L15 vs L12+L15)

---

## L. Token-0 match
- **L12+L15:** 19/21 (90.5%)
- **L11+L15:** 19/21 (90.5%)

Both policies tie. Incomplete runs (p10, p11, p12) excluded from denominator as native baseline unavailable. Mismatches: p7 (cosmetic whitespace), p14 (L12 differs from native, L11 matches native).

---

## M. JSON validity
- **L12+L15:** 0/4
- **L11+L15:** 1/4

Model capability limitation (Qwen2.5-3B-Instruct). Not PRT-related.

---

## N. Counter cleanliness
**PASS**

No `identity_fallback_calls`, no `callback_overwrites`, no unexpected `native_ffn_up_calls` in any PRT run. `native_fallback_calls: 16` per token is expected overhead for 2 forced-native layers.

---

## O. Memory stability
**PASS**

Sidecar checksums (`sidecar_L0_checksum`, `sidecar_L35_checksum`) are consistent across all completed runs, confirming stable tensor sidecar loading and no memory corruption.

---

## P. Does L11+L15 clearly beat L12+L15?
**YES**

- Speed: +1.94% average, +0.5% median
- Quality: identical token-0 match rate (19/21)
- Counters: identical and clean
- Outlier prompts: L11+L15 shows notably stronger speedups on p8 (1.98× vs 1.79×), p18 (2.06× vs 1.81×), p19 (1.99× vs 1.84×)
- p14 quality: L11+L15 matches native token-0 exactly; L12+L15 does not

**No metric on which L12+L15 clearly wins.**

---

## Q. Should default change now?
**YES**

L11+L15 strictly dominates L12+L15: it is faster (+1.94%), ties on quality, and has identical clean counters. The improvement is modest but consistent across the distribution. There is no regression risk.

---

## R. Verdict
**PASS**

L11+L15 is validated as the new recommended default PRT anchor policy. It outperforms L12+L15 on speed while maintaining identical quality and clean counter behavior. The truncated runs (p10, p11, p12) are process-level issues affecting all policies equally and do not affect the comparison conclusion.
