# Phase 30A — Low-Precision Base Residency Test

**Branch:** `experimental/prt-phase19a-alt-sidecar-backed`
**Commit:** `1c9f2d9dd` (unchanged — no new commit yet)
**Date:** 2026-05-28

---

## Overview

Test whether lower-bit base models (Q3_K_M, Q2_K) derived from Q4_K_M can achieve lower RSS when combined with reference sidecars from the Q4_K_M base. Goal: determine if PRT sidecar injection is viable on lower-precision bases for memory reduction.

---

## Test Matrix

| Test | Base | Sidecars | n_predict | Description |
|------|------|----------|-----------|-------------|
| A1 | Q4_K_M | No | 1 | Q4 baseline |
| A2 | Q4_K_M | No | 8 | Q4 baseline |
| A3 | Q4_K_M | No | 32 | Q4 baseline |
| B1 | Q3_K_M | No | 1 | Q3 baseline |
| B2 | Q3_K_M | No | 8 | Q3 baseline |
| B3 | Q3_K_M | No | 32 | Q3 baseline |
| C1 | Q3_K_M | attn_out layer0 (true injection) | 8 | Q3 + sidecars |
| D1 | Q2_K | No | 1 | Q2 baseline |
| D2 | Q2_K | No | 8 | Q2 baseline |
| D3 | Q2_K | No | 32 | Q2 baseline |
| E1 | Q2_K | attn_out layer0 (true injection) | 8 | Q2 + sidecars |

---

## RSS Results (Peak Resident Set Size)

All measurements via `llama-cli` with `--log-disable -t 0`. RSS from `/usr/bin/time -v`.

| Test | RSS (KB) | Wall time | Δ vs Q4 baseline n=8 |
|------|---------|-----------|----------------------|
| A1 (Q4 n=1) | 874,720 | 0.68s | — |
| A2 (Q4 n=8) | 874,736 | 0.78s | baseline |
| A3 (Q4 n=32) | 874,616 | 0.80s | — |
| B1 (Q3 n=1) | 984,332 | 0.71s | — |
| B2 (Q3 n=8) | 984,236 | 0.81s | +109,500 (+12.5%) |
| B3 (Q3 n=32) | 984,588 | 0.80s | — |
| C1 (Q3 + sidecars n=8) | 999,064 | 0.80s | +124,328 (+14.2%) |
| D1 (Q2 n=1) | 922,856 | 0.69s | — |
| D2 (Q2 n=8) | 923,320 | 0.81s | +48,584 (+5.6%) |
| D3 (Q2 n=32) | 922,784 | 1.20s | — |
| E1 (Q2 + sidecars n=8) | 938,344 | 0.83s | +63,608 (+7.3%) |

---

## Sidecar Injection Smoke

| Base | Activation successes | Guard reject pattern | Sidecar loaded |
|------|---------------------|----------------------|----------------|
| Q4 + sidecars | 1 | layer0 triggered, all others guard_reject → then g_true_inj=1 g_apply=1 for all | Yes (attn_out layer0 Q5_0) |
| Q3 + sidecars | 1 | Same pattern | Yes |
| Q2 + sidecars | 1 | Same pattern | Yes |

All sidecar smoke: `activation_attempts=1, activation_successes=1, non_null_views>=1, null_views=117`. No NaN/Inf observed.

---

## Analysis

### Finding 1: Q3 and Q2 baselines use MORE RSS than Q4 baseline

Unexpected result. Q3 baseline uses ~110 MB more RSS than Q4 baseline. Q2 baseline uses ~49 MB more.

**Possible explanation:** The embedding dimension (896) is not divisible by 256, causing 144/290 tensors to fall back to Q4_0 (for attn_k, attn_q, ffn_gate, ffn_up) and Q5_0 (for attn_output). The resulting mixed-quant model doesn't have a substantially smaller memory footprint than Q4_K_M, while the requantization process may add overhead.

### Finding 2: Sidecar injection on Q3/Q2 increases RSS further

Q3 + sidecars: 999,064 KB — **+124 MB vs Q4 baseline**
Q2 + sidecars: 938,344 KB — **+64 MB vs Q4 baseline**

Sidecar injection adds overhead on these lower-bit bases, not reduces it.

### Finding 3: Q4 + sidecars is most memory-efficient configuration tested

Q4 baseline n=8: 874,736 KB
Q4 + sidecars n=8: 889,316 KB (+14 MB, +1.7%)

The Q4 base with Q4-sidecars (attn_out Q5_0) maintains the lowest RSS. Adding sidecars to Q4 increases memory by only 1.7%.

---

## Classification

```
SIDECAR_COMPAT_BLOCKED
```

**Conditions met:**
- Q3 + sidecars RSS (999,064 KB) exceeds Q4 baseline (874,736 KB) by +14.2% → RESIDENCY_PATH_NOT_SUPPORTED
- Q2 + sidecars RSS (938,344 KB) exceeds Q4 baseline by +7.3% → RESIDENCY_PATH_NOT_SUPPORTED
- Lower-bit bases derived from Q4_K_M do not reduce baseline RSS → partial base preparation valid but sidecar compatibility blocked

---

## Key Observation: attn_output.weight fallback

The `blk.*.attn_output.weight` tensor has shape [896, 896]. Q3_K and Q2_K require ncols divisible by 256, but 896 is not. This causes `attn_output.weight` to fall back to Q4_0 (not the target Q3_K or Q2_K). Since the sidecar being injected is `attn_out` (which maps to `attn_output.weight`), the exact tensor family being modified falls back to Q4_0 precision — limiting the expected memory advantage.

---

## No Forbidden Claims Made

- ❌ No quality claim
- ❌ No correctness claim
- ❌ No speedup claim
- ❌ No Q2→Q4 recovery claim
- ❌ No production readiness claim

---

## Safe Claim

"Q2/Q3 base models created from Q4 via local quantization. Residency tested under Qwen2.5-0.5B canary conditions. No quality claims."

---

## Next Phase

**Phase 30B:** Generate sidecars directly from Q3 and Q2 bases (base-matched sidecars) rather than using Q4 reference sidecars. Test whether base-matched sidecar injection produces residency advantages. Also consider testing Q5_K_M and Q6_K bases to find the optimal quant level where base + sidecar RSS is minimized.