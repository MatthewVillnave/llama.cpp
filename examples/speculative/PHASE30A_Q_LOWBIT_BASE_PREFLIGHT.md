# Phase 30A-Q — Low-Bit Base Quantization Preflight

**Branch:** `experimental/prt-phase19a-alt-sidecar-backed`
**Commit:** `1c9f2d9dd` (unchanged — no new commit yet)
**Date:** 2026-05-28
**Goal:** Create Q2/Q3 base files from existing Q4_K_M via local quantization. Smoke-only models for memory testing. No quality claims.

---

## Step 1 — Inventory

| Item | Value |
|------|-------|
| Source model | `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf` |
| Source size | 380 MB |
| Quantizer binary | `build_CI/bin/llama-quantize` (build 8856) |
| llama-cli binary | `build/bin/llama-cli` (build 9168) |
| `/` free space | 126 GB |
| Sidecar source | `/tmp/prt_sidecars_0_5b_layer0/` (Q4_K_M reference sidecars) |

---

## Step 2 — Quantize from Q4_K_M → Q3_K_M and Q2_K

**Note:** `llama-quantize` requires `--allow-requantize` when source is already quantized. Without it, error: `requantizing from type q5_0 is disabled`.

### Q3_K_M (with --allow-requantize)
```bash
./build_CI/bin/llama-quantize --allow-requantize \
  Qwen2.5-0.5B-Instruct-Q4_K_M.gguf \
  Qwen2.5-0.5B-Instruct-Q3_K_M-DERIVED_FROM_Q4-SMOKE.gguf \
  Q3_K_M
```
**Result:** Exit 0. 144/290 tensors fell back to Q4_0/Q5_0 (embedding dim 896 not divisible by 256, required for Q3_K/Q2_K). Mixed quant result.

### Q2_K (with --allow-requantize)
```bash
./build_CI/bin/llama-quantize --allow-requantize \
  Qwen2.5-0.5B-Instruct-Q4_K_M.gguf \
  Qwen2.5-0.5B-Instruct-Q2_K-DERIVED_FROM_Q4-SMOKE.gguf \
  Q2_K
```
**Result:** Exit 0. 144/290 tensors fell back. ffn_down used Q3_K, attention used Q4_0.

### Generated Files

| File | Size |
|------|------|
| `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf` (source) | 380 MB |
| `Qwen2.5-0.5B-Instruct-Q3_K_M-DERIVED_FROM_Q4-SMOKE.gguf` | 339 MB |
| `Qwen2.5-0.5B-Instruct-Q2_K-DERIVED_FROM_Q4-SMOKE.gguf` | 323 MB |

---

## Step 3 — Load Verification

Both models load and produce tokens (sanity only, no quality claim).

| Model | Load exit | Token output |
|-------|-----------|-------------|
| Q3_K_M | 0 | `17` (≈) |
| Q2_K | 0 | `785` (≈) |

---

## Step 4 — Baseline RSS (no sidecars)

| Config | RSS (KB) | Wall time |
|--------|---------|-----------|
| Q4 n=1 | 874,720 | 0.68s |
| Q4 n=8 | 874,736 | 0.78s |
| Q4 n=32 | 874,616 | 0.80s |
| Q3 n=1 | 984,332 | 0.71s |
| Q3 n=8 | 984,236 | 0.81s |
| Q3 n=32 | 984,588 | 0.80s |
| Q2 n=1 | 922,856 | 0.69s |
| Q2 n=8 | 923,320 | 0.81s |
| Q2 n=32 | 922,784 | 1.20s |

**Observation:** Q3 and Q2 baselines use MORE RSS than Q4 baseline. This is the opposite of what would be needed for a residency advantage.

---

## Step 5 — Q4 + Sidecars (reference)

| Config | RSS (KB) | Wall time |
|--------|---------|-----------|
| Q4 + sidecars n=8 | 889,316 | 0.79s |

Sidecar smoke: `activation_successes=1`, `trit_validated`, `non_null_views=8`, `null_views=117`. Sidecar loading successful. The sidecars come from Q4_K_M reference — attn_out layer0 (Q5_0, 393,264 bytes).

---

## Step 6 — Q3 + Sidecars

| Config | RSS (KB) | Wall time |
|--------|---------|-----------|
| Q3 + sidecars n=8 | 999,064 | 0.80s |

Sidecar smoke: `activation_successes=1`, sidecar loaded and applied. Guard-reject pattern same as Q4 — `g_true_inj=0 g_apply=0` until layer0 triggers, then `g_true_inj=1 g_apply=1` for remaining layers.

**RSS + sidecars vs Q4 baseline:** 999,064 vs 874,720 → **+124,344 KB (+14.2%)**

---

## Step 7 — Q2 + Sidecars

| Config | RSS (KB) | Wall time |
|--------|---------|-----------|
| Q2 + sidecars n=8 | 938,344 | 0.83s |

Sidecar smoke: `activation_successes=1`, `Loaded 0/24 sidecars`, pager lazy-activated layer0 sidecar. Injection flags `g_true_inj=1 g_apply=1` confirmed.

**RSS + sidecars vs Q4 baseline:** 938,344 vs 874,720 → **+63,624 KB (+7.3%)**

---

## Classification

### RSS Summary

| Config | RSS (KB) | Δ vs Q4 baseline |
|--------|---------|-----------------|
| Q4 baseline (n=8) | 874,736 | — |
| Q3 baseline (n=8) | 984,236 | +109,500 (+12.5%) |
| Q2 baseline (n=8) | 923,320 | +48,584 (+5.6%) |
| Q4 + sidecars (n=8) | 889,316 | +14,580 (+1.7%) |
| Q3 + sidecars (n=8) | 999,064 | +124,328 (+14.2%) |
| Q2 + sidecars (n=8) | 938,344 | +63,608 (+7.3%) |

### Findings

1. **Q3 and Q2 baselines are LESS memory-efficient than Q4 baseline.** This is unexpected — lower-bit models should be smaller. The mixed quant (144/290 tensors fell back to Q4_0/Q5_0) means actual compressed size is not much smaller than Q4_K_M. The overhead from requantization from intermediate Q5_0/q4_K/q6_K sources may also add overhead.

2. **Q3 + sidecars increases RSS vs Q3 baseline** (999,064 vs 984,236) and vs Q4 baseline (999,064 vs 874,736). Sidecar injection is memory-negative for Q3 base.

3. **Q2 + sidecars increases RSS vs Q2 baseline** (938,344 vs 923,320) and vs Q4 baseline (938,344 vs 874,736). Sidecar injection is memory-negative for Q2 base.

4. **Q4 + sidecars is the only configuration where sidecar injection might be considered memory-acceptable** (only +1.7% vs Q4 baseline without sidecars), but this is only tested at n=8.

### Classification

```
SIDECAR_COMPAT_BLOCKED — Q3 + sidecars RSS (999,064 KB) exceeds Q4 baseline (874,736 KB) by +14.2%
                        — Q2 + sidecars RSS (938,344 KB) exceeds Q4 baseline by +7.3%
                        — Lower-bit bases (Q3, Q2) derived from Q4_K_M with mixed quant do not reduce baseline RSS
                        — Sidecar injection with attn_out layer0 adds memory overhead on both Q3 and Q2 bases
```

---

## Safe Claim (if PASS)

"Q2/Q3 base models created from Q4 via local quantization. Residency tested under Qwen2.5-0.5B canary conditions. No quality claims."

---

## Actual Classification

`SIDECAR_COMPAT_BLOCKED`

**Reasoning:** Lower-bit bases (Q3, Q2) derived from Q4_K_M with mixed quant do not produce memory savings on baseline. Sidecar injection on these lower-bit bases further increases RSS above Q4 baseline. The Q4 reference sidecars (Q5_0 attn_out layer0) are not compatible with Q3 or Q2 bases for residency reduction purposes.

---

## Next Recommended Phase

- **Phase 30B:** Generate sidecars directly from Q3 and Q2 bases (rather than reusing Q4 sidecars). Test whether base-matched sidecars reduce RSS on lower-bit bases.
- **Alternative:** Test Q5_K_M or Q6_K bases with same sidecar set to see if intermediate quant levels maintain compatibility.
- **Note:** The attn_output.weight tensor requires 896 cols (not divisible by 256), causing Q3/Q2 attn_output to fall back to Q4_0 — limiting quantization savings on the exact tensor family being sidecar-injected.