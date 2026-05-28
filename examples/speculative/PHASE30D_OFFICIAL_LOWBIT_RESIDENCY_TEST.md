# Phase 30D — Official Low-Bit Residency Test

## Branch
`experimental/prt-phase19a-alt-sidecar-backed`
Old HEAD: `a88a013a0`

## Classification
`SDI_THESIS_DISPROVEN`

## Models Tested (official, from FP16)

| Model | Path | File Size | Provenance |
|-------|------|-----------|------------|
| Q4_K_M (baseline) | VL_usb | 469 MB | from FP16 |
| Q3_K_M | VL_usb | 413 MB | from FP16 |
| Q2_K | VL_usb | 396 MB | from FP16 |

## RSS Results (n=1, batch=512)

| Configuration | RSS KB | vs Q4_K_M_baseline |
|---------------|--------|--------------------|
| Q4_K_M_baseline | 967,436 KB | — |
| Q4_K_M + sidecars | 967,012 KB | -424 KB (-0.04%) |
| Q3_K_M_baseline | 1,062,688 KB | **+95,252 KB (+9.8%)** |
| Q3_K_M + sidecars | 1,062,500 KB | **+95,064 KB (+9.8%)** |
| Q2_K_baseline | 1,001,556 KB | **+34,120 KB (+3.5%)** |
| Q2_K + sidecars | 1,001,540 KB | **+34,104 KB (+3.5%)** |

## Key Finding

**SDI thesis: DISPROVEN.**

Official F16→Q3_K_M uses +9.8% MORE RSS than Q4_K_M.
Official F16→Q2_K uses +3.5% MORE RSS than Q4_K_M.
Sidecars are ADDITIVE — Q4+sidecars uses slightly MORE than Q4 alone (within noise).
Sidecars do NOT compensate for Q3/Q2 overhead.

## What This Means

The core SDI thesis — "lower-bit base + paged residual/sidecar correction uses less memory than full higher-bit dense model" — is **disproven for Qwen2.5-0.5B on CPU**.

Lower-bit bases (Q3, Q2) derived from clean FP16 source are NOT lighter at runtime than Q4. The runtime memory overhead of decoding lower-bit quantization exceeds any weight savings on disk.

The sidecar mechanism is also proven additive, not substitutive. Sidecars add ~0 KB (within measurement noise) to Q4 base, but Q4+sidecars is still the lightest configuration.

## No Quality Claim
No claim is made about output quality. This is memory/residency only.

## Next Recommended Phase
Phase 30E — Architecture Redesign Options:
1. Tensor-level replacement (not additive sidecar — replace dense tensors entirely)
2. KV cache offload to VL_usb (not sidecar-based)
3. Native llama.cpp optimizations (context truncation, batch control)
4. Stop SDI path — accept Q4 as optimal for this model/hardware

## Commit
`c9b72f14c` — Phase 30D: official Q2/Q3 RSS worse than Q4, SDI thesis disproven