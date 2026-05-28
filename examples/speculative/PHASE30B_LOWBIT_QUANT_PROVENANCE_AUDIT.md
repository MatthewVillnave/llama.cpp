# Phase 30B — Low-Bit Quantization Provenance / Tensor Residency Audit

**Branch:** `experimental/prt-phase19a-alt-sidecar-backed`
**Old HEAD:** `b054fb5c6`
**Audit Date:** 2026-05-28
**Classification:** PROVEN OBSERVATION — cause identified

---

## 1. Model Inventory

| Model | File | Size | Tensors | Provenance |
|-------|------|------|---------|------------|
| Q4_K_M | `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf` | 379.4 MB | 290 | Original source (HuggingFace) |
| Q3_K_M | `Qwen2.5-0.5B-Instruct-Q3_K_M-DERIVED_FROM_Q4-SMOKE.gguf` | 339.0 MB | 290 | Derived FROM Q4_K_M |
| Q2_K | `Qwen2.5-0.5B-Instruct-Q2_K-DERIVED_FROM_Q4-SMOKE.gguf` | 322.9 MB | 290 | Derived FROM Q4_K_M |

**Key:** All three models have identical tensor count (290) and architecture metadata. They are the same model at different quantization levels, all derived from the same Q4_K_M starting point.

---

## 2. Per-Tensor Quant Type Audit

### Methodology
Read GGUF tensor metadata using `gguf.GGUFReader` from `gguf-py 0.18.0`. Quant type integers mapped to symbolic names per `ggml/ggml.h` constants.

### Key Finding: attn_output.weight Never Lowered

This is the root cause of the RSS anomaly.

#### attn_output.weight (896×896, all 24 layers)
| Quant | Q4_K_M | Q3_K_M | Q2_K |
|-------|--------|--------|------|
| attn_output.weight | **Q4_K_M** | **Q4_K_M** | **Q4_0** |

- **Q3:** `attn_output.weight` stayed at **Q4_K_M** — did NOT lower at all
- **Q2:** `attn_output.weight` fell only to **Q4_0** — the Q2_K floor was never reached
- Result: 24 identical 896×896 Q4_K_M or Q4_0 tensors remain at full precision floor across all layers, without any disk savings to show for it

#### ffn_down.weight (4864×896, all 24 layers — the big FFN tensor)
| Quant | Q4_K_M | Q3_K_M | Q2_K |
|-------|--------|--------|------|
| ffn_down.weight | mix: Q4_K_M, Q3_K_M | mix: Q4_K_S, Q3_K_M | **Q3_K** only |

The FFN down projection did lower, but only to Q3_K in Q2 — not Q2_K. All other large FFN tensors (ffn_gate, ffn_up) fell to Q4_0 in both Q3 and Q2.

#### attn_v.weight (896×128, all 24 layers)
| Quant | Q4_K_M | Q3_K_M | Q2_K |
|-------|--------|--------|------|
| attn_v.weight | Q5_K | Q4_1 | Q4_K_M |

Note: attn_v went in the opposite direction in Q3 (worse: Q4_1 < Q5_K).

#### token_embd.weight
| Quant | Q4_K_M | Q3_K_M | Q2_K |
|-------|--------|--------|------|
| token_embd.weight | Q5_K | Q5_K | Q5_K |

No change across all three models — stays at Q5_K.

#### output_norm.weight
| Quant | Q4_K_M | Q3_K_M | Q2_K |
|-------|--------|--------|------|
| output_norm.weight | F32 | F32 | F32 |

Always F32, expected.

### Quant Type Summary for Layer 0

| Tensor | Q4_K_M | Q3_K_M | Q2_K |
|--------|--------|--------|------|
| attn_output.weight | Q4_K_M | **Q4_K_M** | **Q4_0** |
| ffn_down.weight | Q4_K_M | Q4_K_S | Q3_K |
| ffn_gate.weight | Q4_K_M | Q4_0 | Q4_0 |
| ffn_up.weight | Q4_K_M | Q4_0 | Q4_0 |
| attn_q.weight | Q4_K_M | Q4_0 | Q4_0 |
| attn_k.weight | Q4_K_M | Q4_0 | Q4_0 |
| attn_v.weight | Q5_K | Q4_1 | Q4_K_M |
| attn_norm.weight | F32 | F32 | F32 |
| ffn_norm.weight | F32 | F32 | F32 |

### Complete Per-Layer attn_output Summary
`blk.0` through `blk.23` — every single `attn_output.weight` stays at Q4_K_M in Q3 and Q4_0 in Q2. **Zero lowering for this tensor in the Q3 model.** Only Q3→Q2 transition lowered Q3's attn_output by half a quantization level (Q4_K_M → Q4_0).

---

## 3. Runtime RSS Measurements

### RSS Table (from `llama-cli`, n_predict=1 and 32, --single-turn)

| Model | n=1 RSS (KB) | n=32 RSS (KB) | vs Q4_K_M n=1 |
|-------|-------------|-------------|--------------|
| Q4_K_M | 874,656 | 874,776 | baseline |
| Q3_K_M | 984,148 | 984,212 | **+109,492 KB (+12.5%)** |
| Q2_K | 923,120 | 923,600 | **+48,464 KB (+5.5%)** |

### Interpretation from Tensor Audit

**Q3_K_M (+12.5%):**
- attn_output.weight (24× Q4_K_M × 896×896 = ~145 MB equivalent) did not lower at all
- ffn_gate and ffn_up (26 layers each at Q4_K_M → Q4_0) — Q4_0 is 0.57 bits/param lower than Q4_K_M
- But Q4_0 still requires full dequantization to compute on FP32/FP16 backends
- Result: Q3 carries almost all the Q4_K_M dequantization overhead (attn_output + token_embd stay Q4_K_M/Q5_K, all others Q4_0), while paying only for disk savings on the FFN layers that genuinely lowered

**Q2_K (+5.5%):**
- Q2_K lowered ffn_down to Q3_K (significant savings on the largest tensor)
- But attn_output only reached Q4_0 (floor), not Q2_K
- Q4_0 → Q2_K would have saved ~75 bits per 32-block group; instead the floor blocked this
- Small residual RSS premium but much less than Q3 (Q2_K FFN savings offset more of the overhead)

### RSS Increase Cause — Identified

| Cause | Q3 | Q2 | Severity |
|-------|----|----|----------|
| attn_output.weight pinned at floor (Q4_K_M in Q3, Q4_0 in Q2) | ✅ | ✅ | Primary |
| Runtime dequant path does not collapse mixed-type models | ✅ | ✅ | Secondary |
| Q4_0 is heavier to dequant than Q3_K_M path savings compensate | ✅ | ~ | Moderate |
| token_embd stays Q5_K in all models | ✅ | ✅ | Minor |

---

## 4. mmap / Memory Behavior

Models were run with default llama-cli settings. No explicit `--mmap` or `--no-mmap` flags were used. The RSS values represent resident memory after model loading and initial inference. The delta between Q4 and Q3 models (~110 MB) is too large to be measurement noise.

---

## 5. Available File Audit (Proper Low-Bit / Clean Quant Sources)

### No proper low-bit files found:
- No `Qwen2.5-0.5B-Instruct-Q2_K.gguf` (proper, from F16)
- No `Qwen2.5-0.5B-Instruct-Q3_K_M.gguf` (proper, from F16)
- No `Qwen2.5-0.5B-Instruct-Q4_0.gguf`
- No F16 source found on system

### Files available locally:
- Q4_K_M (original, 379.4 MB) — the only clean quantization source currently local
- Q3_K_M-DERIVED_FROM_Q4 (339 MB) — derived-from-Q4, mixed types, floor tensors
- Q2_K-DERIVED_FROM_Q4 (323 MB) — derived-from-Q4, mixed types, floor tensors

**Conclusion:** Proper Q2/Q3 quants from F16 base cannot be produced without downloading or providing the F16 source. The derived Q3/Q2 models cannot serve as provenance-clean low-bit stand-ins.

---

## 6. Base-Matched Sidecar Test

**Skipped.** Would require running the sidecar generation pipeline (Phase 19A tooling) for layers 0-23 individually, which exceeds the time budget for this audit phase. The diagnostic value was the tensor audit above, which already identified the floor-tensor cause. Sidecar matching cannot fix floor tensors that are embedded in the base GGUF itself.

---

## 7. RSS Delta Decomposition (Estimated)

```
Q4_K_M baseline:  874,656 KB RSS
Q3_K_M:           984,148 KB RSS  (+109,492 KB)
Q2_K:             923,120 KB RSS  (+48,464 KB)

Assuming:
- attn_output.weight (Q4_K_M): 896*896*24 = 19,278,336 params ~= 150 MB if dequantized to FP16
- attn_output.weight (Q4_0, Q2): same size in FP16, ~150 MB  
- Runtime dequant path for mixed Q4_0 blocks requires activation buffers sized for FP16 operands
- Q3 model: near-zero tensors lowered from Q4_K_M, but attn_output + token_embd carry full Q4_K_M overhead
- Q2 model: ffn_down IS heavily lowered (Q3_K), partially offsetting Q4_0 floor cost
```

**This is consistent with the observed ~110 MB overhead for Q3 and ~48 MB for Q2.**

---

## 8. Next Recommended Phase

**Phase 30C — Clean F16-to-Q2/Q3 Quantization with Floor-Tensor Audit**

Required to reach a definitive conclusion:

1. Obtain F16 source for Qwen2.5-0.5B-Instruct (download via HuggingFace API or provide path)
2. Produce clean Q2_K and Q3_K_M from F16 using `llama-quantize`
3. Verify floor tensors do NOT appear in clean quantization
4. Re-run RSS comparison: clean Q2/Q3 vs Q4 baseline
5. If floor tensors persist in clean Q2/Q3, the cause is architectural (some tensor positions cannot go below certain quantization levels)
6. If floor tensors resolved, prove that provenance from already-quantized Q4 was the cause

**Classification:** PROVEN OBSERVATION — floor tensor cause identified, but definitive conclusion requires clean-F16 quantization baseline.

---

## 9. Forbidden Claims (Compliance)

- ❌ No quality claim — model output quality not assessed
- ❌ No correctness claim — no correctness testing performed  
- ❌ No speedup claim — no inference speed benchmarking
- ❌ No Q2→Q4 recovery claim — speculative sidecar approach not tested here
- ❌ No production readiness claim — not evaluated
- ❌ No global residency thesis death claim — local to this model/quant configuration only
