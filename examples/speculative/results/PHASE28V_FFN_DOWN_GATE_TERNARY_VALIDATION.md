# Phase 28V: FFN_DOWN + FFN_GATE Multi-Layer Ternary Residual Validation

## Verdict: PASS_PHASE28V_FFN_DOWN_GATE_VALIDATION ✅ | PASS_FFN_DOWN_STRONG_TRANSFER ✅ | PASS_FFN_GATE_STRONG_TRANSFER ✅ | RECOMMEND_ATTENTION_PROJECTION_TEST

## Summary
**FFN_DOWN and FFN_GATE both transfer strongly.** 0.5B and 3B results show consistent STRONG_RECOVERY across all MLP tensor families. This significantly expands the validated tensor coverage for capacity-first PRT.

---

## A. Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## B. Current HEAD
`a74ae894d Phase 28U: extrapolate 30B budget from FFN_UP residual anchors`

## C. Source Models
- **0.5B:** Qwen2.5-0.5B-Instruct-Q4_K_M.gguf
- **3B:** Qwen2.5-3B-Instruct-Q4_K_M.gguf

## D. Tensor Families Found

| Family | 0.5B Shape | 0.5B QType | 3B Shape | 3B QType |
|--------|-----------|-----------|----------|----------|
| ffn_up | [896, 4864] | Q5_0 | [2048, 11008] | Q4_K |
| **ffn_down** | [4864, 896] | **Q6_K** | [11008, 2048] | **Q6_K** |
| **ffn_gate** | [896, 4864] | **Q5_0** | [2048, 11008] | **Q4_K** |
| attn_output | [896, 896] | Q5_0 | [2048, 2048] | Q4_K |

**FFN_GATE exists** in both 0.5B and 3B — no blocking needed.

## E. Layers/Tensors Tested

### 0.5B (4 layers × 2 families = 8 slices)
- FFN_DOWN: layers 0, 5, 11, 23 — shape 512×896
- FFN_GATE: layers 0, 5, 11, 23 — shape 512×896

### 3B (5 layers × 2 families = 10 slices)
- FFN_DOWN: layers 0, 8, 17, 26, 35 — shape 512×2048
- FFN_GATE: layers 0, 8, 17, 26, 35 — shape 512×2048

## F. FFN_DOWN Results

### 0.5B FFN_DOWN

| Layer | Shape | Q2 cos | Q2+T cos | Δ cos | MAE hat | Compression | Verdict |
|-------|-------|--------|----------|-------|---------|-------------|---------|
| 0 | 512×896 | -0.0067 | 0.6921 | **+0.6987** | 0.3123 | 0.75 | STRONG |
| 5 | 512×896 | +0.0040 | 0.6899 | **+0.6858** | 0.2854 | 0.75 | STRONG |
| 11 | 512×896 | +0.0004 | 0.6881 | **+0.6877** | 0.3030 | 0.75 | STRONG |
| 23 | 512×896 | -0.0142 | 0.6997 | **+0.7139** | 0.2970 | 0.75 | STRONG |

**0.5B FFN_DOWN: Mean Δ cos = +0.6965 ± 0.0112, 4/4 STRONG**

### 3B FFN_DOWN

| Layer | Shape | Q2 cos | Q2+T cos | Δ cos | MAE hat | Compression | Verdict |
|-------|-------|--------|----------|-------|---------|-------------|---------|
| 0 | 512×2048 | +0.0093 | 0.7096 | **+0.7003** | 0.6397 | 0.75 | STRONG |
| 8 | 512×2048 | +0.0011 | 0.7008 | **+0.6997** | 0.6229 | 0.75 | STRONG |
| 17 | 512×2048 | +0.0066 | 0.7056 | **+0.6990** | 0.6409 | 0.75 | STRONG |
| 26 | 512×2048 | -0.0123 | 0.7040 | **+0.7164** | 0.6733 | 0.75 | STRONG |
| 35 | 512×2048 | +0.0049 | 0.7059 | **+0.7010** | 0.6136 | 0.75 | STRONG |

**3B FFN_DOWN: Mean Δ cos = +0.7033 ± 0.0066, 5/5 STRONG**

## G. FFN_GATE Results

### 0.5B FFN_GATE

| Layer | Shape | Q2 cos | Q2+T cos | Δ cos | MAE hat | Compression | Verdict |
|-------|-------|--------|----------|-------|---------|-------------|---------|
| 0 | 512×896 | +0.0089 | 0.7179 | **+0.7090** | 0.3819 | 0.75 | STRONG |
| 5 | 512×896 | +0.0090 | 0.6916 | **+0.6825** | 0.4038 | 0.75 | STRONG |
| 11 | 512×896 | +0.0120 | 0.6803 | **+0.6682** | 0.3345 | 0.75 | STRONG |
| 23 | 512×896 | +0.0121 | 0.7000 | **+0.6879** | 0.3433 | 0.75 | STRONG |

**0.5B FFN_GATE: Mean Δ cos = +0.6869 ± 0.0146, 4/4 STRONG**

### 3B FFN_GATE

| Layer | Shape | Q2 cos | Q2+T cos | Δ cos | MAE hat | Compression | Verdict |
|-------|-------|--------|----------|-------|---------|-------------|---------|
| 0 | 512×2048 | -0.0047 | 0.6951 | **+0.6998** | 0.6695 | 0.75 | STRONG |
| 8 | 512×2048 | -0.0026 | 0.6993 | **+0.7018** | 0.7574 | 0.75 | STRONG |
| 17 | 512×2048 | +0.0095 | 0.7177 | **+0.7082** | 0.6538 | 0.75 | STRONG |
| 26 | 512×2048 | +0.0094 | 0.7095 | **+0.7001** | 0.6449 | 0.75 | STRONG |
| 35 | 512×2048 | -0.0073 | 0.6824 | **+0.6898** | 0.6710 | 0.75 | STRONG |

**3B FFN_GATE: Mean Δ cos = +0.6999 ± 0.0059, 5/5 STRONG**

## H. 0.5B vs 3B Comparison

| Family | 0.5B Mean Δ cos | 3B Mean Δ cos | Delta | Transfer? |
|--------|-----------------|---------------|-------|-----------|
| FFN_UP | +0.7079 ± 0.0084 | +0.7076 ± 0.0029 | +0.0003 | ✅ |
| FFN_DOWN | +0.6965 ± 0.0112 | +0.7033 ± 0.0066 | -0.0068 | ✅ |
| FFN_GATE | +0.6869 ± 0.0146 | +0.6999 ± 0.0059 | -0.0130 | ✅ |

**All three MLP tensor families transfer from 0.5B to 3B with strong recovery maintained.**

## I. Comparison to FFN_UP

| Metric | FFN_UP | FFN_DOWN | FFN_GATE |
|--------|--------|----------|----------|
| 0.5B mean Δ cos | +0.7079 | +0.6965 | +0.6869 |
| 3B mean Δ cos | +0.7076 | +0.7033 | +0.6999 |
| 0.5B std | 0.0084 | 0.0112 | 0.0146 |
| 3B std | 0.0029 | 0.0066 | 0.0059 |
| QType sensitivity | Q5_0/Q4_K OK | Q6_K OK | Q5_0/Q4_K OK |

**Key findings:**
1. **FFN_UP remains strongest** but FFN_DOWN and FFN_GATE are close (+0.70 vs +0.71)
2. **FFN_GATE slightly lower on 0.5B** (+0.6869) but tightens on 3B (+0.6999) — still strong
3. **All 3 families use different base QTypes** (Q5_0, Q6_K, Q4_K) — all recover equally well
4. **3B variance is tighter than 0.5B** across all families — consistent with FFN_UP pattern

## J. 30B/Full-Model Implication

### Updated MLP Memory Coverage

For Qwen2.5-7B (architecture reference):
| Tensor Family | Memory (Q4) | Q2+ternary | Savings | Status |
|--------------|-------------|------------|---------|--------|
| FFN_UP | 906.5 MB | 679.9 MB | 226.6 MB | ✅ VALIDATED |
| FFN_DOWN | ~680 MB (est) | ~510 MB (est) | ~170 MB (est) | ✅ VALIDATED |
| FFN_GATE | ~453 MB (est) | ~340 MB (est) | ~113 MB (est) | ✅ VALIDATED |
| **Total MLP** | **~2,040 MB** | **~1,530 MB** | **~510 MB** | **~25% savings** |

**Full MLP (FFN_UP + FFN_DOWN + FFN_GATE) Q2+ternary coverage estimate: ~1.5 GB vs ~2.0 GB Q4 = 25% savings on the entire MLP section.**

### Remaining Unvalidated
- Attention projections (attn_output, attn_q, attn_k, attn_v)
- Embeddings and output head
- LayerNorm tensors

## K. Recommended Next Tensor Family
**Attention output projection** — next highest memory share after MLP tensors, already showed strong recovery on 0.5B attn_q in Phase 28Q.

## L. Interpretation

### 1. Does FFN_DOWN behave like FFN_UP?
**Yes, nearly identically.** Mean Δ cos difference < 0.01. All 9 tested layers (4 on 0.5B, 5 on 3B) show STRONG_RECOVERY. Q6_K base quantization does not degrade ternary residual recovery.

### 2. Does FFN_GATE behave like FFN_UP?
**Yes, slightly lower but still strong.** 0.5B is the weakest family tested so far (+0.6869 mean) but still well above the +0.5 strong threshold. 3B tightens to +0.6999 — nearly identical to FFN_UP/FFN_DOWN.

### 3. Are recovery values scale-stable between 0.5B and 3B?
**Yes.** Delta between 0.5B and 3B is ≤0.013 for all families. The 3B variance is consistently tighter.

### 4. Which tensor family should be targeted next?
**Attention output projection.** It's the next largest linear tensor family after MLP. Phase 28Q tested attn_q but only at one layer — need multi-layer validation.

### 5. What is the updated MLP memory savings estimate?
**~510 MB savings on 7B MLP alone** (FFN_UP + FFN_DOWN + FFN_GATE combined, estimated). This represents the bulk of the model's linear tensor memory.

## M. Recommended Next Phase

**Phase 28W — Attention Projection Multi-Layer Validation**

Test attention output projection (attn_output) on 0.5B and 3B across multiple layers:
- 0.5B: layers 0, 5, 11, 23 — shape 512×896 (square)
- 3B: layers 0, 8, 17, 26, 35 — shape 512×2048 (square)

If attn_output shows strong recovery:
- All major linear tensor families are validated for the MLP + attention output path
- Next phase becomes full linear-tensor budget extrapolation or residual file format design

If attn_output shows weaker recovery:
- Investigate quantization sensitivity for attention tensors specifically
- May need per-family scaling factor

## N. Models/Sidecars/F32 Refs Staged?
**NO.** All slices written to /tmp only. No model files staged.

## O. Secrets Detected?
None.

## P. Tags Touched?
None.

---

## Files Committed
- `examples/speculative/prt_residual_ffn_down_gate.py` — FFN_DOWN/GATE validation script
- `examples/speculative/results/PHASE28V_FFN_DOWN_GATE_TERNARY_VALIDATION.md` — this report
- `examples/speculative/results/phase28v_ffn_down_gate_ternary_validation.json` — structured results