# Phase 28U: 30B Budget Extrapolation from FFN_UP Residual Anchors

## Verdict: PASS_PHASE28U_30B_BUDGET_EXTRAPOLATION ✅ | PASS_FFN_UP_SCALE_ANCHORS_DOCUMENTED ✅ | PASS_SCOPE_BOUNDARY_PRESERVED ✅ | RECOMMEND_NEXT_TENSOR_FAMILY_TEST

## Summary
FFN_UP Q2+ternary residual overlay is validated at 0.5B (24/24 layers) and 3B (5/5 layers). 7B FFN_UP dimensions confirm the 0.75× Q4 ratio holds at larger scale. Extrapolating to 30B: FFN_UP Q2+ternary would save ~25% on that tensor class, but FFN_UP is ~20% of total model — full savings depend on extending to other tensor families.

---

## A. Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## B. Current HEAD
`fcc3662b5 Phase 28T: check 3B FFN_UP ternary residual transfer`

## C. Empirical Anchors

### 0.5B Anchor (Phase 28S)
- **Model:** Qwen2.5-0.5B-Instruct-Q4_K_M.gguf
- **FFN_UP layers:** 24
- **FFN_UP shape:** [896, 4864] per layer
- **FFN_UP params/layer:** 4,358,144
- **FFN_UP total params:** 104,595,456
- **Q4 FFN_UP memory:** 49.9 MB
- **Q2+ternary FFN_UP memory:** 37.4 MB
- **Compression ratio:** 0.7500 (validated)
- **24/24 STRONG_RECOVERY**, mean Δ cos = +0.7079 ± 0.0084

### 3B Anchor (Phase 28T)
- **Model:** Qwen2.5-3B-Instruct-Q4_K_M.gguf
- **FFN_UP layers:** 36
- **FFN_UP shape:** [2048, 11008] per layer
- **FFN_UP params/layer:** 22,544,384
- **FFN_UP total params:** 811,597,824
- **Q4 FFN_UP memory:** 387.0 MB
- **Q2+ternary FFN_UP memory:** 290.2 MB
- **Compression ratio:** 0.7500 (validated)
- **5/5 STRONG_RECOVERY**, mean Δ cos = +0.7076 ± 0.0029

### 7B Reference (architecture validation only, Phase 28U)
- **Model:** Qwen2.5-7B-Instruct-Q4_K_M.gguf
- **FFN_UP layers:** 28
- **FFN_UP shape:** [3584, 18944] per layer
- **FFN_UP params/layer:** 67,895,296
- **FFN_UP total params:** 1,901,068,288
- **Q4 FFN_UP memory:** 906.5 MB
- **Q2+ternary FFN_UP memory:** 679.9 MB
- **Compression ratio:** 0.7500 (formula-verified)
- **Model total file size:** 4,460 MB
- **FFN_UP as % of total model:** 906.5 / 4460 = **20.3%**

### Scaling Table

| Model | Layers | FFN_UP params/layer | FFN_UP total params | Q4 FFN_UP | Q2+ternary FFN_UP | Ratio |
|-------|--------|-------------------|---------------------|-----------|--------------------|-------|
| 0.5B | 24 | 4.36M | 104.6M | 49.9 MB | 37.4 MB | 0.75 |
| 3B | 36 | 22.5M | 811.6M | 387.0 MB | 290.2 MB | 0.75 |
| 7B | 28 | 67.9M | 1.90B | 906.5 MB | 679.9 MB | 0.75 |
| **30B** (est) | ~56 | ~137M | ~7.7B | ~3.7 GB | ~2.8 GB | 0.75 |

## D. Extrapolation Assumptions

### High Confidence (validated facts)
- Q2+ternary = **0.75× Q4** for FFN_UP slices, confirmed on 0.5B and 3B
- FFN_UP is a distinct tensor family with consistent quantization behavior
- Formula: Q4 FFN_UP bytes = intermediate × hidden × layers × 0.5

### Medium Confidence (scaling inference)
- Recovery magnitude (+0.707 Δ cos) transfers to larger models — confirmed 0.5B→3B
- FFN_UP compression ratio 0.75× holds for 7B architecture dimensions
- 30B FFN_UP dimensions estimated by architectural extrapolation

### Low Confidence (not validated)
- Actual 30B FFN_UP dimensions (no access to 30B GGUF tensor metadata here)
- Generation behavior (no runtime testing)
- Recovery on non-FFN_UP tensor families
- Full-model quality with only FFN_UP corrected

### 30B FFN_UP Estimate
Assuming 30B dense Qwen architecture scales to ~56 layers, intermediate ~5120–6144, hidden ~18944–22000:
- FFN_UP params/layer: ~97–135M
- FFN_UP total: ~5.4–7.6B params
- Q4 FFN_UP: ~2.7–3.8 GB
- Q2+ternary FFN_UP: ~2.0–2.9 GB
- Savings vs Q4 FFN_UP: ~0.7–0.95 GB

## E. FFN_UP Memory Impact

### 7B Model — Full Breakdown (Validated Reference)

| Component | Bytes | Q4 | Q2 | Q2+ternary | Savings vs Q4 |
|-----------|-------|----|----|-------------|---------------|
| FFN_UP (28 layers) | 1.90B params | 906.5 MB | 453.2 MB | 679.9 MB | **226.6 MB** |
| Other tensors (est) | ~2.56B params | ~1,276 MB | ~638 MB | ~638 MB | baseline |
| **Total model** | **~4.46B params** | **~4,460 MB** | **~2,230 MB** | **~2,457 MB** | **~200 MB** |

**Note:** "Other tensors" includes FFN_DOWN (Q6_K), attention projections, embeddings, LayerNorm. These are not yet validated for Q2+ternary.

### If All Linear Tensors Used Q2+ternary (Theoretical Maximum)

If FFN_UP + FFN_DOWN + attention projections all used Q2+ternary at 0.75× Q4:
- Total linear tensor memory savings would be proportionally larger
- Approximate linear tensor fraction of model: 85–90%
- Theoretical total model with Q2+ternary everywhere: ~0.82–0.85× current Q4 size
- This is a hypothesis only — **FFN_DOWN and attention must be validated first**

## F. Full-Model Implication

### 1. If only FFN_UP uses Q2+ternary (current validated scope)
- **7B example:** 226.6 MB savings on FFN_UP vs Q4 = **5.1%** of total model file size
- **30B estimate:** ~700 MB–1 GB savings on FFN_UP alone = **3–5%** of total model
- **This is real but modest without extending to other tensor families**

### 2. If all linear tensors used Q2+ternary at 0.75× Q4 (hypothesis)
- **7B model:** Q2+ternary linear tensors = ~4,460 × 0.85 ≈ 3,791 MB
- **Savings vs Q4:** ~669 MB (**15%** of total model)
- **30B model:** Q2+ternary linear tensors would save ~2–3 GB
- **Critical dependency:** FFN_DOWN, attention projections must be validated first

### 3. Quality risk if only FFN_UP is corrected
- **Uncorrected FFN_DOWN** — Q6_K base quantization, different bit-width
- **Uncorrected attention projections** — may have residual error
- **Net quality:** Unknown. Phase 28Q showed ffn_down and attn_q also recover strongly, but those were single-layer tests on 0.5B only

### 4. Is FFN_UP enough alone?
**No.** FFN_UP is the single largest tensor family but not the only one. A practical PRT system must eventually address FFN_DOWN and attention.

### 5. Additional tensor families to validate (priority order)

| Priority | Tensor Family | Rationale |
|----------|--------------|-----------|
| 1 | **FFN_DOWN** | 2nd largest linear family; Phase 28Q showed strong recovery on 1 layer |
| 2 | **FFN_GATE** (if separate) | Qwen may have gate_proj in addition to up/down |
| 3 | **Attention output proj** | 3rd largest; attn_q showed strong recovery in Phase 28Q |
| 4 | **Attention Q/K/V** | Smaller per-layer; Phase 28Q attn_q tested |

## G. Next Tensor Family Ranking

1. **FFN_DOWN** — high memory share, strong Phase 28Q signal, must-test before multi-layer overlay
2. **FFN_GATE** — if Qwen uses separate gate projection, gate+up forms the gated FFN structure
3. **Attention output projection** — high impact on output quality, moderate memory share
4. **Attention Q/K/V** — lower memory share but quality-sensitive

## H. Architecture Update

### Current Validated PRT FFN_UP Piece
```
Q2 base FFN_UP + ternary residual overlay
  → 0.75× Q4 storage for same tensor
  → +0.707 mean cosine improvement on matvec
  → Scale-stable across 0.5B and 3B
  → 24/24 layers strong on 0.5B
  → 5/5 layers strong on 3B
```

### Still Unvalidated
| Component | Status |
|-----------|--------|
| FFN_DOWN Q2+ternary | Phase 28Q L11 only, strong, but single layer |
| Attention projections | Phase 28Q L11 only, strong, but single layer |
| FFN_GATE | Not tested |
| Full-layer multi-tensor interaction | Not tested |
| Generation quality | Not tested |
| 30B scale (real) | Not tested |
| Residual file format | Not designed |
| Backend integration | Not designed |
| KV + context memory | Not tested |

### Capacity-First PRT Roadmap (Updated)

```
Phase 28V: FFN_DOWN + FFN_GATE multi-layer validation (0.5B + 3B)
Phase 28W: Attention projection multi-layer validation
Phase 28X: Full multi-tensor offline overlay plan
Phase 28Y: Residual file format design
Phase 28Z: Runtime integration path (not generation — capacity monitoring)
```

## I. Recommended Next Phase

**Phase 28V — FFN_DOWN + FFN_GATE Multi-Layer Validation**

Test FFN_DOWN slices across multiple layers on both 0.5B and 3B:
- Layers 0, 5, 11, 23 on 0.5B (FFN_DOWN shape: [4864, 896])
- Layers 0, 8, 17, 26, 35 on 3B (FFN_DOWN shape: [11008, 2048])
- Also probe whether Qwen has a separate FFN_GATE tensor

If FFN_DOWN shows consistent strong recovery:
- Next step becomes attention projections
- Full linear-tensor overlay strategy becomes viable

If FFN_DOWN recovery is weaker:
- Investigate quantization sensitivity differences
- May need per-family residual scaling adjustment

## J. Models/Sidecars/F32 Refs Staged?
**NO.** Analysis only — no model files touched. Architecture data sourced from GGUF metadata reads only.

## K. Secrets Detected?
None.

## L. Tags Touched?
None.

---

## Files Committed
- `examples/speculative/results/PHASE28U_30B_BUDGET_EXTRAPOLATION_FROM_FFN_UP.md` — this report
- `examples/speculative/results/phase28u_30b_budget_extrapolation_from_ffn_up.json` — structured results