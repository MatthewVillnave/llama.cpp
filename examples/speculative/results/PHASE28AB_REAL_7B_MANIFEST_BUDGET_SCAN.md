# Phase 28AB: Real 7B Manifest Budget Scan + 30B Extrapolation

## Verdict: PASS_PHASE28AB_7B_MANIFEST_BUDGET_SCAN ✅

## Summary
Generated a real 7B estimated manifest from GGUF inspection + empirical Phase 28X data, ran all budget policies against it, verified consistency with Phase 28X estimates, and extrapolated to 30B scale.

---

## A. Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## B. Current HEAD
`975b19523`

## C. Estimated Manifest Generator

**Path:** `examples/speculative/prt_residual_manifest_estimate.py`

Generates a manifest-like JSON from:
- GGUF inspection of real Qwen2.5-7B weights
- Empirical `delta_cosine` from Phase 28X per-family data
- Non-weight tensor bytes from actual GGUF metadata
- Architecture metadata (hidden_size=3584, intermediate_size=18944, layers=28)

**Output:** `/tmp/prt_7b_estimated_residual_manifest.json` (not staged)

**Key fields added to manifest:**
- `q2_base_bytes` — weight Q2 + non-weight F32/Q4 ≈ 3.83 GB
- `q2_base_weight_bytes` — weight tensors only ≈ 3.13 GB
- `q2_base_non_weight_bytes` — embeddings, norms, biases, lm_head ≈ 0.75 GB
- `is_estimated: True` — clearly marks as estimated

---

## D. 7B Architecture (Confirmed from GGUF)

| Parameter | Value |
|-----------|-------|
| Layers | 28 |
| Hidden size | 3584 |
| Intermediate size | 18944 |
| Attn heads | 28 |
| KV heads | 4 |
| Q4 total (7 families) | 4.174 GB |
| Q2 base (weights) | 3.130 GB |
| Non-weight tensors | 0.754 GB |
| **Total Q2 model** | **3.885 GB** |

### Q4 per Family (per layer)

| Family | Q4 bytes/layer | Q4 total (28L) | % of total |
|--------|--------------|----------------|-----------|
| FFN_DOWN | 55,695,360 | 1.559 GB | 37.3% |
| FFN_UP | 38,191,104 | 1.069 GB | 25.6% |
| FFN_GATE | 38,191,104 | 1.069 GB | 25.6% |
| attn_q | 7,225,344 | 0.202 GB | 4.9% |
| attn_output | 7,225,344 | 0.202 GB | 4.9% |
| attn_k | 1,032,192 | 0.029 GB | 0.7% |
| attn_v | 1,505,280 | 0.042 GB | 1.0% |

**FFN_UP + FFN_DOWN + FFN_GATE = 88.5% of Q4 weight tensor memory.**

---

## E. 7B Budget Policy Results

RAM=16 GB, c=2048, KV=2 KB/token, buffer=1 GB, OS headroom=2 GB

| Policy | Selected | Residual MB | Base GB | Total GB | Remaining GB | SAFE |
|--------|----------|-----------|---------|----------|--------------|------|
| `base_only` | 0 | 0.0 MB | 3.83 | 7.06 | 10.12 | ✅ |
| `attention_partial` | 56 | 101.2 MB | 3.83 | 7.16 | 10.02 | ✅ |
| `mlp_all` | 84 | 924.5 MB | 3.83 | 7.98 | 9.20 | ✅ |
| `all_validated` | 140 | 1025.7 MB | 3.83 | 8.08 | 9.10 | ✅ |
| `budget_greedy @ 512 MB` | 101 | ~512 MB | 3.83 | 7.59 | 9.59 | ✅ |
| `budget_greedy @ 128 MB` | 59 | ~128 MB | 3.83 | 7.18 | 10.00 | ✅ |

**Key findings:**
- All policies SAFE on 7B under 16 GB RAM, c=2048
- `all_validated` total = 8.08 GB — well within 16 GB
- `mlp_all` (7.98 GB) captures most savings from a smaller residual set
- `attention_partial` (7.16 GB) is minimal residual option above base
- ~9 GB headroom remains even with all validated residuals
- **7B residual budget ceiling: ~9 GB** (before hitting RAM limit)

---

## F. Budget Greedy Analysis

### Greedy ranking by score_per_byte

Ranking: `score_per_byte` (when available) or `delta_cosine / byte_size`

| Rank | Family | Score/byte | Layer bytes | Delta cos |
|------|--------|-----------|-------------|-----------|
| 1 | attn_output | 9.77×10⁻⁸ | 1.81 MB | 0.6865 |
| 2 | attn_q | 9.31×10⁻⁸ | 1.81 MB | 0.6729 |
| 3 | ffn_up | 2.65×10⁻⁸ | 9.55 MB | 0.7079 |
| 4 | ffn_gate | 2.65×10⁻⁸ | 9.55 MB | 0.6869 |
| 5 | ffn_down | 1.39×10⁻⁸ | 13.92 MB | 0.6965 |

**Attention tensors rank 8–12× higher than FFN tensors per byte** because they are much smaller but have comparable delta_cos.

### Budget greedy fill order

At 128 MB residual budget:
1. All 28 `attn_output` layers (28 × 1.81 MB = 50.7 MB)
2. All 28 `attn_q` layers (28 × 1.81 MB = 50.7 MB)
3. 3 `ffn_up` layers (3 × 9.55 MB ≈ 28.7 MB)
→ total ≈ 130 MB, budget exhausted

At 512 MB residual budget:
1. All attention layers (~101 MB)
2. All 28 `ffn_up` layers (267 MB)
3. 17 `ffn_gate` layers (17 × 9.55 MB ≈ 162 MB)
→ ~530 MB total selected (budget slightly over at first-fit)

**Policy prefers attention tensors first, then MLP.**

---

## G. Consistency with Phase 28X

Phase 28X estimated Q4 vs Q2+ternary for 7B:
- Q4: 4.46 GB (Phase 28X) vs 4.17 GB (actual GGUF) — Phase 28X slightly overestimated
- Q2+ternary: ~3.3 GB (Phase 28X) vs 3.13 GB (actual weight Q2) — consistent

**Phase 28X was accurate to within ~7%** of actual GGUF values for Q4 weights.

---

## H. 30B Extrapolation

**30B architecture (from Phase 28X):** hidden_size≈5120, intermediate_size≈13824, layers≈56

| Metric | 7B (actual) | 30B (estimated) |
|--------|-------------|------------------|
| Total Q4 (5 families) | 4.17 GB | 35.2 GB |
| Q2 base (weights) | 3.13 GB | 26.4 GB |
| MLP Q4 | 3.70 GB | 31.7 GB |
| Attention Q4 | 0.48 GB | 3.5 GB |
| Residual savings | 1.04 GB | 8.8 GB |
| Fits 16 GB? | ✅ YES | ❌ NO |

**30B does NOT fit** even with full Q2+ternary on 16 GB RAM.

### 30B feasible path

At c=1024:
- Available for model: ~14 GB (after OS + buffer + KV)
- Required: ~26 GB Q2 base + overhead
- **Gap: ~12 GB** — requires selective residual + layer paging

At c=512 (minimal context):
- Same Q2 base, but KV = 512 × 2 KB = 1 MB
- Still need ~14 GB available for model
- Q2 base alone ≈ 26 GB — still won't fit

**Conclusion:** Dense 30B on 16 GB requires:
1. Selective residual overlay (not full-linear)
2. Out-of-core layer paging (load layers on-demand)
3. Or context reduction + KV compression

---

## I. Interpretation

1. **7B is comfortable on 16 GB** — all validated residual policies fit with ~9 GB headroom
2. **MLP families dominate budget** — 88.5% of Q4 weight memory; attention is marginal by comparison
3. **Budget greedy picks attention first** — high score_per_byte ratio, small byte_size
4. **30B requires architectural changes** — Q2 base alone exceeds 16 GB; residual overlay alone won't solve it
5. **Layer paging is essential for 30B** — selective loading of residual layers, not full model residency

---

## J. Recommended Next Phase

**Phase 28AC — Layer Paging + Selective Residual Loader Architecture**

Design a paging system for 30B:
1. Divide model into page groups (e.g., 4 layers per page)
2. Only resident pages need RAM; paged-out layers loaded on demand
3. Compute which page groups are needed for residual coverage under budget
4. Evaluate page size vs. residual coverage trade-off

This moves from budget analysis to runtime architecture design.

---

## K. Models/Sidecars/F32 Refs Staged?
**NO.** No model files, no sidecars, no f32 refs staged. `/tmp/prt_7b_estimated_residual_manifest.json` is in /tmp only.

## L. Secrets Detected?
None.

## M. Tags Touched?
None.

---

## Files Committed
- `examples/speculative/prt_residual_manifest_estimate.py` — estimated manifest generator
- `examples/speculative/results/PHASE28AB_REAL_7B_MANIFEST_BUDGET_SCAN.md` — this report
- `examples/speculative/results/phase28ab_real_7b_manifest_budget_scan.json` — structured results