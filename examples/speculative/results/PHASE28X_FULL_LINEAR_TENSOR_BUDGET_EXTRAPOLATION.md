# Phase 28X: Full Linear Tensor Budget Extrapolation + Residual Format Design

## Verdict: PASS_PHASE28X_FULL_LINEAR_BUDGET_EXTRAPOLATION ✅ | PASS_VALIDATED_TENSOR_FAMILIES_CONSOLIDATED ✅ | PASS_RESIDUAL_FORMAT_DESIGN_STARTED ✅ | RECOMMEND_ATTN_KV_VALIDATION

## Summary
Consolidated all validated tensor family results. Q2+ternary recovery validated for FFN_UP, FFN_DOWN, FFN_GATE, attn_q, attn_output — covering ~54% of model memory at 0.75× Q4 compression. 7B full-linear Q2+ternary estimated at ~3.3GB vs 4.46GB Q4. 14B estimated at ~6.4GB vs 8.5GB Q4. 30B dense still exceeds 16GB even with full linear Q2+ternary — out-of-core layer paging still required.

---

## A. Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## B. Current HEAD
`1a2aef0a6 Phase 28W: validate attention projection ternary residuals`

## C. Validated Tensor Family Table

| Tensor Family | 0.5B Tested | 3B Tested | 0.5B Mean Δ cos | 3B Mean Δ cos | Compression vs Q4 | Verdict |
|--------------|-------------|-----------|-----------------|---------------|-----------------|---------|
| **FFN_UP** | 24/24 ✅ | 5/5 ✅ | +0.7079 ± 0.0084 | +0.7076 ± 0.0029 | 0.75× | **STRONG** |
| **FFN_DOWN** | 4/4 ✅ | 5/5 ✅ | +0.6965 ± 0.0112 | +0.7033 ± 0.0066 | 0.75× | **STRONG** |
| **FFN_GATE** | 4/4 ✅ | 5/5 ✅ | +0.6869 ± 0.0146 | +0.6999 ± 0.0059 | 0.75× | **STRONG** |
| **attn_q** | 4/4 ✅ | 5/5 ✅ | +0.6729 ± 0.0422 | +0.6970 ± 0.0063 | 0.75× | **STRONG** |
| **attn_output** | 4/4 ✅ | 5/5 ✅ | +0.6865 ± 0.0068 | +0.6927 ± 0.0051 | 0.75× | **STRONG** |
| attn_k | not tested | not tested | — | — | — | UNVALIDATED |
| attn_v | not tested | not tested | — | — | — | UNVALIDATED |
| token_embd | not tested | not tested | — | — | — | UNVALIDATED |
| output | not tested | not tested | — | — | — | UNVALIDATED |
| LayerNorm | not tested | not tested | — | — | — | NOT NEEDED |

**All 5 major linear tensor families STRONG across both model scales.**

### Unvalidated Families Notes

| Family | Memory Share (7B) | Reason for Non-Validation |
|--------|------------------|--------------------------|
| attn_k | 0.6% | Head-dim constrained (non-square), small share |
| attn_v | 0.8% | Head-dim constrained (non-square), small share |
| token_embd | 6.6% | Different tensor class (vocabulary × hidden) |
| output | 9.6% | LM head, separate from residual overlay target |

**Collectively ~17% of model.** These can be handled separately (lower priority for residual overlay).

## D. Full Linear Tensor Budget Extrapolation

### 7B Qwen2.5-7B-Instruct-Q4_K_M.gguf (Measured)

| Tensor Family | Q4 Bytes | Q4 % | Q2+ternary Est | Notes |
|--------------|---------|------|----------------|-------|
| FFN_UP | 1,019.8 MB | 22.9% | 764.9 MB | ✅ Validated 0.75× |
| FFN_DOWN | 1,253.5 MB | 28.1% | 940.1 MB | ✅ Validated 0.75× |
| FFN_GATE | 1,019.8 MB | 22.9% | 764.9 MB | ✅ Validated 0.75× |
| attn_q | 193.3 MB | 4.3% | ~145 MB est | ✅ Validated (mixed F32/Q4_K) |
| attn_output | 192.9 MB | 4.3% | 144.7 MB | ✅ Validated 0.75× |
| **Validated subtotal** | **3,679.3 MB** | **82.5%** | **~2,759 MB** | |
| token_embd | 292.4 MB | 6.6% | ~219 MB est | Unvalidated, Q4_K |
| output (LM head) | 426.4 MB | 9.6% | 320 MB est | Unvalidated, Q6_K |
| attn_k | 27.6 MB | 0.6% | baseline | Unvalidated, mostly F32 |
| attn_v | 33.9 MB | 0.8% | baseline | Unvalidated, mixed |
| norms | ~0.8 MB | 0.0% | ~0.8 MB | F32, negligible |
| **Total** | **4,460.4 MB** | **100%** | **~3,300 MB** | |

**7B Q2+ternary full-linear estimate: ~3.3 GB vs 4.46 GB Q4 = 26% savings.**

### 14B Qwen2.5-14B-14B.gguf (Estimated from 7B scaling)

| Tensor Family | Q4 Est | Q4 % | Q2+ternary Est | Notes |
|--------------|--------|------|----------------|-------|
| FFN_UP | 1,822.5 MB | 21.3% | 1,366.9 MB | ✅ Validated |
| FFN_DOWN | 2,240.2 MB | 26.2% | 1,680.1 MB | ✅ Validated |
| FFN_GATE | 1,822.5 MB | 21.3% | 1,366.9 MB | ✅ Validated |
| attn_q | 675.9 MB | 7.9% | ~507 MB est | ✅ Validated |
| attn_output | 675.0 MB | 7.9% | 506.3 MB | ✅ Validated |
| **Validated subtotal** | **7,236 MB** | **84.5%** | **~5,427 MB** | |
| output | 609.1 MB | 7.1% | ~457 MB est | Unvalidated |
| token_embd | 417.7 MB | 4.9% | ~313 MB est | Unvalidated |
| attn_k/v | 301.3 MB | 3.5% | baseline | Unvalidated |
| **Total** | **8,566 MB** | **100%** | **~6,200 MB** | |

**14B Q2+ternary full-linear estimate: ~6.2 GB vs 8.57 GB Q4 = 27% savings.**

### 30B Dense Estimate

Using architectural scaling (extrapolation, NOT measured):
- 30B parameters, ~56 layers, hidden ~2048-2200, intermediate ~7168-8192
- Total Q4 estimate: ~18 GB
- Q2+ternary validated linear (~85%): ~15.3 GB × 0.75 = ~11.5 GB
- Still exceeds 16GB RAM with KV/context buffers added

### Summary Table

| Model | Q4 Size | Q2 Base Est | Q2+ternary (validated linear) | Q2+ternary (full-linear est) | Savings |
|-------|---------|------------|------------------------------|---------------------------|---------|
| 7B | 4.46 GB | ~2.2 GB | ~2.76 GB | ~3.3 GB | ~26% vs Q4 |
| 14B | 8.57 GB | ~4.3 GB | ~5.4 GB | ~6.2 GB | ~27% vs Q4 |
| 30B (est) | ~18 GB | ~9 GB | ~11.5 GB | ~13.5 GB | ~25% vs Q4 |
| 32B (est) | ~19 GB | ~9.5 GB | ~12 GB | ~14.3 GB | ~25% vs Q4 |

## E. 30B Feasibility Interpretation

### Key Finding: Q2+ternary Alone Is Not Sufficient for 30B

Even with full linear Q2+ternary at 0.75×:
- 30B Q2+ternary full-linear estimate: **~13–14 GB**
- Plus KV/context at 2K: ~2–4 GB additional
- Total: **~15–18 GB** — borderline or exceeding 16GB RAM

### Path Forward: Combined Strategy

| Strategy | 30B Feasible? | Notes |
|----------|---------------|-------|
| Q4 native | ❌ OOM | ~18-22 GB exceeds 16GB |
| Q2 base only | ⚠️ Marginal | ~9GB base + KV may fit, but quality poor |
| Q2 + tern residual (full linear) | ⚠️ Borderline | ~13-14GB + KV may exceed RAM |
| Q2 base + selective residual | ✅ Possible | Budget-selected subset, lower residual cost |
| Q2 base + selective residual + out-of-core | ✅ Probable | Out-of-core paging for residual + layer streaming |

**Recommended 30B path:**
1. Q2 base weights (fit in RAM)
2. Selective residual overlay on top-K layers (within remaining budget)
3. Out-of-core layer paging for residual overlays not in active set
4. SDI context management for KV pressure

### Remaining Gap Analysis for 30B

| Component | Memory Est | Notes |
|----------|-----------|-------|
| Q2 base (30B) | ~9 GB | All layers, Q2 |
| KV/context (2K) | ~2-4 GB | Varies with sequence length |
| OS headroom | ~2 GB | Linux baseline |
| **Available for residuals** | **~1-3 GB** | 16 - 9 - 2-4 - 2 |
| Full linear ternary residuals | ~5 GB | All linear tensors |
| **Gap** | **~2-4 GB** | Full residuals exceed budget |

**Selective residual is required** — only top layers by score fit in the ~1-3GB residual budget.

## F. Residual Overlay Format Design

### Design Principles
1. **Native-layout aware** — residuals stored in the same orientation as the base tensor
2. **Tensor-family aware** — format metadata knows FFN_UP vs attn_output shapes
3. **Per-layer metadata** — each residual overlay has its own scale/sign metadata
4. **No f32 expansion** — ternary/int2/int4 only, scales stored separately
5. **Mmap-friendly** — can be memory-mapped without full load
6. **Deterministic decode** — same bits always decode to same values
7. **Budget-compatible** — layer selection encoded in manifest, not scattered

### Proposed Format: `prt_residual_<model_hash>.pgr` (Phase, not committed)

```
.prt_residual/
  manifest.json          — index of all residual overlays
  layer_00_ffn_up.bin   — binary residual data
  layer_05_ffn_up.bin
  ...
  layer_NN_attn_output.bin
```

### Manifest Schema

```json
{
  "version": "0.1",
  "model_id": "sha256_of_base_model",
  "created": "ISO-8601",
  "total_layers": 28,
  "residual_format": "ternary",
  "base_quant": "Q2",
  "layers": [
    {
      "layer": 0,
      "tensor": "ffn_up",
      "shape": [896, 4864],
      "slice": [512, 2048],
      "residual_bytes": 131072,
      "scales": {
        "sign": "ternary_sign_plane",
        "magnitude": "mean_of_nonzero_abs"
      },
      "metrics": {
        "validation_cosine_improvement": 0.7073,
        "validation_mae_hat": 0.460,
        "compression_ratio": 0.75
      },
      "paging": "full|mmap|lazy"
    }
  ],
  "total_residual_bytes": 3932160,
  "budget_checked": true,
  "q4_equivalent_bytes": 5242880
}
```

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Storage unit | Per-layer per-tensor | Enables selective activation |
| Ternary encoding | sign × magnitude (mean nonzero) | 1 bit/element, no f32 expansion |
| Scale storage | Per-layer float, not per-block | Low overhead, deterministic |
| Paging strategy | mmap + lazy load | Keeps resident RSS minimal |
| Manifest | JSON, human-readable | Easy inspection, versionable |
| Naming | sha256 model ID | Prevents mismatched base+residual |

### What This Format Does NOT Include

- No runtime integration (future work)
- No KV cache format (handled by SDI layer)
- No computation — purely a storage/manifest design
- No generation or quality guarantees

## G. Runtime Architecture Implication

### Envisioned Stack

```
┌─────────────────────────────────────────────────────────┐
│  Inference Request                                      │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  SDI Layer (Context Management)                         │
│  - Bounded KV: max 2K tokens                          │
│  - KV eviction policy                                  │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  Q2 Base Model (RAM-resident or mmap'd)               │
│  - 30B @ Q2 ≈ 9GB                                     │
│  - All layers available                                │
│  - Base quality only                                   │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  Residual Overlay Manager                              │
│  - Reads manifest.json                                │
│  - Selects top-K layers within residual budget        │
│  - Mmaps active residual planes                        │
│  - Adds to base at matvec time                        │
│  - Budget: ~1-3GB for 30B                            │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  Out-of-Core Pager (for non-active residuals)          │
│  - Streams residual overlays on demand                 │
│  - NVMe-backed, not RAM                                │
│  - Latency-tolerant                                    │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  Output Verifier (canary checks)                       │
│  - Detects quality collapse                            │
│  - Triggers fallback (reduce residual set)             │
└─────────────────────────────────────────────────────────┘
```

### Fallback Hierarchy

| State | Action |
|-------|--------|
| Memory pressure | Deactivate lowest-priority residual overlays |
| Quality collapse detected | Increase residual coverage for critical layers |
| Recovery | Restore deactivated overlays when budget allows |
| OOM imminent | Trigger full residual eviction + base-only mode |

**This is future architecture — NOT implemented yet.**

## H. Recommended Next Phase

**Phase 28Y — Residual Sidecar Format Specification + Budget Calculator**

Rationale: The residual overlay concept is now empirically validated across 5 tensor families and 2 model scales. Before moving to runtime integration or generation testing, we need:

1. **Formalize the sidecar format spec** — define the binary layout, manifest schema, and paging model explicitly
2. **Build a budget calculator** that takes a model path and outputs the Q2+ternary budget analysis
3. **Design the layer-selection algorithm** in code (not just documented heuristically)

This bridges empirical validation → format spec → implementation planning.

**Alternative:** Phase 28Y — attn_k/v validation to complete attention coverage (lower priority since k/v are small memory share).

## I. Models/Sidecars/F32 Refs Staged?
**NO.** Analysis and design only. GGUF metadata reads for architecture data — no model files staged.

## J. Secrets Detected?
None.

## K. Tags Touched?
None.

---

## Files Committed
- `examples/speculative/results/PHASE28X_FULL_LINEAR_TENSOR_BUDGET_EXTRAPOLATION.md` — this report
- `examples/speculative/results/phase28x_full_linear_tensor_budget_extrapolation.json` — structured results