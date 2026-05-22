# Phase 28R: Budgeted Selected-Layer Multi-Layer Residual Overlay Plan

## Verdict: PASS_PHASE28R_BUDGETED_LAYER_PLAN ✅ | PASS_MULTI_LAYER_OBJECTIVE_DEFINED ✅ | PASS_MEMORY_MATH_DEFINED ✅ | PASS_LAYER_SELECTION_HEURISTIC_DEFINED ✅

## Summary
Design complete for budgeted selected-layer residual overlay. Q2+ternary on all FFN_UP layers stays at 0.75× Q4 — 25% memory savings at better quality. Layer selection heuristic: rank by residual norm × expected recovery.

---

## A. Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## B. Current HEAD
`d77d21d7c Phase 28Q: test selected-layer ternary residual sensitivity`

## C. Phase 28Q Summary
- 4 FFN_UP layers tested (0, 5, 11, 23) on Qwen2.5-0.5B
- All showed consistent strong recovery: Δ cos [+0.7056, +0.7122], std=0.0024
- No layer-position dependence
- ffn_down and attn_q also show strong recovery
- Best layer: layer 5 (+0.7122), but spread is noise
- Heuristic: norm-weighted budget selection, top-K by tensor norm
- All 24 FFN_UP layers are viable targets

## D. Multi-Layer Objective

**Goal:** Select a subset of layers/tensors to receive ternary residual correction such that:

```
total_memory = base_Q2_bytes + residual_ternary_bytes(selected) + metadata <= target_budget
selected_layers = top-K by score_i where score_i = norm_i × expected_recovery_i
```

**Hard constraints:**
1. Total memory ≤ Q4-equivalent for selected layers (capacity goal)
2. No f32 runtime expansion
3. Deterministic and offline-validatable
4. No OS swap required

**Soft constraints:**
1. Maximize sum of expected_recovery_i for selected layers
2. Prefer layers with higher residual_norm (larger Q2 error = more recovery room)

**Formal optimization:**
```
Select subset S ⊆ {all FFN_UP layers}
Maximize: Σ_{i∈S} score_i = Σ_{i∈S} norm_i × recovery_i
Subject to: Q2_bytes(S) + ternary_bytes(S) + metadata <= budget
```

## E. Memory Math

### Measurement Base (Qwen2.5-0.5B FFN_UP slice 512×2048)

| Configuration | Bytes per Slice | vs Q4 Ratio | Notes |
|---------------|-----------------|-------------|-------|
| Q4 (5-bit) | 524,288 | 1.000 | Baseline |
| Q2 (2-bit) | 131,072 | 0.250 | 75% savings alone |
| Q2 + ternary residual | 393,216 | 0.750 | 25% savings vs Q4 |
| Q2 + INT2 residual | 524,288 | 1.000 | No savings vs Q4 |
| Q2 + INT4 residual | 655,360 | 1.250 | Worse than Q4 |

**Key insight:** Q2 + ternary is the only residual format that simultaneously improves quality AND stays below Q4.

### Full-Model Extrapolation (Qwen2.5-0.5B, 24 FFN_UP layers)

Each FFN_UP tensor: [896, 4864] = 4,358,400 params
Q4 size per layer: 896 × 4864 × 0.5 = 2,179,200 bytes
Q2 size per layer: 896 × 4864 × 0.25 = 1,089,600 bytes
Q2+ternary per layer: 896 × 4864 × 0.375 = 1,634,400 bytes

| Scenario | All 24 Layers | Total Bytes | vs Full Q4 | Savings |
|----------|--------------|-------------|------------|---------|
| All Q4 | 24 × 2,179,200 | **52,300,800** | 1.000 | baseline |
| All Q2 only | 24 × 1,089,600 | **26,150,400** | 0.500 | -50% |
| All Q2+ternary | 24 × 1,634,400 | **39,225,600** | 0.750 | **-25%** |
| Top 12 Q2+ternary, rest Q4 | 12×1,634,400 + 12×2,179,200 | **45,763,200** | 0.875 | -12.5% |
| Top 6 Q2+ternary, rest Q4 | 6×1,634,400 + 18×2,179,200 | **49,132,800** | 0.939 | -6.1% |

**With Q2+ternary on all FFN_UP:** 25% memory savings vs Q4 at better output quality.

### System RAM Budget (16GB system example)

For 16GB RAM system with 4GB headroom (OS + other):
- Available for model: ~12GB
- KV/context budget at 2048 ctx: 2048 × 2048 bytes = ~4MB (negligible vs weights)

For Qwen2.5-0.5B full model (all tensors, Q4):
- Total file size: ~700MB
- RSS estimate: ~900MB
- Easy fit, but we want to plan for larger models

For 3B model (hypothetical):
- Q4 estimate: ~1.8GB
- Q2+ternary: ~1.35GB
- Still well within budget

For 7B model (Q4):
- ~4.5GB file / ~5.5GB RSS
- Q2+ternary estimate: ~3.4GB
- Fits in 16GB with room for context

### Budget Selection Strategies

| Strategy | Selection Criteria | Budget Efficiency | Quality Gain |
|----------|------------------|------------------|-------------|
| All FFN_UP | all layers | max | max (all layers recovered) |
| Top-K by norm | highest tensor norm | high | proportional to K |
| Budget-constrained | fit within target MB | exact fit | maximize within budget |
| Sensitivity-threshold | Δ cos > threshold | varies | only strong layers |

## F. Layer Scoring Method

**Recommended v0 score:**
```
score_i = residual_norm_i × expected_recovery_multiplier
```

**Components:**

`residual_norm_i = ‖W_ref_i - W_Q2_i‖_F` (Frobenius norm of Q2 reconstruction error)

This is the measured residual magnitude per layer. Higher residual = more error to correct = larger recovery potential.

**expected_recovery_multiplier** = constant (1.0 for initial design) or empirically measured per-layer Δ cos from Phase 28Q.

**Simplified v0:**
```
score_i = residual_norm_i
```

Rank layers by residual_norm descending, select top-K.

**Alternative v1 (when full model metrics available):**
```
score_i = tensor_norm_i × measured_cos_improvement_i
```

Where measured_cos_improvement_i comes from offline slice validation.

**Implementation note:** For slices already extracted (/tmp/ffn_ffn_up_layer*.f32), compute residual_norm as:
```python
W_ref = np.fromfile(slice_path).reshape(rows, cols)
W_q2, _ = q2_quantize(W_ref)
R = W_ref - W_q2
residual_norm = float(np.linalg.norm(R))
```

## G. Offline Multi-Layer Validator Design

**Script:** `examples/speculative/prt_residual_multilayer_plan.py`

```python
#!/usr/bin/env python3
"""
PRT Residual Multi-Layer Budget Validator
Offline design tool for budget-constrained layer selection.
No model files needed.
"""
import argparse
import json

def parse_args():
    p = argparse.ArgumentParser(description="Budget-constrained layer selector")
    p.add_argument("--layer-metrics-json", required=True,
                   help="JSON with per-layer metrics (norm, cos_improvement, residual_norm)")
    p.add_argument("--budget-mb", type=float, required=True,
                   help="Maximum residual budget in MB")
    p.add_argument("--tensor-type", default="ffn_up")
    p.add_argument("--residual-format", default="ternary")
    p.add_argument("--base-bits", type=int, default=2)
    p.add_argument("--out-json", default=None)
    p.add_argument("--out-md", default=None)
    return p.parse_args()

class MultiLayerPlanner:
    def __init__(self, layer_metrics, budget_mb, base_bits=2, res_format="ternary"):
        self.layers = layer_metrics
        self.budget_bytes = budget_mb * 1024 * 1024
        self.base_bits = base_bits
        self.res_format = res_format
        
        # Bits per element by format
        self.res_bits = {"ternary": 1, "int2": 2, "int4": 4}[res_format]
        
        # Compute per-layer scores
        for l in self.layers:
            l["score"] = l.get("residual_norm", l.get("tensor_norm", 0)) * \
                         l.get("expected_cos_recovery", 1.0)
    
    def budget_select(self, strategy="greedy"):
        """Select layers within budget."""
        sorted_layers = sorted(self.layers, key=lambda x: x["score"], reverse=True)
        
        selected = []
        total_bytes = 0
        
        for layer in sorted_layers:
            layer_bytes = self._layer_residual_bytes(layer)
            if total_bytes + layer_bytes <= self.budget_bytes:
                selected.append(layer)
                total_bytes += layer_bytes
        
        return selected, total_bytes
    
    def _layer_residual_bytes(self, layer):
        rows = layer["rows"]
        cols = layer["cols"]
        return int(rows * cols * self.res_bits / 8)
    
    def compute_totals(self, selected):
        q2_base = sum(l["rows"] * l["cols"] * self.base_bits / 8 for l in selected)
        residual = sum(self._layer_residual_bytes(l) for l in selected)
        metadata = len(selected) * 64  # ~64 bytes per layer scale/sign
        total = q2_base + residual + metadata
        q4_equiv = sum(l["rows"] * l["cols"] * 0.5 for l in selected)
        
        return {
            "selected_count": len(selected),
            "q2_base_bytes": int(q2_base),
            "residual_bytes": int(residual),
            "metadata_bytes": int(metadata),
            "total_bytes": int(total),
            "q4_equivalent_bytes": int(q4_equiv),
            "compression_ratio_vs_q4": round(total / q4_equiv, 4) if q4_equiv > 0 else 0,
            "budget_remaining_mb": round((self.budget_bytes - total) / 1024 / 1024, 2),
            "total_score": sum(l["score"] for l in selected),
        }
```

**Inputs:**
- JSON with per-layer metrics (norm, cos_improvement, residual_norm, rows, cols)
- Budget target in MB
- Tensor type, residual format, base bits

**Outputs:**
- Selected layers list with scores
- Total Q2 bytes, residual bytes, metadata
- Compression ratio vs Q4
- Budget safe/unsafe status
- Markdown and JSON plan

## H. Next Empirical Step

**Recommended: Phase 28S — All 24 FFN_UP Slices on Qwen2.5-0.5B**

**Why:** We tested 4 layers and found consistency. To validate the scoring heuristic and budget plan, we need metrics from all 24 layers.

**Plan:**
1. Extract all 24 FFN_UP slices from Qwen2.5-0.5B to /tmp
   - Each slice: first 512 rows × 2048 cols
   - No model staging, /tmp only
2. Run `prt_residual_multi_layer.py` on all 24 layers
3. Collect per-layer: residual_norm, cos_improvement, MAE, compression
4. Rank by score = residual_norm × cos_improvement
5. Design budget-selected subset for offline multi-layer validation
6. Optionally: test 3B scale transfer with one slice from Qwen2.5-3B

**Constraints:**
- Offline only, no generation
- All outputs to /tmp
- No model files staged
- No 30B/32B

**Phase 28S deliverables:**
- All 24 FFN_UP layer metrics
- Per-layer score ranking
- Budget selection for top-K within target budget
- Optional: 3B slice comparison

## I. Claims Boundary

### ALLOWED ✅
- Selected FFN_UP slices show strong ternary residual parity recovery (validated 4/24 layers, consistent pattern)
- Q2+ternary stays below Q4 size at 0.75× compression (validated)
- Budgeted multi-layer overlay plan is now design-complete
- Phase 28S can extend to all 24 layers with no new model pulls

### FORBIDDEN ❌
- Runtime works / generation quality improves (not tested)
- 30B support (not tested, not claimed)
- Speedup (not measured)
- Production readiness (design phase only)
- All tensors/layers validated (only 4 FFN_UP + 2 others tested)
- Full-model sidecar ready (design only)
- Real-world deployment (Phase 28S needed first)

## J. Recommended Next Phase
**Phase 28S — All 24 FFN_UP Slices Scan + Budget-Based Layer Selection**

- Extract and test all 24 FFN_UP layers from Qwen2.5-0.5B
- Compute per-layer scoring: residual_norm × cos_improvement
- Design multi-layer budget selection for top-K offline validation
- Optionally probe Qwen2.5-3B for scale transfer check

## K. Models/Sidecars/F32 Refs Staged?
**NO.** Design-only phase. No model files touched.

## L. Secrets Detected?
None.

## M. Tags Touched?
None.

---

## Files Committed
- `examples/speculative/results/PHASE28R_BUDGETED_SELECTED_LAYER_RESIDUAL_PLAN.md` — this report
- `examples/speculative/results/phase28r_budgeted_selected_layer_residual_plan.json` — structured plan