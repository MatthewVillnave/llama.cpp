# Phase 28N: Small-Model Offline Residual Overlay Prototype Design

## A. Branch & Commit
- Branch: `experimental/prt-phase19a-alt-sidecar-backed`
- HEAD before: `f8ada0ba5` (Phase 28M)
- HEAD after: `??` (docs only, no tag)

## B. Prototype Target
**Start with `qwen2.5:0.5B` — one layer, one tensor, FFN up projection preferred.**

Rationale: smallest available model, easiest memory profile, sufficient to validate ternary/INT2/INT4 residual math before touching 3B or 30B. After 0.5B tensor math works, consider 3B as second validation.

**Do not extract full model. Do not create huge f32 dumps.**

## C. Tensor Scope

### Recommended approach: A then B
1. **Phase A — Synthetic tensor prototype:** Generate synthetic W_ref/W_base tensors (rows×cols, e.g. 512×2048) in Python. Full control, no file IO, validates math pipeline end-to-end. No real model files.
2. **Phase B — Extracted small slice from 0.5B:** Extract a single FFN up projection slice (e.g. 256×512) via Ollama, if extraction is safe and small. Only after Phase A passes.

### Tensor class: FFN up projection
- Most quantization-sensitive layer in transformer
- Large enough to show meaningful residual
- Avoids attention complexity for v0

### Scope constraints
- One layer, one tensor
- Sampled rows/blocks if full tensor is large
- Synthetic activations initially (can use random vectors with fixed seed)

## D. Memory-Budget Calculator Design

### File: `examples/speculative/prt_residual_budget.py`

No model files needed. Pure calculation.

**Inputs:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `model_param_count` | int | e.g. 32_000_000_000 for 32B |
| `base_bits` | int | Quantization bits of base model (e.g. 2 for Q2) |
| `residual_bits` | int | Bits per residual element (2 for INT2, 4 for INT4, 1 for ternary) |
| `residual_layer_count` | int | How many layers get residuals |
| `residual_fraction` | float | Fraction of parameters in residual-selected layers (0.0–1.0) |
| `context_size` | int | Context length in tokens |
| `kv_estimate_per_token` | int | KV bytes per token (default ~2000) |
| `runtime_buffer_estimate` | int | Runtime buffer overhead in bytes (default ~1_000_000_000) |
| `os_headroom` | int | OS reserve in bytes (default ~1_000_000_000) |
| `ram_budget` | int | Available RAM in bytes (default ~16_000_000_000) |

**Outputs:**
| Output | Description |
|--------|-------------|
| `estimated_base_bytes` | Base model memory footprint |
| `estimated_residual_bytes` | Residual overlay memory footprint |
| `estimated_kv_bytes` | KV memory at given context size |
| `estimated_total_resident` | Sum of all components |
| `safe_unsafe` | "SAFE" if total < ram_budget, "UNSAFE" otherwise |
| `residual_budget_remaining` | Headroom after residuals |
| `residual_too_large` | True if residual > (Q4 - Q2) memory saving |

**Formula:**
```
base_bytes = (param_count * base_bits) / 8
kv_bytes = context_size * kv_estimate_per_token
residual_bytes = (param_count * residual_fraction * residual_bits) / 8
total = base_bytes + kv_bytes + residual_bytes + runtime_buffer + os_headroom
```

## E. Offline Prototype Script Design

### File: `examples/speculative/prt_residual_overlay_offline.py`

**Inputs:**
| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--mode` | str | required | `synthetic` or `slice` |
| `--rows` | int | 512 | Matrix rows |
| `--cols` | int | 2048 | Matrix columns |
| `--base-bits` | int | 2 | Base quantization bits (Q2) |
| `--residual-format` | str | `ternary` | `ternary`, `int2`, `int4` |
| `--budget-mb` | int | 2048 | Memory budget in MB |
| `--out-json` | path | required | Output JSON path |

**Behavior pipeline:**
```
1. Generate or load W_ref
   → synthetic: np.random.randn(rows, cols).astype(np.float32)
   → slice: load via Ollama extraction (future)

2. Quantize/reconstruct W_base as Q2-like
   → Round W_ref to nearest Q2 step
   → W_base = round(W_ref / q2_step) * q2_step
   → Simulates Q2 dequantization approximation

3. Compute residual R = W_ref - W_base

4. Compress R into selected format
   → ternary: R_hat = sign(R) * threshold_clip
   → int2: R_hat = clamp(R, -2, 1) as 2-bit signed
   → int4: R_hat = clamp(R, -8, 7) as 4-bit signed

5. Reconstruct W_hat = W_base + R_hat

6. Generate synthetic activations x
   → np.random.randn(cols, 64).astype(np.float32)
   → Fixed seed for determinism

7. Compare matvec outputs
   → y_ref = W_ref @ x
   → y_base = W_base @ x
   → y_hat = W_hat @ x
   → cosine(W_ref, W_base) vs cosine(W_ref, W_hat)

8. Report metrics
```

**Metrics reported:**
- `cosine_base_vs_ref` — cosine similarity of W_base @ x vs W_ref @ x
- `cosine_hat_vs_ref` — cosine similarity of W_hat @ x vs W_ref @ x
- `cosine_improvement` — cosine_hat - cosine_base
- `mae_base_vs_ref` — MAE of W_base @ x vs W_ref @ x
- `mae_hat_vs_ref` — MAE of W_hat @ x vs W_ref @ x
- `mae_improvement_pct` — (mae_base - mae_hat) / mae_base * 100
- `relative_l2_improvement` — ||W_base - W_ref|| / ||W_ref|| vs ||W_hat - W_ref|| / ||W_ref||
- `storage_bytes_base` — W_base in Q2 approximation
- `storage_bytes_residual` — R_hat compressed storage
- `storage_bytes_combined` — base + residual
- `q4_equivalent_bytes` — W_ref if stored as Q4
- `compression_ratio` — combined / q4_equivalent
- `pass` — residual improves cosine AND storage < q4_equivalent
- `fail_reason` — if fail, why

**Output format (JSON):**
```json
{
  "config": {"rows": 512, "cols": 2048, "base_bits": 2, "residual_format": "ternary"},
  "metrics": {
    "cosine_base_vs_ref": 0.73,
    "cosine_hat_vs_ref": 0.91,
    "cosine_improvement": 0.18,
    "mae_base_vs_ref": 0.12,
    "mae_hat_vs_ref": 0.05,
    "mae_improvement_pct": 58.3,
    "storage_bytes_base": 262144,
    "storage_bytes_residual": 32768,
    "storage_bytes_combined": 294912,
    "q4_equivalent_bytes": 524288,
    "compression_ratio": 0.563
  },
  "result": {"pass": true, "fail_reason": null}
}
```

## F. Acceptance Criteria

### Pass if ALL:
- `cosine_improvement > 0` — residual overlay strictly improves matvec parity
- `compression_ratio < 1.0` — combined residual+base storage is smaller than Q4 equivalent
- No f32 expansion in reconstruction (all operations stay in integer/low-bit domain)
- Results are deterministic (fixed seed)
- Budget calculator reports `safe_unsafe = SAFE`

### Fail if ANY:
- `cosine_improvement ≤ 0` — residual makes output worse or unchanged
- `compression_ratio ≥ 1.0` — residual is as large or larger than Q4 savings
- Reconstruction requires f32 expansion
- Results are non-deterministic without clear cause
- Budget calculator reports `UNSAFE`

## G. Recommended Next Phase

**Phase 28O — Implement synthetic residual overlay offline prototype + budget calculator**

Scope:
- No real model files
- Synthetic tensors only (mode=synthetic)
- Small sizes (512×2048 to start)
- JSON/MD report output
- No runtime integration
- Scripts committed to repo

**After Phase 28O passes on synthetic:**
- Phase 28P — add small 0.5B tensor slice extraction (mode=slice)
- Only if synthetic results are clean

## H. Safety Checklist
| Item | Status |
|------|--------|
| 30B/32B files touched? | **NO** ✅ |
| Models/sidecars/f32 refs staged? | **NO** ✅ |
| Raw logs/captures staged? | **NO** ✅ |
| llama.cpp modified? | **NO** ✅ |
| Tags touched? | **NO** ✅ |
| Secrets detected? | **NO** ✅ |

## Verdicts
- `PASS_PHASE28N_PROTOTYPE_DESIGN`
- `PASS_SMALL_MODEL_SCOPE_DEFINED`
- `PASS_BUDGET_CALCULATOR_DESIGNED`
- `PASS_OFFLINE_SCRIPT_DESIGNED`
- `RECOMMEND_SYNTHETIC_OFFLINE_PROTOTYPE`
- `BLOCKED_REPO_STATE_CLEAN`