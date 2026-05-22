# Phase 28Y: PRT Residual Sidecar Format Spec + Budget Calculator Update

## Verdict: PASS_PHASE28Y_SIDECAR_FORMAT_SPEC ✅ | PASS_MANIFEST_SCHEMA_DEFINED ✅ | PASS_TERNARY_ENCODING_DEFINED ✅ | PASS_BUDGET_POLICY_DEFINED

## Summary
Formalized the PRT residual sidecar format specification, updated budget policies, and defined the validation/compatibility rules. This completes the design phase for residual overlay storage and moves the project toward implementation.

---

## A. Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## B. Current HEAD
`e1fc0ea1a Phase 28X: extrapolate full linear tensor residual budget`

## C. Sidecar Directory Layout

```
.prt_residual/<model_hash>/
├── manifest.json                    # Index of all overlays
├── layers/                          # Per-layer residual overlays
│   ├── layer_000/
│   │   ├── ffn_up.trit             # Ternary residual binary
│   │   ├── ffn_down.trit
│   │   ├── ffn_gate.trit
│   │   ├── attn_q.trit
│   │   └── attn_output.trit
│   ├── layer_001/
│   │   └── ...
│   └── layer_NNN/
├── metrics/
│   ├── validation_summary.json     # Overall validation results
│   └── per_tensor_metrics.json     # Per-layer per-tensor metrics
├── README.md                       # Optional documentation
└── compatibility.json             # Format version + compatibility rules
```

### Naming Convention

| Field | Definition |
|-------|-----------|
| `model_hash` | SHA-256 of base model file, first 12 hex chars (e.g., `a74ae894d2a6`) |
| `layer_NNN` | Zero-padded 3-digit layer index (e.g., `layer_000`, `layer_027`) |
| `.trit` extension | Ternary residual file — chosen to distinguish from `.bin`/`.gguf` |

### Compatibility Rules
- Format version stored in `manifest.json` and `.trit` header
- Reader must check version before parsing
- Upgrading format requires version bump + migration path
- Backward compatibility not guaranteed across major versions

---

## D. Manifest Schema

### `manifest.json` — Top-Level Fields

```json
{
  "format_name": "prt-residual-v1",
  "format_version": "0.1.0",
  "source_model": "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
  "source_model_hash": "a74ae894d2a6b4a4ba41660be9555a21194c590a",
  "base_quant": "Q2",
  "residual_format": "ternary",
  "created_at": "2026-05-22T14:20:00Z",
  "generator": "prt-phase28y",
  "layer_count": 28,
  "tensor_families": ["ffn_up", "ffn_down", "ffn_gate", "attn_q", "attn_output"],
  "budget_policy": "budget_greedy",
  "budget_bytes": 1073741824,
  "total_residual_bytes": 3932160,
  "compression_ratio_vs_q4": 0.75,
  "validation_passed": true,
  "layers": [...]
}
```

### Per-Tensor Entry Schema

```json
{
  "layer": 0,
  "tensor_name": "blk.0.ffn_up.weight",
  "tensor_family": "ffn_up",
  "shape": [896, 4864],
  "slice": [512, 2048],
  "dtype": "ternary",
  "residual_encoding": {
    "sign_bits": 1,
    "magnitude_encoding": "mean_nonzero_abs",
    "block_size_rows": 512,
    "block_size_cols": 256
  },
  "scales": {
    "type": "float32",
    "count": 8,
    "encoding": "per_block_row"
  },
  "file_path": "layers/layer_000/ffn_up.trit",
  "byte_size": 131072,
  "compression_ratio_vs_q4": 0.75,
  "q2_base_assumption": "Q2_K",
  "validation_metrics": {
    "q2_cosine_vs_ref": 0.0036,
    "q2_ternary_cosine_vs_ref": 0.7109,
    "delta_cosine": 0.7073,
    "mae_base": 55.807,
    "mae_hat": 0.460,
    "rel_l2_base": 81.84,
    "rel_l2_hat": 0.70,
    "compression_ratio": 0.75,
    "deterministic_seed": 123,
    "verdict": "STRONG_RECOVERY"
  },
  "status": "active"
}
```

### `metrics/validation_summary.json`

```json
{
  "total_tensors": 140,
  "validated_tensors": 140,
  "strong_recovery_count": 140,
  "weak_recovery_count": 0,
  "blocked_count": 0,
  "mean_delta_cosine": 0.6987,
  "std_delta_cosine": 0.0098,
  "all_passed": true
}
```

---

## E. Ternary Binary Encoding (.trit)

### Header (32 bytes, little-endian)

| Offset | Size | Field | Type | Description |
|--------|------|-------|------|-------------|
| 0 | 4 | `magic` | u32 | Magic: `0x54524954` ("TRIT") |
| 4 | 2 | `version_major` | u16 | Format version major |
| 6 | 2 | `version_minor` | u16 | Format version minor |
| 8 | 4 | `rows` | u32 | Logical rows |
| 12 | 4 | `cols` | u32 | Logical cols |
| 16 | 2 | `block_rows` | u16 | Block size rows |
| 18 | 2 | `block_cols` | u16 | Block size cols |
| 20 | 4 | `scale_count` | u32 | Number of scale values |
| 24 | 4 | `payload_offset` | u32 | Offset to packed data from file start |
| 28 | 2 | `flags` | u16 | Reserved flags |
| 30 | 2 | `checksum` | u16 | CRC16 of header |

### Packed Ternary Payload

- **8 trits per byte** (3 values × 8 = 24, fits in a byte with 2 bits unused, or use 3 bits per trit = 1.5 bytes per trit — chosen: 3 bits per trit for simplicity)
- **Packing scheme:** 3 bits per trit, values {0=00, 1=01, -1=11}
- **Row-major order:** row 0 col 0..cols, row 1 col 0..cols, etc.
- **Block padding:** rows padded to `block_rows` multiple; last block zero-padded
- **Scale array:** float32[scale_count] immediately after payload, before any checksum

### Scale Encoding

| Encoding | Storage | Decode |
|----------|---------|--------|
| `mean_nonzero_abs` | float32 per block row | Scale = mean(\|R[nonzero]\|) per row |
| `max_abs` | float32 per block row | Scale = max(\|R\|) per row |
| `fixed` | float32 global | Scale = predefined constant |

### Decoding Algorithm

```python
def decode_trit_block(trit_bytes, scales, rows, cols, block_rows, block_cols):
    result = np.zeros((rows, cols), dtype=np.float32)
    for row_block in range(0, rows, block_rows):
        for col_block in range(0, cols, block_cols):
            br = min(block_rows, rows - row_block)
            bc = min(block_cols, cols - col_block)
            scale_idx = (row_block // block_rows) * (cols // block_cols) + (col_block // block_cols)
            scale = scales[scale_idx]
            
            for r in range(br):
                for c in range(bc):
                    trit_idx = (row_block + r) * cols + (col_block + c)
                    byte_idx = (trit_idx * 3) // 8
                    bit_offset = (trit_idx * 3) % 8
                    bits = (trit_bytes[byte_idx] >> bit_offset) & 0x7
                    sign = {0: 0, 1: 1, 7: -1}.get(bits, 0)
                    result[row_block + r, col_block + c] = sign * scale
    return result
```

### File Structure (Final)

```
[32-byte header]
[payload: packed trits, row-major, 3 bits each]
[scales: float32[scale_count], immediately after payload]
[optional: padding to 4-byte alignment]
[optional: 16-byte trailer with SHA256 of payload]
```

### AVX2 Decode Path (Future)

The packing scheme is designed for SIMD-friendly decode:
- Load 32 bytes → decode 85 trits in parallel
- 3-bit trick enables bitwise decode without table lookups
- Per-row scale broadcast as float32 vector

---

## F. Budget Calculator Update

### New Capabilities for `prt_residual_budget.py`

```python
class ResidualBudgetCalculator:
    def __init__(self, model_path=None, manifest_path=None):
        self.model_path = model_path
        self.manifest_path = manifest_path
        self.layers = {}
    
    def load_manifest(self, manifest_path):
        """Load manifest.json to get residual metadata."""
        with open(manifest_path) as f:
            return json.load(f)
    
    def compute_budget(self, policy, max_residual_bytes, kv_bytes=0):
        """Compute memory budget under a given policy."""
        policies = {
            'base_only': self._policy_base_only,
            'mlp_all': self._policy_mlp_all,
            'attention_partial': self._policy_attention,
            'budget_greedy': self._policy_budget_greedy,
            'manual': self._policy_manual,
        }
        return policies[policy](max_residual_bytes, kv_bytes)
    
    def _policy_mlp_all(self, max_bytes, kv_bytes):
        """FFN_UP + FFN_DOWN + FFN_GATE only."""
        # Select all MLP tensors, sort by score, include until budget
        pass
    
    def _policy_budget_greedy(self, max_bytes, kv_bytes):
        """Rank by (residual_norm × cos_improvement) / byte_size, greedy fill."""
        pass
    
    def estimate_total(self, policy, kv_tokens=2048):
        """Estimate total RSS under policy."""
        kv_bytes = kv_tokens * 2048  # rough estimate
        base = self.get_q2_base_rss()
        residual = self.compute_budget(policy, ...)['total_bytes']
        return base + residual + kv_bytes
```

### Updated CLI Arguments

```bash
python3 prt_residual_budget.py \
  --model-path /path/to/model.gguf \
  --manifest /path/to/manifest.json \
  --policy budget_greedy \
  --max-residual-bytes 2147483648 \
  --kv-tokens 2048 \
  --ram-gb 16 \
  --out-json /tmp/budget_report.json
```

### Output Report Structure

```json
{
  "policy": "budget_greedy",
  "selected_tensors": [...],
  "memory_breakdown": {
    "q2_base_rss_gb": 9.0,
    "residual_rss_gb": 2.0,
    "kv_rss_gb": 0.004,
    "headroom_gb": 2.0,
    "total_rss_gb": 13.0
  },
  "budget_safe": true,
  "compression_vs_q4": 0.75,
  "savings_vs_q4_gb": 3.2
}
```

---

## G. Selected Residual Policies

| Policy | Description | Memory | Quality | Risk |
|--------|------------|--------|--------|------|
| **0 — Base only** | Q2 base, no residuals | ~9GB (30B) | Lowest | None |
| **1 — MLP all** | FFN_UP + DOWN + GATE | ~11-12GB (30B) | High MLP | Low |
| **2 — Attention partial** | attn_q + attn_output | ~10-11GB (30B) | Improved attn | Low |
| **3 — Budget greedy** | Top-K by score/byte | ≤budget | Optimized | Medium |
| **4 — Manual canary** | Specific sensitive layers | Variable | Targeted | Low |

### Policy Selection Heuristic

```
IF memory_critical:
  → Policy 0 (base only) OR Policy 3 (budget greedy at low budget)
ELIF quality_critical:
  → Policy 1 (MLP all) if budget allows
ELIF attention_quality_issue:
  → Policy 2 (attention partial)
ELIF balanced:
  → Policy 3 (budget greedy, ~1-2GB residual budget)
```

---

## H. Validation/Compatibility Rules

### Sidecar is VALID iff ALL:

1. **Hash match:** `source_model_hash` matches SHA-256 of base model file
2. **Shape match:** `shape` in manifest matches actual tensor metadata in base model
3. **Version compatible:** format version readable by current reader
4. **Checksum passes:** `.trit` file CRC16/CRC32 matches
5. **Compression ratio ≤ 0.80:** `compression_ratio_vs_q4` ≤ 0.80 (sanity check)
6. **Validation metrics threshold:** `delta_cosine` ≥ 0.5 AND `verdict` = STRONG_RECOVERY
7. **No missing required tensors:** For selected policy, all required tensors present
8. **Budget within limit:** `total_residual_bytes` ≤ configured `max_residual_bytes`

### Sidecar is INVALID if ANY:

- Hash mismatch → wrong model
- Shape mismatch → incompatible base
- Stale format version → reader can't parse
- Checksum fail → corrupted
- Compression > Q4 → format broken
- Validation below threshold → quality insufficient
- Missing tensors → incomplete sidecar
- Budget exceeded → OOM risk

---

## I. Implementation Order

### Phase 1: Schema + Validator
1. Create `prt_residual_schema.py` — JSON schema validation for manifest
2. Create `prt_validate_manifest.py` — check manifest against model file

### Phase 2: Synthetic .trit Writer/Reader
3. Create `prt_trit_io.py` — write/read synthetic ternary data
4. Test round-trip parity on random tensors

### Phase 3: Real Slice .trit Writer/Reader
5. Extract a real 0.5B ffn_up slice
6. Write `.trit`, read back, verify cosine matches original

### Phase 4: Offline Parity Test
7. Create `prt_offline_parity_test.py` — compare Q2+trit matvec vs Q4 full matvec

### Phase 5: Multi-Layer Manifest Builder
8. Extend to build full manifest from scan results

### Phase 6: Budget Calculator Integration
9. Update `prt_residual_budget.py` to read manifest and compute real budgets

### Phase 7: Runtime Integration (Future)
10. **NOT in current scope** — requires llama.cpp modification and actual generation testing

---

## J. Recommended Next Phase

**Phase 28Z — Manifest Validator + Real Slice .trit Writer**

Build Phase 1 + Phase 2 from the implementation order:
- `prt_residual_schema.py` — manifest validation
- `prt_trit_io.py` — binary .trit writer/reader for synthetic and real slices
- `prt_validate_manifest.py` — check manifest against GGUF model metadata

This creates the first concrete implementation artifact from the format spec.

---

## K. Models/Sidecars/F32 Refs Staged?
**NO.** Design/spec only. No real sidecars generated.

## L. Secrets Detected?
None.

## M. Tags Touched?
None.

---

## Files Committed
- `examples/speculative/results/PHASE28Y_PRT_RESIDUAL_SIDECAR_FORMAT_SPEC.md` — this report
- `examples/speculative/results/phase28y_prt_residual_sidecar_format_spec.json` — structured spec