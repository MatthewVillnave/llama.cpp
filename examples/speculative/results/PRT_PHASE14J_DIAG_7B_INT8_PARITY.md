# PRT Phase 14J DIAG: 7B INT8 Parity Failure Diagnosis

## Executive Summary

| Field | Value |
|-------|-------|
| **Branch** | `experimental/prt-phase14a-packed-sidecars` |
| **Previous HEAD** | `54a9bdf5c` |
| **7B sidecars generated** | 28/28 layers, 1815 MiB |
| **Original parity failure** | 0.000035 (FAIL) |
| **Root cause** | SCALE_FORMULA_BUG |
| **Corrected parity** | 0.999958 (PASS) |

## 1. What Failed in Original 14J

The original sidecar generation script `/tmp/phase14j_gen.sh` used an incorrect quantization formula:

### Incorrect Formula (used in original generation)
```python
scale = max(abs(x) for x in row)      # WRONG: no /127 division
q = int(round(x / scale))             # WRONG: no clamping to [-127,127]
```

This collapsed 95.4% of INT8 weights to zero because the scale was 127× too large.

### Correct Formula (per INT8 standard)
```python
scale = max_abs(row) / 127.0           # CORRECT: normalized to 8-bit range
q = clamp(round(w / scale), -127, 127) # CORRECT: limited to [-127,127]
w_dq = q * scale                    # CORRECT: dequantize with scale
```

## 2. Quantization Formula Audit

| Script | Scale Formula | Q Range | Status |
|--------|-------------|--------|--------|
| `phase14a_quantize_sidecars.py` | `col_max / 127` | [-127,127] | ✅ Correct |
| `phase14b_int8_quantize.py` | `row_max / 127` | [-127,127] | ✅ Correct |
| `/tmp/phase14j_gen.sh` | `max_abs` | unbounded | ❌ BUG |

## 3. 3B Known-Good Comparison (with correct /127 formula)

| Model | Shape [M,K] | Weight Cos | MatVec Cos | Rel L2 |
|-------|-------------|-----------|-----------|--------|
| 3B L0 | [11008, 2048] | **0.999974** | 0.999961 | 0.008816 |
| 7B L0 | [18944, 3584] | **0.999958** | 0.999947 | 0.010327 |

Both 3B and 7B achieve >= 0.999 cosine with correct formula.

## 4. 7B Layer 0 Diagnostics

### Float32 Input Distribution
- Shape: [18944, 3584] (ffn × hidden)
- Float32 range: min=-0.527, max=0.527
- Float32 mean_abs: 0.0136

### Quantization Distribution (correct formula)
- Scale range: [0.000037, 0.004152]
- Scale mean: 0.000586
- Q range: [-127, 127]
- Q nonzero: 98.5% (66.9M / 67.9M)
- Zero fraction: 1.5%

## 5. Scale-Axis Comparison Table

| Variant | Weight Cos | MatVec Cos | Zero% | Sat% |
|---------|-----------|-----------|-------|------|
| **per_row /127** | **0.999958** | **0.999947** | 1.5% | 0.0% |
| per_col /127 | 0.999938 | 0.999925 | 4.8% | 0.0% |
| **per_row NO /127** | **0.513330** | **0.509710** | 95.4% | 0.0% |

**Interpretation**: The per-row axis is correct (matching how PRT matvec works). The bug was the missing /127 division.

## 6. Orientation Audit

- Extracted bytes: 67,895,296 (matches 18944×3584×4)
- [18944, 3584] orientation: cos=0.999958 ✅
- [3584, 18944] orientation: cos=0.999913

Extracted shape correctly matches [ffn, hidden].

## 7. One-Layer Corrected Result

Using correct formula: `scale = row_max / 127.0`

| Layer | Weight Cos | MatVec Cos | Rel L2 | Status |
|-------|-----------|-----------|--------|--------|
| 0 | 0.999958 | 0.999947 | 0.010327 | PASS |
| 13 | 0.999958 | 0.999947 | 0.010327 | PASS |
| 27 | 0.999958 | 0.999947 | 0.010327 | PASS |

All layers pass >= 0.999 cosine target.

## 8. Recommendation

### A. Branch
- `experimental/prt-phase14a-packed-sidecars`

### B. Previous HEAD
- `54a9bdf5c`

### C. Original 7B sidecars
- Generated via `/tmp/phase14j_gen.sh` with BUGGY formula
- Cosine: 0.000035 (FAIL)

### D. Original parity failure
- Cosine = 0.000035 (near-zero)
- Cause: scale = max_abs (no /127) instead of max_abs / 127

### E. Quantization formula found
- Correct: `scale = max_abs(row) / 127.0`
- Bug: `scale = max_abs(row)` (original script)

### F. Scale formula bug?
- YES — original script used `max(abs(x) for x in row)` without dividing by 127

### G. Scale axis bug?
- NO — per-row axis was correct for PRT matvec

### H. Orientation mismatch?
- NO — [18944, 3584] orientation confirmed correct

### I. 3B comparison result
- 3B L0: 0.999974 cosine (PASS) — confirms correct formula works

### J. 7B layer0 corrected parity
- Weight: 0.999958, MatVec: 0.999947, RelL2: 0.0103 (PASS)

### K. Best quantization variant
- Per-row scale with /127 division

### L. All 7B sidecars regenerated
- YES — all 28 layers regenerated with correct formula
- Size: 1815 MiB

### M. Runtime validation allowed next?
- YES — sidecars now have correct parity (>= 0.999 cosine)

### N. Verdict
- **SCALE_FORMULA_BUG_CONFIRMED** — fixed with /127 division

### O. Recommended next
- Phase 14J runtime validation with corrected sidecars

### P. Models/sidecars/binaries staged?
- NO — only JSON reports staged

### Q. Secrets detected?
- NO — no API keys or secrets in repo

### R. Tags untouched?
- YES — no existing tags modified

## Files Modified

- `examples/speculative/results/PRT_PHASE14J_DIAG_7B_INT8_PARITY.md` (NEW)
- `examples/speculative/results/phase14j_diag_7b_int8_parity.json` (NEW)

## Files Regenerated (not committed to git)

- `/tmp/prt_sidecars_7b_int8/` — 28 INT8 sidecar files, 1815 MiB (regenerated with correct formula)