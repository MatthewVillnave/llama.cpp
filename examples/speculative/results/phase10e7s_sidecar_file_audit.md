# Phase 10E-7S: Sidecar File Audit

## Sidecar File Audit Results

**File:** `/tmp/prt_sidecars/ffn_up_layer0_prt.bin`

| Property | Value |
|----------|-------|
| File size | 90,177,536 bytes |
| Expected size | 90,177,536 bytes (2048 × 11008 × 4) |
| Match | YES ✓ |

### First 16 Bytes (hex)
```
80 df bc 3c 80 df bc 3c 00 34 71 3b 00 1e 86 3c
```

### First 8 Floats (interpreted)
| Index | Value |
|-------|-------|
| 0 | 0.023056 |
| 1 | 0.023056 |
| 2 | 0.003680 |
| 3 | 0.016372 |
| 4 | 0.003680 |
| 5 | 0.023733 |
| 6 | 0.023733 |
| 7 | 0.023056 |

### Path String Search
Searched for: "ffn_up", "prt_sidecars", "/tmp", ".bin", "layer35" inside binary data.
**Result: NONE FOUND — CLEAN**

### Statistics
| Metric | Value |
|--------|-------|
| Total elements | 22,544,384 |
| NaNs | 0 |
| Infs | 0 |
| Min | 0.000000 |
| Max | 0.360853 |
| Mean | 0.018424 |

## Verdict
**Sidecar file is VALID.** No path strings embedded. Expected size matches. NaN/Inf-free. Values are reasonable magnitudes (|W| weights between 0 and 0.36). No contamination in the file itself.

The sidecar file is clean. Corruption must be in the loader or PRT compute path.