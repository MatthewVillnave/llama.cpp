# PRT Phase 13B: Dynamic Sidecar Specification

**Date:** 2026-05-03

## What Changes

### Sidecar File Naming

Current:
```
/tmp/prt_sidecars/ffn_up_layer{L}_prt.bin
```

Proposed (unchanged — keep compatible with existing 3B sidecars):
```
/tmp/prt_sidecars/ffn_up_layer{L}_prt.bin
```
(Sidecar files for 3B are already at this path — no rename needed)

### Sidecar Metadata Manifest

The loader already uses file existence + naming convention. Adding validation:

```json
// Manifest (optional, for validation):
{
  "model": "Qwen2.5-0.5B-Instruct-Q4_K_M",
  "architecture": "qwen2",
  "n_layer": 24,
  "hidden_dim": 896,
  "ffn_dim": 4864,
  "sidecar_bytes": 17432576,
  "layers": [
    {"id": 0, "file": "ffn_up_layer0_prt.bin", "size": 17432576},
    ...
  ]
}
```

### Loader Validation Rules

1. **Layer count**: `TOTAL_LAYERS` ← `llama_model_n_layer(model)` — NOT hardcoded
2. **Sidecar M/N**: Read from first loaded tensor, not hardcoded constants
3. **File size check**: `expected = hidden_dim * ffn_dim * sizeof(float32)` — fail loudly if mismatch
4. **No silent fallback**: If `--prt-mode 5700` is set but sidecars are missing/wrong size → ERROR, don't silently skip
5. **Force-native bounds**: Verify forced-native layers are in `[0, n_layer-1]`

### Expected Sidecar Byte Rules

```python
# Per sidecar:
expected_bytes = hidden_dim * ffn_dim * 4  # float32

# Qwen2.5-3B: 2048 * 11008 * 4 = 90,113,024
# Qwen2.5-0.5B: 896 * 4864 * 4 = 17,432,576
# Qwen2.5-7B: (model dependent, likely 2048 * 18944 * 4 = ~155MB)
```

### Anchor Policy Mapping

Current: `--prt-force-native 11,15` (for 36-layer model)

Dynamic equivalent: `--prt-force-native {floor(n_layer/3.2)},{floor(n_layer/2.4)}`

For 24-layer (0.5B): `--prt-force-native 7,10` ≈ 1/3 and 1/2 depth

For 36-layer (3B): `--prt-force-native 11,15` (already validated)

### Compatibility Matrix

| Model | Layers | Hidden | FFN | Sidecar Bytes | Anchor Policy | Status |
|-------|--------|---------|-----|---------------|---------------|--------|
| Qwen2.5-0.5B | 24 | 896 | 4864 | 17.4MB | 7,10 or 8,12 | BLOCKED (fixed by patch) |
| Qwen2.5-1.5B | 28? | 1536? | 8960? | ~55MB? | 9,14? | UNTESTED |
| Qwen2.5-3B | 36 | 2048 | 11008 | 90MB | 11,15 | VALIDATED |
| Qwen2.5-7B | 28? | 3584? | 18944? | ~271MB? | 9,14? | UNTESTED |