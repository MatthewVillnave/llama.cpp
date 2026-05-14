# Phase 10E-3S: Sidecar Coverage Correction

## Corrected Model Layer Count

- Model: Qwen2.5-3B-Instruct-Q4_K_M.gguf
- `qwen2.block_count` from GGUF metadata: **36 layers** (blk.0 through blk.35)

## Current Sidecar Coverage

| Category | Count | Layers |
|----------|-------|--------|
| Sidecars available | 28 | 0–27 |
| Layers missing sidecar | 8 | 28–35 |
| Total sidecars needed | 36 | all layers |
| Coverage | **77.8%** | — |

## Missing Sidecars

No sidecar files exist for:
- `ffn_up_layer28_prt.bin`
- `ffn_up_layer29_prt.bin`
- `ffn_up_layer30_prt.bin`
- `ffn_up_layer31_prt.bin`
- `ffn_up_layer32_prt.bin`
- `ffn_up_layer33_prt.bin`
- `ffn_up_layer34_prt.bin`
- `ffn_up_layer35_prt.bin`

## Status: All-Layer PRT NOT Yet Possible

Full-model all-layer sidecar coverage is incomplete (77.8%). Running all-layer replacement would result in identity fallbacks for layers 28–35, silently degrading PRT quality without producing an error.

## Layer0 Canary Status

The layer0 canary (Phase 10E-3S) **can proceed** with only layer0 sidecar coverage, because:
1. Scope is explicitly restricted to layer 0
2. Only `ffn_up_layer0_prt.bin` is required
3. No all-layer replacement is attempted
4. Non-layer0 layers use standard FFN (no fallback required)

## Required for All-Layer Coverage

To enable all-layer PRT:
- Generate sidecars for layers 28–35 (8 additional files)
- Verify each sidecar has correct dimensions [2048×11008]
- Update `run_batch.py` / `generate_new_fields.py` to produce all 36 layers

## Note on Previous Phase 10E-3R

Phase 10E-3R was incorrectly scoped as "all-layer replacement" with only 28 sidecars. The root cause was state bleed causing wrong layer lookup. The layer0 canary approach (Phase 10E-3S) correctly narrows scope to match available coverage.
