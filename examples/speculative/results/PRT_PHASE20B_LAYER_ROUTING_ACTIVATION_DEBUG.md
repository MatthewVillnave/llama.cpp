# PRT Phase 20B: Layer Routing Activation Debug

**Branch:** `experimental/prt-phase19a-alt-sidecar-backed`  
**HEAD:** `0977819173c3873adc849631761c67f2dcede473`  
**Date:** 2026-05-13

## Verdict: FAIL_LOGGING_AMBIGUITY

## Summary

PRT layer routing IS working correctly. The "prt_layer=0" logs that appeared in Phase 20A were misinterpreted - the field shows a boolean (0/1), not the layer ID. PRT activates on the intended layers as confirmed by:

1. `[PRT_COMPUTE] layer=10 mode=int6 hit=1` - PRT compute for layer 10
2. `[PRT_COMPUTE] layer=20 mode=int6 hit=1` - PRT compute for layer 20
3. `[PRT_SHAPE] IL=10 dst_ne=[18944,N] format=2` - Shape logs showing INT6 activation
4. Clean output: "The capital of France is Paris.", "The largest planet in our solar system is Jupiter."

## Tests Performed

### Phase 20B-A: Binary Verification
- Branch: `experimental/prt-phase19a-alt-sidecar-backed`
- All PRT flags present: `--prt-mode`, `--prt-only-layer`, `--prt-only-layers`, `--prt-disable-layers`, `--prt-sidecar-dir`, `--prt-sidecar-format`

### Phase 20B-E: Minimal Activation Tests

| Test | Command | Output | PRT Hits |
|------|---------|--------|---------|
| 1-layer only | `--prt-only-layer 10` | Paris | layer=10 |
| 2-layers | `--prt-only-layers 10,20` | Paris | layer=10,layer=20 |
| 3-layers | `--prt-only-layers 0,10,20` | Paris (longer) | layer=0,layer=10,layer=20 |
| corrupt control | `--prt-only-layers 0,1` | gibberish | layer=0,layer=1 |

### Timing Comparison

| Policy | Prompt t/s | Gen t/s | Delta |
|--------|------------|--------|-------|
| Native (no PRT) | 9.7 | 4.8 | baseline |
| (10,20) | 6.2 | 3.8 | -21% |

**(10,20) is 21% slower than native**, not faster.

### Output Quality

- P1 (Paris): ✅ Clean - "Paris"
- P2 (Jupiter): ✅ Clean - "Jupiter. It has a diameter of about 139,820 kilometers..."
- P5 (Math): ⚠️ Partial - "Average Speed} = frac{60 text{ miles}}{2"
- P6 (Prose): ⚠️ Partial - "Every morning, the sun would rise, casting a"
- Control (0,1): ❌ Corrupt - "Got it! I see understanding that the number of 2024..."
- Control (0,20,27): ❌ Mixed Chinese + gibberish

## Root Cause Analysis

**Logging field misinterpretation:**
- Field `prt_layer` in log line `[PRT-NATIVE] IL=X prt_layer=Y sidecar=(nil)` is a boolean (0/1)
- NOT the layer ID being processed
- The `IL=X` field shows the actual layer number
- `prt_layer=0` means "PRT not active for this layer" (boolean false)
- `prt_layer=1` means "PRT active for this layer" (boolean true)

## Recommended Next Steps

1. **Phase 20C** - Run full 8-prompt validation with (10,20) using proper output extraction
2. Consider renaming log field from `prt_layer` to `prt_active` to avoid confusion
3. If speedup is required, investigate why (10,20) is 21% slower - may need optimization

## Models/Sidecars/Binaries Staged
- Model: `/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-7B-Instruct-Q4_K_M.gguf`
- Sidecars: `/tmp/prt_sidecars_7b_int6_phase15b_packed`
- Binary: `./build/bin/llama-cli` (fresh build)

## Secrets Detected: None

## Existing Tags Touched: None