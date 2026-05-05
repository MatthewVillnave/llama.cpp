# PRT Phase 13E: Active 0.5B PRT Canary After Shim Fix

## Context
Phase 13D-S attempted to fix PRT shim disabled-mode bug by guarding sidecar loading and callback installation when `g_prt_debug_mode == 0`.

## Test Environment
- Branch: `experimental/prt-phase13-model-generalization`
- HEAD: `60096649e86585e4ccad83318fe67a5e397703a5`
- Model: Qwen2.5-0.5B-Instruct-Q4_K_M
- Sidecar dir: `/tmp/prt_sidecars/` (24 files, nonzero)

## Sidecar Stats (verified nonzero)
| Layer | Size | Non-zero | Mean | Checksum |
|-------|------|----------|------|----------|
| 0 | 17432576 | 4,088,567/4,358,144 | 0.014413 | -13.197754 |
| 1 | 17432576 | 4,088,489/4,358,144 | 0.013463 | 15.113427 |
| 11 | 17432576 | 4,066,398/4,358,144 | 0.014290 | 31.876828 |
| 15 | 17432576 | 4,064,150/4,358,144 | 0.015344 | 6.081878 |
| 23 | 17432576 | 4,088,365/4,358,144 | 0.016037 | 127.408373 |

## Three-Way Test Results

### A. Native llama-cli
```
> The capital of France is
The capital of France is Paris.
[ Prompt: 80.5 t/s | Generation: 39.7 t/s ]
```
**Status:** PASS - Clean output

### B. PRT-disabled (no --prt-mode)
```
[GEN] step 18: token_id=374 is_eog=0
[DEBUG] token 18: n=3 buf_hex: 20 69 73  str=' ispital'
[GEN] step 19: token_id=279 is_eog=0
[DEBUG] token 19: n=4 buf_hex: 20 74 68 65  str=' theital'

=== Results ===
Total PRT replacements: 0  [fallback: 0]
[11BD] prt_true_replacement_calls: 0
[11BD] native_fallback_calls: 0
```
**Status:** FAIL - Output contains garbage tokens ("theital", "ispital")

### C. PRT-active (--prt-mode 5700)
```
[PRT_SHAPE] n_layer=24 M=896 N=4864 expected_bytes=17432576 sidecars=24 force_native=11,15
[PRT] Loaded 24/24 sidecars
[DEBUG] token 0: n=8 ... str=' located_sidecars/ffn_up_layer0_prt.bin'

=== Results ===
[11BD] prt_true_replacement_calls: 638
[11BD] native_fallback_calls: 16
[11BD] sidecar_L0_checksum: -13.197754
```
**Status:** FAIL - Path fragments in output tokens

## Analysis

### What Worked
- Sidecar loading: 24/24 files loaded correctly
- PRT computation: non-zero in_sum and out_sum (confirming PRT active)
- Force-native layers: 11,15 respected (16 fallback calls)

### What Failed
1. **Disabled-mode fix partial failure:** Even without `--prt-mode`, the output shows garbage tokens instead of clean generation
2. **Active-mode path fragments:** With `--prt-mode 5700`, path fragments still appear in generated token text
3. **The "theital" corruption:** Tokens show " theital", " ispital" - not coherent text

## Root Cause Analysis

The Phase 13D-S fix added guards at lines 432 and 443:
```cpp
if (g_prt_debug_mode > 0) {
    load_all_sidecars(model);
    // ...
}
```

But the disabled-mode test still shows corrupted output. This suggests:
1. Either the guard isn't fully preventing the issue
2. Or there's a separate code path causing corruption
3. Or the callback is still installed without guard

## Verdict: FAIL / BLOCKED

The Phase 13E canary shows:
- **Native:** Clean ✅
- **PRT-disabled:** Corrupted ❌ (should be clean after fix)
- **PRT-active:** Path fragments ❌ (active mode broken)

The shim fix did NOT resolve the disabled-mode corruption. Active-mode PRT still produces path fragments in output.

## Recommended Next Steps

1. **Debug disabled-mode guard:** Add printf to confirm guard is actually preventing sidecar loading
2. **Check callback installation:** Verify cb_eval is NOT installed when g_prt_debug_mode == 0  
3. **Active-mode memory corruption:** Debug why path fragments appear in active-mode token output
4. **Possible fix path:** The custom op may be writing to wrong buffer - audit ggml_map_custom2 usage

## Safety Check
- No model files staged: ✅
- No sidecar binaries staged: ✅  
- No secrets: ✅
- Tags untouched: ✅