# PRT Phase 14K: 7B INT8 Single-Prompt Runtime Canary

## Executive Summary

| Field | Value |
|-------|-------|
| **Branch** | `experimental/prt-phase14a-packed-sidecars` |
| **Previous HEAD** | `13ff1d6c7` |
| **New HEAD** | `13ff1d6c7` (same — only CLI fix, no new commit needed) |
| **Model** | Qwen2.5-7B-Instruct-Q4_K_M.gguf (4.4 GB) |
| **Model SHA256** | `1875fb29e8c91c86615c00e92d8b4114e56bc24359adb5a8db8b36452fae4a49` |
| **Model shape** | n_layer=28, hidden=3584, ffn=18944 |
| **Sidecar dir** | `/tmp/prt_sidecars_7b_int8/` |
| **Sidecars loaded** | 28/28 ✅ |
| **Verdict** | **PASS_7B_INT8_SINGLE_PROMPT** |

## Native 7B Baseline

```
Prompt: "The capital of France is"
Output: "The capital of France is Paris."
Timing: 4.337s wall, 9.7 t/s gen
```

## INT8 PRT 7B Canary Result

```
Prompt: "The capital of France is"
Output: "The capital of France is Paris."
Timing: 4.827s wall, 9.7 t/s gen
Sidecars loaded: 28/28 ✅
Format: int8 (per-row scale)
Shape: n_layer=28 hidden=3584 ffn=18944 sidecar_rows=18944 sidecar_cols=3584 runtime_M=18944 runtime_N=3584 format=int8
Force-native layers: 11, 15
Sidecar load: 544.38ms
Build: b8799-13ff1d6c7
```

## Output Quality

- Native output: "The capital of France is Paris."
- INT8 PRT output: "The capital of France is Paris."
- **Exact semantic match** ✅

## Timing Comparison

| Mode | Wall (s) | Gen t/s | Status |
|------|----------|---------|--------|
| Native | 4.337 | 9.7 | ✅ |
| INT8 PRT | 4.827 | 9.7 | ✅ |

**INT8/Native ratio**: 0.90× wall, 1.00× gen (timing informational only, no broad claims)

## PRT_SHAPE_DETAIL

```
n_layer=28
hidden=3584
ffn=18944
sidecar_rows=18944
sidecar_cols=3584
runtime_M=18944
runtime_N=3584
format=int8
```

## Key Fixes Applied

### Fix 1: Sidecar naming
- **Bug**: loader expected `ffn_up_layer{L}_prt.int8`, files were `ffn_up_layer{L}.int8` + `ffn_up_layer{L}.scale` (separate)
- **Fix**: combined into single file per layer: `[M*K int8][M*4 scales]` → `ffn_up_layer{L}_prt.int8`

### Fix 2: 7B shape not recognized in loader
- **Bug**: CLI only recognized 3B (M=11008, K=2048) and 0.5B (M=4864, K=896) — not 7B (M=18944, K=3584)
- **Fix**: added 7B size recognition in `tools/cli/cli.cpp`

## Verdict

**PASS_7B_INT8_SINGLE_PROMPT** ✅

The corrected 7B INT8 sidecars:
1. Load successfully (28/28) through the current llama-cli PRT path
2. Generate clean output matching native model
3. Use correct shape metadata (hidden=3584, ffn=18944)
4. Force-native layers 11 and 15 are honored

## Recommended Next

- Phase 14L: 7B INT8 multi-prompt validation (8 prompts, quality check)
- Do NOT proceed to full speed benchmarking

## Safety

- **Models/sidecars staged?** NO — only reports committed
- **Secrets detected?** NO
- **Tags touched?** NO
- **CLI fix**: `tools/cli/cli.cpp` only — added 7B recognition, compiled in-place
- **Build**: `b8799-13ff1d6c7` — CLI rebuild only, no new commit (minimal patch)