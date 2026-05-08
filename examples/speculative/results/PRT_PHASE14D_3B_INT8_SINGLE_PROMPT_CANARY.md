# PRT Phase 14D — 3B INT8 Single-Prompt Runtime Canary

## Verdict: PASS_3B_INT8_SINGLE_PROMPT

## Context

- Phase 14C-VERIFY: PASS_SHAPE_ORIENTATION_VERIFIED at commit b03b1ca73
- Phase 14C: PASS_05B_INT8_QUALITY_AND_TIMING at commit 104863026
- 0.5B confirmed: hidden=896, ffn=4864, INT8 ~86.7 t/s, 8/8 exact matches
- 3B expected: hidden=2048, ffn=11008

## 3B Model Metadata

| Property | Value |
|----------|-------|
| Model | Qwen2.5-3B-Instruct-Q4_K_M.gguf |
| n_layers | 36 |
| hidden (embedding) | 2048 |
| ffn (intermediate) | 11008 |
| Float32 sidecar bytes/layer | 90,177,536 (2048×11008×4) |
| INT8 sidecar bytes/layer | 22,588,416 (11008×2048 + 11008×4) |
| Compression ratio | **3.99×** |

## Single-Prompt Results

**Native 3B:**
```
> The capital of France is
The capital of France is Paris.
[ Prompt: 77.4 t/s | Generation: 20.9 t/s ]
```
✅ Clean output, correct answer

**Float32 PRT 3B:**
```
> The capital of France is
The capital of France is Paris.
[ Prompt: 10.4 t/s | Generation: 8.5 t/s ]
```
✅ Clean output, correct answer — slower due to PRT overhead with mmap

**INT8 PRT 3B:**
```
> The capital of France is
The capital of France is Paris.
[ Prompt: 75.8 t/s | Generation: 21.1 t/s ]
[PRT_SHAPE] n_layer=36 M=11008 N=2048
[PRT_SHAPE_DETAIL] n_layer=36 hidden=2048 ffn=11008 sidecar_rows=11008 sidecar_cols=2048 runtime_M=11008 runtime_N=2048 format=int8
[PRT] Loaded 36/36 sidecars from /tmp/prt_sidecars_3b_int8/
```
✅ Clean output, correct answer, **nearly matches native speed**

## Shape Orientation Check

| Field | Value | Status |
|-------|-------|--------|
| n_layer | 36 | ✅ |
| hidden | 2048 | ✅ matches model |
| ffn | 11008 | ✅ matches model |
| sidecar_rows (M) | 11008 (ffn) | ✅ consistent with 0.5B convention |
| sidecar_cols (N) | 2048 (hidden) | ✅ consistent with 0.5B convention |
| format | int8 | ✅ |
| Sidecars loaded | 36/36 | ✅ |
| Fallback layers | 11, 15 force-native | ✅ |

## Timing Comparison

| Mode | Generation t/s | vs Native |
|------|---------------|-----------|
| Native 3B | 20.9 | 1.00× (baseline) |
| Float32 PRT 3B | 8.5 | 0.41× |
| **INT8 PRT 3B** | **21.1** | **1.01×** |

INT8 PRT runs at **101% of native speed** on 3B — nearly identical to native!

## Allowed Claims

✅ 3B INT8 single-prompt canary passed — correct output and near-native speed.
✅ INT8 PRT on 3B produces correct output ("Paris") matching native.
✅ INT8 PRT 3B ran at 21.1 t/s vs native 20.9 t/s (~1.01× native).
✅ 36/36 sidecars loaded correctly.
✅ Shape orientation consistent with 0.5B results.

## Forbidden Claims

- ❌ No full 8-prompt quality suite for 3B
- ❌ No production readiness
- ❌ No universal speedup across all models

## Recommended Next

Phase 14E: Full 8-prompt quality validation for 3B INT8 with timing suite.

## Safety

- No models/sidecars/binaries staged ✅
- No secrets detected ✅
- Phase 13 tags untouched ✅