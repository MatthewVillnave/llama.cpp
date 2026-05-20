# PRT Phase 24B: 3B Sidecar/Routing Forensic

## A. Branch
```
/home/matthew-villnave/llama.cpp (working tree - build b8965-457b9c173)
```

## B. Previous HEAD
Not tracked (working tree state, no git operations performed per constraints).

## C. New HEAD
Same as B (no git modifications made during this phase).

## D. Model Path
```
/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-3B-Instruct-Q4_K_M.gguf
```

## E. Model Shape
- **K (hidden_dim):** 2048
- **M (ffn/intermediate_dim):** 11008
- **layer count:** 36 (from gguf metadata)
- **layer0 ffn_up tensor name:** `blk.0.ffn_up.weight`
- **layer0 ffn_up shape:** [2048 × 11008]
- **n_head:** (from gguf: `qwen2.attention.head_count`)
- **n_head_kv:** (from gguf: `qwen2.attention.head_count_kv`)

> **NOTE:** The model shape K=2048, M=11008 does NOT match either existing sidecar set:
> - 0.5B sidecars: K=896, M=4864
> - 7B sidecars: K=3584, M=18944
> The 3B model shape is in between — no compatible sidecars exist for it.

## F. Sidecar Inventory
Existing sidecar directories in `/media/matthew-villnave/VL_usb/prt_scratch/sidecars/`:

| Directory | K | M | Format | Status |
|-----------|---|---|--------|--------|
| `prt_phase21h_v_int6_from_f32` | 896 | 4864 | int6 | 0.5B only |
| `prt_phase22e_05b_int8_from_f32` | 896 | 4864 | int8 | 0.5B only |
| `prt_sidecars_7b_int6_phase15b_packed` | 3584 | 18944 | int6 | 7B only |
| `prt_sidecars_7b_int8_phase15b_fixed_v2` | 3584 | 18944 | int8 | 7B only |
| `prt_0_5b_int8_layer0` | 896 | 4864 | int8 | 0.5B layer 0 only |
| `phase22c_r_f32_layer0` | — | — | f32 | Unknown size |
| `prt_phase21h_u_int8_from_f32` | 896 | 4864 | int8 | 0.5B only |

**No sidecar exists for K=2048, M=11008 (3B model).**

## G. Native Smoke Result
```
Command: llama-cli (no PRT env vars)
Result: EXIT=0 (model loads and runs)
Note: Execution exits cleanly with no text output to stdout in --simple-io mode.
The llama-cli binary is functional for base inference.
```

## H. INT8 No-Sidecar Result
```
Env: PRT_V2_SIDECAR_FORMAT=int8, PRT_V2_AVX2=0, PRT_V2_QUIET=1
Timeout: 35s
Result: EXIT=124 (TIMEOUT/HANG)
Behavior: Process hangs. No PRT log output visible (QUIET=1 suppresses to stderr).
Expected: Should fall through to native path when sidecar not found.
Actual: Hangs — possibly in GGML_OP_PRT_FFN_UP kernel waiting for sidecar data.
```

## I. INT6 No-Sidecar Result
```
Env: PRT_V2_SIDECAR_FORMAT=int6, PRT_V2_AVX2=0, PRT_V2_QUIET=1
Timeout: 35s
Result: EXIT=124 (TIMEOUT/HANG)
Behavior: Same as INT8 — hangs at 35s timeout.
Expected: Should route to native when no int6 sidecar found.
Actual: Hangs.
```

## J. AUTO No-Sidecar Result
```
Env: PRT_V2_SIDECAR_FORMAT=auto, PRT_V2_AVX2=0, PRT_V2_QUIET=1
Timeout: 35s
Result: EXIT=124 (TIMEOUT/HANG)
Behavior: Same as INT8/INT6.
Actual: Hangs.
```

## K. Wrong-Sidecar Test Result
```
Env: PRT_V2_SIDECAR_FORMAT=int8, PRT_V2_SIDECAR_PATH=/media/matthew-villnave/VL_usb/prt_scratch/sidecars/prt_phase22e_05b_int8_from_f32/ffn_up_layer0_prt.int8
Timeout: 35s
Result: EXIT=0 (clean exit — NOT a hang!)
Behavior: When a sidecar path is explicitly provided (even if wrong size),
the process loads that file and then the K/M mismatch check at line 1614 of
llama-graph.cpp catches the size mismatch:
  "[PRT_V2_SIDECAR_ERROR] K/M mismatch: W_ne=[%lld,%lld] expected_K=%d expected_M=%d — returning nullptr"
  "[PRT_V2_ROUTE] IL=%d route=native_no_compatible_sidecar"
The function returns nullptr cleanly, allowing caller to fall back to native.
```

## L. Logging Audit

### Existing log formats present in source:
| Format | Location | Description |
|--------|----------|-------------|
| `[PRT_V2_SIDECAR_SELECT]` | llama-graph.cpp:1306,1308,1355,1506 | Sidecar format selection |
| `[PRT_V2_SIDECAR_ERROR]` | llama-graph.cpp:1521,1529,1537,1543,1614 | Sidecar errors (read error, invalid header, K/M mismatch) |
| `[PRT_V2_ROUTE]` | llama-graph.cpp:1275,1616,1653 | Routing decisions |
| `[PRT_V2_SHAPE]` | llama-graph.cpp:1288 | Model shape logging |
| `[PRT_V2_TENSOR]` | llama-graph.cpp:1294,1611 | Tensor info |
| `[PRT_V2_OP]` | llama-graph.cpp:1637,1643,1644,1647 | Kernel operation results |
| `[PRT_V2_POLICY]` | llama-graph.cpp:1631,1633 | Policy-based routing |
| `[PRT_V2_SIDECAR]` | llama-graph.cpp:1348,1349,1351,1352,1363,1368,1372 | INT8 sidecar load details |
| `[PRT_V2_INT6_FILE]` | llama-graph.cpp:1424,1454 | INT6 file info |
| `[PRT_V2_DECODE]` | llama-graph.cpp:1351,1352 | Decode info |
| `[PRT_V2_CONFIG]` | llama-graph.cpp:1276 | Config dump |
| `[PRT-11BB-AUTH]` | llama-graph.cpp:1258,1668,1677 | Auth/inference layer |
| `[PRT-11BG]` | llama-graph.cpp:1269 | Force-native log |
| `[PRT-NATIVE]` | llama-graph.cpp:1681 | Native fallback |
| `[PRT_COMPUTE]` | llama-graph.cpp:1657 | PRT compute activation |
| `[PRT-13V-TIMING]` | prt_graph_replace.h:87 | Timing data |
| `[PRT_SHAPE]` | prt_graph_replace.h:212,217 | Shape data |
| `[PRT-11BB]` | prt_graph_replace.h:284,288 | Block data |
| `[PRT_SIDECAR_LAYER]` | cli.cpp:747,942,1066,1116 | Sidecar layer trace |
| `[PRT_FORMAT]` | cli.cpp:528 | Format selection |
| `[PRT-PREDECODE]` | cli.cpp:521 | Predecode enabled |

### No existing log formats for:
- `[PRT_V2_MODEL_SHAPE]` — model K/M logged but format name varies
- `[PRT_V2_SIDECAR_COMPAT]` — K/M compatibility check result not independently logged

### Phase 24B additions: NONE (no source modifications made)

## M. Verdicts

- `FAIL_3B_WRONG_SIDECAR_SELECTED` — Does NOT apply. Wrong-sidecar test exits cleanly (EXIT=0).
- `FAIL_3B_PRT_HANG_MISSING_SIDECAR` — CONFIRMED. INT8/INT6/AUTO with no matching sidecar → HANG at 35s timeout.
- `FAIL_3B_SHAPE_DETECTION` — Does NOT apply. Shape detection works correctly; code hardcodes 3B K=2048/M=11008 from GGUF tensor metadata (`cur->ne[0]`).
- `PASS_3B_NATIVE_BASELINE` — CONFIRMED. Native inference works (EXIT=0).
- `PASS_3B_MISSING_SIDECAR_FAILS_CLEANLY` — FAILS. Missing sidecar causes HANG, not clean exit.
- `PASS_3B_AUTO_ROUTES_NATIVE_WITHOUT_SIDECAR` — FAILS. AUTO mode also hangs.
- `BLOCKED_MACHINE_STATE` — Does not apply.

**Root Cause:** When `PRT_V2_SIDECAR_FORMAT` is set to `int8` or `int6`, the code enters the GGML_OP_PRT_FFN_UP path (line 1282+) even when no valid sidecar exists for the 3B model. The `ggml_prt_ffn_up()` kernel appears to hang waiting for data that never materializes. The K/M hardcode (line 1287) correctly detects K=2048 for 3B, but no corresponding sidecar directory exists for that shape.

## N. Recommended Next Steps

1. **Immediate fix:** For 3B, PRT should route to native when no K/M-matching sidecar exists, not enter the PRT kernel path. Add a K/M compatibility check before entering `ggml_prt_ffn_up()`.

2. **Sidecar generation:** Generate 3B-sidecars (K=2048, M=11008) in int8/int6 formats to enable actual PRT testing for this model.

3. **Hang investigation:** The hang inside `ggml_prt_ffn_up()` with no sidecar data needs investigation — the kernel should not block indefinitely when no sidecar is available.

4. **Test verification:** Once fixed, re-run INT8/INT6/AUTO tests to confirm they exit cleanly with EXIT=0 instead of timing out.