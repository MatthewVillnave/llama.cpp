# PRT Phase 21L: 0.5B INT8 PRT-v2 Baseline Checkpoint

## Status: PASS_05B_INT8_PRT_V2_BASELINE_CHECKPOINT

## A. Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## B. Previous HEAD
`7dde4414c` (Phase 21K — 4-prompt INT8 canary)

## C. New HEAD
`7dde4414c` (no code changes — checkpoint only)

## 1. Executive Summary

PRT-v2 layer0 INT8 sidecar path is now a **trusted 0.5B baseline**. It uses corrected [K,M] row-major INT8 decode (`int8[k*M+j]*scale[j]`), works with file-based prompts (`-f`), and is a correctness/plumbing baseline — not a speed baseline.

This checkpoint freezes the working state before INT6 runtime development begins.

## 2. Frozen Technical Baseline

| Parameter | Value |
|-----------|-------|
| **Branch** | `experimental/prt-phase19a-alt-sidecar-backed` |
| **HEAD** | `7dde4414ca40ab92a1475dfd8a1e0f02f52c8433` |
| **Model** | Qwen2.5-0.5B-Instruct-Q4_K_M.gguf |
| **Selected layer** | layer 0 only |
| **PRT-v2 op** | GGML_OP_PRT_FFN_UP |
| **Sidecar source** | `/tmp/prt_phase21h_u_int8_from_f32/ffn_up_layer0_prt.int8` |
| **INT8 storage layout** | [K,M] row-major |
| **Decode formula** | `W[k,j] = int8[k*M + j] * scale[j]` |
| **Prompt method** | File-based prompts only (`-f`) |
| **Offline cosine vs f32** | 0.99996084 |
| **Offline MAE vs f32** | 0.000128 |
| **Kernel acc (INT8)** | -0.222431 |
| **Kernel acc (f32 baseline)** | -0.213825 |
| **Acc diff** | ~4.0% |
| **output_abs_sum** | 3.849652 (consistent across prompts) |

### Route/Op/Kernel Evidence
- `[PRT_V2_ROUTE] IL=0 route=ggml_op reason=selected_layer` ✅
- `[PRT_V2_SIDECAR] layer=0 source=int8_sidecar path=.../ffn_up_layer0_prt.int8 K=896 M=4864` ✅
- `[PRT_V2_DECODE] decoded_to=f32 W_shape=[896,4864] layout=K_M_row_major formula=int8[k*M+j]*scale[j]` ✅
- `[PRT_V2_OP] inserted=true op=GGML_OP_PRT_FFN_UP layer=0` ✅
- `[PRT_V2_KERNEL_ENTER] K=896 M=4864 N=2 x_ne=[896,2] w_ne=[896,4864] dst_ne=[4864,2]` ✅
- `[PRT_V2_KERNEL_EXIT] done=1 output_abs_sum_first4=3.849652` ✅
- IL=1..23: `[PRT_V2_ROUTE] route=native` ✅

### Phase History (key turns)
| Phase | Verdict | Key finding |
|-------|---------|-------------|
| 21H-S | PASS_F32_INT8_PROVENANCE_LAYOUT | f32 is [K,M], INT8 is [K,M] row-major |
| 21H-T | PASS_INT8_DECODE_FIX | Decode formula: `int8[k*M+j]*scale[j]` |
| 21H-U | PASS_REGEN_INT8_FROM_F32 | Offline cosine: 0.99996084 |
| 21H-V | PASS_REGEN_INT6_OFFLINE_PARITY | INT6 cosine: 0.99936563 |
| 21J | PASS_INT8_SHORT_STABLE | File prompts stable; inline unstable |
| 21K | PASS_05B_INT8_FILE_PROMPT_CANARY | 4/4 prompts exit 0, no SIGKILL |

## 3. Corrected Claims

### Phase 21G Semantic Interpretation — INVALIDATED
Phase 21G claimed semantic divergence between INT8 and f32 paths. This was caused by using the **wrong INT8 layout** (treating it as [M,K] when it is actually [K,M]).

**Root cause**: The Phase 21G INT8 sidecar was regenerated with the wrong decode assumption.

**Correction**: INT8 layout is `[K,M] row-major`, decode formula `int8[k*M+j]*scale[j]`.

### Inline Prompt Instability
Inline `-p` prompts with INT8 PRT-v2 are **unstable and trigger SIGKILL**. File-based prompts (`-f`) are the only validated stable method for PRT-v2 INT8 inference.

## 4. Allowed Claims

The following claims are supported by evidence:

- **0.5B layer0 INT8 sidecar-backed PRT-v2 path runs through GGML_OP_PRT_FFN_UP** ✅
- **Corrected regenerated INT8 sidecar has strong offline parity** with confirmed f32 reference (cosine 0.99996084) ✅
- **File-prompt canary passes 4/4 with no SIGKILL** ✅
- **Trusted 0.5B layer0 baseline for further PRT-v2 work** ✅
- **output_abs_sum stable at 3.849652** across all prompts ✅
- **Kernel evidence confirms decode formula** `int8[k*M+j]*scale[j]` ✅
- **This is a correctness/plumbing baseline, not a speed baseline** ✅

## 5. Forbidden Claims

Do **NOT** claim any of the following:

- ❌ Speedup over native path
- ❌ Production readiness
- ❌ 7B/14B support
- ❌ Multi-layer support (layer0 only validated)
- ❌ INT6 runtime support (not yet implemented)
- ❌ Inline prompt stability (unstable, separate issue)
- ❌ Exact/token match unless capture proves it (capture is TTY-limited)
- ❌ Phase 21G semantic divergence claims (invalidated)

## 6. Recommended Next: Phase 21M

**Implement regenerated INT6 runtime path** — layer0 only, file prompts only:

### INT6 Format Details
- **Storage layout**: [M,K] column-major (different from INT8's [K,M])
- **Decode formula**: `W[k,j] = q_flat[j*K + k] * scale[j]`
- **Offline cosine vs f32**: 0.99936563
- **Offline cosine vs INT8**: 0.99933678
- **Source**: `/tmp/prt_phase21h_v_int6_from_f32/ffn_up_layer0_prt.int6`

### Phase 21M Prerequisites
1. Parse PRT6 header (magic 0x36545250, M, K, packed_offset, scale_offset)
2. Unpack packed 6-bit values to int8: `q_flat[j*K + k]`
3. Decode with: `W[k,j] = q_flat[j*K + k] * scale[j]`
4. Verify offline parity before runtime test
5. Run file-prompt canary (4 prompts, n=32, temp=0)
6. Compare output_abs_sum to INT8 baseline (3.849652)

## 7. Safety Scan

| Item | Status |
|------|--------|
| Model files staged | ❌ No |
| Sidecar files staged | ❌ No (reference only) |
| Decoded weights staged | ❌ No |
| Binaries staged | ❌ No |
| Secrets detected | ❌ No |
| Huge logs staged | ❌ No |
| Tag created | TBD |

## 8. Checkpoint Metadata

- **Checkpoint file**: `examples/speculative/results/PRT_PHASE21L_05B_INT8_PRT_V2_BASELINE_CHECKPOINT.md`
- **JSON file**: `examples/speculative/results/phase21l_05b_int8_prt_v2_baseline_checkpoint.json`
- **Checkpoint verdict**: `PASS_05B_INT8_PRT_V2_BASELINE_CHECKPOINT`
- **Frozen HEAD**: `7dde4414ca40ab92a1475dfd8a1e0f02f52c8433`
- **Timestamp**: 2026-05-14T17:15:00-04:00
- **Models staged**: No
- **Secrets**: No
- **Existing tags altered**: No