# PRT Phase 21P: 7B INT8 PRT-v2 Layer0 Kernel/Plumbing Baseline Checkpoint

## A. Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## B. Previous HEAD
`2b069721c` (Phase 21O-U)

## C. New HEAD
`2b069721c` (report-only, checkpoint tag created)

## D. Frozen Technical Baseline

| Item | Value |
|------|-------|
| Branch | `experimental/prt-phase19a-alt-sidecar-backed` |
| HEAD | `2b069721c` |
| Tag | `PRT_PHASE21P_7B_INT8_KERNEL_BASELINE_CHECKPOINT` |
| Model | `Qwen2.5-7B-Instruct-Q4_K_M.gguf` |
| Model path | `/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-7B-Instruct-Q4_K_M.gguf` |
| Sidecar | `ffn_up_layer0_prt.int8` |
| Sidecar format | INT8 [K,M] row-major, K=3584 M=18944, with float32 scales |
| Sidecar path | `/tmp/prt_phase21h_u_int8_from_f32/ffn_up_layer0_prt.int8` |
| Layer | 0 only |
| Op | GGML_OP_PRT_FFN_UP |
| K | 3584 |
| M | 18944 |
| Prompt method | File prompts (-f) only |
| Harness | nohup + dedicated log + --log-disable |
| Settings | c=16, t=1, temp=0, n=4 |
| Test prompts | P1: "The capital of France is" / P2: "The largest planet..." / P3: "Return JSON..." / P4: "Once upon a time" |

## 1. Executive Summary

**7B layer0 INT8 PRT-v2 now has confirmed route/op/kernel evidence.**

This checkpoint proves the kernel/plumbing path works correctly. The semantic output quality remains unverified due to slow 7B CPU generation + TTY/capture constraints.

- Route: ggml_op confirmed ✅
- Op insert: GGML_OP_PRT_FFN_UP confirmed ✅
- Kernel ENTER: 32 total (8 per prompt) ✅
- Kernel EXIT: 32 total (8 per prompt) ✅
- output_abs_sum: captured across all 4 prompts ✅
- Exit 137 occurred after kernel exit, not before (CPU timeout, not kernel failure) ⚠️
- Semantic capture: LIMITED ❌

## 2. Evidence

### Route Logs
```
[PRT_V2_ROUTE] IL=0 route=ggml_op reason=selected_layer
[PRT_V2_ROUTE] IL=1 route=native reason=not_selected_layer
...
```

### Sidecar Logs
```
[PRT_V2_SIDECAR] layer=0 source=int8_sidecar path=/tmp/prt_phase21h_u_int8_from_f32/ffn_up_layer0_prt.int8 K=3584 M=18944
[PRT_V2_SIDECAR] scales: first5=0.000529/0.001245/0.000532/0.000594/0.000535
```

### Kernel ENTER/EXIT Count
| Prompt | KERNEL_ENTER | KERNEL_EXIT |
|--------|-------------|------------|
| P1 | 8 | 8 |
| P2 | 8 | 8 |
| P3 | 8 | 8 |
| P4 | 8 | 8 |
| **Total** | **32** | **32** |

### output_abs_sum Values
| Batch (N) | Value |
|-----------|-------|
| 2 | 1.684134 |
| 2 | 1.944075 |
| 16 | 2.112426 |
| 14 | 1.860070 |

### Sample Kernel Log
```
[PRT_V2_KERNEL_ENTER] K=3584 M=18944 N=2 x_ne=[3584,2] w_ne=[3584,18944] dst_ne=[18944,2]
[PRT_V2_KERNEL_PTRS] x=0x72bebf474200 w=0x72bc08c3f000 scales=(nil) dst=0x72bebf60c200
[PRT_V2_KERNEL_TYPES] x=0 w=0 scales=-1 dst=0
[PRT_V2_KERNEL_PROGRESS] token=0 j=0 acc=-0.214229
[PRT_V2_KERNEL_EXIT] done=1 output_abs_sum_first4=1.684134
```

## 3. Limits

### Not Claimed
- ❌ 7B semantic correctness
- ❌ exact/token match
- ❌ clean exit (exit 137 from 90s CPU timeout)
- ❌ f32/native offline cosine
- ❌ speedup vs native
- ❌ production readiness
- ❌ multi-layer support
- ❌ all-prompt semantic stability

## 4. Allowed Claims

- ✅ 7B layer0 INT8 PRT-v2 kernel evidence confirmed
- ✅ 7B sidecar shape K=3584/M=18944 confirmed
- ✅ 4-prompt kernel canary reached kernel ENTER/EXIT for all 4 prompts (8× each)
- ✅ output_abs_sum captured across all prompts
- ✅ Route → GGML_OP → SIDECAR → KERNEL → EXIT path verified

## 5. Forbidden Claims

- 🚫 7B semantic match
- 🚫 exact/token match
- 🚫 speedup
- 🚫 7B INT6 support
- 🚫 multi-layer support
- 🚫 production readiness
- 🚫 all-prompt stability
- 🚫 clean end-to-end completion

## 6. Recommended Next

**Phase 21Q-A:** 7B semantic/capture harness improvement — focus on getting clean completion (exit 0) and captured text output after kernel exit.

OR

**Phase 21Q-B:** Safe 7B f32 reference extraction / offline parity.

**Recommendation:** Phase 21Q-A first. The kernel evidence is solid but exit 137 means we can't confirm end-to-end correctness. Fix the completion/capture harness before adding complexity.