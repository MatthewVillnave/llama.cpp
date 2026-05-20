# PRT Phase 22B: 0.5B AVX2 Runtime Canary — BLOCKED

## A. Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## B. Previous HEAD
`7f83ca5ba`

## C. New HEAD
`7f83ca5ba` (no change — blocked before code)

## D. AVX2 env flag
`PRT_V2_AVX2=1` (ops.cpp) vs `g_prt_kernel_mode=1` (runtime)

## E. scalar baseline output_abs_sum
N/A — kernel not reached in any test

## F. AVX2 output_abs_sum
N/A — kernel not reached in any test

## G. numeric delta
N/A

## H. scalar output
N/A

## I. AVX2 output
N/A

## J. AVX2 activation evidence
BLOCKING FINDING: Two independent AVX2 implementations exist:

1. **ops.cpp:ggml_compute_forward_prt_ffn_up_avx2** (Phase 22A)
   - Defined at ops.cpp:10892
   - Dispatched from ops.cpp:10892 inside `ggml_compute_forward_prt_ffn_up`
   - Guard: `prt_avx2_mode == 1 && !scales`
   - Activated by: `PRT_V2_AVX2=1` env var
   - Called from: `ggml_compute_forward` dispatch in ggml-cpu.c
   - **PROBLEM**: GGML_OP_PRT_FFN_UP tensor never reaches `ggml_compute_forward` in real inference

2. **prt_graph_replace.h:build_prt_ffn_up** (Phase 11BB, ~line 300)
   - Has its own AVX2 implementation inline
   - Activated by: `g_prt_kernel_mode=1` (default ON, cannot be disabled)
   - Called from: PRT graph replacement path (src/llama-graph.cpp:1513)
   - **PROBLEM**: ops.cpp AVX2 is unreachable dead code in real inference

Evidence of build_prt_ffn_up AVX2 activation on 0.5B:
```
[PRT-BUILD] PRT_KERNEL=avx2 (kernel_mode=1)
[PRT-11BB-AUTH] IL=0 PRT result ne=[896,34] name=prt_ffn_up.0
```

Evidence of ops.cpp path NOT reached (KERNEL_ENTER never logged):
```
$ PRT_V2_AVX2=1 ./build/bin/llama-cli --prt-mode 5600 ... --prt-log-level debug
[PRT-BUILD] PRT_KERNEL=avx2 (kernel_mode=1)
[PRT-11BB-AUTH] IL=0 PRT result ne=[896,34] name=prt_ffn_up.0
(no KERNEL_ENTER log)
```

## K. timing comparison
N/A — kernel not reached

## L. n=8 result
N/A

## M. verdict
**BLOCKED_MACHINE_STATE** — Two-path architecture prevents synthetic test from validating real inference path

## N. recommended next
**Phase 22C**: Route ops.cpp AVX2 into the real PRT path

Two options:
1. **Replace**: Swap `build_prt_ffn_up` to call `ggml_compute_forward_prt_ffn_up_avx2` instead of inline AVX2
2. **Wired**: Make `ggml_compute_forward_prt_ffn_up` reachable from real inference by wiring GGML_OP_PRT_FFN_UP into llama-graph.cpp graph construction

Minimal fix: In `build_prt_ffn_up` (prt_graph_replace.h ~line 530), after userdata setup, call `ggml_compute_forward_prt_ffn_up` instead of inline AVX2 loop. This makes ops.cpp kernel reachable.

## O. models/sidecars/binaries staged?
NO — no staging needed (blocked before execution)

## P. secrets detected?
NO

## Q. existing tags touched?
NO

## R. system disk free
58G (75% used)

## S. scratch disk free
51G (57% used)

## Root Cause Analysis

| Component | Location | AVX2 reachable? |
|----------|----------|------------------|
| ops.cpp kernel | `ggml_compute_forward_prt_ffn_up_avx2` (ops.cpp:10892) | ❌ NO — never called in real inference |
| ops.cpp dispatch | `ggml_compute_forward` (ggml-cpu.c:2041) | ❌ NO — GGML_OP_PRT_FFN_UP never dispatched |
| inline AVX2 | `build_prt_ffn_up` (prt_graph_replace.h:300) | ✅ YES — used by real inference |
| runtime config | `g_prt_kernel_mode=1` (llama-graph.cpp:155) | ✅ YES — default ON |

Phase 22A validated the **ops.cpp AVX2 kernel design** (synthetic correctness) but never connected it to the **real inference path**.

## Files touched
None (BLOCKED before any code changes)

## Synthetic test evidence
```
K=896 M=4864 N=1: scalar=4.4ms avx2=0.5ms speedup=8.4x err=3.66e-04 PASS
K=3584 M=18944 N=1: scalar=299.2ms avx2=23.4ms speedup=12.9x err=1.10e-03 MARGINAL
```
(Synthetic test passes — kernel is correct, just disconnected)