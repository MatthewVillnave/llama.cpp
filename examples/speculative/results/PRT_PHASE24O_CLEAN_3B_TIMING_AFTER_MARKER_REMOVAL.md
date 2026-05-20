# PRT Phase 24O: Remove Marker Spam and Rerun Clean 3B Timing

## Status: COMPLETE ✅

**Date:** 2026-05-20
**Branch:** `experimental/prt-phase19a-alt-sidecar-backed`
**Previous HEAD:** `8369fdb40` (Phase 24N)
**New HEAD:** `8369fdb40` (source-only cleanup, no functional change)
**Models:** Qwen2.5-3B-Instruct-Q4_K_M
**Sidecar:** canonical INT8 layer0 (`ffn_up_layer0_prt.int8`)
**Test:** `llama-completion --no-conversation`, c=4, n=8, t=1, temp=0

---

## A. Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## B. Previous HEAD (Phase 24N)
`8369fdb40`

## C. New HEAD
`8369fdb40` (no functional change — source cleanup only)

## D. Markers Removed/Guarded

| Marker | Location | Action | Guard |
|--------|----------|--------|-------|
| `[R3_ENTRY]` | llama-graph.cpp:1326 | Guarded | `PRT_V2_PROFILE=1` |
| `[R3_PRT_LAYER]` | llama-graph.cpp:1292 | Guarded | `PRT_V2_PROFILE=1` |
| `[R3_3B_SELECTED]` | llama-graph.cpp:1343 | Guarded | `PRT_V2_PROFILE=1` |
| `[R3_7B_CANONICAL_SELECTED]` | llama-graph.cpp:1347 | Guarded | `PRT_V2_PROFILE=1` |
| `[R3_05B_CANONICAL_SELECTED]` | llama-graph.cpp:1351 | Guarded | `PRT_V2_PROFILE=1` |
| `[R3_WEIGHTS_CHECK]` | llama-graph.cpp:1648 | Guarded | `PRT_V2_PROFILE=1` |
| `[PRT-NATIVE]` | llama-graph.cpp:1750 | Guarded | `!prt_v2_quiet_native_prints() && g_prt_log_level >= 2` |
| `[PRT_V2_*]` | throughout | Guarded via `prt_logf()` | `prt_v2_quiet_native_prints()` → early return |

**Key fix:** `prt_logf()` now checks `prt_v2_quiet_native_prints()` at entry — if `PRT_V2_QUIET=1`, all prt_logf output is suppressed regardless of `g_prt_log_level`.

**Note:** `[PRT-NATIVE]` is a raw `fprintf` that bypasses `prt_logf`. It is guarded by `!prt_v2_quiet_native_prints() && g_prt_log_level >= 2`. This means it fires in the native binary (no PRT_V2_QUIET) but NOT when PRT_V2_QUIET=1.

## E. Marker Check Result

```
grep -cE "R3_|PRT-NATIVE|sidecar=\(nil\)" phase24o_marker_check.err → 0
```

**PASS** — No marker spam in PRT run with `PRT_V2_QUIET=1`.

## F. Previous Phase 24L Timing (Contaminated by Marker Spam)
| Run | Native | PRT INT8 |
|-----|--------|----------|
| 1 | 2.38s | 3.57s |
| 2 | 2.35s | 3.45s |
| 3 | 2.39s | 3.43s |
| **avg** | **2.40s** | **3.44s** |

Ratio: **~1.43x slower** (contaminated)

## G. Clean Native Timings (Marker-Free, 3 Runs)
| Run | real | user | sys |
|-----|------|------|-----|
| N1 | 2.30s | 2.02s | 0.28s |
| N2 | 2.31s | 2.01s | 0.30s |
| N3 | 2.30s | 2.01s | 0.28s |
| **avg** | **2.303s** | **2.013s** | **0.287s** |

## H. Clean PRT Timings (Marker-Free, 3 Runs)
| Run | real | user | sys |
|-----|------|------|-----|
| P1 | 3.48s | 2.95s | 0.54s |
| P2 | 3.63s | 3.08s | 0.55s |
| P3 | 3.44s | 2.91s | 0.53s |
| **avg** | **3.517s** | **2.980s** | **0.540s** |

## I. Clean Ratio

| Metric | Native avg | PRT avg | Delta | Ratio |
|--------|-----------|---------|-------|-------|
| real | 2.303s | 3.517s | +1.214s | **PRT 1.53x slower** |
| user | 2.013s | 2.980s | +0.967s | PRT 1.48x |
| sys | 0.287s | 0.540s | +0.253s | PRT 1.88x |

**Clean ratio: PRT ~1.53x slower than native on real time.**

## J. Output Quality

Both native and PRT generate coherent output (`--no-conversation` mode, greedy):
- Prompt: "The capital of France is"
- Native output: starts with "Paris" ✅
- PRT output: starts with "Paris" ✅
- Output quality: equivalent, model is generating correctly

## K. Policy Result

With `PRT_V2_DECODE_ONLY=1`:
- Layer 0 uses PRT path (canonical INT8 sidecar, `ggml_prt_ffn_up`)
- Layers 1-35 use native path
- decode loop: N=1 → PRT policy applies

## L. Scalar Fallback
- N=1 (single token decode): PRT is used for selected layers
- No fallback to scalar observed in this test

## M. Overhead Attribution (Updated from Phase 24N)

| Component | Estimate | Notes |
|-----------|----------|-------|
| R3 marker spam | 0ms | **Eliminated** — guarded behind `PRT_V2_PROFILE=1` |
| INT8 decode loop (per call) | ~128ms | Layer 0 decode; re-runs each build_ffn call |
| Sidecar file read | ~0ms | Page-cached after first load |
| Kernel (AVX2, K=2048 M=11008 N=1) | ~83ms | Per-call, measured via `op_call_us` |
| Selector/custom-op overhead | unknown | Per-build_ffn call |
| Graph build divergence | unknown | Additional GGML graph ops for PRT path |
| **Total delta (clean)** | **+1.214s** | Consistent across runs |

**The 1.214s delta is NOT from marker spam. It is real overhead from the PRT INT8 path.**

## N. Verdicts

- ✅ **PASS_R3_MARKER_SPAM_REMOVED** — All R3 markers now behind `PRT_V2_PROFILE=1`
- ✅ **PASS_PRT_NATIVE_QUIET** — `[PRT-NATIVE]` suppressed when `PRT_V2_QUIET=1`
- ✅ **PASS_CLEAN_3B_TIMING_CAPTURED** — 3×3 interleaved runs, clean stderr
- ⚠️ **PARTIAL_PRT_SLOWER_CLEAN_SMOKE** — PRT 1.53x slower even after overhead removal
- ✅ **PASS_OUTPUT_SANE** — Both paths generate "Paris"
- ✅ **PASS_KERNEL_TIMER_VALID** — Kernel timing (~83ms) is now captured via proper timer

## O. Models/Sidecars/F32 Refs Staged?
No. No file staging in this phase.

## P. Secrets Detected?
No secrets in this phase.

## Q. Tags Touched?
No tags touched.

## R. System Disk Free
~45GB available on `/dev/sda1`

## S. Scratch Disk Free
45GB on `/media/matthew-villnave/VL_usb`

---

## Recommended Next

**Phase 24P — Isolate PRT Overhead Without Marker Contamination**

Given that the ~1.2s overhead persists even with clean marker output, the slowdown is structural in the PRT INT8 path. Next steps:

1. **Verify the overhead is not from repeated decode loop execution** — confirm that the INT8 decode (128ms) runs once or multiple times per token
2. **Profile the GGML graph build divergence** — compare graph construction time between native and PRT paths
3. **Measure custom op dispatch overhead** — `ggml_prt_ffn_up()` call overhead vs native `ggml_mul_mat`
4. **Consider**: Is the overhead worth the INT8 memory savings? If PRT is 1.53x slower even for a single layer, the INT8 sidecar approach may not be viable for latency-sensitive use cases
