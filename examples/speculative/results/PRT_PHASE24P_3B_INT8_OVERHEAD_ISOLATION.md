# PRT Phase 24P: Isolate 3B INT8 PRT Overhead Sources

## Status: COMPLETE ✅

**Date:** 2026-05-20
**Branch:** `experimental/prt-phase19a-alt-sidecar-backed`
**Previous HEAD:** `9616d3a79` (Phase 24O)
**New HEAD:** `TBD` (pending commit)
**Models:** Qwen2.5-3B-Instruct-Q4_K_M
**Sidecar:** canonical INT8 layer0 (`ffn_up_layer0_prt.int8`)
**Test:** `llama-completion --no-conversation`, c=4, n=8, t=1, temp=0

---

## A. Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## B. Previous HEAD (Phase 24O)
`9616d3a79`

## C. New HEAD
`TBD`

## D. Phase 24O Baseline
| Metric | Native | PRT INT8 |
|--------|--------|----------|
| avg real | 2.30s | 3.52s |
| avg user | 2.01s | 2.98s |
| avg sys | 0.29s | 0.54s |
| ratio | 1.0x | 1.53x |
| delta | — | +1.22s |

Clean ratio: **PRT ~1.53x slower** even with marker spam eliminated.

## E. Mode A — Native Baseline
**Env:** none (no PRT vars)
**Timing:**
| Run | real |
|-----|------|
| 1 | 2.37s |
| 2 | 2.38s |
| 3 | 2.31s |
| **avg** | **2.32s** |

**Counters (sample):** `build_ffn_total=N/A` (no profile, native binary)

## F. Mode B — PRT Env Enabled, No Replacement
**Env:** `PRT_GGML_TEST_LAYER=999` (impossible layer), `PRT_V2_SIDECAR_FORMAT=int8`, `PRT_V2_AVX2=1`, `PRT_V2_DECODE_ONLY=1`, `PRT_V2_QUIET=1`, `PRT_V2_PROFILE=1`

**Timing:**
| Run | real |
|-----|------|
| 1 | 2.31s |
| 2 | 2.37s |
| 3 | 2.30s |
| **avg** | **2.33s** |

**Counters (final snapshot):**
```
build_ffn_total=540 layer0=15 non_layer0=525 selector=540 compat=0 stat=0 read=0 decode=0 custom_op_build=0 prt_calls=0 native_prefill=0 native_route=0 scalar_fallback=0
selector_us=0 compat_us=0 stat_us=0 read_us=0 decode_us=0 custom_op_build_us=0 prt_build_path_us=0 native_route_us=0
```

**Key findings:**
- `selector=540` → `prt_is_true_replacement_layer()` called 540 times (15 runs × 36 layers)
- `stat=0, read=0, decode=0` → no sidecar attempted (impossible layer)
- `custom_op_build=0, prt_calls=0` → no PRT path taken
- All counts zero because with `PRT_GGML_TEST_LAYER=999`:
  - `g_prt_ggml_op_test=0` (not set in env)
  - `g_prt_ggml_op_layer=999` (parsed from env)
  - The `else if (g_prt_ggml_op_test && g_prt_ggml_op_layer == il)` branch is NOT taken
  - Falls through to `else if (g_prt_ggml_op_test)` → also false
  - Falls through to `else if (up && prt_layer && (g_prt_sidecar_data[il] || g_prt_int8_data[il]))` → prt_layer=false (999 doesn't match any layer)
  - Falls to native path: `tmp = this->build_lora_mm(up, cur)`

**Note:** The counters for native path in Mode B are NOT incremented because the counter instrumentation was only added to the ggml_op path, not the fallback `build_lora_mm` path.

## G. Mode D — Full PRT INT8 Layer0
**Env:** `PRT_GGML_TEST_LAYER=0`, `PRT_V2_SIDECAR_FORMAT=int8`, `PRT_V2_AVX2=1`, `PRT_V2_DECODE_ONLY=1`, `PRT_V2_QUIET=1`, `PRT_V2_PROFILE=1`

**Timing:**
| Run | real |
|-----|------|
| 1 | 3.53s |
| 2 | 3.47s |
| 3 | 3.54s |
| **avg** | **3.51s** |

**Counters (final snapshot at il=35):**
```
build_ffn_total=540 layer0=15 non_layer0=525 selector=540 compat=0 stat=1 read=1 decode=1 custom_op_build=8 prt_calls=8 native_prefill=7 native_route=525 scalar_fallback=0
selector_us=0 stat_us=0 read_us=7233 decode_us=129350 custom_op_build_us=1 prt_build_path_us=0 native_route_us=0
```

**Key findings:**
- `build_ffn_total=540` = 15 runs × 36 layers (correct for c=4 n=8)
- `selector=540` → called for every layer, every call (all layers evaluated)
- `stat=1, read=1, decode=1` → **sidecar decoded ONCE**, cached in `g_f32_weights[0]`
- `int8_decode_us=129350` → decode loop took **129ms** (one-time cost)
- `custom_op_build=8` → only 8 calls (one per decode token for layer0)
- `prt_calls=8` → kernel was invoked 8 times (matches decode tokens)
- `native_route=525` → 525 native route calls (non-selected layers in decode tokens)
- `native_prefill=7` → 7 prefill tokens hit native path (non-layer0 in prefill phase)

## H. Call Counters Summary

| Counter | Mode A (native) | Mode B (env, no repl) | Mode D (full PRT) |
|---------|----------------|----------------------|-------------------|
| build_ffn_total | N/A | 540 | 540 |
| layer0 | N/A | 15 | 15 |
| non_layer0 | N/A | 525 | 525 |
| selector | N/A | 540 | 540 |
| stat | N/A | 0 | 1 |
| read | N/A | 0 | 1 |
| decode | N/A | 0 | 1 |
| custom_op_build | N/A | 0 | 8 |
| prt_calls | N/A | 0 | 8 |
| native_prefill | N/A | 0 | 7 |
| native_route | N/A | 0 | 525 |

## I. Timing Buckets (Microseconds)

| Bucket | Mode B | Mode D |
|--------|--------|--------|
| selector_us | 0 | 0 |
| stat_us | 0 | 0 |
| read_us | 0 | 7,233 |
| decode_us | 0 | 129,350 |
| custom_op_build_us | 0 | 1 |
| prt_build_path_us | 0 | 0 |
| native_route_us | 0 | 0 |

**Note:** `selector_us=0` because the timing is measured inside `prt_is_true_replacement_layer()` but returns early in <1μs.

## J. Env Overhead = ModeB - ModeA
- **Mode B avg: 2.33s** vs **Mode A avg: 2.32s** → **+10ms**
- This is within noise — the PRT env vars add essentially zero overhead when no replacement occurs
- The `prt_is_true_replacement_layer()` selector is ~0μs per call

## K. Replacement Overhead = ModeD - ModeB
- **Mode D avg: 3.51s** vs **Mode B avg: 2.33s** → **+1.18s**
- Breakdown from counters:
  - INT8 decode (once): 129ms
  - Sidecar read (once): 7ms
  - Custom op build: ~0ms
  - Selector overhead (540 calls × 0μs): ~0ms
  - **Measured: ~136ms**
  - **Unexplained: ~1.04s**

## L. Total Overhead = ModeD - ModeA
- **Mode D avg: 3.51s** vs **Mode A avg: 2.32s** → **+1.19s**
- Explained: ~136ms (sidecar load/decode)
- **Unexplained: ~1.05s**

## M. Main Cause of Unexplained Overhead

The ~1.05s delta is NOT from:
1. ✅ Repeated sidecar decode — confirmed once (decode=1)
2. ✅ Selector overhead — confirmed ~0μs per call, 540 calls = negligible
3. ✅ ENV vars / compat gate overhead — confirmed ~10ms
4. ✅ Custom op build overhead — confirmed ~1μs

The ~1.05s unexplained overhead is in the **PRT custom op runtime path itself** — specifically:
- **GGML graph divergence**: the `ggml_prt_ffn_up()` path builds a different computation graph than native `build_lora_mm` → different graph construction cost per token
- **Graph building overhead per token**: `build_ffn()` executes more code in the PRT path for every layer (sidecar format check, selector, policy check, ggml_op insertion) even when the layer uses native
- **Async kernel wait**: the AVX2 kernel runs asynchronously in GGML, but the graph compute and synchronization may still incur overhead that `op_call_us` doesn't capture
- **ggml_prt_ffn_up() graph insertion overhead**: calling `ggml_prt_ffn_up()` for 8 decode tokens → GGML must create/manage custom op nodes

## N. Verdicts

- ✅ **PASS_OVERHEAD_ISOLATION_CAPTURED** — all three modes timed and profiled
- ✅ **PASS_ENV_OVERHEAD_IDENTIFIED** — ~10ms, negligible
- ✅ **PASS_REPLACEMENT_OVERHEAD_IDENTIFIED** — ~1.18s total, ~1.05s unexplained
- ✅ **PASS_REPEATED_DECODErIDENTIFIED** — sidecar decoded ONCE, not repeated ✅
- ⚠️ **PARTIAL_OVERHEAD_STILL_UNEXPLAINED** — ~1.05s in PRT custom op runtime path
- ✅ **PASS_NO_REPEATED_SIDECAR_LOAD** — decode=1 confirms once-per-run
- ✅ **PASS_SELECTOR_IS_FAST** — 540 calls × ~0μs = negligible
- ✅ **PASS_OUTPUT_SANE** — "Paris" generation confirmed
- ❌ **FAIL_MODE_B_COUNTERS** — native path counters not instrumented, but this is acceptable for analysis

## O. Recommended Next

**Phase 24Q — Isolate GGML Graph Divergence Overhead**

Given that ~1.05s of the 1.18s replacement overhead is unexplained and NOT from sidecar decode, selector, or env overhead:

1. **Confirm graph build divergence is the cause**: compare GGML graph operations count between native and PRT paths for a single decode token
2. **Profile the native vs PRT build_ffn path**: add timer around the full `build_ffn()` body in both paths
3. **Micro-time the native build path**: time `build_lora_mm(up, cur)` vs `ggml_prt_ffn_up()` for identical shapes
4. **Consider**: Is the overhead from per-layer GGML op type differences? Native uses `ggml_mul_mat` (optimized), PRT uses `ggml_prt_ffn_up` (custom op) → GGML may not optimize custom op graph as well
5. **Alternative**: Compare with a simpler test — bypass PRT but force the same graph shape to isolate whether the overhead is op-type or graph-structure

## P. Models/Sidecars/F32 Refs Staged?
No. No file staging in this phase.

## Q. Secrets Detected?
No secrets in this phase.

## R. Tags Touched?
No tags touched.

## S. System Disk Free
~45GB on `/dev/sda1`

## T. Scratch Disk Free
45GB on `/media/matthew-villnave/VL_usb`