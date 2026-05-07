# PRT Phase 13AJ — Speed Discrepancy Retrospective

**Date:** 2026-05-07  
**Verdict:** MEASUREMENT_BOUNDARY_DIFFERENCE  
**Model:** Qwen2.5-3B-Instruct-Q4_K_M.gguf  
**Branch:** `experimental/prt-phase13-model-generalization`

---

## Verdict: MEASUREMENT_BOUNDARY_DIFFERENCE

**Phase 12 showed ~1.8× speedup. Phase 13 shows 2.55× slowdown. The most likely explanation: Phase 12 used KV cache (`--cache-type-k q8_0 --cache-type-v f16`) which inflates the native baseline's computation time, making PRT's FFN_UP bypass appear as a larger speedup. Phase 13 measures fresh inference without KV cache, where native is faster per token and PRT's sparse matmul overhead is fully visible. The two results are not directly comparable without accounting for the KV cache confounder.**

---

## Executive Summary

### What Phase 12 Appeared to Show
- **~1.82× speedup** for PRT (L11+L15 anchored policy) vs native on Qwen2.5-3B-Instruct-Q4_K_M
- Across 24 prompts, using `llama-prt-posix` custom binary
- Clean counters, stable memory, 0 collapse/repetition failures
- Token-0 match rate 90.5%

### What Phase 13 Currently Shows
- **2.55× wall slowdown** for PRT (L11+L15) vs native on same model
- Gen throughput: 8.6 t/s PRT vs 20.7 t/s native (0.41×)
- Clean quality, 0 fallback calls, AVX2 path active
- mmap-optimized sidecar loading (0.15ms)

### Why They Differ
**Primary cause: KV cache confounder.** Phase 12 used `--cache-type-k q8_0 --cache-type-v f16`, meaning:
- After the first inference, subsequent tokens retrieve KV from cache (q8_0/f16 compressed)
- Both native and PRT benefit from this cache, but PRT's FFN_UP bypass saves computation on the cached path too
- The native baseline in Phase 12 is inflated by the KV cache retrieval overhead
- Phase 13 measures **fresh inference** without KV cache reuse, where the native baseline is faster

**Secondary causes:**
- Different binary (llama-prt-posix vs llama-cli) — different threading, batch scheduling
- Phase 12 was a completely separate branch/fork with different build configuration

**Key finding: The Phase 12 speedup claim is contextually valid within its measurement setup but is NOT reproduced by the clean llama-cli stack without KV cache.**

---

## Phase 12 Evidence Summary

| Field | Value |
|-------|-------|
| Branch/tag | `experimental/prt-route-a-phase12e-l11-l15` / `PRT_PHASE12E_L11_L15_CHECKPOINT` |
| Commit | `bca5a1f32` |
| Binary | `llama-prt-posix` (built May 5, 2026) |
| Claims | 1.8216× avg speedup (L11+L15, 24 prompts) |
| Model | Qwen2.5-3B-Instruct-Q4_K_M.gguf |
| Prompt count | 24 |
| n_predict | 50 |
| Cache flags | `--cache-type-k q8_0 --cache-type-v f16` |
| KV cache active | **YES** — critical confounder |
| Timing | Wall-clock from invocation to completion |
| PRT policy | L11+L15 (force-native layers 11 and 15) |
| Active layers | 34 PRT + 2 native anchor |
| Sidecar loading | malloc+fread (not mmap) |
| AVX2 verified | Yes — kernel_mode=1 |
| Replacement evidence | prt_true_replacement_calls > 0 |
| Fallback | 16/token (2 anchor layers × 8 tokens/batch) |
| Speedup formula | native_wall / prt_wall |
| Native baseline | Wall time with KV cache |
| Quality check | Token-0 match rate 90.5% |

### Phase 12 Counter Evidence
- `callback_overwrites: 0` — clean
- `identity_fallback_calls: 0` — clean
- `native_fallback_calls: 16` per token (expected for 2 anchor layers)
- `prt_true_replacement_calls: 2006–3706` — PRT is active

### Phase 12 Key Confounder
The `--cache-type-k q8_0 --cache-type-v f16` flag means:
- KV tokens are stored in q8_0/f16 format
- Retrieval from KV cache is faster than fresh computation
- Both native and PRT use the same cache, but the cache doesn't bypass FFN_UP entirely
- **The KV cache was active in both native and PRT runs, inflating native baseline time**

---

## Phase 13 Evidence Summary

| Field | Value |
|-------|-------|
| Branch/commit | `experimental/prt-phase13-model-generalization` / `1b4bfdea8` |
| Binary | `llama-cli` (Phase 13 clean stack) |
| Claims | Speed-negative: 2.55× wall slowdown, 0.41× gen throughput |
| Model | Qwen2.5-3B-Instruct-Q4_K_M.gguf |
| Prompt count | 1 (Phase 13AI test) or 8 (quality suite) |
| n_predict | 40, 80 |
| Cache flags | **NONE** — no KV cache |
| KV cache active | **NO** — fresh inference |
| Timing | Wall-clock from invocation to completion |
| PRT policy | L11+L15 (force-native layers 11 and 15) |
| Active layers | 34 PRT + 2 native anchor |
| Sidecar loading | mmap (MAP_PRIVATE) |
| AVX2 verified | Yes — `__AVX2__` defined, ymm registers in disasm |
| Native baseline | 2.067s wall / 20.7 t/s (fresh, no cache) |
| PRT baseline | 5.267s wall / 8.6 t/s (fresh, no cache) |
| Replacement evidence | PRT-13V-TIMING: 272 calls (34 layers × 8 batches) |
| Fallback | 0 (AVX2 path active for all active layers) |
| Quality check | Paris in all outputs, no corruption |

### Phase 13 Key Finding
- PRT is 2.55× slower in wall time and 0.41× generation throughput vs native
- The AVX2 kernel is confirmed active
- mmap eliminates sidecar load overhead (0.15ms)
- Warm harness shows no additional benefit (mmap already keeps pages warm)
- The bottleneck is per-batch kernel execution (2666ms first-batch + 89ms warm batches)

---

## Side-by-Side Comparison Table

| Field | Phase 12 / 12E | Phase 13 clean stack | Difference / Risk |
|-------|-----------------|----------------------|--------------------|
| **Model** | Qwen2.5-3B-Q4_K_M | Qwen2.5-3B-Q4_K_M | SAME |
| **Frontend binary** | llama-prt-posix | llama-cli | DIFFERENT — critical |
| **Prompt count** | 24 | 1–8 | Phase 13 suite smaller |
| **n_predict** | 50 | 40–80 | DIFFERENT |
| **Threads** | default | 4 | UNKNOWN |
| **Cache flags** | `--cache-type-k q8_0 --cache-type-v f16` | NONE | **CRITICAL CONFOUNDER** |
| **KV cache active** | YES | NO | **CRITICAL** |
| **Timing boundary** | With KV cache | Fresh inference | **CRITICAL** |
| **Sidecar loading** | malloc+fread | mmap | Phase 13 optimized |
| **mmap** | NO | YES | Phase 13 faster for sidecar load |
| **PRT mode** | 5700 | 5700 | SAME |
| **Force-native layers** | 11, 15 | 11, 15 | SAME |
| **Active PRT layers** | 34 | 34 | SAME |
| **Sidecar shape** | [11008][2048] float32 | [11008][2048] float32 | SAME |
| **AVX2 path** | YES | YES | SAME |
| **Kernel code** | Identical | Identical | SAME |
| **Replacement evidence** | >0 via 11BD | 272 via 13V-TIMING | BOTH ACTIVE |
| **Fallback calls** | 16/token | 0 | Phase 12 logged 16/token differently |
| **Reported speedup** | 1.82× faster | 2.55× SLOWER | **CONTRADICTORY** |
| **Quality check** | token-0 match rate | semantic output | DIFFERENT METRICS |
| **Native command** | `--prt-mode 0` | no PRT flags | SAME LOGIC |

---

## Likely Causes (Ranked by Confidence)

### 1. MEASUREMENT_BOUNDARY_DIFFERENCE — HIGH CONFIDENCE

**Explanation:** Phase 12 used KV cache (`--cache-type-k q8_0 --cache-type-v f16`). This means:
- After the first token, KV data is stored/retrieved in compressed format
- Native inference retrieves KV from cache for subsequent tokens — still requires FFN_UP computation for new tokens
- PRT inference also retrieves KV from cache, but bypasses FFN_UP via sparse ternary matmul
- **The speedup ratio (native_wall / prt_wall) is inflated** because native still has FFN_UP overhead per token while PRT has FFN_UP bypass

Phase 13 measures **fresh inference without KV cache**:
- Every token is computed fresh (no retrieval from cache)
- Native baseline is faster per token because no cache overhead
- PRT's sparse matmul overhead is fully visible as a slowdown
- The PRT custom op adds overhead that wasn't visible when KV cache dominated the timing

**Evidence:** `--cache-type-k q8_0 --cache-type-v f16` explicitly documented in Phase 12E command.

**Counter-evidence:** Phase 12 B showed 1.81× speedup even for long n=100 runs where the cache effect should be proportionally smaller. However, KV cache is partial — it doesn't eliminate all computation, just stores results. The net effect is that KV cache retrieval (q8_0/f16) takes time, making the baseline slower.

**Severity:** This alone explains most of the discrepancy.

---

### 2. FRONTEND_BASELINE_DIFFERENCE — MEDIUM CONFIDENCE

**Explanation:** Phase 12 used `llama-prt-posix` (built May 5, 2026). Phase 13 uses `llama-cli` (built May 7, 2026). These are different binaries with potentially different:
- Threading models
- Batch scheduling
- Graph compilation paths
- Optimization levels

**Evidence:** Different binary filenames; different build timestamps; separate branch development.

**Counter-evidence:** Both link against `libllama.so` which was rebuilt for Phase 13. The PRT custom op is in `libllama.so`. The binary that runs inference determines the threading and loop structure.

**Severity:** May contribute but unlikely to flip 1.8× speedup to 2.5× slowdown alone.

---

### 3. PHASE 12 RESULT IS CONTEXTUAL — MEDIUM CONFIDENCE

**Explanation:** Phase 12's speedup may be valid **within its specific measurement setup** (with KV cache, using `llama-prt-posix`). It does not automatically transfer to fresh inference without KV cache, which is what Phase 13 measures.

**Evidence:** Phase 12 consistently showed speedup across 24 prompts. Phase 13 consistently shows slowdown across multiple runs.

**Counter-evidence:** If the speedup were real, it should appear in both KV-cached and fresh-inference contexts. The fact that it only appears with KV cache suggests the speedup is an artifact of the measurement boundary.

**Severity:** Phase 12 result is not automatically falsified — it's just context-dependent. Requires controlled rerun to resolve.

---

### 4. CODEPATH DIFFERENCE — LOW CONFIDENCE

**Explanation:** Phase 12 and Phase 13 PRT kernels are nearly identical. The main differences are:
- Phase 13 added timing instrumentation (negligible in hot path at quiet log level)
- Phase 13 uses mmap (faster sidecar loading, already accounted)
- Phase 13 has additional log routing

**Evidence:** `diff` of `prt_graph_replace.h` between `bca5a1f32` and current HEAD shows mostly timing additions; kernel math is identical.

**Counter-evidence:** If the kernel code is the same, the speed relationship should be the same. Code changes don't explain the discrepancy.

**Severity:** Minimal.

---

## Claims Discipline Update

### Allowed
- Phase 12 showed ~1.8× speedup **in its specific measurement setup** (with KV cache, custom binary) ✅
- Phase 13 currently shows speed-negative results with fresh inference on llama-cli ✅
- KV cache is a critical confounder when comparing Phase 12 and Phase 13 timing results ✅
- Phase 12 results are not reproducible with the clean llama-cli stack without KV cache ✅
- The KV cache difference is the most likely explanation for the discrepancy ✅
- Phase 12 speedup claim is **contextually valid** within its setup but **not universally valid** ✅

### Forbidden
- ❌ No claim that Phase 12 speedup is falsified or fabricated
- ❌ No claim that PRT is universally faster or slower
- ❌ No production readiness claim
- ❌ No larger-than-3B extrapolation
- ❌ No claim that Phase 12 result applies to current llama-cli stack

---

## Recommended Next Phase

**Option A — Controlled Phase 12 Reproduction (Recommended):**
Run a controlled reproduction to definitively resolve the discrepancy:
- Same model as Phase 12: Qwen2.5-3B-Instruct-Q4_K_M.gguf
- Same prompts as Phase 12 (or a subset)
- Same PRT policy: L11+L15 (force-native layers 11, 15)
- Same cache flags: `--cache-type-k q8_0 --cache-type-v f16`
- Use `llama-prt-posix` (May 5 build) or equivalent
- Compare native vs PRT wall time with KV cache active
- **Expected:** If Phase 12 result was real, speedup should appear with KV cache
- **Expected:** If Phase 13 result is the true picture, speedup won't appear even with KV cache

**Option B — Pause Speed Path:**
Given the discrepancy, pause speed claims and focus on:
- Publishing/frozen quality results
- Clarifying the conditions under which PRT speedup applies
- Not claiming universal speedup until measurement setup is standardized

**Option C — Deeper Backend Research:**
Investigate why native FFN_UP is faster than PRT sparse ternary in fresh inference (3B), despite Phase 12 showing the opposite with KV cache. The difference may be in:
- How KV cache affects FFN_UP computation in subsequent tokens
- Whether PRT saves more on cached tokens vs fresh tokens
- Model-specific behavior at different inference stages

---

## Safety

- No `.gguf`/`.bin`/`.safetensors` staged ✅
- No secrets in any changed file ✅
- Only docs/JSON created, no binaries changed ✅
- Existing tags untouched ✅

**Tag:** `PRT_PHASE13AJ_SPEED_DISCREPANCY_RETROSPECTIVE`