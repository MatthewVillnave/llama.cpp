# PRT Phase 13AF — 3B Latency Profile

**Date:** 2026-05-07  
**Verdict:** PASS_LATENCY_PROFILED ✅  
**Model:** Qwen2.5-3B-Instruct-Q4_K_M.gguf  
**Branch:** `experimental/prt-phase13-model-generalization`

---

## Verdict

3B latency fully profiled. Sidecar load is the dominant cost (46% of wall time, ~4.54s per run). Per-token callback+kernel adds 33% (~3.30s for 80 tokens). Unexplained model inference overhead accounts for 21% (~2.04s). PRT is 4.77× slower in wall time and 0.41× in generation throughput vs native. No speedup claimed.

---

## Model / Sidecar Metadata

| Field | Value |
|-------|-------|
| **Model path** | `/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-3B-Instruct-Q4_K_M.gguf` |
| **Model size** | 1.84 GB |
| **Layers** | 36 |
| **Hidden** | 2048 |
| **FFN** | 11008 |
| **Sidecar dir** | `/tmp/prt_sidecars_3b/` |
| **Sidecars loaded** | 36/36 ✅ |
| **Force-native layers** | 11, 15 |
| **Prompt** | "The capital of France is" |
| **Generation tokens** | 80 |

---

## Native Timing (5 Cold Runs)

| Run | Wall (s) | Gen tok/s | Exit | Paris |
|-----|----------|-----------|------|-------|
| 1 | 2.121 | 21.1 | 0 ✅ | ✅ |
| 2 | 2.078 | 20.8 | 0 ✅ | ✅ |
| 3 | 2.052 | 21.0 | 0 ✅ | ✅ |
| 4 | 2.046 | 21.1 | 0 ✅ | ✅ |
| 5 | 2.066 | 20.8 | 0 ✅ | ✅ |
| **Avg** | **2.073 ± 0.027** | **20.96 ± 0.14** | — | — |

---

## PRT Active Timing (5 Cold Runs)

| Run | Wall (s) | Gen tok/s | Exit | Paris | Sidecar Load (ms) |
|-----|----------|-----------|------|-------|-------------------|
| 1 | 9.792 | 8.7 | 0 ✅ | ✅ | 4472.2 |
| 2 | 9.750 | 8.6 | 0 ✅ | ✅ | 4421.8 |
| 3 | 9.745 | 8.6 | 0 ✅ | ✅ | 4415.0 |
| 4 | 9.751 | 8.6 | 0 ✅ | ✅ | 4335.5 |
| 5 | 10.380 | 8.5 | 0 ✅ | ✅ | 5068.3 |
| **Avg** | **9.884 ± 0.249** | **8.60 ± 0.06** | — | — | **4542.6 ± 266.5** |

---

## PRT Internal Timing (Summary-Level Run)

| Metric | Value |
|--------|-------|
| **Total replacement calls** | 272 (34 active × 8 calls/layer) |
| **Callback total** | 3304.3 ms |
| **Kernel total** | 3304.3 ms |
| **Callback overhead** | 0.0 ms (0.0%) |
| **First-call total** | 2667.1 ms |
| **Later-call total** | 637.2 ms |
| **Avg first-call time** | 78.4 ms/layer |
| **Avg later-call time** | 2.65 ms/call |
| **Force-native first-call** | ~2.6 ms/layer (layers 11, 15) |
| **AVX2 kernel** | `avx2 (kernel_mode=1)` |
| **Compile flags** | `-mavx2 -mfma (LLAMA_PRT_AVX2)` |

---

## Timing Interpretation

### Is 3B PRT faster than native?
**No.** PRT wall is 4.77× slower than native. No speedup claimed.

### How much slower/faster?
- **Wall slowdown:** 9.884s vs 2.073s = **4.77× slower**
- **Generation slowdown:** 8.60 t/s vs 20.96 t/s = **0.41× (PRT is 59% slower in throughput)**

### What dominates PRT latency?

| Cost Component | Time (ms) | % of Wall |
|----------------|-----------|-----------|
| Sidecar load | 4542.6 | 46.0% |
| Callback+kernel total | 3304.3 | 33.4% |
| Unexplained (model inference + dispatch) | 2036.7 | 20.6% |
| **Total** | **9883.6** | **100%** |

**Sidecar load is the single largest cost.** At 4.54s per run, it consumes nearly half the wall time.

### Is sidecar loading material?
**Yes.** At 4.54s per run (46% of wall), sidecar loading is the dominant cost and the primary target for optimization.

### Is kernel time the dominant cost?
**No, but it's second.** Kernel total (3.30s, 33%) is the second-largest cost, but callback overhead is 0% — the kernel time IS the callback time with no extra dispatch overhead.

### Is graph/custom-op overhead measurable?
**No.** Callback total = Kernel total exactly (3304.3ms = 3304.3ms). There is zero measurable graph/custom-op overhead beyond the AVX2 kernel execution itself.

### Does 3B behave better or worse than 0.5B?

| Model | Native Gen | PRT Gen | PRT Ratio | Source |
|-------|-----------|---------|-----------|--------|
| Qwen2.5-0.5B | 94.86 t/s | 49.54 t/s | **0.522×** | Phase 13AA |
| Qwen2.5-3B | 20.96 t/s | 8.60 t/s | **0.410×** | Phase 13AF |

**PRT slowdown worsens by 21.4%** when going from 0.5B to 3B (ratio drops from 0.522 to 0.410). PRT overhead becomes proportionally more expensive as model size increases.

### Next optimization target
**Priority 1: Sidecar load time** (4.54s, 46% of wall). Options:
- Memory-map sidecar files instead of reading into RAM
- Async sidecar loading (start loading before first inference call)
- Persist sidecars in a RAM filesystem (tmpfs)
- Load sidecars once and keep resident across multiple inferences

**Priority 2: Per-token overhead** (3.30s for 80 tokens = 41ms/token). Each token generation triggers 34 PRT calls (8 calls × 34 layers, skipping force-native). First-call penalty (2667ms total) suggests initial kernel warmup is expensive.

---

## Allowed Claims

- 3B latency was profiled under clean PTY runner conditions ✅
- PRT wall time is 4.77× slower than native ✅
- PRT generation throughput is 8.6 t/s vs native 21.0 t/s (0.41×) ✅
- Sidecar load is the single largest cost at 46% of PRT wall time ✅
- AVX2 kernel is active with zero callback overhead ✅
- PRT slowdown worsens by 21% when scaling from 0.5B to 3B ✅

## Forbidden Claims

- ❌ No 3B speedup claim
- ❌ No production readiness
- ❌ No larger-than-3B extrapolation
- ❌ No universal speedup

---

## Recommended Next Phase

**Phase 13AG:** Sidecar load optimization.

Primary focus: reduce or eliminate the 4.54s sidecar load time. Options:
1. **tmpfs / RAM filesystem** — mount `/tmp/prt_sidecars_3b/` as tmpfs, eliminate disk I/O
2. **Memory-mapped I/O** — use `mmap()` instead of `read()` for sidecar loading
3. **Async preload** — load sidecars before the first inference call, keep resident
4. **Single-load reuse** — load once, keep in memory across multiple `llama-cli` invocations within the same process

If sidecar load can be eliminated or reduced to near-zero, PRT overhead drops from 4.77× to ~2.6× wall slowdown (callback+kernel + unexplained only). That would put PRT gen throughput at ~12.8 t/s vs native 21.0 t/s (0.61×), a significant improvement.

---

## Safety

- No `.gguf`/`.bin`/`.safetensors` staged ✅
- No secrets in any changed file ✅
- PRT logs in `/tmp/` ✅

**Tag:** `PRT_PHASE13AF_3B_LATENCY_PROFILE`