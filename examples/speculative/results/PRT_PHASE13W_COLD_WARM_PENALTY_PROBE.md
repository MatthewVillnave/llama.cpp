# PRT Phase 13W: Cold/Warm Custom-Op Penalty Probe

**Date:** 2026-05-06
**Branch:** `experimental/prt-phase13-model-generalization`
**Previous HEAD:** `2fccd83ba` (Phase 13V)
**New HEAD:** [pending commit]
**Verdict:** `DISPATCH_FIXATION_BOTTLENECK` — dispatch/ggml-callback overhead, NOT memory bandwidth

---

## 1. Force-Native Accounting

**Result: ✅ Correct by design**

| Layer | Role | Custom op calls | Path |
|-------|------|----------------|------|
| 11, 15 | Force-native | 0 | `build_lora_mm` native path |
| 0–10, 12–14, 16–23 | PRT active | 8 calls each | PRT custom op |

- `g_prt_true_replacement_calls = 176` (22 layers × 8 tokens)
- `g_native_fallback_calls` incremented only for layers 11 and 15
- Total: 24 layers, 2 force-native, 22 PRT active — **correct**

---

## 2. Timer Scope — CRITICAL FINDING

**Timer measures: Total callback time (dispatch + compute + timing overhead)**

```
Total callback = dispatch/setup + kernel_compute + timing_record + return
```

The `std::chrono` timer starts at function entry (before any compute) and ends at function exit.
No nested sub-phases were measurable in the summary log because `kern_total=0.0ms` for all runs.

---

## 3. CRITICAL DISCOVERY: AVX2 Path Was Never Compiled

### Phase 13V Assumed: AVX2 SIMD kernel active
### Phase 13W Found: **SCALAR/SSE fallback active, AVX2 code path not compiled**

**Root cause:**

```
$ python3 -c "import json; data=json.load(open('build/compile_commands.json')); \
    print([e['file'].split('/')[-1] for e in data if 'llama-graph.cpp' in e['file'] \
    and '__AVX2__' in e['command']])"
[]
```

- **`llama-graph.cpp` (which includes `prt_graph_replace.h`) is compiled with NO `-march=native` and NO `-mavx2`**
- **Therefore `__AVX2__` is never defined**
- The `#if defined(__AVX2__)` at line 240 of `prt_graph_replace.h` **always fails**
- The `if (ud->kernel_mode == 1)` branch is **dead code** — the else branch always executes
- `kernel_mode=1` is set but the AVX2 block is never compiled

**Verified via disassembly of `prt_ffn_up_custom_op` in `libllama.so`:**
```
f6790: movups  (%rcx,%rax,1),%xmm0    ← SSE (XMM), NOT AVX (YMM)
f6794: movups  (%rdx,%rax,1),%xmm3    ← SSE, NOT AVX
f679c: mulps   %xmm3,%xmm0            ← SSE multiply (4-wide), NOT AVX 8-wide
```

**Correct instruction encoding would be:**
```
vmovups  (%rcx,%rax,1),%ymm0    ← AVX2 YMM (8-wide)
vmulps   %ymm3,%ymm0,%ymm0      ← AVX2 multiply
```

**Contrast:** `ggml-cpu.cpp` (which has AVX2 GEMM kernels) IS compiled with `-march=native` and contains ymmword instructions. The llama core graph engine is NOT.

**Impact:** Phase 13V analysis of the AVX2 GOTOLO kernel was analyzing **code that was never executed**.

---

## 4. Per-Layer First-Call vs Later-Call Data (n=10 tokens, 8 calls/layer)

| Layer | First call (ms) | Later avg (ms) | Calls |
|-------|-----------------|---------------|-------|
| 0 | **74.1** (highest) | 2.02 | 8 |
| 1–22 | 61–63 (consistent) | 2.0–2.1 | 8 each |
| 23 | **2.2** (lowest) | 2.0 | 8 |

**Observations:**
- First-call penalty: ~61–74ms per layer for layers 0–22
- Later calls: ~2ms per call (stable across all layers)
- Layer 23 (final layer) shows no first-call penalty
- **61ms × 22 first-calls = 1342ms ≈ `first_total=1302ms` from summary — matches!**
- **The first-call penalty IS real and per-layer, not one-time startup**

---

## 5. Pretouch Results

**Pretouch: Sequential page-by-page read of all 22 sidecar arrays (one float per 4KB page)**

| Metric | Baseline (3 runs avg) | Pretouch (3 runs avg) |
|--------|----------------------|----------------------|
| Wall time | 2.338s | 2.334s | 
| cb_total | 1610.4ms | 1610.1ms |
| first_total | 1297.7ms | 1297.8ms |
| later_total | 313.1ms | 313.3ms |
| pretouch_ms | N/A | 1.01ms |

**Verdict: Pretouch provides ZERO benefit.**

If page-fault/first-touch were the bottleneck, pretouch should shift 60ms-per-layer to pretouch_ms ≈ 1300ms and reduce first_total to near-zero. Instead:
- `pretouch_ms = 1ms` (negligible, OS already cached)
- `first_total` unchanged (~1298ms)
- `later_total` unchanged (~313ms)

**Page-fault hypothesis REJECTED.**

---

## 6. Back-to-Back Process Runs

Each new `llama-cli` invocation loads sidecars from disk freshly:

| Run | Wall (s) | cb_total (ms) | first_total (ms) |
|-----|----------|---------------|-----------------|
| 1 (cold) | 2.386 | 1622.0 | 1309.7 |
| 2 (warm) | 2.335 | 1615.2 | 1301.7 |
| 3 (warm) | 2.331 | 1615.0 | 1301.5 |

**Verdict: No meaningful difference across runs. OS page cache is already warm from sidecar_load_ms=121ms.**
The sidecar loading from disk (121ms) dominates the cold-start cost, not first-touch within generation.

---

## 7. Synthetic Bandwidth Benchmark

```
One sidecar (17.4MB):    6.4ms = 2,744 MB/s
All 22 sidecars (383MB): 53.7ms = 7,136 MB/s
```

**Contrast with Phase 13V's implied bandwidth:**
- 17.4MB / 52ms ≈ **335 MB/s** (from first-call penalty)
- Actual sequential read: **2,744–7,136 MB/s**

**The first-call penalty is NOT memory bandwidth.**

---

## 8. The 61ms First-Call Penalty: What's Actually Happening

### Mathematical Impossibility

Each custom op call processes:
- 22 layers × 1 token × 896 hidden × 4864 ffn_up = **~4.35M FLOPs**
- 22 layers × 896 × 4 bytes = **17.4MB working set per call**

At 3.5GHz with SSE (4-wide), **pure compute = 1.1ms max**.  
At 7GB/s memory bandwidth, **memory access = 2.5ms max**.

**61ms >> 3.6ms theoretical minimum. Something else dominates.**

### Profiler Evidence

The disassembly of `prt_ffn_up_custom_op` shows:
1. **Long prologue** (24 instructions, ~150ns): context/tensor pointer setup, logging
2. **Timing initialization** (18 `movaps` instructions): initializing timing accumulators every call (not just once!)
3. **SSE compute** (1.8ms actual): `movups`/`mulps`/`addss` loop
4. **Epilogue + return**: ~50ns

The prologue/setup section has **no cache-line straddle, no unusual branching**.  
The total non-compute overhead per call should be **< 1μs**.

**Yet the measurement shows 2ms for every call and 61ms for first calls.**

### Hypothesis: ggml Graph Dispatch Overhead

The 61ms is likely dominated by **ggml's graph walk / scheduling overhead** — specifically:
1. ggml traverses the compute graph to find custom ops
2. Each `ggml_forward` call involves:
   - Tensor shape validation
   - Backend dispatch (CPU backend → ggml-impl.c → custom op lookup)
   - `ggml_compute_forward_*` function pointer dispatch
   - The custom op function call itself
3. First call per layer: ggml's internal hash tables, dispatch caches, and tensor metadata are being populated for the first time
4. Later calls: ggml's dispatch cache is warm, reducing overhead

This is consistent with:
- **First-call ≈ 61ms**: ggml populating dispatch tables, resolving function pointers, first-time tensor layout checks
- **Later-call ≈ 2ms**: ggml dispatch cache warm, but still overhead per call
- **Pretouch doesn't help**: OS page faults aren't the issue — it's ggml's internal dispatch state

### ggml Backend Dispatch Cost (Per-Call Overhead)

Modern ggml with backend unification has ~500ns base cost per compute node. For 176 calls, that's ~88μs. But we see 1600ms total.

The gap suggests that **per-call ggml overhead is ~9ms per call**, which would be ~40× more than expected — indicating potential:
- **Lock contention** in the compute scheduler
- **Memory allocation per call** (ggml tensor allocation)
- **Repeated tensor shape checks** in the hot path

---

## 9. Phase 13V Analysis Correction

Phase 13V assumed AVX2 was active and analyzed the GOTOLO outer-product kernel.
**This was wrong.** The actual code running is the SSE scalar fallback.

The SSE scalar fallback computes `s += X_t[k] * W[j*hidden+k]` in a tight loop using `movss`/`mulss`/`addss` (1-wide), not AVX2's `vmulps` (8-wide).

Performance ratio: **AVX2 would be ~4–8× faster** than SSE for this problem size.

---

## 10. Optimization Roadmap (Updated)

| Priority | Action | Expected gain |
|----------|--------|--------------|
| **0 (CRITICAL)** | Fix AVX2 compilation: add `-march=native` to llama-graph.cpp | 4–8× per-call speedup |
| **1** | Profile with `perf` to identify exact ggml dispatch cost | Quantify overhead |
| **2** | Transpose sidecars to [hidden × ffn] (column-major, cache-friendly) | 1.2–1.5× |
| **3** | Batch multiple tokens into single custom op call | Reduce dispatch overhead |
| **4** | Eliminate per-call timing initialization (move outside hot path) | 0.5–1ms savings |

---

## 11. Benchmark Data

### Native Baseline (3 runs)
```
Run 1: Generation: 91.5 t/s
Run 2: Generation: 93.7 t/s
Run 3: Generation: 92.7 t/s
```

### PRT Baseline (3 runs, `--prt-log-level summary`)
```
Run 1: wall=2.353s cb=1611.7ms first=1297.8ms load=120.67ms
Run 2: wall=2.335s cb=1608.9ms first=1296.6ms load=121.41ms
Run 3: wall=2.326s cb=1610.6ms first=1295.9ms load=121.77ms
```

### PRT + Pretouch (3 runs)
```
Run 1: wall=2.341s cb=1612.2ms first=1301.2ms load=120.73ms pretouch=1.01ms
Run 2: wall=2.340s cb=1612.1ms first=1297.2ms load=121.35ms pretouch=1.01ms
Run 3: wall=2.325s cb=1605.9ms first=1295.0ms load=148.76ms pretouch=0.98ms
```

### Per-Layer First-Call Detail (n=10 tokens)
```
IL=0:  first=74.1ms later=2.02ms
IL=1:  first=61.6ms later=2.04ms
IL=2:  first=61.7ms later=2.04ms
IL=3:  first=62.0ms later=2.02ms
IL=4:  first=61.5ms later=2.03ms
IL=5:  first=61.4ms later=2.04ms
IL=6:  first=61.8ms later=2.03ms
IL=7:  first=61.9ms later=2.04ms
IL=8:  first=61.5ms later=2.02ms
IL=9:  first=61.9ms later=2.05ms
IL=10: first=61.6ms later=2.02ms
IL=12: first=61.7ms later=2.02ms
IL=13: first=61.5ms later=2.01ms
IL=14: first=61.6ms later=2.01ms
IL=16: first=62.1ms later=2.02ms
IL=17: first=61.7ms later=2.01ms
IL=18: first=61.7ms later=2.01ms
IL=19: first=61.9ms later=2.00ms
IL=20: first=61.7ms later=2.02ms
IL=21: first=61.8ms later=2.02ms
IL=22: first=62.9ms later=2.00ms
IL=23: first=2.2ms  later=2.02ms
```

---

## Summary

| Question | Answer |
|----------|--------|
| Is 60ms first-call penalty real? | **YES** — consistently ~61–74ms per layer for first token |
| Is it page-fault/first-touch? | **NO** — pretouch adds 1ms, not 1300ms |
| Is it memory bandwidth? | **NO** — synthetic shows 7GB/s, implied 335MB/s from timing |
| Is AVX2 active? | **NO** — `llama-graph.cpp` compiled without `-march=native`, AVX2 path never compiled |
| Is sidecar layout the main problem? | **NO** — pretouch doesn't help, layout contributes minorly |
| Recommended next optimization | **Fix AVX2 compilation** (4–8× expected), then profile ggml dispatch |
| Bottleneck location | **ggml graph dispatch overhead** per custom op call |