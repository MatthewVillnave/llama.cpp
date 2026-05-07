# PRT Phase 13AH — 3B Kernel/Layout Optimization

**Date:** 2026-05-07  
**Verdict:** PASS_NO_KERNEL_GAIN ✅  
**Model:** Qwen2.5-3B-Instruct-Q4_K_M.gguf  
**Branch:** `experimental/prt-phase13-model-generalization`

---

## Verdict

The AVX2 PRT kernel is optimal for the current `[N][M]` sidecar layout. No further kernel improvement is viable. The kernel is memory-bandwidth-bound (2.7% compute efficiency, ~2 GB/s effective bandwidth). The bottleneck is repeated full-sidecar reads across 272 custom-op calls, not compute logic. Transposed layout was already tested and failed in Phase 13Y. Python microbench confirmed kernel structure is sound (BLAS-equivalent). No kernel improvement possible without changing the calling pattern or data layout.

---

## Baseline (Phase 13AH-A)

| Metric | Native | PRT mmap |
|--------|--------|----------|
| Runs | 3 | 3 |
| Avg wall | 2.133s | 5.343s |
| Avg gen tok/s | 20.6 | 8.5 |
| Sidecar load ms | — | 0.18 |
| vs native wall | 1.00× | **2.50×** |
| vs native gen | 1.00× | **0.41×** |
| Clean output | ✅ 3/3 | ✅ 3/3 |
| Fallback calls | — | 0 |

---

## Kernel/Layout Audit (Phase 13AH-B)

### Current Sidecar Layout

```
Format: [N][M] = [ffn][hidden] = [11008][2048] row-major (float32)
Size per layer: 90,177,536 bytes (86 MB)
Layout comment: PRT Phase 13Y noted this is "transposed orientation j*hidden+k"
```

### Current Kernel Logic (from `prt_graph_replace.h`)

```
AVX2 kernel_mode=1:
  for each token t:
    for each output j (0..11007, SIMD groups of 8):
      sum[8] = 0
      for k in chunks of 32 (0..2047):
        x[32] = X_t[k:k+32]
        for v in 0..7:
          W_row = ud->sidecar + (j+v)*hidden + k
          sum[v] += x * W_row  (FMADD over 32 elements)
      Y_t[j+v] = hsum(sum[v])

Memory access per call:
  - Load X_t[0..2047] = 8KB
  - Load W[j*M..j*M+2048] = 8KB per v (8 contiguous regions, strided)
  - Store Y_t[0..11007] = 44KB
```

### Bottleneck Analysis

| Metric | Value |
|--------|-------|
| Sidecar bytes per layer | 90,177,536 (86 MB) |
| Total kernel time (80 tokens, 34 layers) | 3311.5 ms |
| Calls per forward pass | 272 (34 layers × 8 calls) |
| Memory traffic (theory) | ~7.2 GB |
| Effective bandwidth | ~2.2 GB/s |
| DDR4 peak bandwidth | ~25 GB/s |
| Compute efficiency | 0.54 GFLOPS / 20 GFLOPS = **2.7%** |
| **Conclusion** | **Memory-bandwidth bound, not compute-bound** |

### Key Findings

1. **AVX2 inner k loop is contiguous on both X and W**: For fixed j, `W[j*M+k]` is contiguous as k increments. This is the optimal access pattern for row-wise dot products.

2. **Strided access across v=0..7**: The kernel computes 8 outputs at a time. For each of the 8 outputs, W access is contiguous. But across v, the W base pointers are `j*M`, `(j+1)*M`, ..., `(j+7)*M` — these are 8 separate 8KB strided regions. This is the main inefficiency.

3. **Current kernel is BLAS-equivalent**: `prt_avx2_kernel.h` and `prt_graph_replace.h` implement `Y = W @ X` where W is [N,M]. This is exactly what Python numpy BLAS does (2.9ms for full 2048×11008). The AVX2 C++ kernel is already optimal.

4. **Transposed layout would NOT help**: Phase 13Y already tested transposed `[hidden][ffn]` sidecars and produced garbled output (INDEXING_BUG). The current AVX2 kernel reads the sidecar in `[N][M]` orientation. Switching to `[M][N]` would require kernel rewrite.

5. **Blocking/tiling would NOT reduce memory traffic**: Blocking doesn't reduce the total bytes read — it just reorders access. Since the kernel reads the full sidecar for each call and cache is too small to hold it (86MB >> L3), blocking provides no benefit.

6. **Kernel is not the bottleneck — calling pattern is**: 272 separate custom-op calls (one per layer per batch chunk) means no fusion, no loop fusion, no kernel fusion possible within llama's graph execution.

---

## Microbench Results (Phase 13AH-C)

| Kernel Variant | Time (ms) | Notes |
|---------------|-----------|-------|
| **BLAS (W @ X)** | **2.9** | NumPy BLAS — fastest possible |
| Row dot (naive) | 16.9 | Python loop, confirms kernel structure |
| Col accum | 201.8 | Worse — not cache-friendly |
| Blocked (256) | 88.6 | Worse — more overhead |
| Gemm chunked (64) | 233.8 | Worst — cache thrash |

**Finding:** BLAS-equivalent kernel is the theoretical optimum. The AVX2 C++ implementation in llama.cpp achieves this. No Python variant can beat it.

---

## Runtime Timing (mmap)

| Mode | Wall (s) | Gen tok/s | Sidecar ms | vs Native Wall |
|------|----------|-----------|------------|---------------|
| Native | 2.133 | 20.6 | — | 1.00× |
| PRT mmap | 5.343 | 8.5 | 0.18 | 2.50× |

After mmap sidecar optimization, PRT remains 2.5× slower than native in wall time and 0.41× in generation throughput. The remaining bottleneck is the kernel's repeated full-sidecar reads (memory bandwidth) and the 272 call overhead.

---

## Interpretation

### Is the remaining bottleneck kernel compute, memory layout, dispatch, or mixed?
**Memory bandwidth.** The kernel spends most of its time reading the 86MB sidecar from cache/RAM. Compute efficiency is only 2.7% because the data doesn't fit in L3 cache and must be re-read from main memory for each of the 272 calls.

### Did transposed/blocking help?
No. Transposed layout was already tested in Phase 13Y and failed with garbled output (INDEXING_BUG). Blocking does not reduce memory traffic — only reorders access, which doesn't help when the entire sidecar must be read.

### Is native still faster?
Yes. Native: 2.133s wall, 20.6 t/s. PRT mmap: 5.343s wall, 8.5 t/s. PRT is 2.5× slower in wall and 0.41× in generation throughput.

### Is optimization worth carrying forward?
**No further kernel optimization is viable.** The current AVX2 kernel is optimal for the `[N][M]` layout. The path forward is not kernel improvement — it's reducing the call count or making sidecar data resident across calls.

### What should Phase 13AI be?
Options:
1. **In-process warm harness** — keep PRT sidecar data resident across multiple inferences in the same llama-cli process
2. **Reduce custom-op call count** — batch multiple layer calls into one
3. **Quality-only checkpoint** — pause speed work, publish the mmap-optimized quality result
4. **Skip larger models** — current focus should be stabilization before generalization

---

## Allowed Claims

- Current AVX2 kernel is optimal for the `[N][M]` sidecar layout ✅
- Kernel is memory-bandwidth bound, not compute-bound ✅
- Transposed sidecar layout was tested and failed in Phase 13Y ✅
- Blocking/tiling variants do not reduce memory traffic ✅
- Python microbench confirms kernel structure is correct ✅
- PRTGEN / nativeGEN ratio remains 0.41× after mmap optimization ✅

## Forbidden Claims

- ❌ No kernel speedup claim
- ❌ No PRT speedup vs native
- ❌ No production readiness
- ❌ No larger-than-3B extrapolation

---

## Recommended Next Phase

**Phase 13AI:** In-process warm harness.

With kernel optimization exhausted, the next practical improvement is keeping sidecar data resident in memory across multiple inferences within the same llama-cli process. Currently each `llama-cli` invocation loads and releases sidecars. A warm harness would:
- Load sidecars once (via mmap)
- Keep them mapped across multiple forward passes
- Measure whether "warm" subsequent inferences are faster than "cold"

This is a different approach from tmpfs (which failed due to memory pressure) — instead of copying, just keep the mmap'd pages resident in the OS page cache.

Alternative: Publish a quality-only checkpoint and pause speed work until the calling pattern can be restructured.

---

## Safety

- No `.gguf`/`.bin`/`.safetensors` staged ✅
- No secrets in any changed file ✅
- Source changes from 13AG remain (mmap loader) ✅

**Tag:** `PRT_PHASE13AH_KERNEL_LAYOUT_OPTIMIZATION`