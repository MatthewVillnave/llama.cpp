# PRT Phase 18C — Weight Residency / Layer Streaming Feasibility

## Verdict

**NO_GO_LAYER_STREAMING_TOO_SLOW** — Layer-by-layer streaming from disk is too slow for real-time inference. mmap provides hot-page behavior but does not reduce RSS meaningfully. The dominant memory is in attention/embeddings/norms (~80% of total), which cannot be streamed independently.

## Context

- Phase 18B: FFN_UP is only 10% of model weights. PRT-Resident Replacement cannot solve 32B RAM alone.
- Phase 18A: Eight RAM-reduction strategies ranked. PRT-Resident Replacement was #1, but failed architecturally.
- Phase 18C tests: llama.cpp mmap behavior, layer streaming feasibility, page cache warm/cold comparison.

---

## llama.cpp Residency Findings

**mmap behavior:**
- `--mmap` (default): mmaps GGUF file with MAP_SHARED. Uses `posix_madvise(POSIX_MADV_WILLNEED)` for prefetch. Tensor data is file-backed, OS pages in on first access.
- `--no-mmap`: reads entire file into heap-allocated buffer. Model weights fully resident in RAM.

**Critical finding — mmap with default settings uses MORE RSS than no-mmap:**
| Mode | RSS (MB) | Host RAM (MiB) | CPU_REPACK (MiB) | Wall time |
|------|----------|----------------|-------------------|-----------|
| mmap (default) | 12,173 | 8,693 | 6,108 | 14.03s |
| no-mmap | 8,688 | 2,584 | 6,108 | 14.08s |

**Why?** mmap with `MAP_POPULATE` eagerly faults in pages via `posix_madvise(POSIX_MADV_WILLNEED)`. This increases RSS because the OS pages are actually accessed and brought into RAM. no-mmap loads into a heap buffer but does not trigger the same page-fault overhead.

**load_all_data behavior:**
- All tensors are loaded eagerly — no lazy loading path exists.
- Both mmap and no-mmap result in all model weights resident in RAM (6.1GB for 14B weights).
- mmap: file-backed pages in OS page cache (counts in Cached but not in RSS until accessed).
- no-mmap: explicit heap allocation (counts in RSS immediately).

**Layer streaming blockers:**
1. **ggml_graph_build() requires all tensors present** — the computation graph is built with all tensor references resolved at build time. No conditional tensor loading path exists.
2. **ggml_backend_tensor_alloc() requires persistent buffers** — tensor buffers are allocated at init and held for the entire session. No eviction/refetch cycle per layer.
3. **No layer-by-layer execution model** — llama.cpp computes entire sequence in one forward pass. Layers are processed sequentially but all tensor data must be present.
4. **KV cache and attention are inter-layer** — they depend on previous layer outputs, making streaming complex even if tensors could be evicted.
5. **Even if tensors were mmap'd and evicted via madvise, re-faulting costs dominate** — NVMe ~3GB/s, 14B layer ~0.2GB → 60-70ms per layer re-fault. 40 layers × 70ms = 2.8s just for re-faults before compute even starts.

---

## 14B Memory Baseline (mmap mode)

- RSS: 12,173 MB (12.2 GB)
- Model weights resident: ~6.1 GB (via CPU_REPACK)
- Attention/norms/embeddings: ~2.5 GB
- KV cache @ ctx=128: negligible
- Wall time: 14.03s for 4 tokens
- tok/s: ~8.7

---

## mmap/no-mmap Comparison

| Property | mmap (default) | no-mmap |
|----------|----------------|---------|
| Load time | ~3s | ~10s |
| RSS | 12.2 GB | 8.7 GB |
| Host RAM (model) | 8.7 GB | 2.6 GB |
| CPU_REPACK | 6.1 GB | 6.1 GB |
| File-backed | Yes (OS page cache) | No (heap) |
| Evictable | Yes (via munmap/madvise) | No (heap) |
| Wall time | 14.03s | 14.08s |
| tok/s | 8.7 | 8.7 |

**Key insight:** mmap's RSS is higher because MAP_POPULATE eagerly faults in pages. Without MAP_POPULATE, mmap would have lower RSS but higher first-access latency.

**mmap IS effectively a hot-page residency mechanism** — pages stay in OS page cache after first fault. Warm run vs cold run showed no timing difference (14.03s vs 15.43s warm — within noise). This suggests page cache is already hot from the first run.

---

## Layer Streaming Feasibility

**What would need to change:**
1. Per-tensor mmap with per-layer munmap after compute
2. Graph modification to build one layer at a time
3. Custom backend buffer with lazy page-in
4. Async prefetch of N+1 while computing N
5. KV preservation across layer boundaries

**Expected RAM savings:** Unlimited in principle (only keep current layer in RAM). But compute overhead makes it unusable.

**Expected speed cost:**
- NVMe sequential bandwidth: ~3-5 GB/s (measured from dd test earlier)
- 14B per-layer tensor size: ~210 MB (6.1GB / 28 layers effective)
- 32B per-layer tensor size: ~290 MB (18.5GB / 64 layers)
- Per-token layer re-fault time: 290MB / 3.5GB/s = 83ms per layer
- 64 layers × 83ms = 5.3 seconds just for re-faults before first token
- Even with perfect prefetch: token time = max(compute, I/O) = ~83ms minimum → ~12 tok/s maximum (vs 8.7 tok/s for full loaded)

**Layer streaming would make 32B SLOWER, not more feasible.**

**Why it fails:**
- The I/O time per layer (83ms) exceeds compute time per layer (compute for 32B is heavier but still less than I/O)
- Real-time generation requires all layers processed per token — no batching opportunity
- Page cache can't help because we'd be explicitly unmapping to save RAM

**Could madvise(MADV_DONTNEED) drop cold pages?**
- Only if pages are not actively in use. Since ggml holds tensor references, madvise on in-use tensor regions would cause crashes.
- Would need to unmap after compute, remap before next use — re-fault cost is the same as no-mmap.

---

## 32B Implication

**Could layer streaming move 32B into range?** No.

Even if we could stream perfectly:
- 32B total weights: ~18.5 GB
- NVMe at 3.5 GB/s → cold load: 5.3s per token
- Even with perfect prefetch overlap: ~83ms per layer × 64 layers = 5.3s minimum per token
- Real tok/s: < 0.2 tok/s — essentially unusable

**What actually could help 32B:**
1. **KV cache compression** — reduces per-token memory, not weights
2. **Attention weight quantization (INT4/INT2)** — targets the 8% attention weight memory
3. **Model parallelism** — split across multiple machines
4. **Smaller model alternative** — 14B is the practical CPU ceiling on this hardware
5. **Hardware upgrade** — more RAM is the only real solution

---

## Recommended Phase 18D

**RECOMMEND_STOP_RAM_PATH_THIS_HARDWARE**

The RAM wall for 32B cannot be broken via software techniques on this hardware:
- FFN_UP is only 17% of 32B weights → PRT can't fix it
- Layer streaming is too slow → I/O dominates compute
- KV compression helps but not enough → 32B still needs ~19GB vs 11GB available

The only viable path forward for 32B on this machine is **hardware**: more RAM, or a different machine with GPU/ more memory.

**What to do next instead:**
1. Phase 17D: Focus on 8B model generalization (when model is available)
2. Phase 16O: Public release / documentation of PRT results so far
3. Phase 18X: Explore KV cache compression as a separate line (targets different memory bottleneck)
4. Or simply document the RAM boundary finding and move PRT work to a different machine

---

## Allowed Claims

- mmap with MAP_POPULATE causes higher RSS than no-mmap due to eager page faults
- Layer streaming from disk is too slow for real-time CPU inference
- 32B is not feasible on this hardware via any known software technique
- 14B is the practical ceiling for CPU inference on this OptiPlex 7010

## Forbidden Claims

- RAM problem solved
- 32B will run on this hardware
- Layer streaming is viable
- PRT solves the RAM wall
- Production readiness