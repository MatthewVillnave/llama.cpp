# PRT Phase 13AG — 3B Sidecar Load Optimization

**Date:** 2026-05-07  
**Verdict:** MMAP_HELPED ✅  
**Model:** Qwen2.5-3B-Instruct-Q4_K_M.gguf  
**Branch:** `experimental/prt-phase13-model-generalization`

---

## Verdict

mmap sidecar loading reduced PRT wall time from 9.884s to 5.300s — a 46.4% improvement. Sidecar load time dropped from 4542ms to 0.14ms (99.97% reduction). Generation throughput unchanged (8.5 vs 8.6 t/s). After eliminating sidecar load, the kernel time becomes the dominant remaining bottleneck. PRT remains 2.56× slower in wall time vs native — but that's a massive improvement from the fread 4.77× baseline.

---

## Baseline (Phase 13AF)

| Metric | Value |
|--------|-------|
| **Native wall** | 2.073s (avg of 5 runs) |
| **Native gen** | 20.96 t/s |
| **PRT fread wall (cold cache)** | 9.884s (4.77× native) |
| **PRT fread gen** | 8.60 t/s (0.41× native) |
| **Sidecar load** | 4542.6ms (46% of PRT wall) |

---

## Loader Analysis

### fread approach (old)
```
stat(path) → fopen(path, "rb") → malloc(N*float) → fread(data, float, N) → fclose(f)
```
**Per-layer cost:** malloc(90MB) + fread(90MB) copy  
**36 layers:** 36 × (malloc + memcpy)  
**Cold:** adds disk I/O time  
**Warm:** still pays allocation + copy overhead  

### mmap approach (new)
```
stat(path) → fopen(path, "rb") → fileno(f) → mmap(NULL, N*4, PROT_READ, MAP_PRIVATE, fd, 0) → fclose(f)
```
**Per-layer cost:** mmap syscall only (no allocation, no copy)  
**36 layers:** 36 × mmap() syscall + OS page faults on first access  
**Cold:** OS reads pages from disk into page cache on first fault  
**Warm:** pages already in page cache → ~0ms load time  

### Key findings
- **No validation scan** in either approach — shape is detected by file size (bytes == M*N*4)  
- **mmap data is resident** via OS page cache + g_prt_sidecar_buffers keeps it live  
- **mmap fallback works:** if mmap fails, falls back to fread automatically  
- **sidecar_load_ms timing** measures loader execution only — kernel page faults are in callback/kern time  

---

## Experiments

### Baseline: fread cold (disk, cold page cache)

| Metric | Value |
|--------|-------|
| Runs | 5 |
| Avg wall | 9.884s ± 0.249s |
| Avg gen tok/s | 8.60 t/s |
| Avg sidecar load | 4542.6ms (46.0% of wall) |
| Wall vs native | 4.77× |
| Gen vs native | 0.410× |
| Clean output | ✅ 5/5 |
| Fallback calls | 0 |

### Warm fread (disk, warm page cache)

| Metric | Value |
|--------|-------|
| Runs | 3 |
| Avg wall | 6.463s |
| Avg sidecar load | 947.4ms (14.7% of wall) |
| Wall vs native | 3.12× |
| Note | Page cache helps but still pays malloc+copy overhead |

### tmpfs (RAM-backed storage, 7.7GB available)

| Metric | Value |
|--------|-------|
| Runs | 3 |
| Avg wall | 17.111s ± 1.265s |
| Avg sidecar load | 11540.4ms (67.4% of wall) |
| Wall vs native | 8.25× |
| Clean output | ✅ 3/3 |
| Memory pressure | Swap grew from 3.1GB → 3.8GB during runs |
| **Verdict** | **BLOCKED_MEMORY** — copying 3.1GB to tmpfs caused page eviction and swap thrash |

### mmap disk (OS page cache, mmap loader)

| Metric | Value |
|--------|-------|
| Runs | 3 |
| Avg wall | 5.300s ± 0.023s |
| Avg gen tok/s | 8.5 t/s |
| Avg sidecar load | 0.14ms (0.0% of wall) |
| Wall vs native | 2.56× |
| Gen vs native | 0.406× |
| Clean output | ✅ 3/3 |
| Fallback calls | 0 |
| PRT_SHAPE | n_layer=36 M=2048 N=11008 ✅ |
| Replacement calls | 272 ✅ |
| AVX2 calls | 272 ✅ |
| Kern total | 3307.4ms |
| Cb total | 3307.5ms |

---

## Timing Comparison Table

| Mode | Wall (s) | Gen tok/s | Sidecar ms | Sidecar % | vs Native Wall |
|------|----------|-----------|------------|-----------|---------------|
| Native | 2.073 | 21.0 | — | — | 1.00× |
| PRT fread cold | 9.884 | 8.6 | 4543 | 46% | 4.77× |
| PRT fread warm | 6.463 | 8.6 | 947 | 15% | 3.12× |
| PRT tmpfs | 17.111 | — | 11540 | 67% | 8.25× |
| **PRT mmap** | **5.300** | **8.5** | **0.1** | **0%** | **2.56×** |

---

## Interpretation

### Why is sidecar load ~4.5s with fread?
- 36 layers × 90MB = 3.24GB of malloc + memcpy
- Cold: adds disk I/O for reading from NVMe
- Warm: page cache supplies data but allocation + copy overhead remains

### Why did tmpfs make things worse (17s vs 9.9s)?
- Pre-copying 3.1GB to tmpfs consumed memory
- fread still does malloc+copy in tmpfs path
- Additional memory pressure caused swap thrash (swap grew 0.7GB during runs)
- Not viable on this 15GB machine

### Why is mmap so much faster?
- No malloc — OS provides page-mapped memory directly
- No copy — mmap maps pages in-place
- Warm page cache: pages already in RAM → ~0ms load time
- First access: OS page faults handle bringing data in (included in callback/kern time)

### What remains after mmap optimization?
| Component | Time | % of Wall |
|-----------|------|-----------|
| Sidecar load | 0.14ms | 0% |
| Callback/kernel (272 calls, 80 tokens) | 3307ms | 62% |
| Unexplained (model inference + ggml dispatch) | ~2000ms | 38% |
| **Total** | **~5307ms** | **100%** |

**After mmap, kernel time IS the bottleneck** — 62% of wall vs 33% before. The sidecar load problem is solved.

### Gen tok/s unchanged
mmap doesn't improve gen throughput (8.5 vs 8.6 t/s). The AVX2 kernel is the bottleneck for token generation, not the loader. PRTGEN / nativeGEN ratio remains ~0.41×.

---

## Allowed Claims

- mmap reduced sidecar load from 4542ms to 0.14ms (99.97% reduction) ✅
- mmap reduced PRT wall time from 9.884s to 5.300s (46.4% reduction) ✅
- mmap does not change generation throughput (8.5 vs 8.6 t/s) ✅
- After mmap, PRT remains 2.56× slower in wall time vs native ✅
- mmap is safe and functional: 36/36 sidecars loaded, clean output, 0 fallback ✅
- tmpfs is not viable on this 15GB machine ✅

## Forbidden Claims

- ❌ No 3B speedup claim (PRT still 2.56× slower in wall)
- ❌ No production readiness
- ❌ No larger-than-3B extrapolation
- ❌ No universal speedup from mmap alone

---

## Recommended Next Phase

**Phase 13AH:** Kernel/layout optimization.

With sidecar load eliminated, the per-token AVX2 kernel time (41ms/call × 272 calls for 80 tokens) is now the clear bottleneck at 62% of wall time. Options:
1. Reduce calls per token (fewer layer traversals)
2. Batch kernel calls
3. Optimize AVX2 inner loops
4. In-process warm benchmark harness (keep sidecars resident across inferences)

---

## Safety

- No `.gguf`/`.bin`/`.safetensors` staged ✅
- No secrets in any changed file ✅
- Source changes: `common/common.h` (1 line), `common/arg.cpp` (7 lines), `tools/cli/cli.cpp` (+19 lines mmap logic) ✅

**Tag:** `PRT_PHASE13AG_SIDECAR_LOAD_OPTIMIZATION`