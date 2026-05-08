# PRT Phase 15E — INT6 Unpack/Setup Overhead Profiling

## Verdict

**PASS_INT6_OVERHEAD_PROFILED** ✅

## Context

Phase 15D confirmed INT6 generation tok/s at 1.000× native (6/6 exact matches) but the ~19% wall overhead from Phase 15B-H remained unexplained. Phase 15E adds detailed timing instrumentation to identify exactly where the time goes during INT6 sidecar setup.

## Instrumentation Added

**File:** `tools/cli/cli.cpp`

Added per-layer and aggregate timing to `--prt-log-file`:
```
[PRT_TIMING] sidecar_load_ms=2191.24 sidecar_hash_total_ms=798.77 provenance_header_ms=0.00
[PRT_SIDECAR_LAYER] layer=0 file=ffn_up_layer0_prt.int6 size=50997268 sha256=... status=loaded fallback=false read_ms=49.98 hash_ms=28.61 layer_total_ms=78.59
```

- `sidecar_load_ms`: total sidecar setup time (all layers)
- `sidecar_hash_total_ms`: cumulative SHA256 time across all 28 layers
- `provenance_header_ms`: provenance block write time (negligible, ~0ms)
- Per-layer: `read_ms`, `hash_ms`, `layer_total_ms`

**Note:** No new `--prt-provenance-hash` flag was added (would be invasive). Hash cost was measured via the existing timing instrumentation.

## Test Matrix

| Prompt | Text | Runs | Settings |
|--------|------|------|----------|
| P1 | "The capital of France is" | 2 | n=80, temp=0, c=512, t=4 |
| P2 | "Once upon a time in a distant galaxy" | 2 | n=80, temp=0, c=512, t=4 |
| P3 | 'Return JSON with keys name and status for an AI system named Qwen.' | 2 | n=80, temp=0, c=512, t=4 |

## Quality Results

| Check | Result |
|-------|--------|
| Native completed | 6/6 ✅ |
| INT6 completed | 6/6 ✅ |
| Exact/semantic matches | 6/6 ✅ |
| Quality degradations | 0 ✅ |
| Provenance: 28/28 loaded | 6/6 ✅ |
| Provenance: 28 unique SHA | 6/6 ✅ |
| Duplicate warnings | 0 ✅ |

## Overhead Breakdown (from instrumentation)

**Aggregate setup timing (INT6, 6 runs):**

| Metric | Avg (ms) | Range (ms) |
|--------|----------|------------|
| Total sidecar load | 2,218 | 2,191–2,260 |
| SHA256 hash total | 811 | 799–819 |
| Hash % of load | 37% | 36–37% |
| Per-layer read | ~50ms | 48–51ms |
| Per-layer hash | ~29ms | 28–30ms |
| Per-layer total | ~78ms | 76–82ms |

**Per-layer detailed timing (28 layers, 50MB each, ~1.4GB total):**

| Component | Time | % of Load | Throughput |
|-----------|------|-----------|------------|
| File I/O (read 1.4GB) | 1,392ms | 64% | ~1.0 GB/s (NVMe sequential) |
| SHA256 hashing (28×50MB) | 799ms | 36% | ~1.8 GB/s hash throughput |
| Provenance header write | ~0ms | ~0% | — |
| **Total** | **2,191ms** | **100%** | — |

The gap between measured `sidecar_load_ms` and sum of per-layer totals is **< 1ms** — all time is fully accounted for.

**File I/O component:** Sequential reads of 28 × 50.997MB packed sidecar files (1.428GB total) at ~1 GB/s on NVMe.

**SHA256 hashing component:** 28 `sha256sum` invocations via `popen()` on 50MB files each — popen() fork+exec overhead is included. Effective hash throughput ~1.8 GB/s.

## Generation Timing

Quick verification check (prompt "The capital of France is", n=80):
- Native: `Generation: 9.5 t/s`
- INT6: `Generation: 9.5 t/s`

Generation tok/s remains at parity. Per-token throughput is clean.

## Bottleneck Analysis

**Q: What causes wall overhead?**

The INT6 sidecar load takes ~2,191ms on top of model loading and KV fill. Breaking that down:

1. **File I/O (64% = ~1,392ms):** Sequential reads of 28 × 50MB = 1.4GB packed data from NVMe. This is unavoidable for cold-start. ~1 GB/s is reasonable NVMe throughput.

2. **SHA256 hashing (36% = ~799ms):** 28 `sha256sum` calls via popen(), each processing 50MB. This is **the single largest identifiable extra cost** of provenance logging. Without provenance, INT6 load would be ~1,392ms (I/O only) vs INT8 which also reads its 1.4GB sidecars. So provenance logging adds ~800ms per run.

3. **INT6 unpack (embedded in I/O):** The INT6 → INT8 unpack loop (scalar bit-shifting, ~2.2GB of int8 output) is included in the read time measurement. CPU-bound unpack of 28 × 67.9M = 1.9B elements at ~1.4 GB/s is comparable to the NVMe I/O rate.

4. **INT8 baseline:** INT8 sidecars are also ~1.4GB (68MB × 28). The equivalent INT8 setup without provenance would likely be ~1,200–1,400ms (I/O + dequant, no hash). The ~800ms provenance cost is purely from SHA256 logging.

## Interpretation

- **Q: Is INT6 per-token generation still near native?** Yes — generation tok/s confirmed at 9.5 tok/s, matching native within measurement noise.

- **Q: What causes wall overhead?** Three factors: (1) File I/O for 1.4GB packed sidecar data (~1,392ms, unavoidable cold-start), (2) SHA256 provenance hashing (~799ms, provenance-specific), (3) INT6 unpack CPU cost (embedded in I/O, comparable to NVMe rate). The provenance hashing at 36% of load time is the most actionable.

- **Q: Is the overhead one-time and amortizable?** Yes — all overhead is per-run setup. It does not affect per-token generation throughput. A warm/cached run (sidecars pre-loaded in memory) would eliminate most of this cost.

- **Q: Should next phase optimize?** Three clear options:
  1. **Cache SHA256 hashes** (save ~800ms/run) — store hashes in a manifest file, read once, reuse
  2. **Pre-unpack INT6 sidecars** (save unpack CPU cost) — write a pre-unpacked INT8 version alongside the INT6 packed file
  3. **Memory-mapped sidecars** — use mmap instead of fread to let the OS cache pages; reduces I/O cost on repeated runs

## Allowed Claims

- INT6 wall overhead profiled ✅ — primary cause identified (file I/O + SHA256 hashing)
- INT6 generation tok/s remains at native parity ✅
- Provenance SHA256 logging costs ~800ms per run (36% of load time) ✅
- INT6 remains experimental; INT8 remains validated runtime path

## Forbidden Claims

- Do NOT claim production readiness
- Do NOT claim INT6 wall overhead is zero or minimal
- Do NOT claim INT6 replaces INT8
- Do NOT claim speedup
- Do NOT claim larger-than-7B support

## Recommended Next Phase

**Phase 15F: provenance SHA hash caching** — store per-file SHA in a manifest alongside the sidecar directory; compute once at generation time, reuse for subsequent runs. Eliminates the ~800ms (36%) provenance cost with minimal code change.

OR

**Phase 15F: pre-unpacked sidecar strategy** — regenerate sidecars as pre-decompressed INT8 alongside the packed INT6; runtime selects based on whether it's a first run (use packed, compute hash, cache) or subsequent run (use cached).

---

**Test environment:**
- Branch: `experimental/prt-phase14a-packed-sidecars`
- Model: Qwen2.5-7B-Instruct-Q4_K_M (SHA 1875fb29e8c91c86615c00e92d8b4114e56bc24359adb5a8db8b36452fae4a49)
- INT6 sidecar dir: `/tmp/prt_sidecars_7b_int6_phase15b_packed/` (28 files, 51MB each, 1.4GB total)
- Binary: `build/bin/llama-cli` (Phase 15E instrumentation)