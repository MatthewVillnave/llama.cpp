# PRT Phase 15I — INT6 Optimized Pipeline Checkpoint

## Checkpoint

- **Tag:** `PRT_PHASE15I_INT6_OPTIMIZED_PIPELINE_CHECKPOINT`
- **Branch:** `experimental/prt-phase14a-packed-sidecars`
- **Base commit:** `fdf657e1ee329acc1bfcfb31f86879bf59a75836`
- **Verdict:** `PASS_INT6_OPTIMIZED_PIPELINE_CHECKPOINT` ✅

---

## Context

- **INT8:** validated runtime path — not changing
- **INT6:** experimental compressed path — passed offline parity, runtime canary, 8-prompt validation, longer-gen smoke, and timing repeatability; now further optimized at CLI level
- **INT4 per-row:** offline NO-GO
- Phase 15C fixed provenance logging (eliminates the "28/28 loaded but duplicated" blind spot)
- Phase 15F eliminated repeated SHA hashing via manifest cache (~811ms/run saved)
- Phase 15G added mmap sidecar loading (~416ms further saved)
- Phase 15H validated LUT4x unpack optimization — correct but delivers modest gains (~32ms)
- CLI-level INT6 setup optimization is now exhausted; remaining ~910ms is memory-bandwidth-bound

---

## Frozen Result Chain (Phase 15B–H)

| Phase | Result | Key Metric |
|-------|--------|------------|
| **15B-J** | PASS — INT6 experimental freeze | Generation at native parity, 28/28 sidecars |
| **15C** | PASS — Runtime provenance logging | 6/6 runs, 28/28 unique SHAs, 0 duplicates |
| **15D** | PASS — INT6 timing repeatability | 6/6 exact, gen tok/s 1.000× ratio |
| **15E** | PASS — INT6 overhead profiled | Setup ~2,218ms, SHA ~811ms, I/O ~1,392ms |
| **15F** | PASS — Manifest cache | Setup ~1,361ms, hash ~0ms, ~39% reduction |
| **15G** | PASS — mmap I/O improvement | Setup ~945ms, ~30% further reduction |
| **15H** | PASS — LUT4x unpack optimization | Setup ~913ms, ~3.4% further reduction, functionally correct |

---

## Setup Time Progression

| Phase | Mode | Setup (ms) | Delta | Change |
|-------|------|-----------|-------|--------|
| **15E** | fread + full hash | 2,218 | baseline | |
| **15F** | fread + manifest | 1,361 | −857ms | SHA eliminated |
| **15G** | mmap + manifest | 945 | −416ms | I/O reduced |
| **15H** | mmap + LUT4x | 913 | −32ms | unpack micro-opt |
| **Total** | | **−1,305ms** | **−59%** | |

---

## Key Results

### Provenance Logging (Phase 15C)
Runtime provenance logs include:
- Sidecar directory, format, expected layer count
- Per-layer: filename, size, SHA256, loaded status, fallback flag, load_mode
- Per-run: `loaded_count`, `unique_sha_count`, duplicate hash warnings
- Stale manifest → `hash_mode=full` fallback
- `load_mode=mmap` or `load_mode=read` per layer

### Manifest Cache (Phase 15F)
- File: `prt_sidecar_manifest.json` alongside sidecar directory
- Validation: file existence + per-layer size match
- Stale fallback: recompute SHA on mismatch
- Pairs with provenance to eliminate repeated hashing on warm runs

### mmap Loading (Phase 15G)
- MAP_PRIVATE + MADV_SEQUENTIAL for INT6 sidecar files
- fread path preserved as fallback
- ~416ms improvement vs fread path (~30% of Phase 15F baseline)

### LUT4x Unpack Optimization (Phase 15H)
- 64-entry lookup table replaces `(bits & 0x3F) - 32` arithmetic
- 4× loop unrolling (16 elements per iteration)
- Correctness: mathematically exact — `lut[x] = x - 32` for x in [0, 63]
- Result: ~32ms improvement (~3.4%), functionally correct but modest
- Dominant remaining cost: memory-bandwidth-bound CPU unpack (~1.9B elements across 28 layers)

### Generation Quality
- All Phase 15 test runs: generation tok/s at near/native parity
- No collapse, repetition, or contamination observed
- Valid JSON outputs on structured prompts
- Fallback only to force-native layers 11 and 15

---

## Allowed Claims

✅ INT6 experimental path passed the frozen Phase 15 validation chain on Qwen2.5-7B in Matt's measured CPU setup

✅ Provenance logging prevents the duplicated-sidecar blind spot

✅ Manifest cache and mmap reduced repeated SHA overhead and I/O overhead

✅ LUT4x unpack optimization was functionally correct

✅ INT6 remains experimental; INT8 remains the validated runtime path

✅ INT4 per-row is offline NO-GO

---

## Forbidden Claims

❌ Do NOT claim production readiness

❌ Do NOT claim universal speedup

❌ Do NOT claim INT6 replaces INT8

❌ Do NOT claim full deployment readiness

❌ Do NOT claim all-long-context stability

❌ Do NOT claim larger-than-7B support

❌ Do NOT claim GPU comparison

❌ Do NOT claim broad benchmark dominance

❌ Do NOT claim sidecars or manifests are public/staged

---

## Known Caveats

1. INT6 still has one-time setup/unpack overhead (~913ms on measured NVMe/CPU)
2. Remaining ~910ms is memory-bandwidth-bound — CLI-level loop tweaks unlikely to yield further gains
3. No larger-than-7B model validation yet
4. No production hardening (error handling, graceful degradation, multi-model support)
5. INT8 remains the safer validated path for any real deployment
6. Sidecar and manifest files are local only — not staged, not public
7. LUT4x delivered modest gains because the bottleneck is memory throughput, not instruction count

---

## Recommended Next Phase

**Phase 16A: Backend/ggml integration design OR 14B feasibility canary**

Rationale: CLI-level INT6 setup optimization is exhausted. Future gains likely require:
- Backend/ggml integration (avoid per-run file open + mmap overhead)
- Model generalization (does Phase 14B architecture support arbitrary models?)
- A comprehensive technical writeup of the Phase 15 series

---

**Test environment:**
- Branch: `experimental/prt-phase14a-packed-sidecars`
- Commit: `fdf657e1ee329acc1bfcfb31f86879bf59a75836`
- Model: Qwen2.5-7B-Instruct-Q4_K_M (SHA 1875fb29)
- INT6 sidecar dir: `/tmp/prt_sidecars_7b_int6_phase15b_packed/`
- All 7 phase files (15B–H) verified present