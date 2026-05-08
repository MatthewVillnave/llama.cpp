# PRT Phase 15F — Provenance + Manifest Cache Checkpoint

## Checkpoint

- **Tag:** `PRT_PHASE15F_PROVENANCE_MANIFEST_CACHE_CHECKPOINT`
- **Branch:** `experimental/prt-phase14a-packed-sidecars`
- **Base commit:** `6972de062cb37bdf76b007721b4a52fe85dcf0bf`
- **Verdict:** `PASS_PROVENANCE_MANIFEST_CACHE_CHECKPOINT` ✅

---

## Context

- **INT8:** validated runtime path — not changing
- **INT6:** experimental compressed path, passes offline parity, runtime canary, 8-prompt validation, longer-gen, repeatability, and overhead profiling
- **INT4 per-row:** offline NO-GO
- The duplicated-sidecar issue (Phase 15C discovery) required runtime provenance logging as a countermeasure
- Phase 15F implemented manifest caching to eliminate repeated SHA hashing while preserving provenance guarantees

---

## Frozen Result Chain (Phase 15C–F)

| Phase | Result | Key Metric |
|-------|--------|------------|
| **15C** | PASS — Runtime sidecar provenance logging | 6/6 runs logged 28/28 unique SHAs |
| **15D** | PASS — INT6 timing repeatability | 6/6 exact, gen tok/s 1.000× avg/median |
| **15E** | PASS — INT6 overhead profiled | Setup ~2,218ms, SHA ~811ms, I/O ~1,392ms |
| **15F** | PASS — Manifest cache | Setup ~1,361ms, hash ~0ms, ~811ms saved (~39% setup reduction) |

---

## Key Results

### Provenance Logging (Phase 15C)
Runtime provenance logs include:
- Sidecar directory and format
- Expected layer count and force-native layer list
- Per-layer: filename, size, SHA256, loaded status, fallback flag
- Per-run: `loaded_count`, `unique_sha_count`, duplicate hash warnings
- Stale manifest → `hash_mode=full` (fallback to real SHA computation)

### INT6 Repeatability (Phase 15D)
- 6/6 runs: exact output match
- Generation tok/s: native avg 8.93, INT6 avg 8.93 → **1.000× ratio**
- 28 unique sidecar SHA256 verified every run
- Fallback only to force-native layers 11 and 15

### Overhead Profile (Phase 15E)
Sidecar setup breakdown (INT6, 28×50MB = 1.4GB total):
| Component | Time | % |
|-----------|------|---|
| File I/O (sequential NVMe read) | ~1,392ms | 64% |
| SHA256 hashing (28× `popen("sha256sum")`) | ~811ms | 36% |
| Provenance header write | ~0ms | ~0% |

Generation tok/s remained at **9.5 tok/s** (parity).

### Manifest Cache (Phase 15F)
| Metric | Full Hash (15E) | Manifest (15F) | Delta |
|--------|-----------------|----------------|-------|
| Setup total | ~2,218ms | ~1,361ms | **−857ms** |
| Hash/provenance | ~811ms | 0.00ms | **−811ms** |
| Setup reduction | — | — | **~39%** |

- Validation: file existence + size match
- Stale fallback: verified (manifest_status → full hash on mismatch)
- Duplicate detection: preserved via manifest SHA array
- All 6 runs: `manifest_status=valid`, `hash_mode=manifest`

---

## Remaining Overhead

File I/O / sequential read of ~1.4GB packed sidecars (~1 GB/s NVMe throughput). This is the dominant remaining cost in sidecar setup.

---

## Allowed Claims

✅ Runtime sidecar provenance logging prevents the prior "28/28 loaded but duplicated" blind spot

✅ Manifest cache eliminated repeated SHA hashing overhead in measured INT6 runs

✅ Packed INT6 retained native-parity generation tok/s in the measured repeatability suite

✅ INT6 remains experimental; INT8 remains the validated runtime path

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

1. INT6 still has one-time setup/file-I/O overhead (~1,360ms on current NVMe)
2. Manifest validation uses file existence + size match; stale fallback to full hash recomputation
3. No larger-than-7B model validation
4. No production hardening (error handling, graceful degradation)
5. INT8 remains the safer validated path for any real deployment
6. Sidecar and manifest files are local only — not staged, not public

---

## Recommended Next Phase

**Phase 15G — I/O optimization via mmap sidecar loading**

Rationale: After eliminating SHA hashing, the dominant remaining setup cost (~1,360ms) is file I/O. Memory-mapping sidecar files would let the OS manage page caching and avoid repeated `fread()` syscalls on warm runs. This would further reduce setup overhead without changing INT6 format or PRT math.

**Alternative:** Phase 15G — backend/ggml integration design for production deployment path.

---

**Test environment:**
- Branch: `experimental/prt-phase14a-packed-sidecars`
- Commit: `6972de062cb37bdf76b007721b4a52fe85dcf0bf`
- Model: Qwen2.5-7B-Instruct-Q4_K_M (SHA 1875fb29)
- INT6 sidecar dir: `/tmp/prt_sidecars_7b_int6_phase15b_packed/`
- All 4 phase files: 15C, 15D, 15E, 15F verified present