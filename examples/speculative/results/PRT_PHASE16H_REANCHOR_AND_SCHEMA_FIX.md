# PRT Phase 16H — Re-anchor + 14B INT6 Schema Fix

**Date:** 2026-05-09
**Session:** sharp-nudibranch
**Branch:** experimental/prt-phase14a-packed-sidecars

---

## A. Starting branch/head
- Branch: `experimental/prt-phase14a-packed-sidecars`
- HEAD: `8130cb039` (Phase 16C: 14B single-tensor FFN_UP extraction probe)

## B. Final branch/head
- Same: `experimental/prt-phase14a-packed-sidecars` at `8130cb039`
- No branch switch needed

## C. Branch mismatch real or stale?
- **Stale report text** — Phase 16G report incorrectly referenced `phase10e-archived-20260430` but actual branch was always correct

## D. Runtime magic confirmed
- **PRT6** — runtime loader checks for magic `{'P','R','T','6'}` at bytes 0-3
- Both C++ v4 writer and Python regen.py write PRT6 (no magic contradiction)

## E. Header size confirmed
| Component | Runtime expects | Writer produced | Match? |
|-----------|--------------|-----------------|-------|
| Header bytes | **16** | **20** | **NO** |
| Scale offset | byte 16 | byte 20 | NO (+4) |
| Payload offset | 16+M×4 | 20+M×4 | NO (+4) |
| File size (14B) | 53,139,472 | 53,139,476 | NO (+4) |

**Root cause:** Writer (all variants) includes 4-byte `clear` field at bytes 16-19

## F. Writer fixed?
- **YES** — Fixed in previous turn:
  - `phase16e_14b_int6_v4.cpp`: Changed `SIDECAR_SIZE` from `20 + M×4 + SIDECAR_PACKED` to `16 + M×4 + SIDECAR_PACKED`
  - Removed `*(uint32_t*)(buf + 16) = 0;` line
  - Changed `memcpy(buf + 20, ...)` to `memcpy(buf + 16, ...)`
  - Changed `memcpy(buf + 20 + M×4, ...)` to `memcpy(buf + 16 + M×4, ...)`
- Files changed: `examples/speculative/phase16e_14b_int6_v4.cpp`
- Binary rebuilt: `build/bin/phase16e_14b_int6_v4` (16:09 on May 9)

## G. Synthetic roundtrip
- **PASSED** — Dump tool on synthetic test (M=2, K=16, scales=[0.5, 1.0]):
  - File size: 48 bytes (correct)
  - scale[0]: 0.5000 ✓
  - scale[1]: 1.0000 ✓
  - q decoded: exact match ✓
  - size_match: true ✓

## H. Layer0 regenerated parity
- **File size:** 53,139,472 bytes ✓ (correct)
- **Scale stats:**
  - scale[0]: 0.00274243 (non-zero) ✓
  - nonzero: 13824/13824 ✓
- **GGUF mapping:** Phase 16D accesses GGUF tensor ROWS (flat positions r*K : r*K+K)
- **Full matvec cosine (vs GGUF ROWS):**
  - mean: **0.999234**
  - median: **0.999365**
  - min: **0.987199**
- **MAE:**
  - mean: 0.000733
  - median: 0.000691

| Sidecar | vs GGUF Mapping | Cosine Mean | MAE Mean |
|--------|---------------|------------|----------|
| v4 (old 20-byte header) | ROW (wrong header) | ~0.000005 | ~0.05 |
| v4 (old 20-byte header) | COL (wrong mapping) | ~0.999 | ~0.007 (false positive) |
| **New (fixed 16-byte header)** | **ROW (correct mapping)** | **0.999234** | **0.000733** |

## I. Layer1 result
Not generated (per user request: layer1 only if layer0 passes)

## J. Overall verdict
**PASS_SCHEMA_FIX_LAYER0**

The INT6 sidecar writer schema has been fixed to match runtime loader expectations. The 16-byte header format now produces numerically correct sidecars with cosine ~0.999 vs GGUF reference when using the correct mapping (GGUF tensor ROW r, not column r).

## K. Recommended next
- **Phase 16I: full 40-layer 14B INT6 sidecar generation** — The schema fix is verified, ready for full layer regeneration
- Alternatively: fix if any remaining schema issues found

## L. Models/sidecars/binaries staged?
- No new staging
- New layer0 at `/tmp/ffn_up_layer0_prt.int6` (not staged to repo)
- Old v4 sidecar at `/tmp/prt_phase16e_forensic/` (not staged)

## M. Secrets detected?
None

## N. Existing tags touched?
None

---

## Interpretation

**Is schema fixed?**
- YES — 16-byte header matches runtime loader

**Is v4/GGUF-direct writer now compatible with runtime schema?**
- YES — New sidecar parses correctly with dump tool, matches runtime loader's expected layout

**Is full 40-layer regeneration justified next?**
- YES — Layer0 shows 0.999 cosine. Other layers should be similar.

**What remains blocked?**
- Runtime canaries (not requested)
- 8-prompt validation (not requested)

---

## Files Changed
- `examples/speculative/phase16e_14b_int6_v4.cpp` — Schema fix
- `build/bin/phase16e_14b_int6_v4` — Binary rebuilt
- `examples/speculative/phase16g_int6_sidecar_dump.cpp` — Dump tool (unchanged, works correctly)