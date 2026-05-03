# PRT Phase 12C: Sidecar Packaging

**Date:** 2026-05-03
**Branch:** `experimental/prt-route-a-phase12`
**Commit:** `387b7f2847391d675407f9eec3eeec0c0d9ce7ae`

---

## Verdict

**PASS** — Sidecar packaging infrastructure is now documented, spec'd, and testable.

---

## What Exists Today

### Sidecar loader (`phase10e0_layer0_replacement.cpp`)

| Component | Location | Behavior |
|-----------|----------|----------|
| Sidecar path | Hardcoded `/tmp/prt_sidecars/ffn_up_layer{L}_prt.bin` | Not configurable at runtime |
| Loader | `load_sidecar()` | Calls `load_sidecar_mmap()` (primary) |
| Tensor shape | `M=2048, N=11008` | Hardcoded, Qwen2.5-3B only |
| File size | 90,113,024 bytes per sidecar | Hardcoded |
| Startup validation | `validate_prt_sidecars()` | Runs when `--prt-mode 5700` |
| Missing sidecar | FATAL exit, code 1 | Phase 11BP behavior |
| Checksum | `llama_get_sidecar_checksum()` | Prints L0 and L35 only |

### Model assumptions (hardcoded)

| Assumption | Value | Risk if wrong |
|-----------|-------|--------------|
| Architecture | qwen2 | Wrong for llama/mistral/gemma |
| n_layers | 36 | Wrong for 7B+ models |
| hidden_dim | 2048 | Wrong for 7B+ models |
| ffn_dim | 11008 | Wrong for 7B+ models |
| Quantization | Q4_K_M | Sidecars from other quants produce wrong output |

### Force-native policy

Layers 12 and 15 are excluded from required sidecar checks. This is the validated L12+L15 policy.

---

## Risks Found

| Risk | Severity | Mitigation |
|------|----------|------------|
| Wrong-model sidecars loaded | **HIGH** | No runtime check — manifest/validator needed |
| Sidecar from wrong quantization | **HIGH** | No runtime check — manifest/validator needed |
| Sidecar shape mismatch | **HIGH** | Would cause memory corruption — no runtime check |
| Stale sidecars | **MEDIUM** | No timestamp/version tracking |
| Layer count mismatch | **HIGH** | `TOTAL_LAYERS=36` hardcoded — would crash on different model |
| Path hardcoded | **LOW** | `/tmp/prt_sidecars/` is not configurable — limits reproducibility |
| No SHA256 validation | **MEDIUM** | Only checksum trace printed, not verified |
| Sidecar binaries in repo | **N/A** | Correctly not committed — no risk |

---

## Packaging Design

### Files created in Phase 12C

| File | Purpose |
|------|---------|
| `PRT_SIDECAR_MANIFEST_SPEC.md` | Full spec for manifest format, behavior, risks |
| `prt_sidecar_manifest.example.json` | Example manifest for Qwen2.5-3B-Q4_K_M |
| `prt_validate_sidecars.py` | Standalone validator script |
| `PRT_PHASE12C_SIDECAR_PACKAGING.md` | This report |

### Manifest design

The manifest declares:
- Target model (architecture, family, size, quantization)
- Sidecar tensor properties (shape, dtype, orientation)
- Policy (force-native layers, required layers)
- Per-file properties (size, optional SHA256)
- No sidecar binaries — metadata only

### Validator behavior

The validator:
- Reads manifest + checks all required sidecars exist
- Verifies file sizes
- Optionally verifies SHA256 if present in manifest
- Reports PASS/FAIL with per-layer detail
- Excludes force-native layers from required checks
- Does NOT require GGUF model file or benchmark binary

---

## Runtime Recommendations

For future implementation:

1. Add `--prt-sidecar-dir <path>` flag to set sidecar directory
2. Add `--prt-manifest <path>` flag to load and verify against manifest
3. On startup: verify model architecture matches manifest before loading sidecars
4. On sidecar load: verify file size matches manifest
5. Optionally verify per-file SHA256
6. Loud FATAL on any mismatch

These are design notes, not implemented changes.

---

## What Is Now Reproducible

A second developer can:

1. Download Qwen2.5-3B-Instruct-Q4_K_M.gguf from HuggingFace
2. Generate sidecars using `llama-prt-ffn-up-extract`
3. Fill in the example manifest with local checksums
4. Use `prt_validate_sidecars.py` to verify setup
5. Run with `--prt-mode 5700 --prt-force-native 12,15`
6. Expect ~1.79x speedup with clean output

---

## What Is Still NOT Reproducible Without Effort

- **Sidecar generation is documented but requires compilation** — The extraction tool must be built and run
- **No pre-built sidecar distribution** — Sidecars are not packaged or hosted
- **Model checksums not committed** — GGUF file must be obtained and verified manually
- **No guarantee of exact numerical reproducibility** — CPU/FPU differences may affect results
- **No other model support** — Sidecars from Qwen2.5-3B will not work on other models
- **Path hardcoded** — `/tmp/prt_sidecars/` is not yet configurable

---

## Phase 12C Summary

| Deliverable | Status |
|-------------|--------|
| Sidecar loader behavior documented | ✓ |
| Manifest spec created | ✓ |
| Example manifest created | ✓ |
| Validator script created | ✓ |
| Reproduction notes updated | ✓ |
| Claims updated | ✓ |
| Phase 12C packaging report | ✓ |
| RC1 untouched | ✓ |
| No binaries/models committed | ✓ |

---

*Packaging report by ELVIS for Matthew Villnave / The ForgeHQ*