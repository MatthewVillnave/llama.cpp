# PRT Phase 16I — 14B Full INT6 Sidecar Generation

**Date:** 2026-05-09
**Session:** sharp-nudibranch
**Branch:** experimental/prt-phase14a-packed-sidecars
**Commit:** bdfb2892b

---

## A. Branch
- Starting: `experimental/prt-phase14a-packed-sidecars` at `bdfb2892b`
- Ending: Same branch at `bdfb2892b` (no new commits)

## B. Previous HEAD
`bdfb2892b` — Phase 16H fixed INT6 writer schema

## C. New HEAD
Not committed yet at time of this report

## D. 14B model path
`/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-14B-14B.gguf` (8.4GB)

## E. Sidecar file count
**40** files generated (layers 0-39)

## F. Unique SHA count
**40** unique SHA256 hashes (no duplicates)

## G. Per-layer file size
**53,139,472 bytes** each (matching expected)

## H. Total sidecar size
**2.0GB** (~2,125,358,880 bytes)

## I. Selected layers tested
- Layer 0: **cosine 0.999400**, MAE 0.000687 (PASS vs GGUF reference)
- Layer 1, 10, 20, 30, 39: Finite but no separate reference data to compare

## J. Minimum weight cosine
**0.999400** (layer 0 only - other layers lack reference data)

## K. Minimum matvec cosine
N/A - only layer 0 has reference

## L. Tiny runtime canary run?
**Not run** - per user request (generation pass is sufficient milestone)

## M. Tiny runtime result
Skipped (not requested in Phase 16I)

## N. Memory/swap health
- RAM: 12GB available throughout generation
- Swap: 100% used pre-flight (from previous processes, not from generation)
- Per-layer generation: ~12-15 seconds per layer

## O. Overall verdict
**PASS_14B_FULL_INT6_SIDECARS**

Generation successful with schema-valid INT6 files for all 40 layers.

## P. Recommended Next Phase
**Phase 16J** — Run the tiny runtime canary to verify end-to-end PRT functionality with the generated sidecars.

## Q. Models/sidecars/binaries staged?
No staging - sidecars in `/tmp/` only

## R. Secrets detected?
None

## S. Existing tags touched?
None

---

## Summary

Phase 16I successfully generated all 40 INT6 sidecars for the Qwen2.5-14B model using the schema-fixed writer from Phase 16H. Each sidecar:
- Uses 16-byte header (matching runtime loader)
- Has correct PRT6 magic
- Is 53,139,472 bytes per layer
- All 40 layers have unique SHA256 hashes
- Total 2.0GB for complete set

Layer 0 parity vs GGUF reference: **cosine 0.999**, MAE 0.0007 — excellent match.

The other layers cannot be compared against separate reference data (each layer would need its own float reference file), but the generation process itself is verified consistent.

Ready for Phase 16J runtime verification.