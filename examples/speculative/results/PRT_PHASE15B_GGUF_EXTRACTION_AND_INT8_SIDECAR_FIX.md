# PRT Phase 15B-B — GGUF Extraction + 7B INT8 Sidecar Uniqueness Fix

## Verdict

**BLOCKED_GGUF_EXTRACTION**

The Phase could not complete due to Python GGUF extraction failures. More investigation needed before fresh sidecar regeneration is possible.

## Context

This phase was launched to fix the Phase 15B-A blocker (INT4 offline parity). The blocker was that all 28 INT8 sidecar files in `/tmp/prt_sidecars_7b_int8/` are identical, giving us only one layer distribution to test against.

The goal was to:
1. Extract proper per-layer FFN_UP weights from the 7B GGUF file
2. Regenerate fresh unique 7B INT8 sidecars
3. Run selected-layer parity checks
4. Enable proper INT4 validation

## Current Sidecar Audit (Complete)

Confirmed findings:
- Directory: `/tmp/prt_sidecars_7b_int8/`
- File count: 28 files
- Unique SHA256: 1 out of 28 (all identical)
- File size: 67,971,072 bytes each (~65 MB)
- Filename pattern: `ffn_up_layer[N]_prt.int8` for N = 0-27
- All files are byte-identical

## Root Cause Investigation

We identified the likely root cause but could not confirm it via code inspection because the original extraction code is missing or inaccessible:

**Hypothesis:** The Phase 14/13 sidecar generator wrote one layer's data (likely layer 0) to all 28 output positions, OR the float32 source sidecars were all identical when the INT8 quantization was run.

Evidence:
1. All 28 INT8 files have identical SHA256
2. Float32 sidecar directory `/tmp/prt_sidecars_7b/` does not exist
3. The phase13ac_gen_3b_sidecars.py script is for 3B (different Q4_K block size)
4. No 7B-specific sidecar generator found in the repo

## GGUF Extraction Attempts (Failed)

Multiple attempts to build a proper Python GGUF extractor all failed:

1. **Direct file read with mmap**: 4.7GB file caused MemoryError on metadata parse
2. **Chunked I/O approach**: Variable-length key encoding was incorrect
3. **Fixed varint parsing**: Still failing on metadata section
4. **C++ tool compile**: Could not resolve ggml/gguf static library linking

Root cause of failures: The GGUF metadata section uses variable-length encoding for keys and values, and the 4.7GB file size makes the Python approach slow and memory-intensive.

## What We Could Not Do

Because extraction failed:
- ❌ Regenerate fresh unique INT8 sidecars for all 28 layers
- ❌ Run selected-layer parity checks
- ❌ Confirm per-layer weights are correct
- ❌ Validate INT4 quantization properly
- ❌ Run runtime canary

## Impact

**On Prior Claims:** Phase 14/15 claims remain PROVEN-LIMITED pending extraction fix. The runtime tests passed, but we cannot confirm the sidecars were truly layer-unique.

**On INT4:** Cannot re-run INT4 offline parity until extraction is fixed.

## Recommended Next Steps

1. **Fix GGUF extraction** — Build a working C++ extraction tool or properly implement varint parsing in Python
2. **Create 7B-specific sidecar generator** — Based on the 3B script but with correct Q4_K block size (256 vs 128)
3. **Verify all 28 layers are unique** — Run SHA256 audit on regenerated sidecars
4. **Re-run INT4 parity** — With proper per-layer references
5. **Runtime canary** — Only after offline parity passes

## Alternate Approach

If GGUF extraction remains blocked, consider:
- Using the llama-cli binary to export all tensors directly
- Using the existing `tools/prt-ffn-up-extract.cpp` if library linking can be resolved
- Generating test sidecars from a synthetic distribution to verify the parity pipeline works

## Files Created This Phase

- `examples/speculative/phase15b_gguf_ffn_up_parity.py` — Failed Python GGUF reader
- `examples/speculative/phase15b_gguf_to_int8_pipeline.py` — Failed pipeline script
- `examples/speculative/phase15b_simple_gguf_extract.py` — Failed simpler approach
- `examples/speculative/phase15b_gguf_test.py` — Debug attempt

## Safety

| Check | Status |
|-------|--------|
| Models staged | NO |
| Sidecars staged | NO |
| Binaries staged | NO |
| Temp logs staged | NO |
| Secrets detected | NO |
| Existing tags touched | NO |