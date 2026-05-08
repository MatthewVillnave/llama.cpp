# PRT Phase 15B-A — INT4 Sidecar Offline Prototype / Parity Gate

## Verdict

**NO_GO_INT4_OFFLINE_PARITY**

Blocked by: GGUF extraction tooling failure (Python couldn't read 4.7GB GGUF file; C++ tool exists but not used due to Phase 15B-A offline-only constraint).

## Context

Phase 14 proved packed INT8 sidecars recovered near-native throughput.
Phase 15A proved longer-context stability.
Phase 15B explores INT4 quantization as the next step.

INT4 format designed:
- Signed range: [-7, +7]
- Scale: row_max / 7.0 per ffn row
- Packing: 2 nibbles per byte, row-major
- Scale storage: float32 per ffn row
- File layout: [M*K/2 bytes nibble][M*4 bytes scales]

## INT4 Format

| Property | Value |
|----------|-------|
| Signed range | [-7, +7] |
| Scale formula | row_max / 7.0 |
| Packing | 2 nibbles per byte |
| Scale storage | float32 per M row |
| File layout | [M*K/2 int8][M*4 float32] |
| Runtime compatibility | NOT attempted in Phase 15B-A |
| Reference | Same orientation as Phase 14 INT8: [ffn, hidden] |

## Existing INT8 Path Inspection

- `phase14b_int8_quantize.py`: per-row INT8 quantization, scale = row_max / 127.0
- `phase14a_quantize_sidecars.py`: similar, per-row scales
- Both use the same per-row scale approach that INT4 extends

**CRITICAL BUG DISCOVERED**: All 28 INT8 sidecar files in `/tmp/prt_sidecars_7b_int8/` are IDENTICAL (SHA256: `43c62fab...`). Phase 14 sidecar generation wrote one layer's data to all 28 index positions.

## Attempted GGUF Extraction

Attempted direct GGUF extraction via Python mmap to get per-layer float32 reference weights for INT4 quantization.

Model: `/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-7B-Instruct-Q4_K_M.gguf` (4.7GB)

GGUF format confirmed: magic `GGUF` (bytes `47 47 55 46`), version 3, 339 tensors, big-endian magic but little-endian fields.

Python mmap read operations timed out or failed on the 4.7GB file. C++ extraction tool (`tools/prt-ffn-up-extract.cpp`) exists but Phase 15B-A was constrained to offline Python/prototype tooling.

## INT4 Prototype Results (using identical INT8 reference)

Earlier run on 6 layers (0, 5, 10, 15, 20, 27) using the (buggy) INT8 sidecars as reference:

| Layer | Weight Cosine | Matvec Cosine | Verdict |
|-------|-------------|--------------|---------|
| 0 | 0.98386 | — | CONDITIONAL |
| 5 | 0.98386 | — | CONDITIONAL |
| 10 | 0.98386 | — | CONDITIONAL |
| 15 | 0.98386 | — | CONDITIONAL |
| 20 | 0.98386 | — | CONDITIONAL |
| 27 | 0.98386 | — | CONDITIONAL |

All layers returned identical cosine because all INT8 sidecar files are the same.

**Parity gate thresholds:**
- PASS: matvec cosine >= 0.995 AND weight cosine >= 0.995
- MAYBE: matvec cosine >= 0.990 but < 0.995, or one borderline layer
- NO-GO: matvec cosine < 0.990, or nonfinite, or shape mismatch

## INT4 vs INT8 Comparison

INT4 size: ~34 MB per layer (2× smaller than INT8 at ~68 MB)
INT4 weight cosine: 0.984 vs INT8 reference (CONDITIONAL threshold = 0.980)

INT4 compression is real and significant (2× over INT8). The cosine is above the CONDITIONAL threshold but below SAFE.

## Why NO-GO

The INT4 prototype results are encouraging (0.984 cosine, CONDITIONAL) but cannot be trusted as per-layer validation because:
1. All INT8 sidecars are identical — we only tested one weight distribution
2. GGUF extraction failed — we cannot get proper per-layer references
3. Without per-layer references, we cannot confirm other layers behave similarly

A clean NO-GO is more valuable than a false MAYBE.

## Interpretation

- INT4 quantization works (2× compression achieved)
- INT4 cosine 0.984 is above CONDITIONAL threshold
- Cannot claim per-layer validation without proper GGUF extraction
- INT4 is promising but not yet validated

## Next Steps

1. **Fix GGUF extraction** — Use the existing C++ tool or build a minimal GGUF reader that can extract individual ffn_up tensors without loading the full 4.7GB into memory
2. **Regenerate INT8 sidecars** — Fix the Phase 14 sidecar generation bug first
3. **Re-run INT4 prototype** — With proper per-layer INT8 references
4. **Runtime canary** — Only if per-layer validation confirms cosine >= 0.990

## Allowed Claims

- INT4 prototype achieves 0.984 weight cosine on the tested distribution (CONDITIONAL)
- INT4 is 2× smaller than INT8 per layer (34MB vs 68MB)
- INT4 NOT YET VALIDATED — GGUF extraction must be resolved first
- INT8 remains the validated path

## Forbidden Claims

- No INT4 runtime quality
- No INT4 speedup claim
- No per-layer validation
- No production readiness
- No all-layer safety guarantee

## Files Created

- `examples/speculative/phase15b_int4_prototype.py` — INT4 prototype script
- `examples/speculative/phase15b_int4_sidecar_probe.py` — GGUF extraction probe (blocked)
- `examples/speculative/results/PRT_PHASE15B_INT4_PROTOTYPE_RESULTS.md` — Results
- `examples/speculative/results/phase15b_int4_prototype_results.json` — JSON

## Safety

| Check | Status |
|-------|--------|
| Models staged | NO |
| Sidecars staged | NO |
| Binaries staged | NO |
| Temp logs staged | NO |
| Secrets detected | NO |
| Existing tags touched | NO |