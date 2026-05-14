# PRT Phase 16G: INT6 Sidecar Format Schema Audit

**Date:** 2026-05-09
**Session:** sharp-nudibranch
**Branch:** phase10e-archived-20260430

---

## A. Branch
`phase10e-archived-20260430`

## B. Previous HEAD
See Phase 10E archive. No new commits in this phase.

## C. New HEAD
N/A — forensic only, no code changes.

---

## D. Runtime Schema Documented

**Source:** `tools/cli/cli.cpp` lines 688–950 (INT6 mmap and fread paths)

### Exact Runtime Loader Schema (16-byte header)

```
Offset  Size  Field
------  ----  -----
0       4     magic[4] = {'P','R','T','6'}
4       4     version  = 1  (uint32_t, native endian)
8       4     rows     = M  (uint32_t, native endian)
12      4     cols     = K  (uint32_t, native endian)
16      M*4   scales   (float32[M], one scale per row)
16+M*4  P     payload  (packed INT6, ceil(M*K/4)*3 bytes)

Total expected = 16 + M*4 + ceil(M*K/4)*3 bytes
```

**For 14B (M=13824, K=5120):**
- Scales: 13824 × 4 = 55,296 bytes
- Payload: ceil(70778880/4) × 3 = 53,084,160 bytes
- Total expected: 16 + 55296 + 53084160 = **53,139,472 bytes**
- Actual KG/v4 file: 53,139,476 bytes (+4 bytes)

### INT6 Packing Scheme

- 4 INT6 values → 3 bytes
- Offset-32 encoding: stored = (value + 32), range [0, 63]
- Bit layout per 3 bytes:
  - Byte 0: bits 0-5 = v0, bits 6-7 = v1[0:1]
  - Byte 1: bits 0-3 = v1[2:5], bits 4-7 = v2[0:3]
  - Byte 2: bits 0-1 = v2[4:5], bits 2-7 = v3[0:5]
- Row-major traversal (consecutive groups of 4 elements)
- Decoded range: [-32, 31]
- LUT used in runtime: `lut[ui] = ui - 32` for ui ∈ [0, 63]

### Unpack Algorithm (C++)
```cpp
int8_t lut[64];
for (int ui = 0; ui < 64; ui++) lut[ui] = (int8_t)(ui - 32);
const uint8_t * p_src = packed_src;
int64_t i = 0;
for (; i + 15 < total; i += 16) {
    // 4x unrolled groups of 4
    uint8_t b0 = *p_src++; uint8_t b1 = *p_src++; uint8_t b2 = *p_src++;
    int8_data[i]     = lut[b0 & 0x3F];
    int8_data[i+1]  = lut[((b0 >> 6) | ((b1 & 0x0F) << 2)) & 0x3F];
    int8_data[i+2]  = lut[((b1 >> 4) | ((b2 & 0x03) << 4)) & 0x3F];
    int8_data[i+3]  = lut[(b2 >> 2) & 0x3F];
    // ... 3 more groups ...
}
```

---

## E. Dump Tool Created

**Source:** `examples/speculative/phase16g_int6_sidecar_dump.cpp`
**Binary:** `build/bin/phase16g_int6_sidecar_dump`

Tool parses sidecar **exactly** as runtime loader does:
- Reads 16-byte header (magic at 0, version at 4, rows at 8, cols at 12)
- Reads scales at offset 16 (M × float32)
- Reads payload at offset 16 + M*4
- Unpacks INT6 using identical LUT + bit-shuffle decode
- Computes SHA256 checksum of decoded int8 array
- Outputs JSON with full diagnostics

### Synthetic Test Results

**Correct 16-byte file (RT):**
```
scales: [0.5000, 1.0000] ✓
q decoded: exact match ✓
size_match: true ✓
```

**Wrong 20-byte file (KG style):**
```
scales: [0.0000, 0.5000] ← NULL scale[0]!
size_match: false (52 vs 48 expected) ← +4 bytes
q decoded: garbage (misaligned payload)
```

---

## F. Synthetic Roundtrip Result

**Status: PARTIAL PASS**

The dump tool correctly parses 16-byte files and detects 20-byte files as malformed.

Synthetic test (M=2, K=16, scales=[0.5, 1.0]):
- Dump tool with 16-byte header: **PASS** — scales correct, q exact, size exact
- Dump tool with 20-byte header: **FAIL** — scale[0]=0.0000, size mismatch, payload misaligned

The 4-byte padding in the writer's 20-byte header causes:
1. Scale[0] reads as 0.00000000 (null) — padding bytes interpreted as scale
2. Payload starts 4 bytes before actual data — all decoded values garbage
3. File is 4 bytes larger than expected

---

## G. Header Size Confirmed

| Component | Runtime Expects | Writer Produces | Match? |
|-----------|----------------|-----------------|--------|
| Header | **16 bytes** | **20 bytes** | **NO** |
| Scale offset | byte 16 | byte 20 | NO (+4) |
| Payload offset | 16+M*4 | 20+M*4 | NO (+4) |
| File size (14B) | 53,139,472 | 53,139,476 | NO (+4) |

**Root cause:** Writer (Python v2 and C++ v4) uses 20-byte header with `clear` field at bytes 16-19. Runtime loader expects no `clear` field — scales start immediately at byte 16.

---

## H. Scale Offset Confirmed

- **Runtime:** scale[0] at byte 16, scale[m] at byte 16 + m*4
- **Writer:** scale[0] at byte 20, scale[m] at byte 20 + m*4 (if reader used correct offsets)
- **Effect:** Reader interprets padding/null-bytes as scale[0]=0.00000000

---

## I. Payload Offset Confirmed

- **Runtime:** payload at byte 16 + M*4
- **Writer:** payload at byte 20 + M*4
- **Effect:** Reader starts reading payload 4 bytes before actual data. All decoded values are garbage.

---

## J. Layer0 Real Roundtrip Parity

**Reference:** `/tmp/p16c_layer0_float.bin` (GGUF ffn_up dequantized, M=13824, K=5120)

**Mapping:** Phase 16D/16E extraction produces tensor reshaped as (M, K) row-major. In GGUF terms, this means sidecar row r = GGUF tensor column r (TRANSPOSED layout). Validation must compare against GGUF flat at positions [r, r+M, r+2M, ...].

### v4 extraction (with corrected 20-byte offsets for decode)
| Metric | Value | Gate | Result |
|--------|-------|------|--------|
| Cosine mean | **0.999210** | ≥0.995 | ✅ PASS |
| Cosine median | 0.999323 | — | — |
| Cosine min | 0.993107 | — | — |
| Cosine 5th pct | 0.998532 | — | — |
| MAE mean | **0.000729** | <0.01 | ✅ PASS |
| MAE median | 0.000686 | — | — |

### KG sidecar (Phase 16D reference, with corrected 20-byte offsets)
| Metric | Value | Gate | Result |
|--------|-------|------|--------|
| Cosine mean | **0.000109** | ≥0.995 | ❌ FAIL |
| Cosine median | 0.000084 | — | — |
| Cosine min | -0.056445 | — | — |
| Cosine 5th pct | -0.022732 | — | — |
| MAE mean | **0.049346** | <0.01 | ❌ FAIL |

**Conclusion:** v4 extraction is **numerically correct** against GGUF (cosine ~0.999, MAE ~0.0007). The previous near-zero cosine readings were due to incorrect transpose mapping in the Python validator, NOT extraction failure.

KG sidecar is **completely broken** vs GGUF — cosine near zero, MAE ~0.05. The Phase 16D extraction produced fundamentally different values from the GGUF reference, independent of header format.

---

## K. Root Cause

### Schema Mismatch (Writer vs Runtime)

| | Runtime Loader | Writer |
|--|--|--|
| **Header** | 16 bytes | 20 bytes |
| **Scale offset** | 16 | 20 |
| **Payload offset** | 16+M*4 | 20+M*4 |
| **File size (14B)** | 53,139,472 | 53,139,476 |

The writer writes a 4-byte `clear`/`reserved` field that the runtime loader does not expect. This causes:
1. scale[0] reads as 0.0 (padding/null bytes)
2. All subsequent scales shifted by 1 index
3. Payload pointer 4 bytes before actual data
4. Every decoded row is garbage

### Fix Required

**Writer (Python v2, C++ v4):** Remove the 4-byte `clear` field. Write 16-byte header:
```cpp
// CORRECT (16-byte):
header[0:4] = b'PRTG';      // or b'PRT6' per runtime
struct.pack_into('<I', header, 4, 1);   // version
struct.pack_into('<I', header, 8, M);  // rows
struct.pack_into('<I', header, 12, K); // cols
// scales at offset 16, payload at offset 16+M*4

// WRONG (current, 20-byte):
header[0:4] = b'PRTG';
struct.pack_into('<I', header, 4, 1);
struct.pack_into('<I', header, 8, M);
struct.pack_into('<I', header, 12, K);
struct.pack_into('<I', header, 16, 0);  // ← REMOVE THIS
// scales at offset 20, payload at offset 20+M*4 ← MISMATCH
```

### KG Sidecar Additional Failure

Even with header fix, KG sidecar produces near-zero cosine vs GGUF. The extraction approach used in Phase 16D produced a fundamentally different tensor from the GGUF reference. Possible causes:
1. Wrong tensor extracted (ffn_up vs ffn_down vs different layer)
2. Dequantization using different scale computation
3. Transpose interpretation mismatch
4. Quantization parameters (PRESCALE, clip range) different from GGUF's internal Q4_K

**KG sidecar is not a reliable numeric reference.**

---

## L. Verdict

| Verdict | Status |
|---------|--------|
| PASS_SCHEMA_ROUNDTRIP | ❌ **BLOCKED** — writer schema mismatch prevents roundtrip |
| FAIL_HEADER_MISALIGNMENT | ✅ **CONFIRMED** — 4-byte offset |
| FAIL_PACKING_ORDER | ❌ **NOT CONFIRMED** — packing order matches runtime |
| FAIL_SCALE_OFFSET | ✅ **CONFIRMED** — scales at byte 20, not byte 16 |
| FAIL_RUNTIME_READER_MISMATCH | ✅ **CONFIRMED** — 20-byte writer vs 16-byte reader |
| BLOCKED_TOOLING | ❌ Not blocked — dump tool works correctly |

---

## M. Recommended Next Steps

### Must Fix (before any runtime testing)
1. **Fix writer schema:** Remove 4-byte `clear` field from Python v2 and C++ v4 writers. Use 16-byte header matching runtime loader.
2. **Verify synthetic roundtrip:** After writer fix, re-run synthetic test with dump tool. Must pass with scale[0]=expected, q exact, size exact.
3. **Verify v4 real extraction parity:** With header fix, v4 extraction should achieve cosine ≥ 0.999 against GGUF reference (already confirmed with corrected offsets, just need correct header).
4. **Stop using KG as reference:** KG sidecar is broken regardless of header format.

### Do Not Do Yet
- Do not generate all 40+ sidecars until header fix is verified
- Do not run layer1 forensic (blocked per user request)
- Do not run runtime canaries
- Do not run 8-prompt validation
- Do not claim 14B PRT works end-to-end

---

## N. Models/Sidecars/Binaries Staged?
No new staging. Existing artifacts:
- `/tmp/prt_test_runtime/` (KG sidecar) — broken, do not trust
- `/tmp/prt_phase16e_forensic/` (v4 sidecar) — numerically valid but header malformed
- `/tmp/p16c_layer0_float.bin` — GGUF reference, valid

## O. Secrets Detected?
None.

## P. Existing Tags Touched?
None.

---

## Summary

The INT6 sidecar writer produces a 20-byte header but the runtime loader expects 16 bytes. This 4-byte mismatch causes scale[0] to be read as null (0.0), all scales to be misaligned by one index, and the payload pointer to be 4 bytes off. File is 4 bytes too large. **Fix: remove the 4-byte `clear`/`reserved` field from the writer header.**

v4 extraction (C++ with ggml::to_float) is **numerically correct** against the GGUF reference (cosine ~0.999, MAE ~0.0007) once the transpose mapping is accounted for. The previous near-zero cosine readings were a validator bug, not an extraction bug.

KG sidecar from Phase 16D is **completely broken** against GGUF (cosine ~0.0001) — unrelated to header format. Cannot be used as numeric reference.
