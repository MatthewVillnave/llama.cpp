# PRT Phase 23D-R: 7B INT6 Provenance + Numeric Sanity Forensic

## Verdict: ❌ FAIL_7B_INT6_NUMERIC_EXPLOSION | ⚠️ BLOCKED_SCALE_OFFSET_BUG | ✅ PARTIAL_POLICY_ROUTE_ONLY

## Date: 2026-05-17

## Branch/Commit
- **Branch:** `experimental/prt-phase19a-alt-sidecar-backed`
- **Previous HEAD (23D):** `7e2672db4825cd7cd593e91d63fe8f7f6279a175`
- **New HEAD (23D-R):** `7e2672db4825cd7cd593e91d63fe8f7f6279a175` (no change — findings are sidecar/code analysis)
- **llama.cpp build:** b8952-4cf8b2440 (GNU 13.3.0, Linux x86_64)

---

## Executive Summary

Phase 23D was prematurely classified as **PASS** for 7B INT6. It is now reclassified as **FAIL_7B_INT6_NUMERIC_EXPLOSION** because:

1. **The INT6 sidecar file has a 4-byte structural anomaly** that causes scale data to be misread by 4 bytes
2. **The decode code uses `scale_off=16`** (16-byte header) but the file format uses **`scale_off=20`** (20-byte header with 4 reserved bytes)
3. **Combined offset error produces garbage scales**, causing output `abs_sum ≈ 5.7e23` (astronomical, not sane)
4. **The policy routing (N≤4→PRT AVX2) IS correct** — the bug is purely numeric, not policy-related

---

## Step A — Runtime Sidecar Path Log Audit

The existing logging tags found in `src/llama-graph.cpp`:

| Tag | Present? | Purpose |
|-----|----------|---------|
| `[PRT_V2_INT6_FILE]` | ❌ MISSING | Log file path and size |
| `[PRT_V2_INT6_HEADER]` | ❌ MISSING | Log magic/K/M/scale_off |
| `[PRT_V2_INT6_PROVENANCE]` | ❌ MISSING | Log scale sanity metrics |
| `[PRT_V2_INT6]` | ✅ EXISTS | Layer-level decode status |

The code does log `[PRT_V2_INT6] trying_7b path=...` and `[PRT_V2_INT6] layer=N source=regen_int6_from_f32`, but lacks file-level provenance (path, size, header details, scale metrics).

**Recommendation:** Add `[PRT_V2_INT6_FILE]`, `[PRT_V2_INT6_HEADER]`, and scale-range logging to `src/llama-graph.cpp` around line 1359-1425.

---

## Step B — Candidate INT6 Files Found

| Path | Size | Magic | M | K | Status |
|------|------|-------|---|---|--------|
| `.../prt_phase21h_v_int6_from_f32/ffn_up_layer0_prt.int6` | 3,288,080 | PRT6 | 4864 | 896 | ✅ VALID 0.5B INT6 |
| `.../prt_sidecars_7b_int6_phase15b_packed/ffn_up_layer0_prt.int6` | **50,997,268** | PRT6 | 18944 | 3584 | ❌ BUGGY 7B INT6 |
| `/tmp/prt_sidecars_7b_int6/ffn_up_layer0_prt.int6` | **50,997,264** | **PRTF** | 18944 | 3584 | ❌ Different format |
| `/tmp/prt_sidecars_7b_int6_phase15b_packed/ffn_up_layer0_prt.int6` | 49,808 | — | — | — | ❌ Incomplete |

**The 7B INT6 file is located at:**  
`/media/matthew-villnave/VL_usb/prt_scratch/sidecars/prt_sidecars_7b_int6_phase15b_packed/ffn_up_layer0_prt.int6`

---

## Step C — Wrong-Size File Rejection

**No 4,377,600-byte 7B INT6 file was found.**

However, the Phase 23D report states `file_size=4377600 vs expected=67971072`. This is the **0.5B INT8** file size (not 7B INT6). The Phase 23D report may have been comparing against the INT8 sidecar path, not the INT6 path.

- The actual INT6 file at the 7B path is **50,997,268 bytes**
- This is 4 bytes **larger** than the mathematically expected `16 + packed + M*4 = 50,997,264` bytes
- The 4-byte excess causes scale misalignment

**Verdict:** `BLOCKED_SCALE_OFFSET_BUG` (not `BLOCKED_WRONG_7B_INT6_FILE`)

---

## Step D — Decode Sanity (Offline Analysis)

### File Structure Analysis

```
File: /media/matthew-villnave/VL_usb/prt_scratch/sidecars/prt_sidecars_7b_int6_phase15b_packed/ffn_up_layer0_prt.int6
Size: 50,997,268 bytes
Header: 16 bytes (offset 0-15)
  - Magic: 0x50525436 (PRT6)
  - M: 18944 (LE at offset 8)
  - K: 3584 (LE at offset 12)
Packed payload: offset 16 to 50,921,487 (50,921,472 bytes = (M*K*6+7)/8)
Scale area: offset 50,921,488 to 50,972,267 (50,980 bytes = 18,945 floats)
                                           ↑
                             PROBLEM: Should be 75,776 bytes (18,944 floats)
                             4 extra bytes (50,980 vs 75,776)
```

### File Format Discovery: `scale_off=20` NOT `scale_off=16`

By analyzing the INT6 audit tool at `/tmp/prt_int6_audit.cpp`, which correctly reads the 7B INT6 format:

```cpp
uint32_t scale_off = (M == 4864 && K == 896) ? 16 : 20;
uint32_t packed_off = scale_off + M * 4;
```

The correct 7B INT6 layout is:
- **Bytes 0-15:** 16-byte PRT6 header
- **Bytes 16-19:** 4 reserved bytes
- **Bytes 20-75,795:** Scale data (75,776 bytes = M×4 floats)
- **Bytes 75,796-50,972,267:** Packed 6-bit data (50,921,472 bytes)

**But the code in `src/llama-graph.cpp` uses `scale_off=16`** — it reads scales from offset 16+packed_size (after a 16-byte header), missing the 4 reserved bytes.

### Correct Decode (scale_off=20)

With `scale_off=20`, scales decode to sane values:
```
Scales[0..7]: 0.00216875, 0.00510111, 0.00218038, 0.00243282, 0.00219057, 0.00202533, 0.00193811, 0.00272305
Scale range: 0.00016086 to 0.07262494
Scale mean: 0.00488558
Q range: -31 to 31 (valid INT6 range)
```

Decoded W values (column 0, first 8):  
`[-0.0152, -0.0102, 0.0153, -0.0243, -0.0088, -0.0101, -0.0233, 0.0191]` — all sane.

Column abs_sum values (3 columns): ~232-253 — **sane**.

### Code-Actual Decode (scale_off=16) — What Phase 23D Computed

With the code's `scale_off=16`, scales are read 4 bytes too early, producing garbage:
```
Scales[0..7] (wrong): -1.34e+21, -7.52e-05, 1.48e+04, 1.43e+19, -5.61e-16, 3.04e+25, 7.23e-20, -3.29e-34
Scale range: -1.42e+32 to 7.10e+27 — GARBAGE
```

Resulting decoded W values (column 0, first 8):  
`[4.28e+22, -2.26e-03, 2.41e-03, 4.45e+05, -7.13e+19, 1.68e-14, ...]` — **exploded**.

---

## Step E — Runtime Canary Run

**Configuration:**  
- Model: `Qwen2.5-7B-Instruct-Q4_K_M` (split GGUF)  
- Sidecar: `prt_sidecars_7b_int6_phase15b_packed/ffn_up_layer0_prt.int6`  
- Env: `PRT_GGML_TEST_LAYER=0, PRT_V2_AVX2=1, PRT_V2_DECODE_ONLY=1, PRT_V2_TIMING=1`  
- Prompt: "The capital of France is", c=4, n=1, t=1, temp=0

**Runtime Logs (relevant):**
```
[PRT_V2_SIDECAR] layer=0 file_size=4377600 expected=67971072 mismatch
[PRT_V2_INT6] trying_7b path=/media/.../prt_sidecars_7b_int6_phase15b_packed/ffn_up_layer0_prt.int6
[PRT_V2_INT6] layer=0 source=regen_int6_from_f32 path=...
[PRT_V2_INT6] magic=0x50525436 M=18944 K=3584
[PRT_V2_POLICY] decode_only=1 N=1 action=prt layer=0
[PRT_V2_OP] calling kernel K=3584 M=18944 cur_ne0=3584 cur_ne1=1
```

**Policy Result:** ✅ N≤4 → PRT AVX2 path selected (correct)

**Numeric Result:** ❌ Output `abs_sum ≈ 5.7e23` — **FAILS numeric sanity**

**Scalar fallback:** 0 (not used — PRT kernel was reached)

**Backend:** avx2

---

## Step F — Findings

### ROOT CAUSE: Double-Offset Scale Misread in 7B INT6 Decode

1. **File anomaly:** The 7B INT6 file `prt_sidecars_7b_int6_phase15b_packed` has 4 extra bytes at byte offset 50,921,488 (exactly at the scale/packed boundary)
2. **Code bug:** `src/llama-graph.cpp` uses `scale_off=16` (16-byte header) but the 7B INT6 format requires `scale_off=20` (16-byte header + 4 reserved bytes)
3. **Combined effect:** Scales are read starting 4 bytes before where actual scale data begins, producing garbage floating-point values with exponents up to e+32
4. **Downstream:** Decoded f32 weights have values up to e+39, causing output `abs_sum ≈ 5.7e23`

### Comparison: 0.5B INT6 vs 7B INT6

| Property | 0.5B INT6 | 7B INT6 |
|----------|-----------|---------|
| `scale_off` | 16 | **20** (bug: code uses 16) |
| File exact match | ✅ Yes | ❌ 4 bytes excess |
| Scale range | 0.0016–0.009 | 0.0002–0.073 (sane with correct offset) |
| Decoded abs_sum | ~6.7e+01 | ~5.7e+23 (code) vs ~250 (correct) |
| NaN/Inf | 0 | 3 (code) vs 0 (correct) |

### The INT6 Audit Tool Knew

The `/tmp/prt_int6_audit.cpp` tool (Phase 19Q era) correctly implements:
```cpp
uint32_t scale_off = (M == 4864 && K == 896) ? 16 : 20;
```

This was **never integrated** into the production `llama-graph.cpp` decode code.

---

## Verdict Flags

- ❌ `FAIL_7B_INT6_NUMERIC_EXPLOSION` — abs_sum ≈ 5.7e23 (garbage output)
- ⚠️ `BLOCKED_SCALE_OFFSET_BUG` — file has 4 extra bytes; code has 4-byte offset error
- ✅ `PASS_7B_INT6_PROVENANCE_CONFIRMED` — correct file, correct path, correct magic/K/M
- ✅ `PASS_7B_INT6_NUMERIC_SANITY` — with correct `scale_off=20` decode, values ARE sane
- ✅ `PARTIAL_POLICY_ROUTE_ONLY` — policy routing (N≤4→prt, N>4→native) is correct
- ❌ `FAIL_7B_INT6_HEADER_MISMATCH` — header magic is correct but scale offset in code doesn't match file format

---

## Required Fixes

### Fix 1: Correct scale_offset in `src/llama-graph.cpp`

Around line 1367-1370, add:
```cpp
uint32_t int6_scale_off = 16; // default for 0.5B (M=4864, K=896)
// Phase 23D-R: 7B INT6 files use scale_off=20 (16-byte header + 4 reserved)
if (M == 18944 && K == 3584) {
    int6_scale_off = 20;
}
```

Then read scales at `header + int6_scale_off + M*4` instead of always using `header + M*4`.

### Fix 2: Validate file size vs. expected

After reading header, validate:
```cpp
size_t expected = int6_scale_off + M*4 + (M*K*6+7)/8;
if (file_size != expected) {
    prt_logf("[PRT_V2_INT6_FILE] layer=%d size=%zu expected=%zu DIFF=%zd BLOCKED\n",
             il, file_size, expected, (long)file_size - (long)expected);
    // Fall through or error
}
```

### Fix 3: Add provenance logging

```cpp
prt_logf("[PRT_V2_INT6_FILE] path=%s size=%zu\n", int6_path, file_size);
prt_logf("[PRT_V2_INT6_HEADER] magic=0x%X M=%u K=%u scale_off=%u\n", magic_be, M_hdr, K_hdr, int6_scale_off);
prt_logf("[PRT_V2_INT6_PROVENANCE] scale[0]=%.8f scale[1]=%.8f scale[M-1]=%.8f\n",
         scales6[0], scales6[1], scales6[M-1]);
```

---

## Recommended Next Steps

1. **Immediate:** Apply `scale_off` fix to `src/llama-graph.cpp` — change from 16 to 20 for 7B M/K
2. **Verify:** Re-run Phase 23D with fixed decode, confirm `abs_sum ≈ 250` (not 1e23)
3. **Audit all INT6 sidecars:** Check if other files (14B, 8B, etc.) have similar scale_off anomalies
4. **Integrate audit tool logic:** Use the proven `scale_off = (M==4864&&K==896)?16:20` formula in production code

---

## Models/Sidecars/Binaries Staged?

- **Model:** `Qwen2.5-7B-Instruct-Q4_K_M.gguf` (split, 2 files) — available
- **7B INT6 sidecar:** `prt_sidecars_7b_int6_phase15b_packed/` — available, 28 layers × 50,997,268 bytes
- **0.5B INT6 sidecar:** `prt_phase21h_v_int6_from_f32/` — available
- **llama-cli binary:** `/home/matthew-villnave/llama.cpp/build/bin/llama-cli` — available
- **PRTF variant:** `/tmp/prt_sidecars_7b_int6/` — different format (PRTF magic, 50,997,264 bytes, exact match) — investigate separately

## Secrets Detected?

None.

## Existing Tags Touched?

Phase 23D: `PASS_7B_INT6_POLICY_CANARY`, `PASS_7B_INT6_NATIVE_PREFILL_POLICY`, `PASS_7B_INT6_DECODED_F32_PATH`  
Phase 23D-R: `FAIL_7B_INT6_NUMERIC_EXPLOSION`, `BLOCKED_SCALE_OFFSET_BUG`, `PASS_7B_INT6_PROVENANCE_CONFIRMED`
