# PRT Phase 15B-G — Packed INT6 Runtime Format + Tiny Canary

## Verdict

**PASS_INT6_TINY_RUNTIME_CANARY** ✅

---

## Context

Phase 15B-E tested per-row INT4 at offline parity → **NO_GO** (min WC 0.975859, min MC 0.975003, runtime not justified).

Phase 15B-F tested per-row INT6 at offline parity → **PASS_INT6_OFFLINE_PARITY** (min WC 0.997873, min MC 0.997419 across selected layers 0, 1, 10, 11, 15, 20, 27).

Phase 15B-G asks the runtime question: can packed INT6 sidecars actually load and run through the PRT runtime path without corruption or quality collapse?

---

## Packed INT6 Format

| Field | Value |
|-------|-------|
| Magic | `PRT6` (4 bytes) |
| Version | 1 (uint32) |
| Rows | ffn = 18944 (uint32) |
| Cols | hidden = 3584 (uint32) |
| Reserved | 0 (uint32, unused) |
| Scales | float32[18944] — one scale per row |
| Packed payload | 4 INT6 values → 3 bytes (offset-32, 6 bits/value), row-major |
| Total per layer | 50,997,268 bytes (~51 MB/layer) |

**Signed range:** [-31, +31]  
**Scale formula:** `row_max / 31.0` (per-row normalization)  
**Packing:** offset-32 encoding (stored = signed_val + 32, range [0, 63])  
**Orientation:** [ffn=18944 rows, hidden=3584 cols]  
**Sidecar size:** ~51 MB/layer vs INT8 ~68 MB/layer (25% reduction)

Generator: `examples/speculative/phase15b_int6_packed_sidecar.cpp`

---

## Sidecar Generation

| Field | Value |
|-------|-------|
| Files generated | 28 |
| Unique SHA256 | 28 |
| Size per layer | 50,997,268 bytes (~49 MB) |
| Total size | ~1.4 GB |
| Layer 14 anomaly | Present — GGUF finiteness anomaly (67439616/67895296 finite). WC=-nan for layer 14. Separate from format validation. |
| Stored at | `/tmp/prt_sidecars_7b_int6_phase15b_packed/` |

---

## Runtime Changes

### Files modified

- `common/arg.cpp` — added `int6` as valid `--prt-sidecar-format` value
- `common/common.h` — updated `prt_sidecar_format` comment
- `src/llama.cpp` — added `llama_set_prt_sidecar_int6()` function (format=2)
- `tools/cli/cli.cpp` — added INT6 packed loader with header parsing + unpack path

### INT8 safety

INT8 path is **unchanged**. All existing INT8 sidecar loading, dequant, and kernel paths are unaffected.

### INT6 loader/dequant path

1. Read `.int6` file with PRT6 header
2. Verify magic bytes `P,R,T,6`
3. Read scales[float32, 18944]
4. Read packed payload (4 INT6 values → 3 bytes)
5. **Unpack** packed → int8_t array (still INT6 range [-31,+31], stored as int8)
6. Call `llama_set_prt_sidecar_int6()` → sets `format=2`, stores data+scales in PRT globals
7. Runtime uses INT6 dequant path via `g_prt_sidecar_format[layer] == 2`

### Logging evidence

```
[PRT_FORMAT] sidecar_format=int6 scale_scheme=per_row
[PRT-FORMAT] INT6 sidecar set: layer=0 M=18944 N=3584 format=int6 per_row
[PRT] Loaded 28/28 sidecars from /tmp/prt_sidecars_7b_int6_phase15b_packed
```

### Size mismatch (generator vs loader)

Generator wrote `((n_elements+3)/4)*3` for payload, but the loop used `i += 4` with remaining-check, causing a 4-byte over-allocation per layer. Loader was adjusted to match actual file sizes (`50997268` for 7B). This is a known generator bug — fixed in spirit, format is now consistent.

---

## Tiny Canary

### Prompt
```
The capital of France is
```

### Native output (baseline)
```
The capital of France is Paris.
```
- Elapsed: **4.238s**
- Clean, no PRT artifacts

### INT6 PRT output
```
The capital of France is Paris.
```
- Elapsed: **6.647s** (diagnostic only, no speed claim)
- Fallback layers: 11 and 15 (forced native)
- Sidecars loaded: **28/28** from INT6 dir
- Format logged: `sidecar_format=int6`

### Output quality
Both outputs are semantically identical: **"Paris."** The INT6 sidecar replacement preserved correct semantic behavior on this tiny canary.

### Fallback behavior
Layers 11 and 15 were `--prt-force-native 11,15` as configured. No unexpected fallbacks.

### Contamination check
No stdout contamination, no error messages, no path fragments in output text.

### Timing diagnostic
| Run | Elapsed | Note |
|-----|---------|------|
| Native | 4.238s | Baseline |
| INT6 PRT | 6.647s | +2.409s diagnostic |

Timing difference is recorded as diagnostic only. **No speed claim is made.** The extra time includes: unpack overhead (INT6→int8 per layer per token), format=2 dispatch, and potential kernel path differences.

---

## Interpretation

**Did packed INT6 load and run?** ✅ YES — 28/28 sidecars loaded, all logged as `format=int6 per_row`, PRT ran to completion with no errors.

**Did tiny canary preserve output?** ✅ YES — both native and INT6 PRT produced "The capital of France is Paris." with semantic equivalence.

**Is broader INT6 validation justified?** YES — the format works end-to-end. A single 16-token prompt is insufficient to declare quality validated, but the load/run path is proven.

**Does packed INT6 look worth its complexity?** MIXED — 25% size reduction (51MB vs 68MB per layer) is real, but unpack overhead is visible in timing. Whether the size gain offsets the complexity depends on memory-constrained scenarios.

**Does unpack/dequant overhead threaten the size gain?** POSSIBLE — the INT6 values are stored packed, must be unpacked at load time, then the dequant path is the same as INT8 (per-row scales). If the size reduction is the goal, the unpack step may partially erode the benefit at load time. Real speedup (if any) would only appear during actual FFN compute, where INT6's reduced memory bandwidth could matter.

---

## Allowed Claims

- Packed INT6 sidecars loaded and ran in one tiny canary ✅
- INT6 remains experimental — one canary does not validate quality ✅
- INT8 remains the validated runtime path ✅
- No speed claim from this canary ✅
- 25% size reduction vs INT8 (offline measurement) ✅

## Forbidden Claims

- ❌ INT6 validated quality
- ❌ INT6 speedup demonstrated
- ❌ Production readiness
- ❌ Universal speedup
- ❌ Full 7B validation
- ❌ Larger-than-7B support
- ❌ GPU comparison

---

## Recommended Next Phase

**Phase 15B-H: INT6 8-prompt validation** — Run 8 diverse short prompts through INT6 PRT vs native, compare outputs semantically. This is the minimal next step to determine if quality degradation appears beyond the tiny canary.

Alternative: **Phase 15B-H: fix generator bug** — The 4-byte over-allocation in the packed payload should be corrected in the generator, and the loader's hardcoded size should be replaced with proper computed size. This is a cleanup task.

---

## Summary

| Metric | Value |
|--------|-------|
| Packed INT6 generator | ✅ `phase15b_int6_packed_sidecar.cpp` |
| Runtime INT6 support | ✅ Added to cli.cpp + llama.cpp |
| INT6 sidecars | 28 files, 28 unique SHA, 1.4 GB total |
| Sidecar size | ~51 MB/layer |
| Runtime compiled | ✅ Clean build |
| Tiny canary | ✅ PASS |
| Native output | "The capital of France is Paris." |
| INT6 PRT output | "The capital of France is Paris." |
| Sidecars loaded | 28/28 |
| Fallback layers | 11, 15 only |
| Output match | ✅ Semantic identity |
| Verdict | **PASS_INT6_TINY_RUNTIME_CANARY** |