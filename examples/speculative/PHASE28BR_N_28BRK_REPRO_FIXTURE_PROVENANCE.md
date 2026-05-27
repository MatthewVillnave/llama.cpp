# Phase 28BR-N: 28BR-K Reproduction / Fixture Provenance Audit

## Summary

**Classification:** `FIXTURE_FORMAT_INCOMPATIBLE` + `LOST_MANIFEST`

28BR-K's token 374 result was **NOT** reproduced. The root cause is fixture format incompatibility: the `.bin` files in `/tmp/phase28bo_layer0_multi_family/` are **int8+scale quantized format**, but `prt_trit_decoder::decode_bytes()` expects **ternary/trit format** (30-byte header + packed trits + scale array). Additionally, no manifest.json exists in that directory.

---

## Key Findings

### 1. Fixture Provenance: LOST_MANIFEST

```
$ ls /tmp/phase28bo_layer0_multi_family/
attn_out_layer0_prt.bin   ffn_up_layer0_prt.bin
ffn_down_layer0_prt.bin   ffn_gate_layer0_prt.bin

$ cat /tmp/phase28bo_layer0_multi_family/manifest.json
MISSING — no manifest.json in this directory
```

Without a manifest, the pager falls back to the **legacy manifest parser** which hardcodes `e.family = "ffn_up"` for ALL entries. This means:

- `attn_out` tensor family is **never registered** in the entries list
- `covers(layer_idx, "attn_out")` returns **false**
- `get_residual(layer_idx, "attn_out")` returns **null view with reason: "not_in_manifest"**
- True injection cannot fire for `attn_out` in this state

### 2. Format Incompatibility — Int8+Scale vs Trit

The `.bin` files in `/tmp/phase28bo_layer0_multi_family/` are **NOT trit files**:

```
File: attn_out_layer0_prt.bin
Magic bytes: 0x3fbc9f7b (float32 bit pattern) → INT8+SCALE format header
Size: 804,944 bytes

Layout:
  [0..2127]     Scale array: 532 scales × 4 bytes = 2,128 bytes
  [2128..804943] Int8 quantized payload: 802,816 int8 values
```

Expected trit layout for `prt_trit_decoder::decode_bytes()`:
```
  [0..31]       32-byte trit header (magic, rows, cols, block_dims, n_scales)
  [32..N]       Packed trits: ceil(802816 × 3 / 8) = 301,056 bytes
  [N+1..]       Scale array: 532 × 4 = 2,128 bytes
  Total expected: 303,216 bytes << actual: 804,944 bytes
```

When `load_trit_header()` reads the int8+scale file as a trit header:
- `th.magic` → garbage (int8 value 0x7b = 123, not 0x54525448)
- `th.block_rows`, `th.block_cols`, `th.n_scales` → garbage
- Validation fails; `decode_bytes()` returns `is_null=true`

### 3. Code Diff: No Regression

```
$ git diff 1d29cad2c..20f8a4c4b --stat
common/sampling.cpp | +18/-2 (logit capture instrumentation only)
```

No changes to:
- `src/llama-graph.cpp`
- `src/prt_sidecar_pager_globals.cpp`
- `examples/speculative/prt_sidecar_pager.cpp`
- `common/arg.cpp`
- `tools/cli/cli.cpp`

**Verdict: Code did not change the injection behavior.** The difference is entirely in the fixture state.

### 4. Legacy Parser Bug

From `load_manifest_legacy()` at commit 1d29cad2c:

```cpp
// Line 226 in prt_sidecar_pager.cpp (1d29cad2c)
e.family = "ffn_up";  // HARDCODED — ignores actual tensor family
```

The legacy parser extracts layer and filename but **ignores the actual tensor family** (attn_out, ffn_down, ffn_gate). All entries are registered as `ffn_up`.

### 5. Fixture File Integrity

```
/tmp/phase28bo_layer0_multi_family/attn_out_layer0_prt.bin:
  md5sum = e7e34b5ab2efdd6a7da315c0a745079a
  Size  = 804,944 bytes

/tmp/phase28br_l_sidecars/attn_out_layer0_prt.bin:
  md5sum = e7e34b5ab2efdd6a7da315c0a745079a  ← IDENTICAL
  Size  = 804,944 bytes
```

The files match between 28BO and 28BR-L fixtures. They are the same int8+scale binary, just placed in different directories.

### 6. Why 28BR-K Produced Token 374 (Hypothesis)

Given that:
- No manifest exists in `/tmp/phase28bo_layer0_multi_family/`
- The legacy parser hardcodes `family="ffn_up"` for all entries
- `attn_out` would not be registered via the legacy parser

The most likely explanation is that **28BR-K used a different manifest path** than `/tmp/phase28bo_layer0_multi_family/manifest.json`. The 28BR-K run likely used the Phase28Y manifest (with `format_version: 1` and proper tensor families in the `files[]` array) that was present during the 28BO run on May 23.

The regeneration script (`gen_sidecars_28br_m.py`) may have overwritten or deleted the original manifest when regenerating files on May 27.

---

## Required Next Phase: 28BR-O — Fixture Reconstruction + Provenance

Do NOT proceed with math or format theory until this is resolved.

### Goal
Reconstruct the working 28BR-K fixture state and verify injection fires for attn_out.

### Required steps

1. **Find original manifest**:
   - Search for any manifest from phase 28BO with `format_version: 1` and `attn_out` in `files[]`
   - Check if `/tmp/phase28bo_layer0_multi_family/manifest.json` was ever created
   - Look in `examples/speculative/results/` for any backup manifest

2. **If original manifest is found**:
   - Restore it to `/tmp/phase28bo_layer0_multi_family/manifest.json`
   - Verify `format_version: 1` is present (triggers Phase28Y parser with correct families)
   - Re-run D mode: `llama-cli ... --prt-mode 5700 --prt-sidecar-manifest /tmp/phase28bo_layer0_multi_family/manifest.json --enable-prt-sidecar-pager --prt-sidecar-true-injection --prt-sidecar-apply-family attn_out`
   - Expected: `sidecar_math_influenced=1`, token 374

3. **If original manifest is LOST**:
   - Create a new Phase28Y manifest for `/tmp/phase28bo_layer0_multi_family/` with:
     - `format_version: 1`
     - `files: [{filename: "attn_out_layer0_prt.bin", tensor_family: "attn_out", ...}]`
   - Or: Create a trit-format sidecar for attn_out with proper 30-byte header

4. **Do NOT**:
   - Use random uniform f32 files as a substitute
   - Run benchmarks or quality evaluations
   - Expand layers/families beyond attn_out
   - Claim f32 sidecar is too small until `sidecar_math_influenced=1` and `injection_successes>=1`

---

## Files Examined

- `/tmp/phase28bo_layer0_multi_family/` — fixture directory (int8+scale format, no manifest)
- `/tmp/phase28br_l_sidecars/` — byte-identical to phase28bo fixture
- `examples/speculative/prt_sidecar_pager.cpp` (1d29cad2c) — legacy parser hardcodes `family="ffn_up"`
- `src/prt_sidecar_pager_globals.cpp` — `prt_true_apply()` and decode path
- `examples/speculative/prt_trit_decode.cpp` — `decode_bytes()` expects trit header + packed trits

---

## Report

- **Classification:** `FIXTURE_FORMAT_INCOMPATIBLE` + `LOST_MANIFEST`
- **Token 374 reproduced:** NO
- **True injection fired:** NO (attn_out not in manifest)
- **sidecar_math_influenced_output:** N/A (injection blocked at manifest lookup)
- **Fixture provenance result:** `LOST_MANIFEST`; int8+scale files exist but format is incompatible with trit decoder
- **Code regression:** NONE — sampling.cpp instrumentation only
- **Next recommended phase:** `28BR-O` — fixture reconstruction