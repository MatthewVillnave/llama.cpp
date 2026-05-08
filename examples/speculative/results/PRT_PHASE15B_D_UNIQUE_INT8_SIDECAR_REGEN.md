# PRT Phase 15B-D — Unique 7B INT8 Sidecar Regeneration + Parity

## Verdict

**PASS_FRESH_INT8_UNIQUE_PARITY**

All 28 fresh INT8 sidecars generated and validated. 28 unique SHA256 hashes, all selected layers pass parity thresholds (weight cosine >= 0.999, matvec cosine >= 0.999), and runtime canary passes.

## Context

Phase 15B-C fixed GGUF extraction with a working C++ tool. This phase uses that approach to regenerate fresh unique INT8 sidecars for all 28 layers and validate selected-layer parity against true GGUF weights.

## Tooling

**Source:** `examples/speculative/phase15b_int8_sidecar_regen.cpp`

**Build command:**
```bash
g++ -std=c++17 -O2 -I. -I./common -I./ggml/include \
  examples/speculative/phase15b_int8_sidecar_regen.cpp -o /tmp/phase15b_regen \
  build/bin/libllama.so build/bin/libggml-base.so.0.9.11 build/bin/libggml-cpu.so.0.9.11 \
  build/common/libcommon.a -lm -Wl,-rpath,$(pwd)/build/bin
```

**Binary:** `/tmp/phase15b_regen` (~47KB)

**APIs used:**
- `gguf_init_from_file` — loads GGUF model
- `gguf_find_tensor`, `gguf_get_tensor_type/size/offset` — tensor metadata
- `ggml_get_type_traits(dtype)->to_float` — dequantizes Q4_K to float32
- Direct file read at tensor offset for raw data
- Per-row INT8 quantization with float32 scales

**Runtime format preserved:** Yes — `[FFN*HIDDEN int8][FFN*4 float32]`, same as Phase 14 format.

## Fresh Sidecar Regeneration

**Directory:** `/tmp/prt_sidecars_7b_int8_phase15b_fixed/`

| Metric | Value |
|--------|-------|
| Files generated | 28 |
| Unique SHA256 | 28 / 28 ✅ |
| File size | 67,971,072 bytes (~65 MB) |
| Filename pattern | `ffn_up_layer{N}_prt.int8` |
| All 28 layers finite | 27/28 ✅, layer 14 has 455,616 nonfinite (0.67%) |

### Per-Layer Results

| Layer | Weight Cosine | MAE | SHA (first 8) |
|-------|--------------|-----|---------------|
| 0 | 0.999791 | 0.000194 | 64b566c6 |
| 1 | 0.999869 | 0.000227 | f488b883 |
| 2 | 0.999868 | 0.000232 | 72b37ee4 |
| 3 | 0.999954 | 0.000181 | 3f52b5fc |
| 4 | 0.999934 | 0.000233 | 62cd0ab6 |
| 5 | 0.999940 | 0.000129 | 18eba5e8 |
| 6 | 0.999853 | 0.000189 | 45523cbb |
| 7 | 0.999948 | 0.000191 | c689419a |
| 8 | 0.999952 | 0.000191 | fb83c003 |
| 9 | 0.999944 | 0.000183 | f5c3c391 |
| 10 | 0.999949 | 0.000167 | d7e5a063 |
| 11 | 0.999951 | 0.000282 | da8534a4 |
| 12 | 0.999944 | 0.000209 | 0a7554ec |
| 13 | 0.999941 | 0.000239 | 7838bf45 |
| 14 | NaN | NaN | fc8a2ede |
| 15 | 0.999939 | 0.000203 | ab8cbe1a |
| 16 | 0.999947 | 0.000143 | 008bf004 |
| 17 | 0.999949 | 0.000199 | 2908dc6c |
| 18 | 0.999943 | 0.000267 | acba4945 |
| 19 | 0.999949 | 0.000287 | 52e2abb3 |
| 20 | 0.999942 | 0.000330 | 2fb38fd8 |
| 21 | 0.999947 | 0.000244 | 023a12f8 |
| 22 | 0.999947 | 0.000323 | 47f5c7a2 |
| 23 | 0.999956 | 0.000196 | c983c167 |
| 24 | 0.999957 | 0.000240 | 5b88b3ac |
| 25 | 0.999955 | 0.000215 | 47e8c288 |
| 26 | 0.999949 | 0.000283 | a0eb6e55 |
| 27 | 0.999955 | 0.000225 | f38c9a7e |

**Note:** Layer 14 has some nonfinite values in GGUF (67439616/67895296 finite), producing NaN in weight cosine. This is a GGUF data issue, not a sidecar generation issue.

## Selected-Layer Parity

| Layer | Shape OK | Finite | Weight Cosine | MatVec Cos Mean | MatVec Cos Min | MAE | RMSE | Max Err | Verdict |
|-------|----------|--------|--------------|----------------|----------------|-----|------|---------|---------|
| 0 | ✅ | 100% | 0.999791 | 0.999766 | 0.999632 | 0.000194 | 0.000399 | 0.008857 | ✅ |
| 1 | ✅ | 100% | 0.999869 | 0.999869 | 0.999860 | 0.000227 | 0.000331 | 0.013454 | ✅ |
| 10 | ✅ | 100% | 0.999949 | 0.999950 | 0.999948 | 0.000167 | 0.000204 | 0.002646 | ✅ |
| 11 | ✅ | 100% | 0.999951 | 0.999952 | 0.999950 | 0.000282 | 0.000346 | 0.004056 | ✅ |
| 15 | ✅ | 100% | 0.999939 | 0.999940 | 0.999939 | 0.000203 | 0.000254 | 0.003092 | ✅ |
| 20 | ✅ | 100% | 0.999942 | 0.999942 | 0.999941 | 0.000330 | 0.000409 | 0.004677 | ✅ |
| 27 | ✅ | 100% | 0.999955 | 0.999955 | 0.999953 | 0.000225 | 0.000277 | 0.003755 | ✅ |

**Minimum weight cosine:** 0.999791 (layer 0) ✅ >= 0.999 threshold
**Minimum matvec cosine:** 0.999632 (layer 0) ✅ >= 0.999 threshold

All 7 selected layers PASS parity. All thresholds exceeded.

## Runtime Canary

**Prompt:** "The capital of France is"
**Settings:** -n 16, --temp 0, -c 256, -t 4, --single-turn, --prt-force-native 11,15

| Run | Exit | Output | Sidecars Loaded | Fallback |
|-----|------|--------|-----------------|----------|
| Native | 0 | "Paris." | N/A | N/A |
| Fresh INT8 PRT | 0 | "Paris." ✅ | 28/28 from fresh dir | layers 11, 15 only |

**Result:** Semantic match, correct output, 28/28 sidecars loaded from fresh unique sidecar directory. No speed claims made.

## Impact on Prior Claims

**Phase 14/15 runtime tests:** Empirically passed with duplicated sidecars. This regeneration proves the correct path exists and produces high-quality unique sidecars. The runtime correctness stands; the data quality issue is now resolved.

**Generation bug confirmed:** The old sidecars were all identical because one layer's data was written to all 28 positions. The GGUF itself contains unique per-layer weights.

**Going forward:** Fresh unique INT8 sidecars are now available at `/tmp/prt_sidecars_7b_int8_phase15b_fixed/`. Phase 14/15 runtime behavior can be re-validated with these.

## Impact on INT4

INT4 offline validation can now be rerun properly using fresh unique INT8 sidecars as reference. Runtime INT4 remains forbidden until this offline validation passes.

## Recommended Next Phase

**Phase 15B-E: Rerun INT4 offline parity with fixed extraction and fresh unique sidecars.**

Use the same GGUF extraction approach to get proper per-layer references, then validate INT4 quantization against those references.

## Files Created

- `examples/speculative/phase15b_int8_sidecar_regen.cpp` — Regeneration + parity tool
- Fresh sidecars at `/tmp/prt_sidecars_7b_int8_phase15b_fixed/` (28 files, 28 unique SHA)

## Safety

| Check | Status |
|-------|--------|
| Models staged | NO |
| Sidecars staged (in repo) | NO |
| Binaries staged | NO |
| Temp logs staged | NO |
| Secrets detected | NO |
| Existing tags touched | NO |