# PRT Phase 13AC — Qwen2.5-3B Sidecar Generation

**Date:** 2026-05-07  
**Verdict:** PASS_3B_SIDECARS_READY ✅  
**Model:** Qwen2.5-3B-Instruct-Q4_K_M.gguf  
**Branch:** `experimental/prt-phase13-model-generalization`

---

## Verdict

3B PRT sidecars generated, validated, and loader-dry-verified. Runtime canary confirmed clean generation with AVX2 kernel active on 3B model. Ready for Phase 13AD.

---

## Model Metadata

| Field | Value |
|-------|-------|
| **Path** | `/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-3B-Instruct-Q4_K_M.gguf` |
| **File size** | 1,929,903,264 bytes (1.84 GB) |
| **Architecture** | Qwen2 (qwen2) |
| **Layers** | 36 (blk.0 – blk.35) |
| **Hidden/embedding** | 2048 |
| **FFN/intermediate** | 11008 |
| **Quantization** | Q4_K_M (GGML type 12) |
| **FFN_UP tensor format** | Q4_K_M in GGUF, `[hidden=2048, ffn=11008]` in memory |

---

## Sidecar Generation

### Method

Used `ggml_get_tensor()` + `ggml_type_traits[Q4_K_M].to_float()` dequantization via a compiled C++ tool (`/tmp/extract_3b_all`). Each ffn_up tensor was extracted from the loaded model and dequantized to float32.

### Results

| Metric | Value |
|--------|-------|
| **Output directory** | `/tmp/prt_sidecars_3b/` |
| **Sidecars generated** | 36/36 ✅ |
| **Sidecars missing/bad** | 0 ✅ |
| **Bytes per sidecar** | 90,177,536 (86.0 MB) |
| **Total sidecar storage** | 3,246,391,296 bytes (3.02 GB) |
| **All finite** | 36/36 ✅ |
| **All nonzero** | 36/36 ✅ |

### Layer Stats (Selected)

| Layer | Size (MB) | Min | Max | Mean | Std | sum_abs | finite | zero_frac | neg_frac |
|-------|-----------|-----|-----|------|-----|---------|--------|-----------|----------|
| 0 | 86.00 | -0.3285 | 0.3609 | -0.000016 | 0.0235 | 415367 | 22.5M | 0.0001 | 0.5001 |
| 1 | 86.00 | -0.4016 | 0.3961 | 0.000000 | 0.0095 | 64638 | 22.5M | 0.0005 | 0.5001 |
| 11 | 86.00 | -0.8036 | 0.6034 | -0.000003 | 0.0255 | 454150 | 22.5M | 0.0001 | 0.5000 |
| 15 | 86.00 | -0.5965 | 0.6213 | -0.000005 | 0.0272 | 484950 | 22.5M | 0.0001 | 0.4999 |
| 23 | 86.00 | -0.6796 | 0.6632 | -0.000013 | 0.0266 | 471156 | 22.5M | 0.0000 | 0.5000 |
| 35 | 86.00 | -0.4626 | 0.5884 | 0.000002 | 0.0272 | 485034 | 22.5M | 0.0001 | 0.5000 |

---

## Loader Dry Validation

| Check | Result |
|-------|--------|
| PRT log level | summary |
| Sidecars loaded | 36/36 ✅ |
| Per-layer M | 2048 |
| Per-layer N | 11008 |
| Bytes per sidecar | 90,177,536 |
| Missing sidecars | none ✅ |
| Dimension mismatch | none ✅ |
| Fallback due to missing | none ✅ |
| AVX2 kernel active | yes ✅ |
| kernel_mode | 1 (LLAMA_PRT_AVX2) |
| compile_flags | `-mavx2 -mfma` |

---

## Runtime Canary (Single Prompt)

| Metric | Value |
|-------|-------|
| **Model** | Qwen2.5-3B-Instruct-Q4_K_M.gguf |
| **Prompt** | "The capital of France is" |
| **Output** | "The capital of France is Paris." ✅ |
| **Exit code** | 0 |
| **Generation tok/s** | 8.8 |
| **Prompt eval tok/s** | 11.2 |
| **Sidecar load ms** | 4685.11 |
| **Pretouch ms** | 7.46 |
| **Active layers** | 34 |
| **Force-native layers** | 17, 18 |
| **Total custom op calls** | 272 (34 × 8) |
| **AVX2 calls** | 272 ✅ |
| **Fallback calls** | 0 ✅ |
| **Clean output** | yes ✅ |

Sample timing (per-layer):
- IL=0: `calls=8 avg_ms=0.012 kern_total=99.052 first_ms=81.280 later_avg=2.539`
- IL=1: `calls=8 avg_ms=0.012 kern_total=97.745 first_ms=79.890 later_avg=2.551`
- IL=2: `calls=8 avg_ms=0.012 kern_total=99.796 first_ms=81.629 later_avg=2.595`

---

## Layout Analysis

| Item | Value |
|------|-------|
| GGUF tensor ne | `{2048, 11008}` = `[ne0=hidden, ne1=ffn]` |
| GGUF memory order | `[hidden, ffn]` by ggml row-major |
| Sidecar memory layout | `[hidden=2048, ffn=11008]` — matches GGUF order |
| PRT stride0 | `ud->N = ffn = 11008` |
| PRT kernel access | `base[k*stride0 + (j+v)]` with k=hidden_dim, j+v=ffn_dim |
| Layout verification | **SUCCESS** ✅ — PRT kernel accesses `[hidden, ffn]` correctly |
| No transpose needed | Original extraction produces correct layout |
| 0.5B comparison | 0.5B sidecars also `[hidden, ffn]` (896, 4864); same runtime |

---

## What This Phase Proves

- 3B sidecars exist and match expected shape (2048 × 11008 × 36) ✅
- Sidecars are finite and nonzero across all 36 layers ✅
- Per-layer M=2048, N=11008 confirmed in loader dry run ✅
- No dimension mismatch at load time ✅
- Runtime AVX2 kernel is active on 3B model ✅
- 3B PRT produces correct output ("Paris") ✅
- Layout is compatible with PRT runtime indexing ✅
- Ready for Phase 13AD runtime canary and quality validation ✅

---

## What This Phase Does NOT Prove

- ❌ No 3B runtime quality claim yet
- ❌ No 3B speed claim
- ❌ No larger-model generalization claim
- ❌ No exact output equivalence to native

---

## Recommended Next

**Phase 13AD:** 3B runtime canary + 4-prompt quality validation:
1. Native baseline run (1-2 prompts)
2. PRT active run with 3B sidecars
3. Verify PRT_SHAPE n_layer=36 M=2048 N=11008
4. Confirm sidecars loaded 36/36
5. Extract replacement/AVX2 evidence
6. Verify clean output, no fallback
7. Run small quality canary (4 prompts)

---

## Safety

- No `.gguf`/`.bin`/`.safetensors` staged ✅
- No secrets in any changed file ✅
- Sidecar files in `/tmp/` (not committed) ✅
- PRT logs in `/tmp/` (not committed) ✅

**Tag:** `PRT_PHASE13AC_3B_SIDECARS_GENERATED`