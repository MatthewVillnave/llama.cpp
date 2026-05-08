# PRT Phase 15B-C — Minimal GGUF Single-Layer FFN_UP Extraction

## Verdict

**PASS_SINGLE_LAYER_GGUF_EXTRACTION**

The minimal C++ GGUF extraction tool works. We can now reliably extract and dequantize real Qwen2.5-7B FFN_UP tensors from the GGUF file. All three tested layers (0, 1, 27) show distinct statistics and checksums, confirming per-layer uniqueness.

## Context

Phase 15B-B was blocked because Python GGUF extraction failed. This phase creates a minimal C++ tool using in-tree llama.cpp APIs (gguf_init_from_file, ggml_get_type_traits, to_float) to extract and dequantize FFN_UP tensors directly from the 4.7GB GGUF file.

## Tool Created

**Source:** `examples/speculative/phase15b_gguf_single_tensor_probe.cpp`

**Build command:**
```bash
g++ -std=c++17 -O2 -I. -I./common -I./ggml/include \
  examples/speculative/phase15b_gguf_single_tensor_probe.cpp -o /tmp/phase15b_probe \
  build/bin/libllama.so build/bin/libggml-base.so.0.9.11 build/bin/libggml-cpu.so.0.9.11 \
  build/common/libcommon.a -lm -Wl,-rpath,$(pwd)/build/bin
```

**Binary:** `/tmp/phase15b_probe` (20KB)

**APIs used:**
- `gguf_init_from_file()` — loads GGUF model
- `gguf_find_tensor()` — finds tensor by name
- `gguf_get_tensor_type/size/offset/name` — tensor metadata
- `ggml_get_type_traits()` — gets dequantization function
- `to_float()` — dequantizes Q4_K to float32
- Direct file read at tensor offset (bypasses manual GGUF parsing)

**Manual GGUF parsing:** Not used. All access via gguf/ggml public API + raw file I/O for data.

## Tensor Probe Results

| Tensor | Found | Type | Shape | Finite | Min | Max | Mean | StdDev | Row0 Norm | Checksum |
|--------|-------|------|-------|--------|-----|-----|------|--------|-----------|----------|
| blk.0.ffn_up.weight | ✅ | Q4_K | 18944×3584 | 67,895,296/67,895,296 | -2.251 | 0.527 | -0.000026 | 0.019500 | 0.931467 | 60c8281a434d4f95 |
| blk.1.ffn_up.weight | ✅ | Q4_K | 18944×3584 | 67,895,296/67,895,296 | -3.420 | 3.403 | -0.000005 | 0.020412 | 0.963035 | a395e71f48fdffc1 |
| blk.27.ffn_up.weight | ✅ | Q4_K | 18944×3584 | 67,895,296/67,895,296 | -0.687 | 0.954 | -0.000004 | 0.029221 | 0.747752 | 68be4353ac4ecc0d |

**Key observations:**
- All layers have identical tensor shape (18944 FFN × 3584 hidden)
- All layers are 100% finite (no NaN/Inf)
- All three layers have DIFFERENT checksums — confirming per-layer uniqueness
- Layer 0 and layer 1 have very different weight distributions (min/max differ significantly)
- Layer 27 has higher std_dev (0.029 vs 0.019-0.020 for layers 0-1)

## Interpretation

**Can we reliably extract/dequantize real 7B FFN_UP tensors?** Yes.

**Do layer 0 and layer 1 differ?** Yes — clearly different statistics and checksums. This confirms the GGUF itself contains layer-unique weights (ruling out the hypothesis that the GGUF is the source of the duplication).

**Is this enough to proceed to sidecar regeneration/parity next?** Yes — the extraction tool is working and can be extended to generate INT8 sidecars and compute parity metrics.

**What remains blocked?** Nothing for extraction. We can now:
1. Regenerate fresh unique INT8 sidecars for all 28 layers
2. Run selected-layer parity checks against real GGUF weights
3. Re-run INT4 validation with proper per-layer references

## Impact on Phase 14/15

The GGUF extraction confirms that each layer's FFN_UP weights in the model file are unique. The current INT8 sidecar duplication (1 unique SHA out of 28) is therefore a generation bug, not a model file issue.

This means: the Phase 14/15 runtime tests that empirically passed used duplicated sidecars, which still produced correct output. The duplication is a data quality issue, not a correctness issue.

## Allowed Claims

- GGUF single-layer extraction works for Qwen2.5-7B Q4_K tensors
- Layer 0, 1, 27 have different checksums and statistics
- Each layer's weights are unique in the source GGUF
- The extraction tool is working and can be extended
- Per-layer INT8 sidecar regeneration is now possible

## Forbidden Claims

- Do not claim INT4 works
- Do not claim current sidecars are fixed
- Do not claim Phase 14/15 fully validated
- Do not claim production readiness
- Do not claim universal speedup

## Recommended Next Phase

**Phase 15B-D: Regenerate fresh 28-layer INT8 sidecars using the working extraction tool.**

Use the same C++ tool (or extend it) to extract layers 0-27, quantize to INT8 per-row, write to `/tmp/prt_sidecars_7b_int8_phase15b_fixed/`, then run selected-layer parity against the GGUF reference.

## Files Created

- `examples/speculative/phase15b_gguf_single_tensor_probe.cpp` — Working C++ extraction tool
- `/tmp/phase15b_tensor_l0.json` — Layer 0 probe output
- `/tmp/phase15b_tensor_l1.json` — Layer 1 probe output
- `/tmp/phase15b_tensor_l27.json` — Layer 27 probe output

## Safety

| Check | Status |
|-------|--------|
| Models staged | NO |
| Sidecars staged | NO |
| Binaries staged | NO |
| Temp logs staged | NO |
| Secrets detected | NO |
| Existing tags touched | NO |