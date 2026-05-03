# PRT Sidecar Manifest Specification

**Version:** 1.0
**Date:** 2026-05-03
**Scope:** PRT Route A on llama.cpp with Qwen2.5-3B-Instruct-Q4_K_M

---

## Overview

A PRT sidecar manifest describes the sidecar files required for a given model, their expected properties, and the force-native policy. It is a machine-readable declaration of what is needed and what checksums to verify against.

The manifest does **not** contain sidecar binaries. It contains metadata only.

---

## Current Sidecar Loader Behavior

### File naming convention
```
/tmp/prt_sidecars/ffn_up_layer{L}_prt.bin
```
Where `{L}` is the layer index (0–35). The path is currently hardcoded.

### Sidecar tensor properties

| Field | Value |
|-------|-------|
| Tensor name | `ffn_up` (from GGUF) |
| Storage shape | `[11008, 2048]` — [ffn_dim, hidden_dim] |
| Access pattern | `sidecar[j * hidden_dim + k]` where j ∈ [0, ffn), k ∈ [0, hidden) |
| Dtype | `float32` (4 bytes per element) |
| File size | 11008 × 2048 × 4 = **90,113,024 bytes** (~90MB per layer) |
| Total for 36 layers | ~3.2 GB |

### Layer policy

| Layer type | Policy |
|------------|--------|
| Layers 0–11, 13–14, 16–35 | Required PRT (34 layers) |
| Layers 12, 15 | Force-native via `--prt-force-native 12,15` |

**Note:** Force-native layers do not require sidecar files for the L12+L15 policy.

### Validation behavior (Phase 11BP)

- `validate_prt_sidecars()` runs at startup when `--prt-mode 5700` is set
- Checks that all **required** layer sidecars (34 layers) are present
- Excludes force-native layers from the required check
- Missing required sidecar → **FATAL error**, process exits with code 1
- Prints sidecar checksums for L0 and L35 after successful load

### Loader functions

| Function | Status | Purpose |
|----------|--------|---------|
| `load_sidecar_mmap()` | Primary (PRODUCTION) | Memory-mapped file read |
| `load_sidecar_posix()` | Fallback | POSIX open/read |
| `load_sidecar_static()` | Dead code | Synthetic zeros, unused |
| `load_sidecar_fopen()` | Dead code | fopen, causes corruption (Phase 10E-5) |

---

## Model Assumptions (Hardcoded)

| Assumption | Current value | Risk |
|-----------|--------------|------|
| Layer count | 36 | Wrong for other models |
| Hidden dim | 2048 | Wrong for 7B+ models |
| FFN dim | 11008 | Wrong for 7B+ models |
| Architecture | qwen2 | Wrong for llama/mistral |
| Quantization | Q4_K_M | Sidecars from other quantizations will produce incorrect results |

---

## Manifest Format

```json
{
  "format_version": 1,
  "prt_name": "Progressive Residual Ternary",
  "target_runtime": "llama.cpp",
  "target_model": {
    "architecture": "qwen2",
    "model_family": "Qwen2.5",
    "model_size": "3B",
    "quantization": "Q4_K_M",
    "n_layers": 36,
    "hidden_dim": 2048,
    "ffn_dim": 11008,
    "model_file_note": "Model file is NOT distributed in this repo. Obtain separately.",
    "model_sha256": "<optional: sha256 of the GGUF file>"
  },
  "sidecar_tensor": {
    "name": "ffn_up",
    "dtype": "float32",
    "shape": [11008, 2048],
    "orientation": "[ffn_dim, hidden_dim] — row-major, row = ffn dimension",
    "access_pattern": "sidecar[j * hidden_dim + k]",
    "file_size_bytes": 90113024,
    "layer_count": 36
  },
  "policy": {
    "prt_mode": 5700,
    "force_native_layers": [12, 15],
    "required_prt_layers": "<auto: all layers except force_native_layers>"
  },
  "paths": {
    "sidecar_dir": "/tmp/prt_sidecars/",
    "sidecar_filename_pattern": "ffn_up_layer{L}_prt.bin"
  },
  "validation": {
    "check_sidecar_exists": true,
    "check_file_size": true,
    "check_sha256": "<optional: per-file sha256 if provided>",
    "fail_on_missing": true,
    "fail_on_size_mismatch": true,
    "fail_on_checksum_mismatch": "<if sha256 provided>"
  },
  "files": [
    {
      "layer": 0,
      "filename": "ffn_up_layer0_prt.bin",
      "required": true,
      "bytes": 90113024
    },
    {
      "layer": 12,
      "filename": "ffn_up_layer12_prt.bin",
      "required": false,
      "note": "Force-native layer — sidecar optional"
    }
  ]
}
```

---

## Creating a Manifest

1. Generate sidecars from a native reference run on the target model
2. Compute sha256 of each sidecar file
3. Fill in the `files` array with actual checksums
4. Set `model_sha256` to the GGUF file checksum
5. Commit the manifest to the repo as a reference

Do **not** commit sidecar binaries. Only the JSON manifest.

---

## Using a Manifest

A validator script reads the manifest and verifies:
1. All required sidecar files exist in the declared directory
2. File sizes match
3. Optional: SHA256 checksums match
4. Force-native layers are excluded from required checks

The validator does **not** require the GGUF model file or the binary benchmark tool.

---

## Sidecar Generation (Reproduction Note)

Sidecars must be generated locally from a native reference run. The extraction tool is:

```
llama-prt-ffn-up-extract
```

Built from: `tools/prt-ffn-up-extract.cpp`

Run:
```bash
./build/bin/llama-prt-ffn-up-extract \
  -m /path/to/Qwen2.5-3B-Instruct-Q4_K_M.gguf \
  -o /tmp/prt_sidecars/
```

This produces 36 binary files, one per layer.

---

## Risks if Wrong Sidecars Are Used

| Risk | Result |
|------|--------|
| Sidecar from wrong model | Incorrect output, quality failure |
| Sidecar from different quantization | Incorrect output, possible crash |
| Stale sidecar (regenerated with different code) | Unpredictable output |
| Sidecar shape mismatch | Memory corruption or crash |
| Sidecar from different architecture | Incorrect output |

**There is currently no runtime check that the sidecar came from the correct model.** The manifest + validator is the design for adding this.

---

## Future Runtime Recommendations

1. Accept `--prt-manifest /path/to/manifest.json`
2. Accept `--prt-sidecar-dir /path/to/sidecars/`
3. On load: verify all required sidecars exist and match manifest
4. Optionally verify model SHA256 matches before generation starts
5. Loud fatal error on any mismatch

---

*Spec by ELVIS for Matthew Villnave / The ForgeHQ*