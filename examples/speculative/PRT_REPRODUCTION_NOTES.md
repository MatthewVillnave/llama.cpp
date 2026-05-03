# PRT Route A — Reproduction Notes

**Version:** 1.0
**Date:** 2026-05-02
**Branch:** `experimental/prt-route-a-rc1`
**Tag:** `PRT_ROUTE_A_RC1`

---

## What This Document Is

Notes for reproducing PRT Route A RC1 results. Read before attempting to run or validate.

---

## System Requirements

| Requirement | Value |
|------------|-------|
| CPU | x86_64 (AVX2 recommended) |
| RAM | 16GB minimum, 32GB recommended |
| OS | Linux |
| Disk | ~90MB per sidecar × 36 layers ≈ 3.2GB |
| Model | Qwen2.5-3B-Instruct-Q4_K_M.gguf |

---

## Expected Branch and Tag

```bash
# Checkout the correct branch
git checkout experimental/prt-route-a-rc1

# Or checkout the tag
git checkout PRT_ROUTE_A_RC1
```

Do not use `master` or `main` — RC1 changes are on the experimental branch.

---

## Model

**Used in RC1 validation:** Qwen2.5-3B-Instruct-Q4_K_M.gguf

Other models will require their own sidecars. Sidecars from this model will not produce correct results on other models.

Model file is **not included in the repo**. Obtain separately.

---

## Sidecars (Required but Not Included)

PRT requires sidecar weight matrices for each layer. These are **large binary files (~90MB each)** and are **NOT committed to the repo**.

### Generating Sidecars

Sidecars must be extracted from a native reference run on the same model. The extraction tool is in this repo:

```bash
# Build the extraction tool
cd build/bin
cmake --build . --target llama-prt-ffn-up-extract -j$(nproc)

# Extract sidecars for all 36 layers
./build/bin/llama-prt-ffn-up-extract \
  -m /path/to/Qwen2.5-3B-Instruct-Q4_K_M.gguf \
  -o /tmp/prt_sidecars/
```

This will produce 36 files:
```
/tmp/prt_sidecars/ffn_up_layer0_prt.bin  (90MB)
/tmp/prt_sidecars/ffn_up_layer1_prt.bin  (90MB)
...
/tmp/prt_sidecars/ffn_up_layer35_prt.bin (90MB)
```

### Sidecar Location

The benchmark binary (`llama-prt-posix`) looks for sidecars at:
```
/tmp/prt_sidecars/ffn_up_layer{L}_prt.bin
```

This path is hardcoded. Change it in `phase10e0_layer0_replacement.cpp` if needed.

---

## Command Pattern

```bash
./build/bin/llama-prt-posix \
  -m /path/to/Qwen2.5-3B-Instruct-Q4_K_M.gguf \
  -p "Once upon a time in a" \
  -n 100 \
  --prt-mode 5700 \
  --prt-force-native 12,15
```

### Required Flags

| Flag | Value | Purpose |
|------|-------|---------|
| `--prt-mode` | `5700` | Enable Route A (all layers PRT) |
| `--prt-force-native` | `12,15` | L12 and L15 use native FFN_UP |
| `-n` | `100` | Generate 100 tokens |

### Why These Exact Flags

- `--prt-mode 5700` activates PRT on all 36 layers
- `--prt-force-native 12,15` marks L12 and L15 as native anchors
- Without the force-native flag, pure Route A fails on specific narrative prompts
- Mode `0` disables PRT entirely (native baseline)

---

## Expected Counters

Run with the command above and look for these in stderr:

```
[11BD] callback_overwrites: 0
[11BD] native_fallback_calls: 16
[11BD] identity_fallback_calls: 0
[11BD] native_ffn_up_calls: 0
[11BD] prt_true_replacement_calls: 3706
```

`prt_true_replacement_calls` will be ~3706 for n=100, ~2006 for n=50.

---

## Expected Output (Known Failure Prompt)

Prompt: `"Once upon a time in a"`

| Mode | Token 0 | Result |
|------|---------|--------|
| Native | " small" | ✓ Correct |
| Route A + L12+L15 | " small" | ✓ Matches native |
| Pure Route A (no force-native) | " far" | ✗ WRONG |

If you get "far" with L12+L15, something is wrong with the sidecar or build.

---

## Memory Prerequisites

Before running, ensure:
- 12GB+ available RAM (or killing Ollama if running)
- Swap configured (if RAM < 16GB)
- No other heavy processes

Run `free -h` before and after to confirm no leaks.

---

## Warnings

### Do NOT Commit Sidecars or Model Files

```bash
# These files are huge and not yours to distribute
*.bin        # sidecar files
*.gguf       # model files
*.safetensors
*.pt
*.pth
```

Check your `.gitignore` before staging files.

### Do NOT Claim Model Independence

Sidecars generated from Qwen2.5-3B will not work correctly on other models. Each model requires its own sidecar extraction.

### Do NOT Claim Production-Ready

This is experimental research code. Do not represent it as production-ready or deployment-safe.

### Sidecar Exact Reproduction

The exact numerical values in PRT matmul depend on:
- CPU architecture (AVX2 vs scalar)
- Compiler optimization
- Floating-point rounding

Checksum-level reproducibility across different machines is not guaranteed. Functional equivalence (matching native output) is the goal.

---

## Troubleshooting

### "PRT mode not recognized"
- Ensure `--prt-mode 5700` is passed before generation starts
- Check that the build includes PRT code

### "Missing sidecar" error
- Verify sidecar files exist at `/tmp/prt_sidecars/`
- Verify all 36 layers are present
- Check file permissions

### Output doesn't match native
- Verify sidecars were extracted from the same model and quantization
- Verify `--prt-force-native 12,15` is set
- Verify model file is Qwen2.5-3B-Instruct-Q4_K_M

### Speedup seems lower than expected
- Check that AVX2 kernel is enabled (`--prt-kernel 1`)
- Verify PRT custom op is actually running (`prt_true_replacement_calls > 0`)
- Check that Ollama or other GPU processes aren't using memory

---

## What's Included in the Repo

| File | Purpose |
|------|---------|
| `examples/speculative/phase10e0_layer0_replacement.cpp` | Benchmark binary source |
| `examples/speculative/prt_graph_replace.h` | PRT custom op + AVX2 kernel |
| `examples/speculative/prt_avx2_kernel.h` | AVX2 SIMD implementation |
| `examples/speculative/PRT_*.md` | Documentation |

## What's NOT Included

- Sidecar binary files (~3.2GB)
- GGUF model files (~2GB)
- Extraction tool binary (build from source)
- Pre-built benchmark binary (build from source)

---

*For claims and allowed statements, see PRT_CLAIMS.md*

---

## Sidecar Packaging Notes

PRT sidecars are external local artifacts that must be generated from the same model used for inference.

### Key properties

| Property | Value |
|----------|-------|
| Sidecar tensor | `ffn_up` from GGUF |
| Storage shape | `[11008, 2048]` — [ffn_dim, hidden_dim] |
| Access pattern | `sidecar[j * hidden_dim + k]` |
| Dtype | float32 (4 bytes/element) |
| File size | 90,113,024 bytes (~90MB) per layer |

### Force-native layers and sidecars

Under the validated L12+L15 policy, layers 12 and 15 use native computation and **do not require** PRT sidecar files. Only 34 layers require sidecars. Layers 12 and 15 are optional.

### Missing required sidecars

If a required sidecar is missing at startup, the process exits with a **FATAL error**:

```
[PRT-ERROR] FATAL: N required sidecar(s) missing. Exiting.
```

This is the loud failure behavior (Phase 11BP).

### Validator

```bash
python3 examples/speculative/prt_validate_sidecars.py \
  --manifest examples/speculative/prt_sidecar_manifest.example.json \
  --sidecar-dir /tmp/prt_sidecars/
```

### Do NOT commit sidecar binaries

Sidecar files are large (~90MB each) and model-specific. Never commit them to the repo.
