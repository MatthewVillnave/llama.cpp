# PRT Phase 14H — 7B Feasibility + Sidecar Plan

## Verdict

**BLOCKED_NO_LOCAL_7B_MODEL**

## Context

Phase 14G froze the 3B INT8 repeatability checkpoint — INT8 PRT matches native at 1.000× on Qwen2.5-3B. The natural next question is: does this scale to 7B? Before any 7B runtime test, we must confirm the model exists locally and that sidecar generation is feasible on this machine.

This phase evaluates feasibility only. No sidecars are generated.

## Machine Health

| Resource | Value |
|----------|-------|
| RAM total | 15 GiB |
| RAM available | 10 GiB |
| RAM used | 5.2 GiB |
| Swap used | 2.9 GiB / 4.0 GiB |
| Disk free (/home) | **141 GiB** |
| Swap status | Stable, not increasing |
| Top memory process | openclaw-gateway (1.1 GiB RSS) |

**Machine health: GOOD** — 10 GiB RAM free, 141 GiB disk free, no swap pressure.

## Local 7B Model Search

**Result: NO LOCAL 7B MODEL FOUND**

Searched locations:
- `/home/matthew-villnave/models/gguf/` — only 0.5B and 3B Qwen2.5 variants
- `/home/matthew-villnave/models/gguf/qwen2.5/` — Qwen2.5-0.5B, Qwen2.5-3B
- `/home/matthew-villnave/models/gguf/qwen25_coder/` — Qwen2.5-Coder-0.5B, Qwen2.5-Coder-3B
- Ollama model registry — no model files present

No 7B GGUF model exists on this machine.

## Known Qwen2.5-7B Metadata (from llama.cpp source)

Qwen2.5-7B architecture (case 28, n_embd=3584):

| Parameter | Value |
|-----------|-------|
| Layers | 28 |
| Hidden size (n_embd) | 3584 |
| FFN intermediate size | 18944 |
| Attention heads | 28 |
| GQA groups | 4 (n_head_kv=7) |
| vocab size | ~151936 |

**Note:** These are architectural defaults from llama.cpp source. Must verify actual GGUF metadata when a real model is acquired.

## Sidecar Size Estimates (Qwen2.5-7B)

Using architecture estimates: layers=28, hidden=3584, ffn=18944.

| Metric | Formula | Result |
|--------|---------|--------|
| **Float32 per layer** | hidden × ffn × 4B | 3584 × 18944 × 4 = **271,564,544 bytes (~259 MiB)** |
| **Float32 total (28 layers)** | per_layer × 28 | **~7,603 MiB (~7.4 GiB)** |
| **INT8 per layer** | hidden × ffn × 1B + scale | 3584 × 18944 + 18944 × 4 = **67,891,136 + 75,776 ≈ 68 MiB** |
| **INT8 total (28 layers)** | 68 MiB × 28 | **~1,900 MiB (~1.85 GiB)** |

### Comparison to 3B Baseline

| Model | Layers | Hidden | FFN | FP32/layer | FP32 total | INT8/layer | INT8 total |
|-------|--------|--------|-----|-----------|-----------|-----------|-----------|
| Qwen2.5-3B | 36 | 2048 | 11008 | 90 MiB | 3.24 GiB | 22.5 MiB | ~0.76 GiB |
| **Qwen2.5-7B** | **28** | **3584** | **18944** | **~259 MiB** | **~7.4 GiB** | **~68 MiB** | **~1.9 GiB** |

### Disk Requirement Breakdown

| Item | Size |
|------|------|
| Float32 temp (per layer, max) | ~259 MiB |
| INT8 sidecar output (final total) | ~1.9 GiB |
| Peak temp disk (float32 + INT8 simultaneously) | ~260 MiB (layer-by-layer streaming) |
| Peak temp disk (all float32 + INT8) | ~9.3 GiB (7.4 + 1.9) |
| **Safe recommendation: layer-by-layer** | **~260 MiB peak** |

### RAM Requirement Breakdown

| Stage | Estimated RAM |
|-------|--------------|
| Model loading (Q4_K_M 7B GGUF) | ~4-5 GiB |
| Layer dequant extraction | ~259 MiB (float32) + ~68 MiB (INT8) |
| Quantization buffer | ~68 MiB |
| **Peak layer-by-layer** | **~5.5 GiB total** |
| Available | **10 GiB** |

**Layer-by-layer streaming: SAFE** — peak ~5.5 GiB vs 10 GiB available.

## Risk Classification

| Risk | Assessment |
|------|------------|
| **Disk risk** | LOW — 141 GiB free. Even storing all float32 (~7.4 GiB) + INT8 (~1.9 GiB) is fine. Layer-by-layer reduces to ~260 MiB peak. |
| **RAM/swap risk** | LOW — model load ~4-5 GiB + per-layer buffers ~330 MiB = ~5.5 GiB peak. With 10 GiB available, safe. |
| **Model availability** | BLOCKED — no 7B GGUF locally |
| **Tooling** | SAFE — existing 3B extraction pipeline should work for 7B |

**Verdict: BLOCKED_NO_LOCAL_7B_MODEL** — cannot proceed without acquiring the model.

## If a 7B Model Were Available: Generation Plan

### Output Path
`/tmp/prt_sidecars_7b_int8/` (260 MiB peak disk, /tmp on same 233 GiB NVMe partition)

Fallback: `/home/matthew-villnave/prt_sidecars_7b_int8/` if /tmp is insufficient.

### Layer-by-Layer Streaming Strategy

For layers 0 through 27:

1. **Extract** ffn_up layer N from GGUF → float32 temp buffer (~259 MiB)
2. **Quantize** float32 → INT8 per-row (scales: 18944 × 4 bytes)
3. **Write** INT8 sidecar to `/tmp/prt_sidecars_7b_int8/ffn_up_layer{N}.int8`
4. **Validate** — per-layer checks:
   - File size matches expected (~68 MiB)
   - All int8 values in [-127, 127]
   - All scales finite and > 0
   - Scale count = 18944
   - No NaN/Inf in weights
5. **Delete** float32 temp buffer immediately after quantization
6. **Repeat** for next layer

### Per-Layer Validation

Each layer produces:
- `ffn_up_layer{N}.int8` — INT8 weights (18944 × 3584 bytes)
- `ffn_up_layer{N}.scale` — per-row scales (18944 × 4 bytes float32)
- Metadata line in validation log

Target: rel_l2 vs dequant reference ≤ 0.02 (same threshold as 3B validation)

### Abort Conditions

Stop immediately if:
- Swap usage increases by >500 MiB during extraction
- RAM available drops below 4 GiB
- Any layer fails validation
- Disk write fails

### Cleanup on Success

After all 28 layers:
- Verify 28/28 files present
- Verify total size ≈ 1.9 GiB
- Delete any intermediate float32 files

## What This Phase Proves

- ✅ Machine health confirmed (10 GiB RAM, 141 GiB disk free, stable swap)
- ✅ No 7B GGUF model exists locally
- ✅ Disk/RAM risk is LOW if a 7B model is acquired
- ✅ Layer-by-layer streaming plan is feasible
- ✅ Estimated sizes calculated

## What This Phase Does NOT Prove

- ❌ No 7B sidecars generated
- ❌ No 7B runtime
- ❌ No 7B quality
- ❌ No 7B speed
- ❌ No production readiness

## Recommended Next Phase

**Phase 14I-A: Acquire candidate 7B model**

Options in order of preference:
1. **Download Qwen2.5-7B-Q4_K_M from HuggingFace** (requires explicit approval — file size ~4.2 GiB)
2. **Use Ollama to pull qwen2.5:7b** and locate GGUF in Ollama cache
3. **Use existing Qwen2.5-3B as proxy** for tooling validation while 7B is acquired

After model acquisition:
- Verify GGUF metadata (layers, hidden, ffn, tensor shapes)
- Confirm tensor orientation matches Qwen2.5-3B pattern
- Generate sidecars layer-by-layer per plan above

## Alternative: 7B Without Download

If Matt approves, the fastest path to 7B validation:
```
ollama pull qwen2.5:7b
# Ollama stores GGUF in ~/.ollama/models/
# Find it and use as source for sidecar extraction
```

However: Ollama's GGUF storage format and tensor layout may differ from the llama.cpp extraction pipeline. Must verify compatibility before extraction.

## Summary Table

| Item | Status |
|------|--------|
| Machine health | ✅ GOOD |
| Local 7B model | ❌ NOT FOUND |
| Disk space (141 GiB free) | ✅ SUFFICIENT |
| RAM (10 GiB available) | ✅ SUFFICIENT |
| Layer-by-layer plan | ✅ DESIGNED |
| Estimated FP32 total | ~7.4 GiB |
| Estimated INT8 total | ~1.9 GiB |
| Peak temp disk (streaming) | ~260 MiB |
| Peak RAM | ~5.5 GiB |
| BLOCKED BY | No local 7B GGUF |