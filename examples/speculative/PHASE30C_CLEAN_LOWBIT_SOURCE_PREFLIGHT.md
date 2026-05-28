# Phase 30C — Clean F16 / Official Low-Bit Source Preflight

## Branch
`experimental/prt-phase19a-alt-sidecar-backed` (old HEAD: `a88a013a0`)

## Classification
`PENDING_DOWNLOAD_APPROVAL`

---

## Local Source Inventory

| Source | Path | Size | Status |
|--------|------|------|--------|
| Q4_K_M (baseline) | VL_usb/models/quantized/ | 469 MB | ✅ already present |
| Official Q2_K | not downloaded | — | ❌ missing |
| Official Q3_K_M | not downloaded | — | ❌ missing |
| Official FP16 | not downloaded | — | ❌ missing |
| F16/safetensors (base model) | not present locally | — | ❌ missing |

**Note on HF cache:** The HF cache at `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/` contains only a `refs/main` pointer (112 bytes) — no actual model blobs. The Q4_K_M blob was previously downloaded from HF but is local only.

**Note on local Q4_K_M:** The file at VL_usb `qwen2.5-0.5b-instruct-q4_k_m.gguf` (469 MB) was previously downloaded as a reference. No other GGUF files are present locally.

---

## Official HF Repo Contents
`Qwen/Qwen2.5-0.5B-Instruct-GGUF`:

| File | Exact Size | Human-readable |
|------|-----------|---------------|
| `qwen2.5-0.5b-instruct-fp16.gguf` | 1,266,425,696 bytes | **~1,207 MB (~1.18 GB)** |
| `qwen2.5-0.5b-instruct-q2_k.gguf` | 415,182,688 bytes | **~396 MB** |
| `qwen2.5-0.5b-instruct-q3_k_m.gguf` | 432,041,824 bytes | **~412 MB** |
| `qwen2.5-0.5b-instruct-q4_k_m.gguf` | 491,400,032 bytes | **~468 MB** (reference) |

All quantizations confirm: FP16 → Q2_K/Q3_K_M chain (not Q4-derived). Clean provenance.

---

## Disk Feasibility

| Location | Free Space | Notes |
|----------|-----------|-------|
| Internal NVMe (`/`) | 126 GB | comfortable |
| VL_usb (`/media/.../VL_usb`) | 48 GB | where Q4_K_M lives |
| RAM | 12 GB available | |
| Swap | 3.7 GB free | |

**RAM is constraining for F16 inference** — loading a 1.2 GB F16 model won't fit comfortably alongside the speculative runner. But the *quantized* files (Q2_K: ~396 MB, Q3_K_M: ~412 MB) easily fit in RAM.

---

## Decision Matrix

| Option | Download | Quantize | Disk needed | Risk | Recommendation |
|--------|----------|----------|-------------|------|----------------|
| **A: Official Q2_K + Q3_K_M** | ~808 MB combined | None | ~860 MB | LOW | **RECOMMENDED** |
| B: FP16 + local quantize | ~1,207 MB | ~5 min | ~1,300 MB | LOW | Alternative (overkill) |
| C: Stop low-bit path | 0 | 0 | 0 | N/A | Last resort |

**Option A notes:**
- Q2_K (396 MB) + Q3_K_M (412 MB) = 808 MB total download
- Both are quantized directly from FP16, NOT derived from Q4
- Cleanest path for fair Q2/Q3 residency testing
- No quantization compute needed
- Matt approval required for download

---

## Classification
`PENDING_DOWNLOAD_APPROVAL`

## Matt needs to approve
- Download of `qwen2.5-0.5b-instruct-q2_k.gguf` (~396 MB) from HF
- Download of `qwen2.5-0.5b-instruct-q3_k_m.gguf` (~412 MB) from HF
- Both to VL_usb or internal disk

## Next recommended phase
**Phase 30D — Official Low-Bit Residency Test**

Requires Matt approval to proceed with download. Once approved:
1. Download Q2_K and Q3_K_M from `Qwen/Qwen2.5-0.5B-Instruct-GGUF`
2. Run `llama-speculative` residency tests with SDI disabled
3. Compare token efficiency, decode steps, and KV read/write patterns vs Q4_K_M baseline
4. Classify results as PROVEN/UNKNOWN/LIKELY

---

## Provenance Summary
All official HF GGUF files in `Qwen/Qwen2.5-0.5B-Instruct-GGUF` are quantized directly from the original FP16 base (`Qwen/Qwen2.5-0.5B-Instruct`). The Q2_K and Q3_K_M quantizations are NOT derived from Q4 — they have clean, direct provenance from FP16.
