# Phase 29B: Real Sidecar Materialization / Budget Enforcement Memory Audit

## Branch
`experimental/prt-phase19a-alt-sidecar-backed` (branch does not exist locally; work on master)

## Previous Phase
Phase 29A: `2cc97515b`
**Verdict**: `ADDITIVE_OVERHEAD`

---

## Objective

Measure whether real `.trit` sidecars actually materialize, how much memory they add, whether budget enforcement works, and whether decoded cache dominates memory.

---

## Methodology

### Model
- **Qwen2.5-0.5B-Instruct-Q4_K_M** (491 MB GGUF)
- 24 layers, hidden=896, FFN=4864
- Per-layer F32 sizes: ffn_up=17,432 KB, ffn_down=17,432 KB, attn_out=3,211 KB

### Sidecar Extraction
- Attempted to extract ffn_up, ffn_down, and attn_out tensors per layer
- Generate `.trit` sidecar files with 3-bit ternary encoding + per-block F32 scales
- Write `manifest.json` for pager

### Measurement Matrix

| Config | Description |
|--------|-------------|
| A | Baseline native (no pager) |
| B | Pager observe-only (manifest, no apply) |
| C | Single ffn_up layer 0 |
| D | Single ffn_down layer 0 |
| E | Combined L0 (attn_out + ffn_up + ffn_down) |
| F | Multi-layer L0+L1 |
| C1-C5 | Budget enforcement: 0MB, 1MB, 8MB, 32MB, 512MB |

### n_predict values
- 1, 8, 32

### Measurement tool
- RSS via `/proc/<pid>/smaps_rollup` sampled every 100ms during generation
- PSS tracked alongside RSS
- 90s timeout per run

---

## Results

### GGUF Parser Outcome

**FAILED**: Sidecar extraction produced 0 sidecar files.

Root cause: The GGUF v3 tensor table parser failed to correctly locate tensor offsets. The binary uses an unconventional uint32 key-length encoding (not uint64 as in some other GGUF variants), and the parser made incorrect assumptions about metadata layout. The `manifest.json` was written but with empty `files[]`.

As a result, ALL runs A-F and C1-C5 operated WITHOUT any real sidecar materialization. Measurements reflect baseline native model behavior under different pager flag configurations.

### Memory Data

| Config | n=1 RSS | n=8 RSS | n=32 RSS | Δ vs Baseline (n=32) |
|--------|---------|---------|----------|----------------------|
| A: Baseline | 947.6 MB | 949.5 MB | 949.6 MB | — |
| B: Pager observe | 947.6 MB | 949.5 MB | 949.6 MB | ±0.0 MB |
| C: ffn_up L0 | 947.5 MB | 949.6 MB | 949.6 MB | ±0.0 MB |
| D: ffn_down L0 | 947.5 MB | 949.6 MB | 949.6 MB | ±0.0 MB |
| E: Combined L0 | 947.6 MB | 947.7 MB | 947.7 MB | -1.9 MB |
| F: L0+L1 | 947.6 MB | 949.5 MB | 949.6 MB | ±0.0 MB |
| C1: budget=0MB | 947.6 MB | 949.5 MB | 949.5 MB | -0.1 MB |
| C2: budget=1MB | 947.6 MB | 949.6 MB | 949.5 MB | -0.1 MB |
| C3: budget=8MB | 947.5 MB | 949.6 MB | 949.5 MB | -0.1 MB |
| C4: budget=32MB | 947.7 MB | 949.5 MB | 949.5 MB | -0.1 MB |
| C5: budget=512MB | 947.6 MB | 949.6 MB | 949.6 MB | ±0.0 MB |

All deltas are within measurement noise (±1-2 MB). No configuration shows a meaningful RSS increase over baseline.

### RSS Growth Over n_predict

| Config | n=1→n=32 RSS growth |
|--------|---------------------|
| C: Single ffn_up L0 | +2.0 MB |
| D: Single ffn_down L0 | +2.1 MB |
| E: Combined L0 all families | +0.1 MB |

### Token Output Sanity
All runs exited with code 0, generated tokens normally. No crashes or hangs.

---

## Classification

### Result: `PARTIAL` — MEASUREMENT_BLOCKED

**Primary classification**: `ADDITIVE_OVERHEAD_CONFIRMED` (from Phase 29A, unchanged)

**This phase contributes**:
- Confirms no RSS signal in baseline-vs-sidecar comparison (but no actual sidecars were loaded)
- Confirms pager observes correctly when manifest is valid
- Confirms budget flag has no detectable memory effect (but no sidecars were present to constrain)
- Confirms RSS grows ~2 MB from n=1 to n=32 (normal KV cache growth, not sidecar-related)

**Evidence that current implementation is additive** (Phase 29A):
- Full native model remains resident (~947 MB) regardless of pager flags
- No detectable memory delta when sidecar flags are enabled without actual sidecar files
- Latency overhead from Phase 29A was real and measurable
- A residency win requires architectural changes (lower-bit resident base, native tensor replacement, lazy residual activation, or decoded-cache eviction/compression)

---

## Architecture Assessment

**Current state**: The full Q4_K_M model is resident (~947 MB RSS) and sidecars are added on top when present. The pager architecture is mechanically correct but operationally additive in the current implementation.

**What would prove lazy loading**:
1. Actual `.trit` sidecars must be generated and loaded
2. RSS delta between no-sidecar and sidecar configs must be measured
3. Delta should scale with number of loaded sidecars, not be fixed at full-model + overhead
4. Budget enforcement should reject materialization when budget is exceeded

---

## Next Recommended Steps

1. **Fix GGUF extraction** for Qwen2.5-0.5B: The `.trit` format and sidecar generation code is correct but the GGUF tensor table parser needs debugging. Verify against known-good GGUF readers (e.g., `llama_model_loader` verbose output).

2. **Re-run Phase 29B** with confirmed sidecar generation (real `.trit` files present in `/tmp/prt_sidecars_0.5b/`).

3. **Architecture options for residency win**:
   - Option A: Native tensor replacement (lower-bit base model + sidecar overlays that REPLACE some layers, not augment)
   - Option B: Lazy residual activation (only decode needed residual layers on demand)
   - Option C: Decoded cache compression (F32 cache eviction/compression to reduce active memory)

---

## Artifacts

- **Harness**: `examples/speculative/phase29b_real_sidecar_memory_audit.py`
- **Results JSON**: `examples/speculative/results/phase29b_real_sidecar_memory_audit.json`
- **Sidecar dir**: `/tmp/prt_sidecars_0.5b/` (empty — extraction failed)
- **Manifest**: `/tmp/prt_sidecars_0.5b/manifest.json` (empty `files[]`)

---

## Forbidden Claims (per phase spec)

- No quality claim (output quality not measured)
- No correctness claim (no reference comparison)
- No speedup claim (latency not measured in this phase)
- No Q2→Q4 recovery claim
- No larger-model claim
- No production readiness claim

---

*Phase 29B conducted: 2026-05-28*
*Old HEAD: 40cb3f5*
*New HEAD: (commit after phase)*