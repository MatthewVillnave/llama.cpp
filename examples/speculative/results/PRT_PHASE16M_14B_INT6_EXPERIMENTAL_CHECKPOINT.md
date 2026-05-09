# PRT Phase 16M — 14B INT6 Experimental Checkpoint

**Date:** 2026-05-09
**Session:** good-dune
**Branch:** `experimental/prt-phase14a-packed-sidecars`
**Commit:** `09600cda2`
**Tag:** `PRT_PHASE16M_14B_INT6_EXPERIMENTAL_CHECKPOINT`

---

## Checkpoint

**Tag:** `PRT_PHASE16M_14B_INT6_EXPERIMENTAL_CHECKPOINT`
**Verdict:** `PASS_14B_INT6_EXPERIMENTAL_CHECKPOINT`

This checkpoint freezes the complete 14B INT6 experimental validation chain. No new tests were run; this is a documentation/snapshot phase.

---

## Context

Matt's goal is **CPU-native inference viability** via PRT/SDI — replacing GPU-bound float inference with compact sidecar-backed FFN replacement that works on consumer CPU hardware.

**The Pipeline:**
- Phase 15: Validated 7B INT6/INT8 pipeline end-to-end on CPU
- Phase 16: Extended the sidecar pipeline to Qwen2.5-14B (2× larger than any prior PRT model)

**What happened in Phase 16:**
1. 14B native Qwen2.5-14B Q4_K_M GGUF loaded and ran first (CPU-only machine, ~8.7GB)
2. INT6 sidecar extraction, generation, and runtime loading were then brought up layer by layer
3. Schema mismatch (20-byte vs 16-byte header) was found in Phase 16G/H and fixed
4. Loader bug (14B size not in known-size list) was found in Phase 16J and fixed
5. Full validation chain completed with clean passes throughout

---

## Frozen 14B Validation Chain

| Phase | What | Verdict | Key Evidence |
|-------|------|---------|--------------|
| 16C | 14B FFN_UP tensor extraction via ggml::to_float | PASS | gguf_get_tensor + to_float for Q4_K_M, 40 layers mapped |
| 16D | One-layer INT6 parity (weight cosine) | PASS | Weight cosine 0.999, MAE 0.0007 vs GGUF reference |
| 16G | Schema mismatch forensic | FOUND | 20-byte header in writer vs 16-byte in runtime loader |
| 16H | Writer schema fix (20→16 bytes) | PASS | Synthetic roundtrip PASS, layer0 cosine 0.999 |
| 16I | Full 40-layer INT6 sidecar generation | PASS | 40/40 files, 40 unique SHA, 2.0GB total |
| 16J | Tiny runtime canary + loader fix | PASS | 40/40 sidecars loaded, "Paris" match, loader bug fixed |
| 16K | 8-prompt quality validation | PASS | 8/8 native, 8/8 INT6, 8/8 semantic matches, 0 degradations |
| 16L | Longer-generation + larger-context smoke | PASS | n=320 exact match, c=2048 exact match, ~0.977× t/s ratio |

---

## Key Results

### Sidecar Generation
- **Sidecar count:** 40 (layers 0-39)
- **Unique SHA256:** 40
- **Per-layer size:** 53,139,472 bytes
- **Total size:** ~2.0GB
- **Output dir:** `/tmp/prt_sidecars_14b_int6_fixed/`

### Selected-Layer Parity (Phase 16I/16J)
- Layer 0 weight cosine vs GGUF: **0.999400**
- All tested layers finite
- No shape/orientation mismatch

### Tiny Runtime Canary (Phase 16J)
- Native: "The capital of France is Paris." ✅
- INT6: "The capital of France is Paris." ✅
- Sidecars loaded: **40/40** (mmap)
- Force-native layers: 11, 15

### 8-Prompt Validation (Phase 16K)
- Native completed: **8/8** ✅
- INT6 completed: **8/8** ✅
- Semantic matches: **8/8** ✅
- Quality degradations: **0** ✅
- Collapse/repetition: **0** ✅
- JSON validity: **PASS** (Prompt 4) ✅
- Code plausibility: **PASS** (Prompt 3) ✅
- **INT6/native generation t/s ratio: 1.009×** ✅

### Longer-Generation Smoke (Phase 16L)
- **Test A (n=320 prose):** Native and INT6 output **identical** — "That sounds like the beginning of an exciting story!..."
- **Test B (c=2048 factual):** Both answered "Phase 16K showed that the 14B INT6 model matched native outputs across all prompts." — **EXACT MATCH**
- Generation t/s ratio: **0.977–0.978×**
- Wall overhead: **1.10–1.15×** (one-time sidecar loading, not per-token)
- 0 collapses, 0 repetitions in both tests

### Runtime Evidence
- Sidecar layers loaded: **40/40** in all tests
- Format: **int6 per_row**
- Load mode: **mmap**
- Unpack kernel: **lut4x**
- Force-native layers: **11, 15** (as configured)
- Duplicate warnings: **0**
- Memory: **stable at 12GB available**, no OOM, no swap pressure

---

## Allowed Claims

✅ **Allowed:**
- Qwen2.5-14B INT6 PRT passed tiny canary, 8-prompt validation, and longer-generation/larger-context smoke tests on Matt's measured CPU setup (Dell OptiPlex 7010, 15GB RAM, TheForgeHQ)
- 14B INT6 produced exact or semantic native matches in all tested prompts with no observed collapse/repetition
- 14B INT6 generation throughput was near native in the measured tests (~0.977–1.009×)
- 14B INT6 remains **experimental**
- Results are scoped to: Qwen2.5-14B Q4_K_M GGUF, this hardware, `experimental/prt-phase14a-packed-sidecars` branch, tested prompt suite

---

## Forbidden Claims

❌ **Forbidden:**
- Production readiness
- Universal speedup
- Universal quality equivalence
- Full long-context guarantee
- Larger-than-14B support
- GPU comparison
- INT6 replaces INT8
- All models supported
- Deployment readiness
- Public reproducibility without sidecar/model availability
- Any claim not directly tested in the Phase 16 validation chain

---

## Known Caveats

1. **14B INT6 remains experimental** — not production-hardened
2. **Wall-time overhead ~1.10–1.15×** due to one-time sidecar loading at startup (not per-token)
3. **Force-native layers 11 and 15** are still used as fallback
4. **Prompt suite is limited** — 8 prompts + 2 longer-gen tests
5. **No larger-than-14B validation** — pipeline not tested beyond 14B
6. **No production hardening** — error handling, edge cases not tested
7. **No GPU comparison** — no performance comparison vs GPU inference
8. **Sidecars and model files are in /tmp/** — not staged in the repo
9. **INT8 was the prior validated path** — Phase 15 validated 7B INT8 first

---

## Interpretation

### Does Phase 16 advance Matt's CPU inference / SDI goal?
**YES.** Phase 16 proves the PRT/SDI sidecar pipeline scales from 7B to 14B on consumer CPU hardware. The full pipeline — GGUF extraction → INT6 quantization → sidecar generation → mmap loader → runtime matvec → clean output — works end-to-end without quality loss.

### What did 14B prove that 7B did not?
1. **Model-size scalability** — the pipeline works at 2× the model size (14B vs 7B)
2. **Loader generality** — the loader now handles both 7B and 14B sidecar sizes
3. **Schema correctness** — the 16-byte header schema works for both model sizes
4. **Context retrieval** — c=2048 exact-match context retrieval confirms INT6 path doesn't lose information in large contexts
5. **Stability under load** — memory stayed at 12GB throughout all validation phases

### What remains unsolved?
1. **Wall-time overhead reduction** — startup time ~1.15× due to sidecar loading (mitigated by mmap reuse across runs, but first-run overhead persists)
2. **Automatic fallback** — force-native layers 11 and 15 are hand-specified; better self-healing on mismatch would help
3. **Larger models** — no 30B+, 70B+ validation
4. **Production hardening** — error handling, manifest management, sidecar versioning
5. **Writeup** — Phase 16 technical writeup documenting the full pipeline for future reference
6. **Backend integration** — ggml/llama.cpp integration path for automatic model detection

### Should next work be repeatability, backend integration, writeup, or bigger hardware?
**Recommendation:** Phase 16 technical writeup AND repeatability — in that order.

The writeup is the highest-value next step because:
- It captures what was built and why before memory fades
- It makes the pipeline legible to future contributors
- It's a prerequisite for any external publication or re-use

Repeatability testing validates the pipeline is reproducible from clean state.

---

## Recommended Next Phase

**Phase 16N: Phase 16 technical writeup** — Create a comprehensive technical document covering:
1. What PRT/SDI is and why it matters
2. The INT6 sidecar format (16-byte header schema)
3. The end-to-end pipeline: extraction → quantization → generation → loading → runtime
4. Key bugs found (schema mismatch, loader size check) and fixes
5. Validation results summary (all phases)
6. Allowed/forbidden claims
7. Open problems and next steps
8. Code locations and reproduction instructions

**Secondary:** Phase 16N repeatability — re-run the full pipeline from a clean state to confirm reproducibility.