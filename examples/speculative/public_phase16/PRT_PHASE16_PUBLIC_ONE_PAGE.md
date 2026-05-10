# PRT Phase 16 — One-Page Summary

## One-Sentence Thesis
> PRT (Progressive Residual / Packed-sidecar path) is a credible experimental SDI (Sub-Dense Inference) implementation: sidecar-backed compressed FFN replacement that preserved tested behavior up to Qwen2.5-14B Q4_K_M on a CPU-only machine, with near-native generation throughput in the measured prompt suite.

---

## What Was Tested

| Component | Detail |
|-----------|--------|
| **Model** | Qwen2.5-14B Q4_K_M (8.4GB GGUF) |
| **Hardware** | Dell OptiPlex 7010, 15GB RAM, CPU-only |
| **Sidecar format** | INT6 packed per-row (4:3 compression) |
| **Sidecar count** | 40 files, 53MB each, ~2.0GB total |
| **Force-native layers** | 11, 15 (fallback) |

---

## Key Result Table

| Phase | Model | Format | Validation | Result | Throughput |
|-------|-------|--------|------------|--------|------------|
| 14Q | Qwen2.5-7B | INT8 packed | 8-prompt + repeatability | 6/8 exact, 8/8 semantic | 0.993× native |
| 15I | Qwen2.5-7B | INT6 packed | 8-prompt + overhead | All pass, INT4 no-go | ~1.000× native |
| 16K | Qwen2.5-14B | INT6 packed | 8-prompt quality | 8/8 semantic, 0 degradations | 1.009× native |
| 16L | Qwen2.5-14B | INT6 packed | n=320 prose + c=2048 factual | Exact matches | 0.977× native |
| 16M | Qwen2.5-14B | INT6 packed | Checkpoint freeze | 40/40 loaded | Stable memory |

---

## Allowed Claims

✅ **The following are supported by the evidence:**
- Packed INT8 and INT6 sidecar paths preserved tested behavior on Qwen2.5 models in Matt's measured CPU setup
- Qwen2.5-14B INT6 PRT passed tiny canary, 8-prompt validation, and longer-generation/larger-context smoke tests on Matt's measured CPU setup
- Generation throughput was near native in measured tests (~0.977–1.009×)
- SDI as a research direction is supported by the Phase 14–16 evidence

✅ **Softer claim (if framed carefully):**
- "This suggests CPU-only inference may be more viable than the usual dense-transformer-on-CPU framing implies, if the runtime and model representation are designed around CPU constraints."

---

## Forbidden Claims

❌ Production readiness
❌ Universal speedup (across all prompts, hardware, models)
❌ GPU comparison
❌ Larger-than-14B support
❌ All-model support
❌ Broad quality equivalence beyond tested prompt suites
❌ Full long-context guarantee beyond c=2048
❌ Deployment readiness
❌ Public reproducibility without model/sidecar availability
❌ "Drop-in replacement for llama.cpp"
❌ "Solved CPU inference"
❌ "No quality loss generally"
❌ "14B fully validated for all use cases"

---

## Key Caveats

1. **Experimental only** — not production-hardened
2. **Wall/setup overhead** — ~1.10–1.15× on first run (one-time sidecar loading)
3. **Force-native fallback** — layers 11 and 15 are hand-specified, not automatic
4. **Limited prompt suite** — 8 prompts + 2 longer-gen tests only
5. **No larger-than-14B validation** — 30B+, 70B+ untested
6. **No backend/ggml integration** — CLI-level only
7. **No public reproducibility package** — sidecars and model in /tmp/
8. **Results scoped** — Qwen2.5 models, measured CPU hardware, tested prompt suite only

---

## What Comes Next

1. Backend/ggml integration design
2. Technical writeup package for public communication
3. 14B repeatability suite from clean state
4. Next-model feasibility (30B+ or 70B+)

---

*Phase 16O | Branch: experimental/prt-phase14a-packed-sidecars | Results scoped to tested setup*
