# PRT Phase 16 — Strict Claims Boundary

This document defines exactly what can and cannot be claimed from the Phase 16 results. Be strict. No overclaiming.

---

## Allowed Claims

### Core result
✅ "Qwen2.5-14B INT6 PRT passed tiny canary, 8-prompt validation, and longer-generation/larger-context smoke tests on Matt's measured CPU-only setup (Dell OptiPlex 7010, 15GB RAM)."

✅ "In the tested prompt suite, Qwen2.5-14B INT6 PRT produced exact or semantic matches with native inference, with near-native generation throughput (~0.977–1.009×)."

✅ "The tested behaviors were preserved across: tiny (8 tokens), 8-prompt suite, n=320 longer prose generation, and c=2048 larger-context factual retrieval."

### Representation approach
✅ "Changing the weight representation — not just the quantization depth — recovered near-native behavior while reducing memory bandwidth pressure."

✅ "Packed INT6 sidecars at 14B scale preserved tested behavior on CPU-only hardware."

### Research direction
✅ "The Phase 14–16 results support SDI (Sub-Dense Inference) as a viable research direction for CPU-native inference viability."

✅ "PRT is a credible experimental SDI path at 14B scale."

### Softer framing (with explicit scope)
✅ "This suggests CPU-only inference may be more viable than the usual dense-transformer-on-CPU framing implies, if the runtime and model representation are designed around CPU constraints."

### Scope reminder
✅ "All results are scoped to: Qwen2.5 Q4_K_M models, Matt's measured CPU hardware (Dell OptiPlex 7010, 15GB RAM), and the tested prompt suite."

---

## Forbidden Claims

### Production / deployment
❌ **"Production ready"** — No production hardening, error handling, or edge-case testing has been done.
❌ **"Deployment ready"** — Same as above.
❌ **"Ready for production use"** — Same.

### Speed / performance
❌ **"Faster than GPU"** — No GPU comparison exists.
❌ **"Universal speedup"** — Only tested on one CPU setup, limited prompt suite.
❌ **"No speed penalty"** — Wall/setup overhead is ~1.10–1.15× on first run.
❌ **"Beats native llama.cpp"** — Generation throughput is equivalent, not faster.
❌ **"Speeds up inference universally"** — Not tested broadly.

### Model / scale coverage
❌ **"Works on all models"** — Only Qwen2.5 Q4_K_M tested.
❌ **"Larger-than-14B support"** — 30B+, 70B+ not tested.
❌ **"Validated for all use cases"** — 14B only tested on limited prompt suite.
❌ **"Universal quality preserved"** — Only tested on the measured prompt suite.

### GPU comparison
❌ **"CPU matches GPU"** — No GPU comparison exists.
❌ **"Better than GPU inference"** — Same.
❌ **"GPU inference unnecessary"** — Same.

### Quality / accuracy
❌ **"No quality loss"** — Only measured on the tested prompt suite, not broadly.
❌ **"Full accuracy preserved"** — Same.
❌ **"Bit-for-bit identical to native"** — Only semantic/exact matches on limited prompts.
❌ **"No hallucinations with PRT"** — Not tested.

### Integration
❌ **"Drop-in replacement for llama.cpp"** — CLI-level only, no ggml integration.
❌ **"Works in any llama.cpp build"** — Requires specific PRT-enabled build.
❌ **"Automatic model detection"** — Uses hardcoded model size checks.

### Reproducibility
❌ **"Anyone can reproduce this"** — No public reproducibility package exists. Sidecars and model not provided.
❌ **"Reproducible without model files"** — Qwen2.5-14B GGUF required.
❌ **"Reproducible without sidecar files"** — Sidecar generation pipeline not packaged.

### The "solved" framing
❌ **"Solved CPU inference"** — This is one experimental result, not a solved problem.
❌ **"CPU inference problem solved"** — Same.
❌ **"The CPU inference problem is addressed"** — Overclaim. One experimental path validated.

---

## Why These Restrictions Exist

### Why no production claim
PRT is experimental:
- No error handling for malformed sidecars
- No graceful fallback for model sizes not in the loader's size list
- No version management for sidecar formats
- No tests for edge cases or corruption detection

### Why no GPU comparison
The Phase 16 testing was entirely CPU-only. There is no GPU measurement in this data. Any GPU comparison would be speculative.

### Why no larger-than-14B claim
The loader uses hardcoded size checks for 7B and 14B. Adding support for larger models would require changes. The pipeline has not been tested beyond 14B.

### Why no universal speedup claim
Throughput was measured on one CPU (Dell OptiPlex 7010), one model (Qwen2.5-14B Q4_K_M), and one prompt suite. Different hardware, models, or prompts may perform differently.

### Why no public reproducibility
The sidecar files and model are not staged in the repository. A public reproducibility package would need: model download, sidecar generation scripts, runtime build, and test harness — none of which are packaged here.

---

## What Was Tested

| Test | Result |
|------|--------|
| Tiny canary (8 tokens) | ✅ Pass — exact match |
| 8-prompt quality suite | ✅ Pass — 8/8 semantic matches |
| n=320 longer prose | ✅ Pass — exact match |
| c=2048 larger context | ✅ Pass — exact match |
| Memory stability | ✅ Pass — 12GB available, no OOM |
| Sidecar loading | ✅ Pass — 40/40 loaded |
| Force-native fallback | ✅ Pass — layers 11, 15 logged |

---

## What Was NOT Tested

| Untested area | Why it matters |
|---------------|----------------|
| Production hardening | Edge cases not tested |
| GPU inference | No comparison data |
| Larger-than-14B models | Unknown behavior |
| Broader benchmark suite (MMLU, HumanEval) | Quality not broadly validated |
| Long-context beyond c=2048 | Context limit unknown |
| Different quantization formats | Only Q4_K_M tested |
| Different hardware | May perform differently |
| Concurrent multi-request load | No stress testing |
| Sidecar format versioning | No version management |

---

*Phase 16O | Strict claims boundary | Branch: experimental/prt-phase14a-packed-sidecars*
