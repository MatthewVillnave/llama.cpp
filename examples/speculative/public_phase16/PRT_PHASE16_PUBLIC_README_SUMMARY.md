# PRT Phase 16 Public Release — README Summary

## What This Folder Is

This folder contains the **Phase 16 public release package** for the PRT/SDI project.

PRT (Progressive Residual / Packed-sidecar path) is an experimental sidecar-backed FFN replacement implementation for CPU-native inference. SDI (Sub-Dense Inference) is the broader research philosophy.

This package documents the Phase 16 result: PRT working on Qwen2.5-14B Q4_K_M on a CPU-only machine.

## What's In This Folder

| File | Description |
|------|-------------|
| `PRT_PHASE16_PUBLIC_X_THREAD.md` | Twitter/X thread, 10 tweets |
| `PRT_PHASE16_PUBLIC_ARTICLE.md` | Full article on SDI and PRT |
| `PRT_PHASE16_PUBLIC_ONE_PAGE.md` | One-page summary with key results |
| `PRT_PHASE16_PUBLIC_README_SUMMARY.md` | This file |
| `PRT_PHASE16_PUBLIC_CLAIMS_BOUNDARY.md` | Strict allowed/forbidden claims |
| `PRT_PHASE16_PUBLIC_TECHNICAL_FAQ.md` | Technical FAQ |

## What PRT/SDI Is

**SDI** means making inference less wasteful by changing what gets computed and how weights are represented — not by brute-forcing GPU-shaped dense inference on CPU hardware.

**PRT** is the concrete implementation: pre-computed compressed sidecar files for FFN_UP weights that are loaded at startup and swapped into the inference kernel at generation time.

## What the Phase 16 Package Says

- PRT at 14B scale (Qwen2.5-14B Q4_K_M) preserves tested behavior in all validated tests
- Generation throughput is near native in the measured suite (~0.977–1.009×)
- The pipeline is experimental and carefully scoped
- No production, GPU comparison, or universal speedup claims

## What Is NOT Included

This folder contains **documentation only**. It does **NOT** include:
- ❌ GGUF model files
- ❌ Generated sidecar files
- ❌ Compiled binaries
- ❌ Reproducibility scripts
- ❌ Private logs or paths
- ❌ API keys or secrets

## Reproducibility Limits

This package documents the methodology and results but cannot serve as a standalone reproducibility package. Reproducing the full pipeline requires:
- The Qwen2.5-14B Q4_K_M GGUF model (from HuggingFace)
- Sidecar generation from source (tools in `examples/speculative/`)
- A compatible CPU-only runtime environment

## Claim Boundary

See `PRT_PHASE16_PUBLIC_CLAIMS_BOUNDARY.md` for the strict allowed/forbidden claims.

**Allowed:** Results scoped to Qwen2.5 models, measured CPU hardware, tested prompt suite.
**Forbidden:** Production readiness, GPU comparison, universal speedup, larger-than-14B support.

## For More Details

Full technical details and source documentation:
- `examples/speculative/results/PRT_PHASE16N_PRT_SDI_TECHNICAL_WRITEUP.md` — Full technical writeup
- `examples/speculative/results/PRT_PHASE16M_14B_INT6_EXPERIMENTAL_CHECKPOINT.md` — Frozen checkpoint
- `examples/speculative/results/PRT_PHASE16K_14B_INT6_8PROMPT_VALIDATION.md` — 8-prompt validation
- `examples/speculative/results/PRT_PHASE16L_14B_INT6_LONGER_GENERATION_SMOKE.md` — Longer-gen smoke test

---

*Phase 16O | Branch: experimental/prt-phase14a-packed-sidecars*
