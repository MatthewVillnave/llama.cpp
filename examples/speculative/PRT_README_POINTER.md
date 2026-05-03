# PRT Route A Documentation

**⚠️ This is an experimental fork. Not production-ready. Not upstream-ready.**

This fork contains PRT (Progressive Residual Ternary) Route A implementation — an experimental CPU inference acceleration layer for llama.cpp.

---

## Where to Start

| Document | Purpose |
|----------|---------|
| **[PRT_OVERVIEW.md](./examples/speculative/PRT_OVERVIEW.md)** | What PRT is, how Route A works, why L12+L15 |
| **[PRT_PHASE12_FINAL_WRAPUP.md](./examples/speculative/PRT_PHASE12_FINAL_WRAPUP.md)** | Full Phase 12 validation summary |
| **[PRT_PHASE12E_POSTMORTEM.md](./examples/speculative/results/PRT_PHASE12E_POSTMORTEM.md)** | Phase 12E L11+L15 vs L12+L15 comparison, L11+L15 recommended |
| **[PRT_ROUTE_A_RC1_SUMMARY.md](./examples/speculative/PRT_ROUTE_A_RC1_SUMMARY.md)** | RC1 baseline results |
| **[PRT_CLAIMS.md](./examples/speculative/PRT_CLAIMS.md)** | What you can and cannot claim |
| **[PRT_REPRODUCTION_NOTES.md](./examples/speculative/PRT_REPRODUCTION_NOTES.md)** | How to reproduce results |
| **[PRT_SIDECAR_MANIFEST_SPEC.md](./examples/speculative/PRT_SIDECAR_MANIFEST_SPEC.md)** | Sidecar format specification |
| **[PRT_PUBLIC_READINESS_CHECKLIST.md](./examples/speculative/PRT_PUBLIC_READINESS_CHECKLIST.md)** | Pre-public safety checklist |

---

## Quick Summary

- **Branch:** `experimental/prt-route-a-phase12e-l11-l15` (L11+L15 candidate branch)
- **Tags:** `PRT_ROUTE_A_RC1` (frozen RC1 baseline), `PRT_PHASE12_VALIDATION_CHECKPOINT` (Phase 12 checkpoint)
- **Phase 12 checkpoint default policy:** `--prt-mode 5700 --prt-force-native 12,15`
- **Phase 12E recommended policy:** `--prt-mode 5700 --prt-force-native 11,15` (L11+L15 validated as stronger on Qwen2.5-3B)
- **Validated speedup:** ~1.79x Phase 12 checkpoint | ~1.82x Phase 12E L11+L15
- **Model:** Qwen2.5-3B-Instruct-Q4_K_M (not included — obtain separately)
- **Sidecars:** Not committed (generate locally from model)

---

## Important Caveats

- **Experimental only** — do not claim production-ready
- **Single model validated** — not tested on all models
- **Sidecars required** — must be generated from the same model
- **Not upstream** — this is a fork, not a PR to llama.cpp
- **L12+L15 is Phase 12 checkpoint default** — historically validated
- **L11+L15 is Phase 12E recommended** — validated as stronger candidate on this tested setup
- **Start with Phase 12 final wrap-up**, then Phase 12E postmortem for latest results

---

## File Index

- `examples/speculative/PRT_OVERVIEW.md` — High-level explanation
- `examples/speculative/PRT_PHASE12_FINAL_WRAPUP.md` — This phase's complete results
- `examples/speculative/PRT_CLAIMS.md` — Allowed and forbidden claims
- `examples/speculative/PRT_REPRODUCTION_NOTES.md` — Reproduction guide
- `examples/speculative/PRT_SIDECAR_MANIFEST_SPEC.md` — Sidecar format
- `examples/speculative/prt_validate_sidecars.py` — Sidecar validator
- `examples/speculative/results/` — All phase result files

---

*For the main llama.cpp README, see the parent directory.*