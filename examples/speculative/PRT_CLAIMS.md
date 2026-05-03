# PRT Route A — Allowed and Forbidden Claims

**Version:** 1.0
**Date:** 2026-05-02
**Scope:** PRT_ROUTE_A_RC1 and experimental/prt-route-a-phase12

---

## Allowed Claims

The following claims are supported by RC1 validation and may be used in documentation, PR descriptions, or public statements:

### Core Results
- "PRT Route A + L12/L15 achieved ~1.84x average speedup on a controlled mini-suite of 6 prompts"
- "PRT Route A + L12/L15 achieved ~1.79x average speedup across a broader 24-prompt validation suite"
- "Route A replaces native FFN_UP matmul with a GGML custom op in the compute graph — no callback overwrite"
- "Callback overwrite counter is 0 in L12+L15 mode on all tested prompts"
- "Missing required sidecars produce a fatal startup error"
- "No collapse or repetition failures observed across 24 prompts"
- "JSON validity: 4/4 on structured output prompts"

### Technical
- "34/36 layers use PRT (94.4% coverage), 2/36 layers use native fallback (L12+L15)"
- "L12 and L15 are force-native anchors required because pure all-36 PRT produces incorrect output on specific narrative prompts"
- "Sidecar validation runs before generation and exits with code 1 if any required sidecar is missing"
- "The native-anchor policy (which layers are native) is explicit and configurable via --prt-force-native"

### Status
- "Release-candidate experimental branch"
- "Production-candidate fallback policy"
- "Experimental CPU inference acceleration for llama.cpp"
- "Validated on Qwen2.5-3B-Instruct-Q4_K_M model only"

### Reproduction
- "Results reproducible using --prt-mode 5700 --prt-force-native 12,15 on the same model"
- "Requires locally generated sidecar files (~90MB per layer) not included in the repo"

---

## Forbidden Claims

The following claims are NOT supported by RC1 validation and must NOT be used:

### Production Overclaiming
- ~~"Production-ready"~~
- ~~"Production-viable"~~
- ~~"Ready for deployment"~~
- ~~"Safe for general use"~~
- ~~"Production-quality software"~~

### Speed Overclaiming
- ~~"Universal 2x speedup"~~
- ~~"Consistent 2x speedup across all prompts"~~
- ~~"2x speedup guaranteed"~~
- ~~"Speedup guaranteed on all models"~~
- ~~"Near-native quality at 2x speedup"~~

### Validation Overclaiming
- ~~"Validated on all prompts"~~
- ~~"Full broader suite passed"~~
- ~~"All structured outputs are valid JSON"~~
- ~~"Code outputs are syntactically correct"~~
- ~~"No edge cases remain"~~
- ~~"Fully tested"~~
- ~~"JSON/structured fully validated"~~

### Model Generalization
- ~~"Works on all GGUF models"~~
- ~~"Generalizes to larger models"~~
- ~~"Validated on Qwen2.5-7B"~~
- ~~"Tested on Llama models"~~
- ~~"Sidecars are universal across models"~~

### Technical Overclaiming
- ~~"Pure all36 PRT is safe"~~
- ~~"Route A all36 works on all prompts"~~
- ~~"No fallback required"~~
- ~~"Callback path is safe"~~
- ~~"Callbacks are recommended"~~

### Process Overclaiming
- ~~"Upstream-ready"~~
- ~~"Ready for pull request"~~
- ~~"No further testing needed"~~
- ~~"Sidecars are ready to distribute"~~

---

## Why These Boundaries Matter

PRT Route A is research code at a specific stage:
- Speedup is real (~1.79–1.84x) but validated on a local CPU/model setup only
- Quality fix is real but the L12+L15 anchor requirement is not fully understood
- Sidecar validation is real but sidecars are not packaged for distribution
- Generalization is unproven beyond Qwen2.5-3B-Instruct-Q4_K_M
- Token-level divergence is expected (PRT is approximate compute, not bit-exact replacement)
- JSON validity at n > 50 is untested on this machine (memory limited)

Claiming more than what was tested damages credibility when reality arrives.

---

## Claim Review Process

Before publishing anything about PRT Route A, ask:
1. Is this claim in the Allowed list above?
2. Does the RC1 data actually support this number/assertion?
3. Is this tested on the same model and prompt types?

If unsure, err toward less. "Experimental" is a feature, not a liability.

---

## Contact for Questions

For questions about PRT Route A or to report results on different models/prompts:
Open an issue on the fork or contact the maintainer.

---

*Claims document maintained by ELVIS for Matthew Villnave / The ForgeHQ*
