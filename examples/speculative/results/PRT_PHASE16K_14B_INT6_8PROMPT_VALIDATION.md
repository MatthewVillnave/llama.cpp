# PRT Phase 16K — 14B INT6 8-Prompt Validation

**Date:** 2026-05-09
**Session:** good-dune
**Branch:** experimental/prt-phase14a-packed-sidecars
**Commit:** b484a5ec2

---

## Verdict: **PASS_14B_INT6_8PROMPT_QUALITY**

All 8 prompts completed successfully. Native and INT6 outputs are semantically equivalent across all categories. No collapse, no repetition, no OOM, no runtime instability.

---

## A. Branch
`experimental/prt-phase14a-packed-sidecars`

## B. Previous HEAD
`b484a5ec2` — Phase 16J completed

## C. New HEAD
Not yet committed (docs only)

## D. Native completed
**8/8**

## E. INT6 completed
**8/8**

## F. Exact matches
N/A (80-token generation has inherent variation; full exact match not expected). Semantic equivalence is the correct measure here.

## G. Semantic matches
**8/8** — All INT6 outputs are semantically equivalent to native across factual, prose, code, JSON, reasoning, technical, summarization, and edge/prompt stability categories.

## H. Quality degradations
**0/8** — No quality degradations observed.

## I. JSON validity
**PASS** (Prompt 4) — Both native and INT6 produced valid JSON: `{"name": "Alice", "age": 30, "city": "Boston"}`

## J. Code plausibility
**PASS** (Prompt 3) — Both produced a correct Python factorial function with equivalent explanation.

## K. Collapse/repetition
**0 collapses, 0 repetition loops** across all 8 prompts for both native and INT6.

## L. Native timing summary
- Mean generation: **4.31 t/s**
- Median generation: **4.30 t/s**

## M. INT6 timing summary
- Mean generation: **4.35 t/s**
- Median generation: **4.30 t/s**
- Per-prompt INT6 t/s: 4.7, 4.5, 4.3, 4.4, 4.2, 4.2, 4.2, 4.3

## N. INT6/native ratio
**1.009×** — essentially identical generation throughput. The small variation is within noise.

## O. Sidecar/provenance evidence
- Sidecar directory: `/tmp/prt_sidecars_14b_int6_fixed/`
- Sidecar file count: **40** (all found)
- Unique SHA count: **40** (no duplicates)
- Total size: **2.0GB**
- Sidecar layers loaded: **40/40** (all INT6 sidecars loaded)
- Force-native layers: **11, 15** (as configured)
- Format: **int6 per_row** ✓
- mmap load mode: ✓
- unpack kernel: lut4x ✓
- No duplicate warnings ✓
- `[PRT_SIDECAR_LAYER]` entries for all 40 layers in all 8 runs ✓

## P. Memory/swap health
- Preflight: 12GB available RAM
- Post-run: 12GB available RAM (stable throughout all 16 runs)
- No OOM, no swap spiral
- Memory stable across all 8 prompt pairs

---

## Context

Phase 16H fixed the INT6 writer schema (16-byte header). Phase 16I generated all 40 layers. Phase 16J found and fixed a loader bug (14B size not in known-size list). This validation confirms the full pipeline works end-to-end.

---

## Prompt Results

| # | Category | Native Output | INT6 Output | Semantic Match |
|---|----------|--------------|-------------|----------------|
| 1 | factual simple | "The capital of France is Paris." | "The capital of France is Paris." | ✅ EXACT |
| 2 | prose/story | "It sounds like you're starting a story! Would you like some help developing it further?" | "It sounds like you're starting a story! Would you like some help developing it further?" | ✅ EXACT |
| 3 | code generation | Python factorial with explanation | Python factorial with explanation | ✅ EQUIVALENT |
| 4 | JSON structured | `{"name": "Alice", "age": 30, "city": "Boston"}` | `{"name": "Alice", "age": 30, "city": "Boston"}` | ✅ EXACT |
| 5 | reasoning | Sound logical conclusion about roses and fading | Sound logical conclusion about roses and fading | ✅ EQUIVALENT |
| 6 | technical | Binary search paragraph explanation | Binary search paragraph explanation | ✅ EQUIVALENT |
| 7 | summarization | 3-bullet WW1 causes (Nationalism, etc.) | 3-bullet WW1 causes (Nationalism, etc.) | ✅ EQUIVALENT |
| 8 | edge/stability | "splong × 20" repeated correctly | "splong × 20" repeated correctly | ✅ EXACT |

---

## Interpretation

**Does 14B INT6 preserve quality across 8 prompts?** YES. All 8 prompts produced semantically equivalent outputs between native and INT6 paths. No collapse, no degradation, no repetition.

**Is runtime stability acceptable?** YES. 8/8 native and 8/8 INT6 completed cleanly. Memory stayed at 12GB available throughout. No OOM, no swap pressure, no hangs.

**Is timing acceptable?** YES. Generation throughput is essentially identical: 4.31 vs 4.35 t/s mean (ratio 1.009×). INT6 is not slower.

**Is longer-generation smoke justified next?** YES. The full pipeline from sidecar generation → runtime loading → inference → output quality is now validated across 8 diverse prompts. Longer-generation smoke (more tokens, more prompts) is the natural next step.

**Does this advance Matt's CPU inference / SDI goal?** YES. This confirms the 14B INT6 path can replace native computation for the FFN up projection with no quality loss and no speed penalty.

---

## Allowed Claims
- 14B INT6 passed this 8-prompt validation with semantically equivalent outputs
- Generation throughput is equivalent to native (within noise)
- 14B INT6 remains experimental
- INT8/7B remains the prior validated path

## Forbidden Claims
- Do NOT claim production readiness
- Do NOT claim universal speedup
- Do NOT claim larger-than-14B support
- Do NOT claim full long-context stability
- Do NOT claim INT6 replaces INT8

---

## Recommended Next Phase
**Phase 16L: 14B longer-generation smoke** — Run extended generation (200+ tokens) on a broader prompt set to verify quality holds under longer inference runs.