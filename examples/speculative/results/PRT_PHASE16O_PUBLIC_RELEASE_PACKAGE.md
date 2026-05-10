# PRT Phase 16O — Public Release Package Report

**Date:** 2026-05-09
**Branch:** `experimental/prt-phase14a-packed-sidecars`
**Previous HEAD:** `07273a449` (Phase 16N)
**New HEAD:** pending commit
**Tag:** None (docs-only commit)

---

## Phase 16O Summary

Phase 16O created a claim-disciplined public release package for the Phase 16 14B PRT/SDI result. No benchmarks, no generation runs, no runtime code changes, no tag creation.

---

## A. Public Files Created (Phase 16O-B through 16O-G)

| File | Size | Purpose |
|------|------|---------|
| `public_phase16/PRT_PHASE16_PUBLIC_X_THREAD.md` | 4.1K | Twitter/X thread, 10 tweets + one-paragraph summary |
| `public_phase16/PRT_PHASE16_PUBLIC_ARTICLE.md` | 8.9K | Full article: "Sub-Dense Inference: Making CPU Inference More Viable" |
| `public_phase16/PRT_PHASE16_PUBLIC_ONE_PAGE.md` | 3.5K | One-page summary with thesis, results table, claims, caveats |
| `public_phase16/PRT_PHASE16_PUBLIC_README_SUMMARY.md` | 3.1K | Repo-facing README: what's in the folder, what's not included |
| `public_phase16/PRT_PHASE16_PUBLIC_CLAIMS_BOUNDARY.md` | 6.0K | Strict allowed/forbidden claims, tested/untested breakdown |
| `public_phase16/PRT_PHASE16_PUBLIC_TECHNICAL_FAQ.md` | 5.9K | 15-question technical FAQ |

**Total public files:** 6 files, ~31.5K

---

## B. Source Docs Used

- `examples/speculative/results/PRT_PHASE16N_PRT_SDI_TECHNICAL_WRITEUP.md` — Full 17K technical writeup
- `examples/speculative/results/PRT_PHASE16M_14B_INT6_EXPERIMENTAL_CHECKPOINT.md` — Frozen 8.1K checkpoint
- `examples/speculative/results/PRT_PHASE16L_14B_INT6_LONGER_GENERATION_SMOKE.md` — 6.0K longer-gen results
- `examples/speculative/results/PRT_PHASE16K_14B_INT6_8PROMPT_VALIDATION.md` — 5.5K 8-prompt validation

---

## C. Public Claims Included

### Core claim (allowed)
> "PRT is now a credible experimental SDI path. In Matt's measured CPU-only setup, sidecar-backed compressed FFN replacement preserved tested behavior up to Qwen2.5-14B Q4_K_M, with near-native generation throughput in the tested prompt suite."

### Softer framing (included with explicit scope)
> "This suggests CPU-only inference may be more viable than the usual dense-transformer-on-CPU framing implies, if the runtime/model representation is designed around CPU constraints."

### Key results documented in public files
- Tiny canary (8 tokens): exact match ✅
- 8-prompt quality suite: 8/8 semantic matches ✅
- n=320 longer prose: exact match ✅
- c=2048 larger context: exact match ✅
- Generation throughput: ~0.977–1.009× native ✅
- Memory stable at 12GB, no OOM ✅
- Phase progression: float32 (too heavy) → INT8 (recovered) → INT4 (too lossy) → INT6 (passed) ✅

---

## D. Forbidden Claims Preserved

All of the following are explicitly excluded from all public files:

| Forbidden claim | How it's handled |
|-----------------|------------------|
| "Production ready" | Explicitly called experimental in all docs |
| "Universal speedup" | Results scoped to measured CPU setup and tested prompt suite only |
| "GPU comparison" | No GPU data referenced anywhere |
| "Larger-than-14B support" | Documented as untested, not claimed |
| "All-model support" | Scoped to Qwen2.5 Q4_K_M only |
| "Broad quality equivalence" | Limited to tested prompt suite |
| "Drop-in replacement for llama.cpp" | CLI-level only, no ggml integration acknowledged |
| "Public reproducibility" | Reproducibility limits documented; no sidecar/model package |
| "Solved CPU inference" | Called "one experimental result" not "solved problem" |
| "No quality loss generally" | Scoped to tested behaviors only |

---

## E. Private-Info Scan Result

**Command:** `grep -R -i -E "api_key|secret|password|github_pat|OPENAI_API_KEY|ANTHROPIC_API_KEY|MINIMAX|BEARER|/home/matthew|/tmp/prt_sidecars|tailscale|telegram|ssh|private key|token" examples/speculative/public_phase16/`

**Result:** CLEAN — all matches are legitimate technical text:
- "per-token" / "tokens" / "tokenization" — normal LLM terminology
- "decompress full GGUF weights per token" — technical description
- "tokens/sec" — throughput unit
- No actual private paths, API keys, Bearer tokens, or secrets detected

---

## F. Models/Sidecars/Binaries Included?

**No.** This is a documentation-only package.

- GGUF model files: NOT included
- Sidecar files: NOT included
- Binaries: NOT included
- Temp logs: NOT included
- Manifests from /tmp: NOT included
- Huge files: NOT included

Public folder contains only: `.md` documentation files (7 total).

---

## G. Deliverables Created Outside Repo

| Deliverable | Path | Size |
|-------------|------|------|
| Zip archive | `~/.openclaw/workspace/prt_phase16_public_release.zip` | pending |
| Combined markdown | `~/.openclaw/workspace/PRT_PHASE16_PUBLIC_RELEASE_COMBINED.md` | pending |

Both contain: 6 public files + 1 internal release report.

---

## H. Safety Scan Before Commit

Pending — to be executed in Phase 16O-K.

Expected state:
- Only `.md` and `.json` files staged from `public_phase16/` and `results/`
- No `.gguf`, `.bin`, `.safetensors`, `.pt`, `.pth` files staged
- No `/tmp/` paths, no private info
- Only 2 new files from Phase 16O (internal report + JSON) staged alongside the 6 public docs

---

## I. Commit Plan

**Files to stage (7 total):**
1. `examples/speculative/public_phase16/PRT_PHASE16_PUBLIC_X_THREAD.md`
2. `examples/speculative/public_phase16/PRT_PHASE16_PUBLIC_ARTICLE.md`
3. `examples/speculative/public_phase16/PRT_PHASE16_PUBLIC_ONE_PAGE.md`
4. `examples/speculative/public_phase16/PRT_PHASE16_PUBLIC_README_SUMMARY.md`
5. `examples/speculative/public_phase16/PRT_PHASE16_PUBLIC_CLAIMS_BOUNDARY.md`
6. `examples/speculative/public_phase16/PRT_PHASE16_PUBLIC_TECHNICAL_FAQ.md`
7. `examples/speculative/results/PRT_PHASE16O_PUBLIC_RELEASE_PACKAGE.md`
8. `examples/speculative/results/phase16o_public_release_package.json`

**Commit message:** "PRT Phase 16O: create public release package"

**No tag created.**

---

## J. Recommended Next Phase

**Phase 16P: Optional Matt Review — Public Package Polish**

The public package is claim-disciplined and technically grounded. Next steps if Matt wants to use it:
1. Review the X thread for length/tone fit
2. Review the article for accuracy
3. Decide if/where to publish (X, blog, arXiv, etc.)
4. Optional: create a minimal reproducibility note (linking to model download + sidecar generation steps)

If no review needed: Phase 16 is complete. The full chain 14→15→16→N→O is documented.

---

*Phase 16O complete* | Branch: `experimental/prt-phase14a-packed-sidecars` | Docs-only commit