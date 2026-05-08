# PRT Phase 14S — Public Release Package

## Verdict

**PASS_PUBLIC_RELEASE_PACKAGE_READY** ✅

## What was created

| File | Purpose |
|------|---------|
| `public/PRT_PUBLIC_X_THREAD.md` | Two X thread versions (Version A: credibility tone; Version B: punchier) |
| `public/PRT_PUBLIC_ARTICLE.md` | Full polished article: "Packed Sidecars Made the Replacement Path Viable" |
| `public/PRT_PUBLIC_README_SUMMARY.md` | GitHub-safe README-style summary (no local paths, no private infrastructure) |
| `public/PRT_PUBLIC_ONE_PAGE.md` | Compact one-page summary for quick reference |
| `public/PRT_PUBLIC_CLAIMS_BOUNDARY.md` | Safe/unsafe phrase guide with approved copy-paste claim |
| `results/PRT_PHASE14S_PUBLIC_RELEASE_PACKAGE.md` | Internal Phase 14S report (this file) |
| `results/phase14s_public_release_package.json` | Structured JSON summary |

All files are in `examples/speculative/public/` or `examples/speculative/results/`.

## Safety scrubbing

- No local machine paths (`/home/`, `/tmp/prt_sidecars`) appear in public files ✅
- No model download paths appear in public files ✅
- No private infrastructure names appear in public files ✅
- No raw logs appear in public files ✅
- No secrets, tokens, or API keys in public files ✅
- Only generic "consumer CPU setup" and "measured CPU setup" language used ✅

## Recommended first public post

**X thread (Version A or B)** — either is claims-disciplined and ready to post.

Version A: credibility/measured tone (safer for technical audience)  
Version B: punchier (higher engagement, still no overclaiming)

## Claim boundary

### Safe public claim (approved for use anywhere)

> "Packed INT8 sidecars let PRT preserve tested output quality while recovering near-native llama.cpp throughput on Qwen2.5-3B and 7B in my measured CPU setup."

### Forbidden in all public contexts

- Universal speedup
- Production readiness
- GPU comparison
- Larger-than-7B claims
- All-task equivalence
- Phase 12 speedup reproduction

## Key numbers (for reference)

| Model | Native t/s | INT8 PRT t/s | Ratio |
|-------|-----------|-------------|-------|
| 3B | 21.0 | 21.0 | 1.000× |
| 7B | 8.75 | 8.69 | 0.993× |

## Recommended next technical phase

**Phase 15A: Longer-context / larger-n stability study**

Before broader publication or deeper optimization work, test whether the near-native result holds at production-scale generation lengths (n=512+) and with longer-context prompts. This is the most important remaining signal before claiming broader applicability.

Alternative next: **INT4 sidecar prototype** (if longer-context is deprioritized).

## Phase 14 series summary

All phases 14A through 14S: **PASS** ✅

5 new public files + 2 internal result files created in this phase.

## Safety

| Check | Status |
|-------|--------|
| Models staged? | NO |
| Sidecars staged? | NO |
| Binaries staged? | NO |
| Temp logs staged? | NO |
| Secrets detected? | NO |
| Private paths in public files? | NO |
| Existing tags touched? | NO |