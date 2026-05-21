# Phase 27C: SDI v0.1.1 Doc Update — 3B Edge-Case Validation

## A. Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## B. Current HEAD
`d1e8313ed` (Phase 27B-R commit)

## C. Phase 27B-R Commit
`d1e8313ed`

## D. Docs Updated
- `examples/speculative/SDI_RUNTIME_V0_1_1_CHECKPOINT.md` — added Phase 27B-R section, updated Sc25 limitation wording
- `examples/speculative/SDI_RUNTIME_V0_1_1_WRITEUP.md` — added Phase 27B-R allowed claim, updated Sc25 limitation wording
- `examples/speculative/SDI_RUNTIME_V0_1_1_PUBLIC_SUMMARY.md` — updated Sc25 paragraph to reflect confirmed ceiling

## E. Sc21 Result
3B auto=recent_only score=1.000 — same as 0.5B. Policy validated on both model sizes.

## F. Sc25 Result
3B auto=0.850 via simple_summary, matching no_packet=0.850. Gap closed: 0.5B auto=0.750 vs 3B auto=0.850. MODEL CEILING CONFIRMED — policy correct, model was the limiter.

## G. Sc26 Result
3B auto=recent_only score=1.000 — same as 0.5B. Policy validated on both model sizes.

## H. Interpretation
The Sc25 failure in Phase 26V/26W was a **qwen2.5:0.5b natural-answer capability ceiling**, not an SDI auto-policy flaw. The auto-policy was correctly routing to `simple_summary`. Only the 0.5B model's natural language generation could not produce "No, never" in the expected format simultaneously. Phase 27B-R targeted comparison with qwen2.5:3b confirmed this: gap fully closed at 0.850 on 3B.

This is NOT broad 3B validation. This is NOT a 7B claim.

## I. Allowed Claim Update
Added narrow claim:
- "Phase 27B-R showed qwen2.5:3b closes the Sc25 hard-constraint natural-answer gap entirely on the targeted edge-case test, supporting the interpretation that the prior Sc25 miss was a qwen2.5:0.5b model ceiling, not an SDI auto-policy failure."

## J. Forbidden Claims Preserved
All prior forbidden claims maintained:
- Production readiness, speedup, 7B/14B validation, KV modification, universal superiority, token savings on small contexts, qwen2.5:0.5b as universal proxy

## K. Recommended Next Phase
Phase 27D: 7B safe-context validation planning only — design a minimal 1-2 scenario eval with strict RAM/swap guard, explicit approval required before any 7B run.

## L. Models/sidecars/f32 refs staged?
No. qwen2.5:3b is in Ollama's model store, not staged to repo.

## M. Secrets detected?
No.

## N. Tags touched?
No. No existing tags modified. No new tag created.

---

## Verdict
**PASS_PHASE27C_DOC_UPDATE**
**PASS_MODEL_CEILING_DOCUMENTED**
**PASS_CLAIM_BOUNDARIES_PRESERVED**