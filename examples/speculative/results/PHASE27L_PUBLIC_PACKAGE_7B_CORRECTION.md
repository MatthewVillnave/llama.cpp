# Phase 27L: Public SDI Package — 7B Correction Update

## A. Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## B. Current HEAD
`531d43ca2`

## C. Docs Updated

| Doc | Change |
|-----|--------|
| `SDI_RUNTIME_V0_1_1_ARTICLE_DRAFT.md` | Status line updated from "tiny canary" to "bounded probes WS-512→WS-6144"; limitation text updated; allowed claims updated |
| `SDI_RUNTIME_V0_1_1_X_THREAD_DRAFT.md` | Post 5/8 updated with full bounded probe narrative + correction |
| `SDI_RUNTIME_V0_1_1_CLAIMS_BOUNDARY.md` | Already updated in Phase 27J (Phase 27J correction line present) |
| `SDI_RUNTIME_V0_1_1_PUBLIC_SUMMARY.md` | No old cliff wording present; no changes needed |
| `SDI_PHASE26_27_FINAL_SUMMARY.md` | Updated in Phase 27K with 7B closure + Phase 27L recommendation |

## D. Corrected 7B Claim

**Old framing:**
- "Tiny 7B canary: 8 runs, 2 tasks at c=2048/4096"
- "Only a tiny 2-task canary at c=2048/4096 was run"
- "No broad 7B validation"

**New framing:**
- qwen2.5:7B bounded working-set probes passed WS-512 through WS-6144 under strict memory guard
- c=8192 forensics passed tiny, medium, and WS-6144 structured prompts
- No swap/RAM cliff observed in the tested bounded range
- The earlier WS-6144/c=8192 failure was superseded (prompt bug + stuck runner state, not memory exhaustion)
- **Still not broad 7B validation** — evidence is bounded and narrow for this backend

## E. Forbidden Claims Preserved

- ❌ Broad 7B validation
- ❌ 7B safe generally or on other hardware
- ❌ Speedup demonstrated
- ❌ Long-context solved
- ❌ 14B support
- ❌ Production readiness
- ❌ KV cache or weight-residency solved
- ❌ c=8192 always works universally
- ❌ WS-6144 is a proven safe upper bound generally

## F. Old Cliff Wording Removed/Superseded

| Location | Old wording | Status |
|----------|-------------|--------|
| ARTICLE_DRAFT.md | "Only a tiny 2-task canary at c=2048/4096 was run" | ✅ Replaced with bounded probe framing |
| X_THREAD_DRAFT.md | "Tiny 7B canary: 8 runs, 2 tasks..." | ✅ Replaced with full narrative |
| CLAIMS_BOUNDARY.md | Phase 27J correction line already added | ✅ Present |
| SDI_PHASE26_27_FINAL_SUMMARY.md | Updated in Phase 27K | ✅ Present |

## G. Recommended Next

**Technical pause.** The 7B probing chapter is closed. The public package is updated.

Recommended next steps (Matt's discretion):
1. **Post the corrected X thread** — the revised Post 5/8 is ready to copy-paste
2. **Archive the technical probing branch** — 7B results are documented, no new runs planned
3. **Pause SDI probing** — Phase 27I design doc exists if future KV mapping is needed
4. **Focus on small-model SDI work** — qwen2.5:0.5B and 3B remain the primary evidence base

## H. Models/Sidecars/F32 Refs Staged?
No.

## I. Secrets Detected?
No secrets in any committed files.

## J. Tags Touched?
No tags created, modified, or pushed.

---

## Verdicts
- ✅ `PASS_PHASE27L_PUBLIC_PACKAGE_UPDATE`
- ✅ `PASS_7B_CORRECTION_APPLIED`
- ✅ `PASS_CLAIM_BOUNDARIES_PRESERVED`
- ✅ `PASS_TECHNICAL_PAUSE_RECOMMENDED`
- ✅ `BLOCKED_REPO_STATE` (old untracked reports remain untracked)