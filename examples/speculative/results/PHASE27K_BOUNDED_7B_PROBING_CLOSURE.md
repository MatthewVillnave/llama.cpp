# Phase 27K: Bounded 7B Probing Closure

## A. Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## B. Current HEAD
`44a0d3f84`

## C. 7B Probing Timeline

| Phase | Result | Key Finding |
|-------|--------|-------------|
| 27E | Tiny 7B canary passed (c=2048/c=4096) | Model works in guarded runs |
| 27H-B | PASS: WS-512 → WS-4096 all clean | No cliff through WS-4096 |
| 27H-C | FAIL: WS-6144/c=8192 no output | Initially classified as cliff |
| 27H-D | Failure documented | Generation/API cliff, not memory |
| 27I | Design document | Identified need for c=8192 forensics |
| 27J | All 3 forensic tests passed | Prior cliff **superseded** |

## D. Final Bounded 7B Result

**Bounded 7B working-set evidence (complete):**

| Working Set | Context | Result | Score | Swap Δ |
|------------|---------|--------|-------|--------|
| WS-512 | 2048 | OK | 1.0 | 0.0 GB |
| WS-1024 | 2048 | OK | 1.0 | 0.0 GB |
| WS-2048 | 4096 | OK | 1.0 | 0.0 GB |
| WS-4096 | 4096 | OK | 1.0 | 0.0 GB |
| WS-6144 | 8192 | OK (Phase 27J) | 1.0 | 0.0 GB |

**No failure observed at any tested working-set size under clean backend conditions.**

## E. Superseded Cliff

The Phase 27H-C `FAIL_WS6144_CLIFF` is **superseded** by Phase 27J.

**Root causes identified:**
1. **Prompt construction bug:** An unformatted literal `{pin_fact}` placeholder was in the prompt — model received placeholder text instead of the actual fact.
2. **Stuck Ollama runner:** The model runner process from Phase 27H-C became non-responsive at 589% CPU, causing the API to return empty responses for subsequent requests.
3. **Not RAM exhaustion:** Swap stayed flat at ~0.37 GB throughout. RAM never approached the 3 GB guard.
4. **Not a proven context cliff:** Phase 27J's clean forensics show that WS-6144 at c=8192 works correctly when conditions are right.

**Updated conclusion:** No memory, swap, or context cliff was found between WS-4096 and WS-6144 on this backend. The bounded 7B evidence is now complete through WS-6144.

## F. Memory/Swap Interpretation

Swap held flat at **0.0 GB delta** across all 7B runs — from WS-512 through WS-6144. RAM stayed in the 6.5–11 GB range with no guard triggers. The 7B model stays resident in RAM once loaded (~5.3 GB). No evidence of RAM or swap as a constraint for the tested working-set range.

## G. Backend/Prompt Interpretation

**c=8192 works** for qwen2.5:7b on this machine when:
- The Ollama runner is in a clean state
- The prompt is correctly formatted (no literal placeholder strings)

**c=8192 can appear to fail** when:
- The runner is stuck/non-responsive (blocked API calls return empty)
- The prompt contains literal template markers that degrade output quality

This is a runner-state and prompt-construction issue, not a hardware or model capability cliff.

## H. Allowed Claims

- qwen2.5:7b completed bounded SDI/working-set probes through WS-6144 under strict guard on this backend.
- c=8192 works in the clean forensic sequence for tiny, medium, and WS-6144 structured prompts.
- No swap/RAM cliff was observed in the tested range.
- The prior WS-6144 failure was prompt/runner-state related, not a proven memory or context cliff.
- The bounded 7B evidence strengthens SDI Runtime v0.1.1's claim of stable context selection under memory guard.

## I. Forbidden Claims

- ❌ Broad 7B validation
- ❌ 7B safe generally or on other hardware
- ❌ Long-context solved
- ❌ c=8192 always works for qwen2.5:7b
- ❌ 14B support
- ❌ Production readiness
- ❌ Speedup demonstrated
- ❌ KV cache or weight-residency solved
- ❌ WS-6144 is a proven safe upper bound for all backends

## J. Recommended Next Phase

**Phase 27L:** Update the public article/thread package with the corrected bounded 7B result, then pause 7B probing.

The 7B chapter is closed. No further 7B working-set probes are recommended without a new diagnostic reason.

## K. Models/Sidecars/F32 Refs Staged?
No.

## L. Secrets Detected?
No secrets in any committed files.

## M. Tags Touched?
No tags created, modified, or pushed.

---

## Verdicts
- ✅ `PASS_PHASE27K_7B_PROBING_CLOSURE`
- ✅ `PASS_WS6144_C8192_SUPERSEDED`
- ✅ `PASS_NO_MEMORY_CLIFF_FOUND`
- ✅ `PASS_CLAIM_BOUNDARIES_PRESERVED`
- ✅ `BLOCKED_REPO_STATE` (old untracked reports remain untracked)