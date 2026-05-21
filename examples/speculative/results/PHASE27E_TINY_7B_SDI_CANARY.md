# Phase 27E: Tiny 7B SDI Safe-Context Canary

## A. Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## B. Current HEAD
`77295f0f1` (Phase 27D commit)

## C. qwen2.5:7b Pull/Status
- Pulled: 4.7 GB (qwen2.5:7b Q4_K_M)
- Ollama verified: `qwen2.5:7b 845dbda0ea48 4.7 GB`
- Pull completed cleanly at ~13 MB/s

## D. Preflight RAM/Swap

| Metric | Value |
|--------|-------|
| RAM total | 15 GB |
| RAM available (pre-pull) | ~12 GB |
| RAM used (post-pull, idle) | 7.1 GB |
| RAM available (post-pull) | 8.2 GB |
| Swap total | 4 GB |
| Swap used (pre-7B) | 168 MB |
| Swap used (post-pull, idle) | 677 MB |
| Swap delta from 7B pull | ~509 MB |
| Abort threshold | 1 GB |

**Hard blockers checked:** None triggered. Swap < 1GB, RAM available > 3GB.

## E. Smoke Test Result
**PASS**

- Prompt: "Return only the capital of France."
- Model: qwen2.5:7b, temp=0, output capped
- Output: `"Paris"` — clean, no runaway
- Swap delta: +516 MB (model load into RAM, within expected bounds)
- No timeout, no OOM, no sluggishness

## F. Sc21 Result (c=2048)

| Baseline | Score | Policy Selected | Swap Delta |
|----------|-------|-----------------|------------|
| recent_only | 1.000 | recent_only | -12 MB |
| simple_summary | 1.000 | simple_summary | -1 MB |
| auto | 1.000 | recent_only | 0 MB |

All 3 baselines pass at c=2048. Auto correctly selects recent_only.

## G. Sc23 Result (c=2048)

| Baseline | Score | Policy Selected | Swap Delta |
|----------|-------|-----------------|------------|
| recent_only | 0.850 | recent_only | 0 MB |
| simple_summary | 0.850 | simple_summary | 0 MB |
| auto | 0.850 | recent_only | 0 MB |

Stable at 0.850. Same score as 0.5B and 3B on this scenario. Score limitation is model output format (missing "F1" in response), not policy.

## H. c=2048 Summary

All 6 runs completed cleanly:
- Sc21: 3/3 baselines pass (1.000 each)
- Sc23: 3/3 baselines stable (0.850 each)
- Total swap delta across all c=2048 runs: ~0 MB net
- No abort criteria triggered
- No stale processes
- Output sane throughout

## I. c=4096 Result (auto only, per Phase 27E mini-check)

| Scenario | Baseline | Score | Policy | Swap Delta |
|----------|----------|-------|--------|------------|
| Sc21 | auto | 1.000 | recent_only | 0 MB |
| Sc23 | auto | 0.850 | recent_only | 0 MB |

Both c=4096 auto runs completed cleanly. Swap delta 0 MB. No machine instability.

## J. Swap Behavior

| Phase | Swap Used |
|-------|-----------|
| Pre-7B pull | 168 MB |
| Post-pull, idle | 677 MB |
| After all c=2048 runs | 677 MB |
| After c=4096 runs | 677 MB |

Swap delta from model load: ~509 MB. Swap delta during all SDI runs: ~0 MB net. Stable throughout.

## K. Abort Criteria Triggered?

**No.** All checks passed:
- Pre-run swap < 1GB: PASS (677 MB)
- Swap delta per run < 250MB: PASS (max observed: 0 MB)
- RAM available > 3GB: PASS (8.2 GB)
- No OOM kills: PASS
- No stale processes: PASS
- Output sane: PASS
- No timeout on any run: PASS

## L. Interpretation

**PASS_TINY_7B_CANARY**

Tiny 7B SDI canary passed at c=2048 and c=4096 auto-only under strict memory guard:
- All 8 runs (6 at c=2048 + 2 at c=4096) completed without triggering any abort criterion
- Swap remained flat during actual SDI runs (~0 MB delta)
- RAM stayed at 8.2 GB available throughout
- Auto policy correctly selected recent_only for both Sc21 and Sc23
- Sc21: perfect score (1.000) at all baselines and both context sizes
- Sc23: stable score (0.850) at all baselines and both context sizes — same as 0.5B/3B

Key observations:
- c=2048 and c=4096 are both safe for qwen2.5:7B under SDI policy
- Swap delta during model load (~509 MB) is the main memory cost; SDI context selection itself adds no measurable swap pressure
- The 7B confirms the same policy behavior as 0.5B and 3B on these tasks

## M. Allowed Claim
- "Tiny 7B SDI canary passed at c=2048 and c=4096 under strict memory guard"
- "qwen2.5:7B ran 8 SDI policy runs with stable swap at c=2048/4096 on 2 tasks"
- "Auto policy correctly selected recent_only for both tasks on 7B"
- "No swap pressure detected during SDI runs on 7B at bounded context"

## N. Forbidden Claims
- 7B validated / 7B safe broadly / 7B production ready
- 7B speedup claims
- Long-context solved
- KV cache modified
- Weight-residency solution
- SDI enables larger models universally
- 14B support
- Universal 7B results

## O. Recommended Next Phase

**Phase 27F — SDI v0.1.1 / Phase 27 Final Writeup**

Current work complete:
- SDI Runtime v0.1.1 frozen and documented
- 3B edge-case confirmed (model ceiling on Sc25)
- 7B canary passed cleanly at c=2048/4096
- All claim boundaries preserved

Phase 27F: Formal writeup synthesizing Phase 26V–27E results for external/internal distribution with strict claim boundaries. No new runs required.

Alternative: If more 7B validation is desired, a bounded Phase 27G could test a few more scenarios at c=4096 only — but only with explicit Matt approval and no rush.

## P. Models/sidecars/f32 refs staged?
No. qwen2.5:7B is in Ollama registry only. No model files staged to repo. No sidecars, no f32 refs, no binaries staged.

## Q. Secrets detected?
No.

## R. Tags touched?
No. No existing tags modified. No new tag created.

---

## Verdict
**PASS_TINY_7B_CANARY**
**PASS_NO_ABORT_CRITERIA_TRIGGERED**
**PASS_SWAP_STABLE_AT_C=2048_AND_C=4096**
**PASS_AUTO_POLICY_CORRECT_ON_7B**