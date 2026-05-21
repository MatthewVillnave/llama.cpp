# Phase 27G: SDI Phase 26/27 Final Summary

## A. Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## B. Current HEAD
`c747e0845`

## C. Checkpoint Tags
- `SDI_PHASE26W_RUNTIME_V0_1_1_CHECKPOINT`

## D. Summary Doc Path
`examples/speculative/SDI_PHASE26_27_FINAL_SUMMARY.md`

## E. Consolidated Evidence

| Phase | Evidence | Status |
|-------|---------|--------|
| 26I | Packet builder deterministic, 14 tests | PASS |
| 26J | Synthetic eval 5/5, 43/43 checks | PASS |
| 26K-R2 | qwen2.5:0.5b packet probe avg 0.83, swap stable | PASS |
| 26M | Standalone runtime built, hostile eval mixed | PARTIAL |
| 26N | Packet v0.2, avg 0.770, wins 4/5 | PASS |
| 26O | Auto policy 9/10 win/tie vs best fixed | PASS |
| 26T | v0.1 checkpoint frozen | PASS |
| 26U | Expanded eval mixed, eager packet issue found | PARTIAL |
| 26V | Targeted gates, avg 0.950, 7/8 win/tie | PASS |
| 26W | v0.1.1 frozen and tagged | PASS |
| 27B-R | 3B closes Sc25 gap, model ceiling confirmed | PASS_NARROW |
| 27E | 7B canary 8 runs, swap stable, no aborts | PASS_NARROW |
| 27E-R | Scoring audit confirms no false pass | PASS |
| 27F | Schema bug fixed, no headline changes | PASS |

## F. Allowed Claims
- SDI Runtime v0.1.1 is a standalone CPU/RAM model-residency prototype
- Structured context packets + memory guard + auto policy reduce harmful context pressure
- Improved auto-policy on qwen2.5:0.5b: 0.950 avg, 7/8 win/tie after targeted gates
- 3B comparison: Sc25 was model ceiling not policy failure
- Tiny guarded 7B canary: stable swap at c=2048/4096 on 2 tasks
- Eval schema fixed with no headline conclusion changes

## G. Forbidden Claims
- Production readiness
- Speedup achieved
- Broad 7B validation
- 14B support
- Long-context solved
- KV cache modified
- Weight-residency solved
- Agent/OpenClaw/SAR integration
- PRT speedup
- Universal superiority
- Token savings on small contexts
- qwen2.5:0.5b solves all exact/natural constraint tasks

## H. Known Limitations
- No broad 7B eval (only tiny 2-task canary at c=2048/4096)
- No 14B tested
- No KV cache modification
- No weight-residency solution yet
- Model ceilings remain on qwen2.5:0.5B
- Packet overhead real on small contexts
- Exact-tool mode has high token cost
- Current evals are local/limited, not public benchmark-level
- PRT custom-op speed path parked

## I. Recommended Next Direction
1. Phase 27H-A: public/internal article package
2. Phase 27H-B: bounded 7B c=4096 limited-task validation (with Matt approval)
3. Parked: PRT v3 weight-residency design
4. Parked: KV memory probe design

## J. Models/sidecars/f32 refs staged?
No. No model files, sidecars, f32 refs, or binaries staged.

## K. Secrets detected?
No. Fake fixture tokens (e.g., sk-fake-...) are clearly marked as test fixtures and are not real secrets.

## L. Tags touched?
No. No existing tags modified. No new tag created.

---

## Verdict
**PASS_PHASE27G_FINAL_SUMMARY**
**PASS_EVIDENCE_CONSOLIDATED**
**PASS_CLAIM_BOUNDARIES_FINALIZED**
**PASS_LIMITATIONS_DOCUMENTED**