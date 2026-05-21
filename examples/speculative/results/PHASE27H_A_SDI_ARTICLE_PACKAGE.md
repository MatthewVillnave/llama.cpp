# Phase 27H-A: SDI Runtime v0.1.1 Article Package

## A. Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## B. Current HEAD
`c8520c7b5`

## C. Article Draft Path
`examples/speculative/SDI_RUNTIME_V0_1_1_ARTICLE_DRAFT.md`

## D. X Thread Draft Path
`examples/speculative/SDI_RUNTIME_V0_1_1_X_THREAD_DRAFT.md`

## E. Claims Boundary Path
`examples/speculative/SDI_RUNTIME_V0_1_1_CLAIMS_BOUNDARY.md`

## F. Allowed Claims
- Standalone CPU/RAM model-residency prototype
- Sub-Dense Inference: reduces active context pressure while preserving correctness
- qwen2.5:0.5B auto policy: 0.950 avg, 7/8 win-tie in local eval
- qwen2.5:3B edge-case confirms 0.5B ceiling, not policy failure
- Tiny qwen2.5:7B canary: stable swap at c=2048/4096 on 2 tasks
- Eval schema fixed with no headline conclusion changes
- Packet builder + memory guard + auto policy + exact-tool mode
- No llama.cpp internals modified, no model weights changed

## G. Forbidden Claims
- Speedup, latency improvement, token throughput gains
- Production readiness, deployment-ready, production-safe
- Broad 7B validation, 14B support, universal model support
- Long-context solved, KV cache modified, weight-residency solved
- Agent/OpenClaw/SAR integration
- PRT as speed path (parked at correctness, no speed win)
- Universal superiority, token savings on small contexts
- qwen2.5:0.5B solves all exact/natural constraint tasks

## H. Limitations
- Local eval only, not public benchmark-level
- Tiny 7B canary (2 tasks at c=2048/4096), no broad 7B validation
- No 14B tested
- No KV cache modification
- No weight-residency solution
- PRT custom-op speed path parked
- Packet overhead on small contexts
- Exact-tool mode high token cost
- qwen2.5:0.5B model ceilings on natural constraint/exact tool tasks

## I. Recommended Next Technical Phase
Option 1: Phase 27H-B — Bounded 7B c=4096 limited-task validation (requires explicit Matt approval, strict memory guard)
Option 2: Phase 27H-C — Structured real-world task eval instead of fixture-based
Option 3: KV/context memory pressure mapping for 7B at higher context lengths
Option 4: Park work until SDI context path stabilizes

## J. Models/sidecars/f32 refs staged?
No. No model files, sidecars, f32 refs, or binaries staged.

## K. Secrets detected?
No. Fake fixture tokens are clearly marked as test fixtures (sk-fake-..., FAKE_DO_NOT_USE, etc.).

## L. Tags touched?
No. No existing tags modified. No new tag created.

---

## Verdict
**PASS_PHASE27H_A_ARTICLE_PACKAGE**
**PASS_CLAIM_BOUNDARIES_INCLUDED**
**PASS_PUBLIC_SAFE_DRAFT**