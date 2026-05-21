# Phase 26X: SDI Runtime v0.1.1 Writeup Package

## A. Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## B. Current HEAD
`aa6ebd460`

## C. Checkpoint tag
`SDI_PHASE26W_RUNTIME_V0_1_1_CHECKPOINT`

## D. Writeup path
`examples/speculative/SDI_RUNTIME_V0_1_1_WRITEUP.md`

## E. Public summary path
`examples/speculative/SDI_RUNTIME_V0_1_1_PUBLIC_SUMMARY.md`

## F. Allowed claims
- SDI Runtime v0.1.1 is a standalone CPU/RAM model-residency prototype
- Improves auto-policy behavior on qwen2.5:0.5b local Ollama evals
- Reached 0.950 average score on expanded 8-scenario real-task eval
- Achieved 7/8 win/tie against best fixed baselines
- Targeted gates reduced eager packet selection
- Swap stable (0 MB delta) in tested qwen2.5:0.5b runs
- Useful for long/noisy/pinned-fact/open-loop/exact-tool contexts

## G. Forbidden claims
- Production readiness
- Token speedup or latency improvement
- 7B/14B validation
- KV cache modification
- Weight-residency solution
- Agent/OpenClaw/Smart Agent Router integration
- PRT speedup (ggml_map_custom2 memory corruption unresolved)
- Universal superiority
- Token savings on small contexts (overhead is real)
- qwen2.5:0.5b as universal proxy
- Scenario 25 "solved"

## H. Limitations
1. Scenario 25 model ceiling: qwen2.5:0.5b cannot produce natural-format hard-constraint answers
2. Packet overhead on small contexts: real, targeted gates mitigate
3. Exact-tool mode high token cost on tiny contexts
4. No 7B long-context validation
5. No KV cache modification
6. No weight-residency mechanism yet
7. PRT speed path parked (ggml_map_custom2 memory corruption)

## I. Recommended next phase
Phase 26X-R: Expand fixture set + investigate natural-constraint-answer prompt variants.

Optional with explicit approval: qwen2.5:3b comparison on Sc21/Sc25/Sc26 only.

## J. Models/sidecars/f32 refs staged?
No. qwen2.5:0.5b via Ollama HTTP API only. No model files, sidecars, f32 refs, or binaries staged.

## K. Secrets detected?
No. Safety scan passed.

## L. Existing tags altered?
No. Existing tags not modified. No new tag created.