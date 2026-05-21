# Phase 26W: SDI Runtime v0.1.1 Checkpoint

## A. Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## B. Current HEAD
`da16a9e2c` (Phase 26V commit)

## C. Previous checkpoint tag
`SDI_PHASE26T_RUNTIME_V0_1_CHECKPOINT` at `9b218ddcb`

## D. v0.1.1 checkpoint doc
`examples/speculative/SDI_RUNTIME_V0_1_1_CHECKPOINT.md`

## E. Phase 26V improvements
- is_simple_factual_recall() gate added: Sc21 fixed (0.850→1.000)
- pinned_parts() fix: HARD CONSTRAINT: prefix detection
- is_hard_constraint_decision() gate: Sc25 partial (0.600→0.750)
- detect_open_loop_continuation() confirmed: Sc26 stable at 1.000
- Gate ordering fix: targeted gates before content-type routing
- Auto avg: 0.894→0.950, win/tie: 5/8→7/8, gap: 0.069→0.013

## F. Allowed claims
- SDI Runtime v0.1.1 improves auto-policy selection over v0.1
- Auto policy reached 0.950 average on 8-scenario qwen2.5:0.5b eval
- Win/tie improved to 7/8 vs best fixed-by-scenario baseline
- Targeted gates reduced eager sdi_packet selection
- Swap stable 0 MB delta on qwen2.5:0.5b
- Standalone, repo-agnostic, non-agentic runtime

## G. Forbidden claims
- Production readiness
- Token speedup or latency improvement
- KV cache modification or weight-residency
- 7B/14B validation
- OpenClaw/Smart Agent Router/PRT integration
- Universal superiority
- Token savings on small contexts
- qwen2.5:0.5b ceiling as universal proxy
- Scenario 25 "solved" (0.750 vs 0.850 gap remains)

## H. Known limitations
1. Scenario 25 model ceiling: qwen2.5:0.5b produces structured constraint output rather than natural answer
2. Packet overhead on small contexts: targeted gates mitigate but don't eliminate
3. Exact-tool mode high token overhead on tiny contexts
4. No 7B long-context validation
5. No KV cache modification
6. No weight-residency mechanism
7. PRT speed path remains parked (ggml_map_custom2 memory corruption unresolved)

## I. Recommended next phase
Phase 26X: Formal writeup of SDI Runtime v0.1.1 with strict claim boundaries. No new eval, no model pulls.

Optional only with explicit approval: Phase 26Y — qwen2.5:3b comparison on Sc21/Sc25/Sc26 only.

## J. Models/sidecars/f32 refs staged?
No. qwen2.5:0.5b via Ollama HTTP API only. No model files, sidecars, f32 refs, or binaries staged.

## K. Secrets detected?
No. Safety scan passed. `secret_warnings` in sdi_packet_builder.py is internal variable naming, not actual secrets.

## L. Tags touched?
No. Phase 26T tag exists at 9b218ddcb and was not modified. New tag SDI_PHASE26W_RUNTIME_V0_1_1_CHECKPOINT created.