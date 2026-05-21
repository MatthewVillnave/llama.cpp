# Phase 26U: Expanded Real-Task SDI Runtime Eval

## Report Fields

**A. Branch:** `experimental/prt-phase19a-alt-sidecar-backed`

**B. Current HEAD:** `9b218ddcb`

**C. Checkpoint tag verified:** `SDI_PHASE26T_RUNTIME_V0_1_CHECKPOINT` ✅

**D. Model/backend:** `qwen2.5:0.5b` / Ollama HTTP API

**E. Task set:** 8 new realistic scenarios (21-28):
- 21: Research/project handoff (MIT license verification)
- 22: Debug root-cause analysis (session.c memory leak)
- 23: Benchmark result interpretation (MODEL-BENCH-4)
- 24: Tool-output exact retrieval (git/service status)
- 25: Constraint trap (/etc/secrets hard constraint)
- 26: Open-loop continuation (API optimization pipeline)
- 27: Conflicting state update (team size v2.4.1 vs v2.4.0)
- 28: Multi-commit code review (PR #847 bugfix commit)

**F. Baseline comparison table:**

| Scenario | no_packet | recent_only | simple_summary | sdi_packet | auto | best_fixed | win? | auto_policy |
|----------|-----------|-------------|----------------|------------|------|------------|------|-------------|
| 21_research_handoff | **1.000** | **1.000** | **1.000** | 0.850 | 0.850 | no_packet(1.000) | TIE | sdi_packet |
| 22_debug_root_cause | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | no_packet(1.000) | WIN | sdi_packet |
| 23_benchmark_interpretation | 0.850 | 0.850 | 0.850 | 0.850 | 0.850 | no_packet(0.850) | TIE | recent_only |
| 24_tool_output_exact | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | no_packet(1.000) | WIN | sdi_packet |
| 25_constraint_trap | **0.850** | **0.850** | 0.750 | 0.600 | 0.600 | no_packet(0.850) | LOSS | sdi_packet |
| 26_open_loop_continuation | 0.925 | **1.000** | **1.000** | 0.850 | 0.850 | recent_only(1.000) | LOSS | sdi_packet |
| 27_conflicting_state | **1.000** | **1.000** | **1.000** | 0.850 | **1.000** | no_packet(1.000) | WIN | recent_only |
| 28_commit_review | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | no_packet(1.000) | WIN | sdi_packet |

**G. Auto average:** 0.894

**H. Best fixed baseline average:** 0.963

**I. Win/tie count:** 5/8 (5 ties, 0 wins, 3 losses)

**J. qwen2.5:3b status:** Not installed; not pulled.

**K. Token reduction/overhead:**

| Scenario | Policy | Raw tokens | Prompt tokens | Token reduction |
|----------|--------|-----------|---------------|-----------------|
| 21_research_handoff | sdi_packet | 1047 | 682 | +34.9% |
| 22_debug_root_cause | sdi_packet | 786 | 661 | +15.9% |
| 23_benchmark_interpretation | recent_only | 445 | 623 | -40.0% |
| 24_tool_output_exact | sdi_packet | 393 | 743 | -89.1% |
| 25_constraint_trap | sdi_packet | 716 | 659 | +8.0% |
| 26_open_loop_continuation | sdi_packet | 621 | 669 | -7.7% |
| 27_conflicting_state | recent_only | 502 | 667 | -32.9% |
| 28_commit_review | sdi_packet | 352 | 763 | -116.8% |

**Token notes:** Negative = overhead (prompt grew vs raw). Positive = compression. Small contexts (<500 tokens) consistently show overhead when sdi_packet is selected. Large/noisy contexts show compression.

**L. Swap behavior:** 0 MB delta across all 40 runs. Stable at 453-455 MB. ✅

**M. Failures/limitations:**

*Failure 1 — Scenario 21 (research handoff):*
- Question: "Can you verify the license one more time? My compliance team needs the exact license name."
- This is a simple self-contained question. Raw context is ~1000 tokens, but the question only needs "MIT" — which appears in the recent conversation window.
- Auto incorrectly picks sdi_packet → 0.850 vs no_packet(1.0)
- Root cause: `exact_tool_question` regex matches "exact" in "exact license name" → routes to sdi_packet even though context doesn't need compression
- Score difference is small but represents unnecessary policy bloat

*Failure 2 — Scenario 25 (constraint trap):*
- Question: "What if we absolutely had to remove /etc/secrets? Is there any exception path?"
- Required: "No" + "never"
- Auto picks sdi_packet (exact-tool mode) → 0.600 vs no_packet(0.850)
- The exact-tool packet format makes model output structured fields instead of a natural answer
- The constraint is already in pinned_facts as a hard constraint — sdi_packet adds no value but changes output format
- Root cause: exact_tool_question regex matches "exception path" → routes to exact-tool sdi_packet, which formats output as structured [EXACT_TOOL_OUTPUTS] fields rather than a direct natural-language answer

*Failure 3 — Scenario 26 (open-loop continuation):*
- Question: "Resume where we left off. What's the next action?"
- This is an open-loop continuation task. The conversation naturally leads to "implement response compression with gzip middleware."
- Auto picks sdi_packet (pinned project/state facts) → 0.850 vs recent_only/simple_summary(1.0)
- The sdi_packet framing causes the model to output the structured packet format with "Expected evidence:", "Pinned facts:", "Current user request:" — completely breaking the natural conversational flow
- Root cause: auto policy routes to sdi_packet because the question references pinned project/state facts, but the question is a pure continuation that benefits from natural conversational context

*Pattern across 3 failures:* All failures involve sdi_packet being selected when no_packet or recent_only would have been better. The common thread: the questions are self-contained (don't need pinned fact compression) and the sdi_packet format itself disrupts output quality for questions expecting natural conversational answers.

**N. Verdict:** `PARTIAL_MIXED_REAL_TASK_RESULTS` — auto achieves 5/8 win/tie but with 3 losses on scenarios where sdi_packet actively hurts output quality. Auto is safe (0 MB swap, no regressions) but not optimal. The policy is too eager to select sdi_packet for exact-tool and pinned-fact questions.

---

## Verdicts

- `PARTIAL_MIXED_REAL_TASK_RESULTS` — auto wins/ties 5/8, loses 3 where sdi_packet is overused
- `PASS_SWAP_STABLE` — 0 MB delta across all 40 runs
- `PASS_NO_SECRETS` — all fixture data uses fake values
- `PASS_NO_MODEL_FILES_STAGED` — no .gguf/.bin staged
- `BLOCKED_Qwen2_5_3B` — not installed; not pulled
- `FAIL_AUTO_POLICY_EAGER_SDI_PACKET` — 3 losses from sdi_packet being too eagerly selected

## Recommended Next Phase

**Phase 26V-A — Policy refinement for sdi_packet overuse:**

Three specific fixes without changing core packet format:
1. **Scenario 21 fix:** Add a "tiny question" gate — if the raw question is short (<15 words) and no risk flags are active, route to `no_packet` even if `exact_tool_question` matches
2. **Scenario 25 fix:** Add a "hard constraint" gate — if pinned_facts contain hard constraints and the question asks for action recommendation, prefer `no_packet` over `exact-tool sdi_packet` (constraints are already visible in raw context)
3. **Scenario 26 fix:** Add an "open-loop continuation" gate — if `detect_open_loop_continuation()` is true (question asks to resume/continue/next step), prefer `recent_only` or `simple_summary` over `sdi_packet`

OR:

**Phase 26V-B — If policy refinement not pursued:**
Phase 26V — consolidate v0.1 results, write public/internal benchmark summary, refine claim boundaries. Accept that auto policy has limitations on small self-contained questions and open-loop continuations.

## Files Changed

New fixtures (8 scenarios × 3 files each = 24 files):
- `fixtures/sdi_packet_eval/21_research_handoff/`
- `fixtures/sdi_packet_eval/22_debug_root_cause/`
- `fixtures/sdi_packet_eval/23_benchmark_interpretation/`
- `fixtures/sdi_packet_eval/24_tool_output_exact/`
- `fixtures/sdi_packet_eval/25_constraint_trap/`
- `fixtures/sdi_packet_eval/26_open_loop_continuation/`
- `fixtures/sdi_packet_eval/27_conflicting_state/`
- `fixtures/sdi_packet_eval/28_commit_review/`

Reports:
- `results/PHASE26U_EXPANDED_REAL_TASK_EVAL.md`
- `results/phase26u_expanded_real_task_eval.json`