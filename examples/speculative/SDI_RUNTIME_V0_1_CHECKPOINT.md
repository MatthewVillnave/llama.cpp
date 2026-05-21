# SDI Runtime v0.1 Checkpoint

**Phase:** 26T | **Branch:** `experimental/prt-phase19a-alt-sidecar-backed` | **HEAD:** `dbbb305ff`
**Date:** 2026-05-21 | **Model:** qwen2.5:0.5b (Ollama) | **Status:** CHECKPOINT

---

## 1. What SDI Runtime v0.1 Is

A standalone Model-Residency Sub-Dense Inference (SDI) prototype that:

- Builds compressed SDI context packets from raw conversation + pinned facts
- Preserves pinned facts, hard constraints, open loops, exact tool outputs, and recent state
- Applies memory/swap guard before running local inference
- Chooses prompt compression policy automatically using content-aware rules
- Compares multiple baselines (no_packet, recent_only, simple_summary, sdi_packet)
- Uses Ollama HTTP API with `qwen2.5:0.5b` for all current evaluations
- Makes no modifications to llama.cpp internals, PRT, OpenClaw, or Smart Agent Router

---

## 2. What It Is NOT

**Not:**
- Agentic or autonomous
- OpenClaw integration
- Smart Agent Router integration
- PRT speedup or kernel work
- MoE implementation
- llama.cpp kernel modification
- Production-ready
- A speedup claim
- A 7B/14B validation
- KV cache modification
- Weight-residency solution

---

## 3. Correct SDI Thesis

> Sub-Dense Inference is a **CPU/RAM survival strategy for dense models**: reduce how much context, KV, weight memory, and dense compute must be active per token while preserving correctness through fallback and verification.

This framing is NOT about agent routing, not about OpenClaw integration, and not about PRT speedup.

---

## 4. Validated Results by Phase

### Phase 26I — Packet Builder Creation
- SDI packet builder (`sdi_packet_builder.py`) created
- Deterministic output
- 14/14 unit tests passed
- Pinned facts preserved in structured format

### Phase 26J — Synthetic Eval
- 5 synthetic scenarios, 43/43 checks passed
- Token reduction 33–94% on synthetic压迫 tests
- Packet format stable under synthetic conditions

### Phase 26K-R2 — Ollama Backend Probe
- Ollama HTTP API with `qwen2.5:0.5b` smoke test passed
- Average score 0.83/1.0 on packet probe
- Packet improved project-specific recall vs no-packet baseline
- Swap stable (423-455 MB range)

### Phase 26L — Measurement Baseline + Hostile Eval Plan
- Defined measurement methodology
- Identified token reduction formula
- Swap behavior characterized: 0 MB per-run delta

### Phase 26M — Standalone Runtime + Memory Guard
- Built standalone runtime + memory guard (`sdi_memory_guard.py`)
- Initial hostile eval: mixed results
- SDI avg 0.535 vs baselines (no_packet 0.470, simple_summary 0.450, recent_only 0.270)
- Win/tie 2/5 on hostile synthetic eval

### Phase 26N — Packet v0.2 Refinement
- Added ANSWER_TARGET, small-model instruction, structured open loops, inferred tool/output facts, current truth/superseded facts
- SDI avg improved: 0.535 → 0.770
- Win/tie 4/5 on hostile eval
- Limitation: tool-output preservation still failed; compact packet bloat on small contexts

### Phase 26O — Auto Policy + Realistic Eval
- Added auto policy selecting among no_packet, recent_only, simple_summary, sdi_packet
- 10 realistic scenarios (5 from Phase 26O, 5 from existing fixture set)
- Auto avg 0.764; fixed SDI avg 0.734; best fixed-by-scenario avg 0.777
- Auto win/tie 9/10 vs best fixed baseline
- Tool-output improved by policy routing (exact questions → sdi_packet, tiny contexts → recent_only/no_packet)
- Swap: 0 MB delta

### Phase 26P — Demo Packaging
- Standalone demo packaged (`run_sdi_demo_eval.py`, `SDI_PACKET_RUNTIME_DEMO.md`)
- Demo runner validated with 10 auto-policy rows
- Claim boundaries documented
- Output written to /tmp only, not staged in repo

### Phase 26Q — Exact Tool-Output Refinement
- Added `[EXACT_TOOL_OUTPUTS]` mini-format with exact command, path, commit, metric, value, unit, status, error fields
- New exact-output fixtures: avg 0.900 with fixed SDI/auto
- All 5 focused tool-output scenarios: fixed SDI/auto avg 0.750
- Limitation: exact mode has high token overhead on tiny contexts (-600% to -727% on tiny exact cases)

### Phase 26R — Real-Task Eval Beyond Fixtures
- 7 new realistic engineering tasks (14-20): partial spec review, benchmark results, config drift, code change review, multi-tool workflow, memory pressure decision, date deadline conflict
- Auto: 5/7 wins/ties vs best fixed baseline
- Scenario 17 (code change review): sdi_packet scored 0.100, simple_summary scored 0.700 — auto overused sdi_packet
- Scenario 15 (benchmark results): auto scored 0.475, sdi_packet scored 0.625
- Swap: 0 MB delta across 35 runs

### Phase 26S — Content-Aware Auto Policy
- Added `detect_multi_commit_git_review()` and `detect_benchmark_result()` content classifiers
- Added `question_wants_summarize()` and `question_wants_exact()` routing functions
- Scenario 17 fixed: auto routes to simple_summary, score 0.700
- Auto: 6/7 wins/ties vs best fixed baseline (auto avg 0.554 vs Phase 26R 0.454)
- Swap: 0 MB delta

### Phase 26S-R — Benchmark Routing Fix
- Swapped `wants_summary`/`wants_exact` branch priority for benchmark results
- Scenario 15: qwen2.5:0.5b ceiling identified at 0.475 (2/5 facts) regardless of routing
- Routing now correct: auto routes to sdi_packet for exact benchmark questions
- Auto: **7/7 wins/ties** vs best fixed baseline
- Scenario 17 stays fixed at 0.700
- Swap: 0 MB delta

---

## 5. Current Policy Behavior

Auto policy selects among:

| Policy | Use When |
|--------|----------|
| `no_packet` | Tiny self-contained context, no fact risk |
| `recent_only` | Recent window smaller than raw, no durable fact risk |
| `simple_summary` | Multi-commit git review, benchmark summarization questions |
| `sdi_packet` | Pinned facts, open loops, conflicting old/new state, long/noisy context, exact tool-output questions |
| exact-tool (via sdi_packet) | Exact path/hash/number/status/error questions |

**Key rules:**
- Do not packetize tiny contexts unless risk flags require it
- Use memory guard to block unsafe local inference (swap pressure, low RAM)
- Route benchmark/score tables to simple_summary for summary questions, sdi_packet for exact questions
- Route multi-commit code review to simple_summary unless exactness is explicitly requested
- Swap must remain at 0 MB delta; if swap grows, context must be compressed

---

## 6. Claims Allowed

✅ **Allowed:**
- SDI Runtime v0.1 is a standalone prototype for CPU/RAM model-residency
- It improves recall on synthetic and realistic hostile context tasks under `qwen2.5:0.5b`
- Auto policy manages packet bloat better than always-on SDI packet mode on small contexts
- SDI packet is useful on long/noisy/pinned-fact/open-loop/exact-tool contexts with `qwen2.5:0.5b`
- Swap remained stable (0 MB delta) in all tested `qwen2.5:0.5b` Ollama eval runs
- Current best evidence supports continuing SDI packet/context work

---

## 7. Claims Forbidden

❌ **Do NOT claim:**
- Production readiness
- Speedup (any model, any task)
- 7B or 14B validated runtime
- Universal packet superiority
- Agent integration
- OpenClaw integration
- Smart Agent Router integration
- PRT speedup or kernel modification
- Token savings apply to small contexts (packet overhead makes small contexts grow)
- `qwen2.5:0.5b` can solve all exact extraction tasks (scenario 15 proves ceiling)
- KV cache modification results
- Weight-residency solution

---

## 8. Known Limitations

1. **qwen2.5:0.5b has a ceiling** on benchmark/result extraction tasks requiring 5+ scattered exact values — scenario 15 ceiling is 0.475 (2/5 facts)
2. **Exact-tool mode has high token overhead** on tiny contexts (-600% to -727% on smallest cases)
3. **Packet bloat on small contexts** — negative token reduction is expected for contexts <350 tokens
4. **No 7B long-context validation** — all results are `qwen2.5:0.5b` only
5. **No KV cache modification** — this is a prompt-level strategy, not a kernel modification
6. **No weight-residency solution** — SDI is a context/prompt strategy, not a weight streaming solution
7. **Results are local/limited** — not public benchmark-level validation
8. **PRT speed path remains parked** — Phase 10E blocked; PRT_3P standalone research only
9. **Swap was already active** (453-455 MB) at eval start — swap behavior under cold start not tested

---

## 9. How to Reproduce

### Run full demo eval (10 scenarios)

```bash
cd /home/matthew-villnave/llama.cpp
python3 examples/speculative/run_sdi_demo_eval.py \
  --model qwen2.5:0.5b \
  --all-baselines \
  --out-dir /tmp/sdi_demo_eval
```

### Run single scenario

```bash
python3 examples/speculative/sdi_packet_runtime.py \
  --conversation examples/speculative/fixtures/sdi_packet_eval/1_pinned_early_fact/conversation.txt \
  --pinned examples/speculative/fixtures/sdi_packet_eval/1_pinned_early_fact/pinned_facts.json \
  --expected examples/speculative/fixtures/sdi_packet_eval/1_pinned_early_fact/expected.json \
  --model qwen2.5:0.5b \
  --baseline auto \
  --packet-style compact \
  --active-context-target 4096 \
  --timeout-s 60 \
  --out /tmp/output.txt \
  --meta /tmp/meta.json
```

### Check memory guard

```bash
python3 examples/speculative/sdi_memory_guard.py \
  --model qwen2.5:0.5b \
  --active-context-target 4096
```

### Prerequisites
- Ollama running with `qwen2.5:0.5b` pulled
- 15 GB RAM available
- Swap configured (4 GB)
- No other local inference workloads during eval

---

## 10. Recommended Next Phase

**Phase 26U** — Expand real-task eval with:
- Current 7 realistic scenarios (14-20) with fixed content-aware policy
- 5 exact tool-output scenarios (11-13 from Phase 26Q + 5 from Phase 26O)
- Token reduction analysis only on large/noisy contexts (not small contexts)
- Swap monitoring under sustained multi-run load
- Optionally: `qwen2.5:3b` if explicitly approved and available (no 7B)

**Do NOT** run 7B, 14B, or modify llama.cpp internals in Phase 26U.

---

## Artifact List

**Core scripts:**
- `examples/speculative/sdi_packet_builder.py`
- `examples/speculative/sdi_packet_runtime.py`
- `examples/speculative/sdi_memory_guard.py`
- `examples/speculative/run_sdi_demo_eval.py`

**Tests/eval:**
- `examples/speculative/test_sdi_packet_builder.py`
- `examples/speculative/evaluate_sdi_packet_builder.py`
- `examples/speculative/fixtures/sdi_packet/` (synthetic)
- `examples/speculative/fixtures/sdi_packet_eval/` (scenarios 1-20)

**Docs/reports:**
- `examples/speculative/SDI_PACKET_RUNTIME_DEMO.md`
- `examples/speculative/SDI_RUNTIME_V0_1_CHECKPOINT.md` ← this file
- `examples/speculative/results/PHASE26T_SDI_RUNTIME_V0_1_CHECKPOINT.md`
- Phase 26H through 26S-R phase reports in `results/`