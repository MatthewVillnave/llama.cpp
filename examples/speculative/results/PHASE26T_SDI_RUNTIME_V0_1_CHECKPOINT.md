# Phase 26T: SDI Runtime v0.1 Checkpoint

## Report Fields

**A. Branch:** `experimental/prt-phase19a-alt-sidecar-backed`

**B. Current HEAD:** `dbbb305ff` (Phase 26S-R: benchmark routing fix)

**C. Checkpoint name:** `SDI_RUNTIME_V0_1_CHECKPOINT`

**D. Artifact list:**
- Core: `sdi_packet_builder.py`, `sdi_packet_runtime.py`, `sdi_memory_guard.py`, `run_sdi_demo_eval.py`
- Tests: `test_sdi_packet_builder.py`, `evaluate_sdi_packet_builder.py`
- Fixtures: `fixtures/sdi_packet/` (synthetic), `fixtures/sdi_packet_eval/` (scenarios 1-20)
- Docs: `SDI_PACKET_RUNTIME_DEMO.md`, `SDI_RUNTIME_V0_1_CHECKPOINT.md`
- Results: Phase 26H through 26S-R reports in `results/`

**E. Consolidated results:**

| Phase | Result |
|-------|--------|
| 26I | Packet builder created, 14/14 tests pass |
| 26J | 5 synthetic scenarios, 43/43 checks, 33-94% token reduction |
| 26K-R2 | Ollama qwen2.5:0.5b probe, avg 0.83/1.0 |
| 26M | Standalone runtime + memory guard built, SDI avg 0.535 |
| 26N | Packet v0.2 refinement, SDI avg 0.535→0.770, win/tie 4/5 |
| 26O | Auto policy, avg 0.764, win/tie 9/10 |
| 26Q | Exact tool-output mode, focused avg 0.900 |
| 26R | 7 realistic tasks, 5/7 wins |
| 26S | Content-aware routing, 6/7 wins |
| 26S-R | Benchmark routing fix, **7/7 wins**, no regression |

**F. Allowed claims:**
- Standalone SDI runtime prototype for CPU/RAM model-residency
- Improves recall on synthetic/realistic hostile context tasks under qwen2.5:0.5b
- Auto policy manages packet bloat better than always-on SDI
- SDI packet useful on long/noisy/pinned-fact/open-loop/exact-tool contexts
- Swap stable (0 MB delta) in all qwen2.5:0.5b Ollama evals

**G. Forbidden claims:**
- Production readiness, speedup, 7B/14B validation, KV modification, weight-residency, agent/OpenClaw/SAR integration, PRT speedup, token savings on small contexts, universal superiority

**H. Known limitations:**
- qwen2.5:0.5b ceiling at 0.475 (2/5 facts) on some benchmark tasks
- Exact-tool mode has high token overhead on tiny contexts
- Packet bloat on small contexts (<350 tokens) — negative token reduction expected
- No 7B long-context validation
- No KV cache modification
- PRT speed path remains parked (Phase 10E blocked)

**I. Recommended next phase:** Phase 26U — expand real-task eval with qwen2.5:0.5b, optionally approved qwen2.5:3b. No 7B/14B.

**J. Models/sidecars/f32 refs staged?** No. Tracked vocab files are in `models/` but no .gguf/.bin/.safetensors staged. build_CI artifacts not staged.

**K. Secrets detected?** No real secrets. All fixture data uses FAKE_DO_NOT_USE markers. Secret scanner correctly flags fake tokens.

**L. Tags touched?** No existing tags altered.

---

## Verdicts

- `PASS_PHASE26T_SDI_RUNTIME_V0_1_CHECKPOINT`
- `PASS_CLAIM_BOUNDARIES_DOCUMENTED`
- `PASS_LIMITATIONS_DOCUMENTED`
- `PASS_REPRODUCTION_INSTRUCTIONS_DOCUMENTED`
- `PASS_NO_SECRETS_DETECTED`
- `PASS_NO_MODEL_FILES_STAGED`
- `PASS_NO_REGRESSION`