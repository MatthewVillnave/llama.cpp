# Phase 26P - Standalone SDI Runtime Demo Package

## A. Branch

experimental/prt-phase19a-alt-sidecar-backed

## B. Current HEAD

7efdab38bf6b1c46c86ea3741e1e1d204e6bc7f5

## C. Demo README Path

`examples/speculative/SDI_PACKET_RUNTIME_DEMO.md`

## D. Runtime Paths

- `examples/speculative/sdi_packet_runtime.py`
- `examples/speculative/sdi_packet_builder.py`
- `examples/speculative/run_sdi_demo_eval.py`

## E. Memory Guard Path

`examples/speculative/sdi_memory_guard.py`

## F. Demo / Eval Command

Memory guard:

```bash
python3 examples/speculative/sdi_memory_guard.py \
  --model qwen2.5:0.5b \
  --active-context-target 4096
```

Single prompt:

```bash
python3 examples/speculative/sdi_packet_runtime.py \
  --conversation examples/speculative/fixtures/sdi_packet_eval/8_tool_output_exactness/conversation.txt \
  --pinned examples/speculative/fixtures/sdi_packet_eval/8_tool_output_exactness/pinned_facts.json \
  --expected examples/speculative/fixtures/sdi_packet_eval/8_tool_output_exactness/expected.json \
  --tier auto \
  --backend ollama \
  --model qwen2.5:0.5b \
  --baseline auto \
  --packet-style compact \
  --active-context-target 4096 \
  --out /tmp/sdi_demo_out.txt \
  --meta /tmp/sdi_demo_meta.json
```

Repro demo eval:

```bash
python3 examples/speculative/run_sdi_demo_eval.py \
  --model qwen2.5:0.5b \
  --out-dir /tmp/sdi_demo_eval
```

Full baseline comparison:

```bash
python3 examples/speculative/run_sdi_demo_eval.py \
  --model qwen2.5:0.5b \
  --all-baselines \
  --out-dir /tmp/sdi_demo_eval_full
```

Validation run completed successfully with 10 auto-policy rows written to `/tmp/sdi_demo_eval_phase26p`.

## G. Phase 26O Carried-Forward Results

- Auto policy average: 0.764.
- Fixed SDI average: 0.734.
- Best fixed-by-scenario average: 0.777.
- Auto win/tie count vs best fixed baseline: 9/10.
- Long/filler token reductions: +69.0%, +76.8%.
- Swap stable around 454MB, max delta 0MB.
- Tool-output preservation improved by policy routing, but exact tool-output packet mode remains weak on `qwen2.5:0.5b`.

## H. Claim Boundaries

Allowed:

- Standalone SDI auto policy works on current hostile synthetic/local tasks.
- Auto policy manages packet bloat better than always-on packet mode.
- SDI packet mode is promising for long/noisy context and pinned-fact/open-loop risk.
- Memory guard prevents unsafe local runs under configured thresholds.

Forbidden:

- Speedup.
- Production readiness.
- 7B scaling.
- 14B support.
- Broad quality.
- Agent integration.
- OpenClaw integration.
- Smart Agent Router integration.
- PRT speed.

## I. Known Limitations

- Exact tool-output packet mode remains weak on `qwen2.5:0.5b`.
- Paths, commits, numbers, and statuses are preserved in packet text, but the small model may fail to extract every exact value.
- Auto policy mitigates some tiny tool-output cases by choosing `recent_only` or `no_packet`.
- Demo uses synthetic/local fixtures only.
- No speedup, 7B, 14B, production, or broad-quality claim is made.

## J. Recommended Next Phase

Phase 26Q - exact tool-output packet refinement.

Alternative next steps:

- Phase 26Q - real-task eval beyond fixtures.
- Phase 26Q - optional `qwen2.5:3b` test only if explicitly approved or already available.

## K. Models / Sidecars / F32 Refs Staged?

No.

## L. Secrets Detected?

No real secrets in staged Phase 26P artifacts. Repo-wide scan shows existing fake fixture tokens marked `FAKE_DO_NOT_USE`, docs/example API-key strings, and existing source references to API-key handling.

## M. Tags Touched?

No.

## Verdicts

- PASS_PHASE26P_REPRODUCIBLE_DEMO_PACKAGE
- PASS_CLAIM_BOUNDARIES_DOCUMENTED
- PASS_TOOL_OUTPUT_LIMITATION_DOCUMENTED
