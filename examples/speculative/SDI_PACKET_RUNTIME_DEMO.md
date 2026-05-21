# Standalone SDI Packet Runtime Demo

## What This Demo Is

This is a standalone Model-Residency SDI packet runtime demo.

It:

- Builds compressed `SDI_CONTEXT_PACKET` prompts from raw context, pinned facts, constraints, open loops, and tool facts.
- Applies a memory/swap guard before local inference.
- Chooses prompt policy automatically with `--baseline auto`.
- Compares `no_packet`, `recent_only`, `simple_summary`, `sdi_packet`, and `auto`.
- Uses the local Ollama HTTP backend.
- Records quality, estimated token reduction, selected policy, swap delta, and output sanity.

The core purpose is to test whether SDI packetization and policy selection can reduce active context/KV pressure while preserving enough answer quality for constrained CPU/RAM local inference.

## What This Demo Is Not

This is not:

- Agentic.
- OpenClaw integration.
- Smart Agent Router integration.
- PRT.
- MoE.
- Production ready.
- A speedup claim.
- A 7B claim.
- A 14B claim.
- A broad quality claim.

## Requirements

- Ollama running locally.
- `qwen2.5:0.5b` installed in Ollama.
- Python 3 standard library.
- Safe swap state: swap used must be <= 1GB for local inference.

Check model availability:

```bash
ollama list
```

Check memory guard:

```bash
python3 examples/speculative/sdi_memory_guard.py \
  --model qwen2.5:0.5b \
  --active-context-target 4096
```

## Run One Demo Prompt

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

## Rerun Demo Eval

The helper script runs the Phase 26O fixture set and writes generated outputs outside the repo.

```bash
python3 examples/speculative/run_sdi_demo_eval.py \
  --model qwen2.5:0.5b \
  --out-dir /tmp/sdi_demo_eval
```

By default it runs `auto` only. To compare all prompt modes:

```bash
python3 examples/speculative/run_sdi_demo_eval.py \
  --model qwen2.5:0.5b \
  --all-baselines \
  --out-dir /tmp/sdi_demo_eval_full
```

If `PRT_SCRATCH` is set, you can use it for outputs:

```bash
python3 examples/speculative/run_sdi_demo_eval.py \
  --model qwen2.5:0.5b \
  --out-dir "$PRT_SCRATCH/phase26p_demo"
```

Generated outputs are intentionally not written to the repo.

## Known Phase 26O Results

Using Ollama HTTP API with `qwen2.5:0.5b`:

- Auto policy average: 0.764.
- Fixed SDI packet average: 0.734.
- Best fixed-by-scenario average: 0.777.
- Auto won or tied the best fixed baseline on 9/10 scenarios.
- Long/filler token reductions: +69.0% and +76.8%.
- Swap delta: 0MB across the measured run.
- Tool-output preservation improved by policy routing, but exact tool-output packet mode remains weak on `qwen2.5:0.5b`.

## Claim Boundaries

Allowed claims:

- Standalone SDI auto policy works on the current hostile synthetic/local tasks.
- Auto policy manages packet bloat better than always-on packet mode.
- SDI packet mode is promising for long/noisy context and pinned-fact/open-loop risk.
- Memory guard prevents unsafe local runs under configured thresholds.

Forbidden claims:

- Speedup.
- Production readiness.
- 7B scaling.
- 14B support.
- Broad quality.
- Agent integration.
- OpenClaw integration.
- Smart Agent Router integration.
- PRT speed.

## Known Limitation: Exact Tool Outputs

Exact tool-output packet mode remains weak on `qwen2.5:0.5b`.

Observed behavior:

- Paths, commits, numbers, and statuses are preserved in the packet.
- The 0.5B model may still fail to extract every exact value.
- Auto policy mitigates this by choosing `recent_only` or `no_packet` for some tiny exact tool-output cases.
- Future work should add a stricter exact tool-output mini-format or test a stronger already-available model.

Do not treat this demo as proof that packet mode solves exact tool-output extraction.

