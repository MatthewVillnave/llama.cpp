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

## Exact Tool-Output Mode

Phase 26Q adds a rigid extractive mini-format for tool outputs:

```text
[EXACT_TOOL_OUTPUTS]
Item 1:
- tool:
- command:
- path:
- commit:
- metric:
- value:
- unit:
- status:
- error:
- user_conclusion:
[/EXACT_TOOL_OUTPUTS]
```

The runtime tells the model to copy exact values from this section for tool-output questions and to avoid inferring missing fields. Auto policy now selects `sdi_packet` for exact tool-output questions that reference paths, commits, hashes, metrics, values, units, statuses, errors, failures, or commands.

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

Exact tool-output packet mode improved in Phase 26Q, but it is still not a general extraction guarantee on `qwen2.5:0.5b`.

Observed behavior:

- Paths, commits, numbers, statuses, and errors are preserved in the packet.
- The 3 new exact-output fixtures averaged 0.900 with fixed `sdi_packet`.
- The older broad tool-result preservation fixture remains weak because the tiny model still omits some required values.
- Exact mode adds prompt overhead, so it should remain gated to tool-output risk cases.
- Future work should test this beyond fixtures and only test a stronger already-available model if explicitly approved.

Do not treat this demo as proof that packet mode solves exact tool-output extraction.
