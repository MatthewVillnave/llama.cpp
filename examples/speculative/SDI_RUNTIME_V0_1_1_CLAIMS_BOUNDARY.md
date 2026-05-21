# SDI Runtime v0.1.1 — Claims Boundary

**What this prototype IS and IS NOT. Strict.**

---

## Allowed Claims

✅ **Core description:**
- SDI Runtime v0.1.1 is a standalone CPU/RAM model-residency prototype
- Sub-Dense Inference: reduces active context pressure while preserving correctness
- Focuses on context selection, not speed, KV modification, or weight residency

✅ **Evidence-based results:**
- qwen2.5:0.5B auto policy: 0.950 average, 7/8 win-tie vs best fixed baseline in local eval
- qwen2.5:3B edge-case comparison: Sc25 limitation confirmed as 0.5B model ceiling, not policy failure
- Tiny qwen2.5:7B canary: 8 runs at c=2048/4096 with stable swap and no abort criteria triggered
- Eval schema bug fixed (required+must_include union): no headline conclusion changes
- Bounded 7B working-set probe (Phase 27H-B/C): qwen2.5:7B passed WS-512 through WS-4096 with stable swap and correct outputs (score 1.0 each); WS-6144 at c=8192 failed by generation/API behavior, not by RAM/swap exhaustion. The observed cliff is between WS-4096 and WS-6144, and the safe bounded zone for this backend is WS<=4096.

✅ **Architecture:**
- Packet builder for structured context summaries
- Memory guard for pre-run RAM/swap safety checks
- Auto policy for per-conversation context strategy selection
- Exact-tool mode for tool-output preservation
- Standalone Ollama-backed eval harness
- No llama.cpp internals modified, no model weights changed

---

## Forbidden Claims

❌ **Performance:**
- Speedup achieved
- Latency improvement
- Token throughput gains
- Any performance benchmark vs no-SDI

❌ **Production:**
- Production ready
- Deployment-ready
- Production-safe

❌ **Model coverage:**
- Broad 7B validation
- 14B support
- Universal model support
- Any model beyond explicitly tested

❌ **Technical scope:**
- Long-context solved
- KV cache modified
- KV cache optimization
- Weight-residency solved
- PRT speedup

❌ **Integration:**
- Agent integration
- OpenClaw integration
- Smart Agent Router integration
- Any production routing layer

❌ **Quality:**
- Universal superiority over any baseline
- Token savings on small contexts
- qwen2.5:0.5B solves all exact/natural constraint tasks

❌ **Path claims:**
- PRT is a speed path (it's parked at correctness-but-no-speed-win)
- SDI enables larger models universally

---

## What SDI Is Not

- Not an agent framework
- Not a routing layer for agents
- Not a KV cache modification
- Not a weight-residency solution
- Not a speedup mechanism
- Not production infrastructure

---

## Prototype Status

SDI Runtime v0.1.1 is a **research prototype**. Local eval signal is real. Public benchmark validation does not exist. Production deployment has not been tested.

The prototype demonstrates:
- Context selection policy matters for CPU/RAM-constrained inference
- Auto policy can improve task scores vs fixed baselines in local eval
- Memory guard can prevent swap-death scenarios
- Some 0.5B failures are model ceilings, not policy failures

The prototype does not demonstrate:
- Speed improvements
- Production readiness
- Broad model coverage
- Any result beyond local eval on qwen2.5:0.5B/3B and a tiny 7B canary

---

## How to Verify

```bash
# Run eval harness
python3 examples/speculative/sdi_packet_runtime.py \
  --conversation fixtures/sdi_packet_eval/<SCENARIO>/conversation.txt \
  --pinned fixtures/sdi_packet_eval/<SCENARIO>/pinned_facts.json \
  --expected fixtures/sdi_packet_eval/<SCENARIO>/expected.json \
  --model qwen2.5:0.5b --baseline auto --backend ollama

# Run regression tests
python3 examples/speculative/test_sdi_packet_builder.py
```