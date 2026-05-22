# Phase 28C: SDI Context/KV Probing — Chapter Closure

## A. Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## B. Current HEAD
`975c7dd42`

## C. Phase 28B Summary

18 external context/memory probe runs across qwen2.5:0.5B and qwen2.5:3B at c=2048/c=4096/c=8192:
- Swap delta: 0.0 GB across all 18 runs
- RAM: no visible correlation with context size (model weights dominate; KV cache not visible externally)
- Time: linear/predictable per-token scaling
- c=8192: reliable for both small models
- 0.5B model ceiling on exact retrieval in noisy contexts confirmed again (not a context cliff)
- 3B: all 9 runs passed sanity
- Deep KV instrumentation: not justified by current data

## D. Consolidated Endpoint

### SDI Runtime v0.1.1
- Status: frozen at `SDI_PHASE26W_RUNTIME_V0_1_1_CHECKPOINT`
- Public package corrected (Phase 27L)
- Claim boundaries established and consistently enforced
- Standalone eval harness, no llama.cpp internals modified

### Bounded 7B Results
- WS-512 through WS-6144 passed under strict memory guard
- c=8192 works under clean runner/prompt conditions
- Phase 27H-C "cliff" superseded — root cause was prompt bug + stuck runner, not memory exhaustion
- Swap delta 0.0 GB throughout all Phase 27H–28B runs
- Broad 7B validation: still forbidden

### Small-Model Context Probe Results
- qwen2.5:0.5B stable across c=2048/c=4096/c=8192 (except model ceiling failures)
- qwen2.5:3B stable across c=2048/c=4096/c=8192 — all 9 runs passed
- No RAM variation visible externally with context size
- Swap flat across all runs
- No justification for deep KV instrumentation

### What This Means
The context/KV probing chapter produced a clean null result: **no memory cliff was found** in the tested ranges for any model. Context length alone does not drive swap or RAM pressure on this machine under the tested conditions. The dominant failure modes were runner state (stuck processes) and prompt construction, not hardware limits.

## E. What We Know

1. **Context selection/policy layer is useful.** SDI v0.1.1 improved auto policy win/tie rate from ~50% to 7/8 on qwen2.5:0.5B local eval.

2. **Swap death is avoidable with guards.** No swap pressure observed across all Phase 26–28B runs with active memory guard.

3. **0.5B failures are often model capability ceilings.** Exact retrieval in noisy contexts fails on 0.5B; 3B handles it correctly. This is model architecture, not a context selection failure.

4. **3B is a stronger local eval model.** Used for edge-case validation and higher-quality local results.

5. **External Ollama telemetry cannot directly expose KV cache behavior.** Only RSS, swap, and timing are externally visible.

6. **Deep KV instrumentation not justified yet.** No failure pattern correlates with context length to justify modifying llama.cpp internals.

7. **Time scaling is linear.** Per-token time scales predictably with context; no anomalous spikes.

8. **c=8192 works reliably** across 0.5B, 3B, and 7B on this machine.

## F. What We Do NOT Know

1. Exact KV cache allocation behavior inside Ollama backend
2. Broad 7B task behavior across many different task types
3. 14B feasibility on this machine
4. Speed/throughput improvements with SDI vs without
5. Weight-residency solution (PRT v3 parked)
6. Production readiness for any real deployment
7. Whether context pressure manifests differently on different hardware
8. Optimal context selection policy for arbitrary user conversations

## G. Technical Pause Recommendation

**Pause further technical probing.**

Reason: SDI v0.1.1 has a complete documented story. Bounded 7B and small-model context probes produced a null result on memory cliffs. More probing has diminishing returns without a new specific hypothesis.

The story to tell: **context selection matters more than context length.** The SDI policy layer improves which facts the model sees; the memory guard keeps the system stable. Neither requires knowing the exact KV allocation to work.

## H. Future Lane Ranking

| Rank | Lane | Rationale | Priority |
|------|------|-----------|----------|
| 1 | **Public package / article polish** | Best next move. Share what was learned. | High |
| 2 | **3B-centered SDI quality eval** | Use 3B as stronger eval model for task quality, not memory cliffs. | Medium |
| 3 | **Backend/KV instrumentation design** | Parked until a real symptom justifies it. | Low |
| 4 | **7B broader validation** | Requires explicit new plan and approval. | Low |
| 5 | **PRT v3 weight-residency** | Parked. | Low |

## I. Allowed Claims

- SDI v0.1.1 is a documented standalone context-selection prototype.
- Small-model external context probes were stable at c=2048/c=4096/c=8192 across 0.5B and 3B.
- Bounded 7B probes passed through WS-6144 under strict guard on this backend.
- No swap/RAM cliff was found in the tested bounded ranges.
- Deep KV instrumentation is not justified by current external probe data.
- Time-per-token scales linearly and predictably with context length.

## J. Forbidden Claims

- ❌ Speedup demonstrated
- ❌ Production readiness
- ❌ KV cache solved or modified
- ❌ Broad 7B validation
- ❌ 14B support
- ❌ Weight-residency solved
- ❌ Universal superiority
- ❌ Long-context solved

## K. Recommended Next

**Phase 29: Public writeup and article polish** — share the SDI v0.1.1 story with corrected bounded 7B results, small-model findings, and the core insight: context selection matters more than context length.

No more technical runs unless a new specific hypothesis emerges.

## L. Models/Sidecars/F32 Refs Staged?
No.

## M. Secrets Detected?
No secrets in any committed files.

## N. Tags Touched?
No tags created, modified, or pushed.

---

## Verdicts
- ✅ `PASS_PHASE28C_CONTEXT_KV_PROBING_CLOSURE`
- ✅ `PASS_TECHNICAL_PAUSE_RECOMMENDED`
- ✅ `PASS_NO_KV_INSTRUMENTATION_JUSTIFIED`
- ✅ `PASS_CONTEXT_PROBE_SUMMARIZED`
- ✅ `BLOCKED_REPO_STATE` (old untracked reports remain untracked)