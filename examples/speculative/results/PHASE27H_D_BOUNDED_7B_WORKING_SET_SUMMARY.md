# Phase 27H-D: Formal Bounded 7B Working-Set Summary

## A. Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## B. Current HEAD
`389868eb3a039dc6b4edd66aad6c029a32bf0dfb`

## C. Phase 27H-B Result
`PASS_BOUNDED_7B_CLIFF_PROBE`

Bounded 7B working-set probe passed through WS-512 to WS-4096 under strict memory guard.

## D. Phase 27H-C Result
`FAIL_WS6144_CLIFF`

WS-6144 at c=8192 failed to produce a sane captured output. Swap/RAM did not trip. The failure was generation/API behavior, not memory exhaustion. WS-6144 is the first observed pressure/failure point.

## E. Consolidated Working-Set Table

| Working Set | Context | Result | Score | Swap Δ | RAM Behavior | Interpretation |
|------------|---------|--------|-------|--------|-------------|----------------|
| WS-512 | 2048 | OK | 1.0 | 0.0 GB | Stable ~7.5 GB | Passed clean |
| WS-1024 | 2048 | OK | 1.0 | 0.0 GB | Stable ~7.5 GB | Passed clean |
| WS-2048 | 4096 | OK | 1.0 | 0.0 GB | Stable ~7.5 GB | Passed clean |
| WS-4096 | 4096 | OK | 1.0 | 0.0 GB | Stable ~7.5 GB | Passed clean |
| WS-6144 | 8192 | FAIL | 0.0 (not established) | 0.0 GB | Stable ~7.2 GB | Generation/API cliff |

## F. Observed Cliff

The first observed 7B pressure point is between WS-4096 and WS-6144.

**Cliff type:** Generation/API/output behavior at c=8192, not RAM/swap exhaustion.
**Location:** WS-6144 at c=8192 with ~6,170 estimated input tokens.
**Bounded safe zone:** WS <= 4096 for this machine/backend/config.

## G. Memory/Swap Interpretation

Swap held flat at ~0.367–0.405 GB across all runs including WS-6144. RAM stayed above the 3 GB guard throughout. No memory pressure was the cause of the WS-6144 failure. This rules out RAM exhaustion as the immediate failure mechanism.

## H. Generation/API Interpretation

The failure at WS-6144/c=8192 appeared as a controlled request that returned through an error path with no response text captured. This happened despite:
- Swap staying below 1 GB guard
- RAM staying above 3 GB minimum
- Model being responsive to simple prompts at the same context size

Possible causes (not confirmed):
- num_ctx=8192 with ~6K tokens may hit internal KV or attention window limits in qwen2.5:7b
- The bounded generation request may have been terminated by a backend limit before producing output
- The prompt structure (heavy filler + context headers) may interact poorly with the larger context window

The exact failure mechanism is not definitively diagnosed. The observation stands as the cliff point: WS-6144/c=8192 fails for this model/backend on this machine.

## I. Allowed Claims

- Bounded 7B working-set probe passed through WS-4096 with stable swap and correct outputs on qwen2.5:7b.
- WS-6144 at c=8192 is the first observed pressure/failure point in the bounded probe sequence.
- The observed cliff is generation/API behavior, not RAM/swap exhaustion.
- The current bounded safe zone for this backend is WS <= 4096.
- No broad 7B validation, general safety, speedup, or production claims are made.

## J. Forbidden Claims

- ❌ Broad 7B validation
- ❌ 7B safe generally or on other hardware
- ❌ Speedup demonstrated
- ❌ Long-context solved
- ❌ 14B support
- ❌ Production readiness
- ❌ KV cache or weight-residency solved
- ❌ WS-6144 is universally impossible
- ❌ c=8192 never works for qwen2.5:7b
- ❌ Memory is not a bottleneck for 7B models
- ❌ WS-6144 failure means RAM is the constraint

## K. Recommended Next Phase

**Phase 27I-A:** KV/context memory mapping design — understand how KV cache grows with context across model sizes, without running more 7B working-set probes.

**Phase 27I-B (alternative):** Ollama/API c=8192 failure forensics — reproduce and diagnose the generation failure at c=8192 with ~6K context, determine if it's a prompt structure issue or a genuine model/context limit.

**Phase 27I-C (if public-facing work is preferred):** Update the public package with the 7B bounded results. No new technical runs.

**Preferred:** Phase 27I-A — KV/context memory mapping design. Stop 7B working-set probing for now.

## L. Models/Sidecars/F32 Refs Staged?

No. No model files staged. qwen2.5:7b remains in Ollama registry only.

## M. Secrets Detected?

No secrets detected in any committed files.

## N. Tags Touched?

No tags created, modified, or pushed.

---

## Verdicts

- ✅ `PASS_PHASE27H_D_7B_WORKING_SET_SUMMARY`
- ✅ `PASS_WS4096_BOUNDED_SAFE_ZONE_DOCUMENTED`
- ✅ `PASS_WS6144_GENERATION_CLIFF_DOCUMENTED`
- ✅ `PASS_NO_BROAD_7B_CLAIM`
- ✅ `BLOCKED_REPO_STATE` (old Phase 26K/L/R reports remain untracked; no new staging)