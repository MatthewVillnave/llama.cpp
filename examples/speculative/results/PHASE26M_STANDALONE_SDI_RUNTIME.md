# Phase 26M - Standalone SDI Runtime + Memory Guard + Baseline Harness

## A. Branch

experimental/prt-phase19a-alt-sidecar-backed

## B. Current HEAD

476e9a99fa1c3b3a217b6ef323449279e67334af

## C. Memory Guard Path

`examples/speculative/sdi_memory_guard.py`

## D. Runtime Path

`examples/speculative/sdi_packet_runtime.py`

## E. Backend Used

Ollama HTTP API /api/generate stream=false

## F. Model(s) Used

- qwen2.5:0.5b
- qwen2.5:3b was not already installed, so it was not pulled or run.
- 7B/14B were not run.

## G. Scenarios

- pinned early fact recall
- long filler + buried hard constraint
- open loop continuation
- conflicting old/new state
- tool output preservation

## H. Baseline Comparison Table

| Scenario | Model | Baseline | Score | Token reduction | Swap delta | Verdict |
| -------- | ----- | -------- | ----- | --------------- | ---------- | ------- |
| Pinned early fact recall | qwen2.5:0.5b | no_packet | 0.100 | -4.7% | 0 MB | - |
| Pinned early fact recall | qwen2.5:0.5b | recent_only | 0.100 | -4.7% | 0 MB | - |
| Pinned early fact recall | qwen2.5:0.5b | simple_summary | 0.100 | -8.9% | 0 MB | - |
| Pinned early fact recall | qwen2.5:0.5b | sdi_packet | 0.625 | 69.7% | 0 MB | best/tied |
| Long filler + hard constraint | qwen2.5:0.5b | no_packet | 0.500 | -4.1% | 0 MB | - |
| Long filler + hard constraint | qwen2.5:0.5b | recent_only | 0.500 | -4% | 0 MB | - |
| Long filler + hard constraint | qwen2.5:0.5b | simple_summary | 0.500 | -6.5% | 0 MB | - |
| Long filler + hard constraint | qwen2.5:0.5b | sdi_packet | 0.750 | 79.3% | 0 MB | best/tied |
| Open loop continuation | qwen2.5:0.5b | no_packet | 0.300 | -30.8% | 0 MB | - |
| Open loop continuation | qwen2.5:0.5b | recent_only | 0.300 | -30.4% | 0 MB | - |
| Open loop continuation | qwen2.5:0.5b | simple_summary | 0.300 | -58.1% | 0 MB | - |
| Open loop continuation | qwen2.5:0.5b | sdi_packet | 0.000 | -82.8% | 0 MB | not best |
| Conflicting old/new state | qwen2.5:0.5b | no_packet | 0.950 | -43.5% | 0 MB | - |
| Conflicting old/new state | qwen2.5:0.5b | recent_only | 0.100 | -42.9% | 0 MB | - |
| Conflicting old/new state | qwen2.5:0.5b | simple_summary | 1.000 | -79.4% | 0 MB | - |
| Conflicting old/new state | qwen2.5:0.5b | sdi_packet | 0.950 | -124.7% | 0 MB | not best |
| Tool output preservation | qwen2.5:0.5b | no_packet | 0.500 | -45.3% | 0 MB | - |
| Tool output preservation | qwen2.5:0.5b | recent_only | 0.350 | -44.7% | 0 MB | - |
| Tool output preservation | qwen2.5:0.5b | simple_summary | 0.350 | -79.2% | 0 MB | - |
| Tool output preservation | qwen2.5:0.5b | sdi_packet | 0.350 | -98.7% | 0 MB | not best |

## I. Token Reduction

SDI packet token reduction vs raw conversation estimate:

- pinned early fact recall: 69.7%
- long filler + hard constraint: 79.3%
- open loop continuation: -82.8%
- conflicting old/new state: -124.7%
- tool output preservation: -98.7%

Interpretation: SDI compressed the two actual long/filler pressure cases. On already-small contexts, packet structure added overhead. That is expected and means the runtime should avoid packet mode for small contexts unless pinned-fact precision is needed.

## J. Memory / Swap Behavior

Swap remained stable across all 20 runs.

- Observed swap used: about 454-455 MB
- Max swap delta per run: 0 MB
- Hard stop threshold: 1 GB
- Memory guard verdict: PASS_MEMORY_GUARD

The guard reported existing local runtime processes but did not kill them.

## K. Quality Results

Average score by baseline:

- no_packet: 0.470
- recent_only: 0.270
- simple_summary: 0.450
- sdi_packet: 0.535

SDI packet record:

- Wins: 2/5
- Ties for best: 0/5
- Losses: 3/5

Scenario notes:

- SDI beat all baselines on pinned early fact recall.
- SDI beat all baselines on long filler + hard constraint.
- SDI was competitive on conflicting state but simple_summary won.
- SDI failed open-loop continuation because qwen2.5:0.5b repeated the question instead of using packet open loops.
- SDI did not improve tool-output preservation; the 0.5B model returned only the two easiest facts.

## L. Runtime Limitations

- Token estimates use chars/4, not tokenizer counts.
- The harness runs one scenario/baseline per invocation; the matrix was orchestrated externally.
- Scoring is assertion substring based.
- qwen2.5:0.5b is weak on multi-fact exact recall.
- No speedup claim was made. Wall time is only recorded as a rough overhead signal.

## M. Verdict

PARTIAL_SDI_PACKET_MIXED_RESULTS

Also passed:

- PASS_PHASE26M_STANDALONE_RUNTIME
- PASS_MEMORY_GUARD

Not claimed:

- PASS_SDI_PACKET_BEATS_BASELINES

## N. Recommended Next Phase

Phase 26N - improve packet format for open loops/tool outputs, expand hostile eval to 5-10 real tasks, and optionally test qwen2.5:3b only if already available or explicitly approved.

## O. Models / Sidecars / F32 Refs Staged?

No.

## P. Secrets Detected?

No real secrets detected. Fixture fake token markers remained test-only.

## Q. Tags Touched?

No.

