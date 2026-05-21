# Phase 26N - SDI Packet Format Refinement + Hostile Eval Rerun

## A. Branch

experimental/prt-phase19a-alt-sidecar-backed

## B. Current HEAD

b808b201c4ee036cf96032123aa9bf7a2c85c05c

## C. Phase 26M Baseline

- sdi_packet avg: 0.535
- no_packet avg: 0.47
- simple_summary avg: 0.45
- recent_only avg: 0.27
- SDI record: 2 wins, 0 ties, 3 losses

## D. Failure Analysis

| Scenario | 26M winning baseline | 26M SDI result | Model missed | Failure mode | Token reduction |
| -------- | -------------------- | -------------- | ------------ | ------------ | --------------- |
| Pinned early fact | sdi_packet | win | initial commit and confidentiality constraint | answer format not explicit enough; expected evidence not prominent enough | positive |
| Long filler constraint | sdi_packet | win | experiment tracker path | 0.5B answered only direct constraint; secondary evidence ignored | positive |
| Open loop continuation | no_packet/recent_only/simple_summary tie | loss | all open-loop details; repeated the question | open loop unclear; current request too prominent relative to answer facts | negative |
| Conflicting state | simple_summary | loss by 0.05 | nothing required; formatting less clean | conflict handling needed current truth/superseded facts section | negative |
| Tool output preservation | no_packet | loss | ctx, working dir, result file, status | tool output buried / model too weak for multi-field exact output | negative |

## E. Packet v0.2 Changes

- Added `ANSWER_TARGET` near the top with `Directly answer` and `Expected evidence`.
- Added small-model instruction: use only packet facts, answer directly, include exact names/paths/constraints/numbers/commits/statuses/next actions.
- Structured open loops as `ID`, `Status`, `Next action`, and `Blocking issue`.
- Inferred compact `Tool/output facts` from pinned facts and recent tool blocks.
- Added `Current truth` and `Superseded/old facts` for conflict tasks.
- Added `--packet-style full|compact`.

## F. Compact Mode Status

Implemented. Phase 26N rerun used `--packet-style compact` for `sdi_packet`.

## G. Rerun Score Table

| Scenario | Model | Baseline | Score | Token reduction | Swap delta | Output |
| -------- | ----- | -------- | ----- | --------------- | ---------- | ------ |
| Pinned early fact | qwen2.5:0.5b | no_packet | 0.100 | -6.8% | 0 MB | sane |
| Pinned early fact | qwen2.5:0.5b | recent_only | 0.100 | -6.7% | 0 MB | sane |
| Pinned early fact | qwen2.5:0.5b | simple_summary | 0.100 | -10.9% | 0 MB | sane |
| Pinned early fact | qwen2.5:0.5b | sdi_packet_v0.2 | 1.000 | 71.4% | 0 MB | sane |
| Long filler constraint | qwen2.5:0.5b | no_packet | 0.250 | -5.6% | 0 MB | sane |
| Long filler constraint | qwen2.5:0.5b | recent_only | 0.250 | -5.5% | 0 MB | sane |
| Long filler constraint | qwen2.5:0.5b | simple_summary | 0.500 | -8% | 0 MB | sane |
| Long filler constraint | qwen2.5:0.5b | sdi_packet_v0.2 | 0.750 | 79.2% | 0 MB | sane |
| Open loop continuation | qwen2.5:0.5b | no_packet | 0.000 | -43.6% | 0 MB | sane |
| Open loop continuation | qwen2.5:0.5b | recent_only | 0.000 | -42.7% | 0 MB | sane |
| Open loop continuation | qwen2.5:0.5b | simple_summary | 0.000 | -70.5% | 0 MB | sane |
| Open loop continuation | qwen2.5:0.5b | sdi_packet_v0.2 | 0.750 | -128.2% | 0 MB | sane |
| Conflicting state | qwen2.5:0.5b | no_packet | 0.950 | -60% | 0 MB | sane |
| Conflicting state | qwen2.5:0.5b | recent_only | 0.900 | -59.4% | 0 MB | sane |
| Conflicting state | qwen2.5:0.5b | simple_summary | 1.000 | -95.9% | 0 MB | sane |
| Conflicting state | qwen2.5:0.5b | sdi_packet_v0.2 | 1.000 | -165.9% | 0 MB | sane |
| Tool output preservation | qwen2.5:0.5b | no_packet | 0.500 | -62.9% | 0 MB | sane |
| Tool output preservation | qwen2.5:0.5b | recent_only | 0.500 | -62.3% | 0 MB | sane |
| Tool output preservation | qwen2.5:0.5b | simple_summary | 0.500 | -96.9% | 0 MB | sane |
| Tool output preservation | qwen2.5:0.5b | sdi_packet_v0.2 | 0.350 | -220.1% | 0 MB | sane |

## H. Phase 26M vs 26N Comparison

| Scenario | 26M SDI score | 26N SDI score | Change | Notes |
| -------- | ------------- | ------------- | ------ | ----- |
| Pinned early fact | 0.625 | 1.000 | +0.375 | win; token 69.7% -> 71.4% |
| Long filler constraint | 0.750 | 0.750 | +0.000 | win; token 79.3% -> 79.2% |
| Open loop continuation | 0.000 | 0.750 | +0.750 | win; token -82.8% -> -128.2% |
| Conflicting state | 0.950 | 1.000 | +0.050 | tie; token -124.7% -> -165.9% |
| Tool output preservation | 0.350 | 0.350 | +0.000 | loss; token -98.7% -> -220.1% |

Average comparison:

- 26M sdi_packet avg: 0.535
- 26N sdi_packet avg: 0.770
- 26N best non-SDI baseline avg: 0.420
- 26N SDI win/tie count: 4/5

## I. Token Reduction

SDI v0.2 token reduction by scenario:

- pinned early fact: 71.4%
- long filler constraint: 79.2%
- open loop continuation: -128.2%
- conflicting state: -165.9%
- tool output preservation: -220.1%

Interpretation: packet mode is clearly useful under long/filler pressure, but still bloats already-small contexts. Phase 26O should add an auto policy: use packet mode only when context pressure or pinned-fact risk justifies the overhead.

## J. Swap Behavior

- Swap remained stable.
- Max per-run swap delta: 0 MB.
- Observed swap used: about 454 MB, below the 1 GB hard stop.

## K. qwen2.5:3b Check Status

skipped_not_installed. It was not pulled.

## L. Verdict

PASS_PHASE26N_PACKET_REFINEMENT_WITH_SMALL_CONTEXT_BLOAT_LIMITATION

Supporting verdicts:

- PASS_PHASE26N_PACKET_REFINEMENT
- PASS_SDI_PACKET_IMPROVED
- PASS_SDI_PACKET_BEATS_BASELINES
- PARTIAL_MIXED_RESULTS_REMAIN

Important limitation: SDI did not fix tool-output preservation, and compact packet mode still bloats small contexts.

## M. Recommended Next Phase

Phase 26O - expand to 8-10 more realistic long/noisy tasks and add an auto policy that only uses packet mode when context pressure or pinned-fact risk justifies the overhead.

## N. Models / Sidecars / F32 Refs Staged?

No.

## O. Secrets Detected?

No real secrets detected. Fixture fake token markers remained test-only.

## P. Tags Touched?

No.

