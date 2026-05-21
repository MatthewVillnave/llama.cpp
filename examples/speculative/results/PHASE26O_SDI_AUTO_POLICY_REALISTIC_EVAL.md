# Phase 26O - SDI Auto Policy + Realistic Long/Noisy Eval

## A. Branch

experimental/prt-phase19a-alt-sidecar-backed

## B. Current HEAD

64447ad3328a98dba3673133835bc870ef0b6d54

## C. Phase 26N Baseline

- SDI packet avg: 0.77
- SDI win/tie count: 4/5
- Verdict: PASS_PHASE26N_PACKET_REFINEMENT_WITH_SMALL_CONTEXT_BLOAT_LIMITATION

## D. Auto Policy Rules

- Use sdi_packet for long context, repeated filler, open loops, conflicting old/new facts, pinned project/state questions, exact tool/path/commit/numeric risk when context is not tiny, or hard memory pressure.
- Use recent_only for tiny exact tool-output contexts when recent view is no larger than raw context.
- Use simple_summary for medium context with low exactness risk.
- Use recent_only/no_packet for small self-contained context when packet overhead is not justified.
- Treat low existing swap as a warning, not by itself a packet trigger; hard pressure remains guarded by sdi_memory_guard.

## E. Tool-Output Preservation Changes

- Auto policy avoids packet overhead for tiny exact tool-output contexts and uses recent_only when it is smallest.
- Packet builder now emits Tool/output fields for command, file/path, commit/hash, numeric result, error/warning, status, and user-facing conclusion.
- Packet instruction explicitly says to preserve paths, command names, numeric values, commit hashes, errors, and statuses exactly.

## F. Scenarios Tested

- Existing 5 hostile scenarios retained.
- Added realistic project handoff.
- Added debug trace root cause.
- Added tool-output exactness.
- Added constraint under contradiction.
- Added realistic open loop.

Total scenarios: 10.

## G. Score Table

| Scenario | Model | Baseline | Effective | Score | Token reduction | Swap delta |
| -------- | ----- | -------- | --------- | ----- | --------------- | ---------- |
| Realistic open loop | qwen2.5:0.5b | auto | sdi_packet | 0.967 | -224.9% | 0 MB |
| Realistic open loop | qwen2.5:0.5b | no_packet | no_packet | 0.000 | -51.7% | 0 MB |
| Realistic open loop | qwen2.5:0.5b | recent_only | recent_only | 0.000 | -51.2% | 0 MB |
| Realistic open loop | qwen2.5:0.5b | sdi_packet | sdi_packet | 0.967 | -224.9% | 0 MB |
| Realistic open loop | qwen2.5:0.5b | simple_summary | simple_summary | 0.000 | -91.5% | 0 MB |
| Pinned early fact | qwen2.5:0.5b | auto | sdi_packet | 1.000 | 69.0% | 0 MB |
| Pinned early fact | qwen2.5:0.5b | no_packet | no_packet | 0.100 | -6.8% | 0 MB |
| Pinned early fact | qwen2.5:0.5b | recent_only | recent_only | 0.100 | -6.7% | 0 MB |
| Pinned early fact | qwen2.5:0.5b | sdi_packet | sdi_packet | 1.000 | 69.0% | 0 MB |
| Pinned early fact | qwen2.5:0.5b | simple_summary | simple_summary | 0.100 | -10.9% | 0 MB |
| Long filler constraint | qwen2.5:0.5b | auto | sdi_packet | 0.750 | 76.8% | 0 MB |
| Long filler constraint | qwen2.5:0.5b | no_packet | no_packet | 0.250 | -5.6% | 0 MB |
| Long filler constraint | qwen2.5:0.5b | recent_only | recent_only | 0.250 | -5.5% | 0 MB |
| Long filler constraint | qwen2.5:0.5b | sdi_packet | sdi_packet | 0.750 | 76.8% | 0 MB |
| Long filler constraint | qwen2.5:0.5b | simple_summary | simple_summary | 0.500 | -8.0% | 0 MB |
| Open loop continuation | qwen2.5:0.5b | auto | sdi_packet | 0.750 | -155.1% | 0 MB |
| Open loop continuation | qwen2.5:0.5b | no_packet | no_packet | 0.000 | -43.6% | 0 MB |
| Open loop continuation | qwen2.5:0.5b | recent_only | recent_only | 0.000 | -42.7% | 0 MB |
| Open loop continuation | qwen2.5:0.5b | sdi_packet | sdi_packet | 0.750 | -155.1% | 0 MB |
| Open loop continuation | qwen2.5:0.5b | simple_summary | simple_summary | 0.000 | -70.5% | 0 MB |
| Conflicting state | qwen2.5:0.5b | auto | sdi_packet | 1.000 | -181.8% | 0 MB |
| Conflicting state | qwen2.5:0.5b | no_packet | no_packet | 0.950 | -60.0% | 0 MB |
| Conflicting state | qwen2.5:0.5b | recent_only | recent_only | 0.900 | -59.4% | 0 MB |
| Conflicting state | qwen2.5:0.5b | sdi_packet | sdi_packet | 1.000 | -181.8% | 0 MB |
| Conflicting state | qwen2.5:0.5b | simple_summary | simple_summary | 1.000 | -95.9% | 0 MB |
| Tool output preservation | qwen2.5:0.5b | auto | no_packet | 0.500 | -62.9% | 0 MB |
| Tool output preservation | qwen2.5:0.5b | no_packet | no_packet | 0.500 | -62.9% | 0 MB |
| Tool output preservation | qwen2.5:0.5b | recent_only | recent_only | 0.500 | -62.3% | 0 MB |
| Tool output preservation | qwen2.5:0.5b | sdi_packet | sdi_packet | 0.350 | -268.6% | 0 MB |
| Tool output preservation | qwen2.5:0.5b | simple_summary | simple_summary | 0.500 | -96.9% | 0 MB |
| Project handoff | qwen2.5:0.5b | auto | sdi_packet | 0.125 | -116.3% | 0 MB |
| Project handoff | qwen2.5:0.5b | no_packet | no_packet | 0.250 | -30.1% | 0 MB |
| Project handoff | qwen2.5:0.5b | recent_only | recent_only | 0.250 | -29.4% | 0 MB |
| Project handoff | qwen2.5:0.5b | sdi_packet | sdi_packet | 0.125 | -116.3% | 0 MB |
| Project handoff | qwen2.5:0.5b | simple_summary | simple_summary | 0.250 | -35.9% | 0 MB |
| Debug root cause | qwen2.5:0.5b | auto | sdi_packet | 0.850 | -114.9% | 0 MB |
| Debug root cause | qwen2.5:0.5b | no_packet | no_packet | 0.850 | -48.1% | 0 MB |
| Debug root cause | qwen2.5:0.5b | recent_only | recent_only | 0.662 | -47.6% | 0 MB |
| Debug root cause | qwen2.5:0.5b | sdi_packet | sdi_packet | 0.850 | -114.9% | 0 MB |
| Debug root cause | qwen2.5:0.5b | simple_summary | simple_summary | 0.662 | -67.8% | 0 MB |
| Tool exactness | qwen2.5:0.5b | auto | recent_only | 0.850 | -78.2% | 0 MB |
| Tool exactness | qwen2.5:0.5b | no_packet | no_packet | 0.700 | -79.7% | 0 MB |
| Tool exactness | qwen2.5:0.5b | recent_only | recent_only | 0.850 | -78.2% | 0 MB |
| Tool exactness | qwen2.5:0.5b | sdi_packet | sdi_packet | 0.700 | -300.8% | 0 MB |
| Tool exactness | qwen2.5:0.5b | simple_summary | simple_summary | 0.700 | -119.5% | 0 MB |
| Constraint contradiction | qwen2.5:0.5b | auto | sdi_packet | 0.850 | -173.1% | 0 MB |
| Constraint contradiction | qwen2.5:0.5b | no_packet | no_packet | 0.100 | -58.1% | 0 MB |
| Constraint contradiction | qwen2.5:0.5b | recent_only | recent_only | 0.350 | -57.5% | 0 MB |
| Constraint contradiction | qwen2.5:0.5b | sdi_packet | sdi_packet | 0.850 | -173.1% | 0 MB |
| Constraint contradiction | qwen2.5:0.5b | simple_summary | simple_summary | 0.350 | -92.8% | 0 MB |

Average scores:

- no_packet: 0.37
- recent_only: 0.386
- simple_summary: 0.406
- sdi_packet: 0.734
- auto: 0.764
- best fixed baseline by scenario: 0.777

## H. Auto Policy Table

| Scenario | Auto selected | Auto score | Best fixed score | Token reduction | Policy reason |
| -------- | ------------- | ---------- | ---------------- | --------------- | ------------- |
| Realistic open loop | sdi_packet | 0.967 | 0.967 | -224.9% | open-loop/conflict state needs structured packet facts |
| Pinned early fact | sdi_packet | 1.000 | 1.000 | 69.0% | context pressure or filler favors packet compression |
| Long filler constraint | sdi_packet | 0.750 | 0.750 | 76.8% | context pressure or filler favors packet compression |
| Open loop continuation | sdi_packet | 0.750 | 0.750 | -155.1% | open-loop/conflict state needs structured packet facts |
| Conflicting state | sdi_packet | 1.000 | 1.000 | -181.8% | open-loop/conflict state needs structured packet facts |
| Tool output preservation | no_packet | 0.500 | 0.500 | -62.9% | tool facts present but raw context is tiny; avoid packet overhead |
| Project handoff | sdi_packet | 0.125 | 0.250 | -116.3% | context pressure or filler favors packet compression |
| Debug root cause | sdi_packet | 0.850 | 0.850 | -114.9% | context pressure or filler favors packet compression |
| Tool exactness | recent_only | 0.850 | 0.850 | -78.2% | tool facts present but raw context is tiny; use smallest raw/recent exact-output view |
| Constraint contradiction | sdi_packet | 0.850 | 0.850 | -173.1% | open-loop/conflict state needs structured packet facts |

Auto record vs best fixed baseline:

- Wins: 0
- Ties: 9
- Losses: 1
- Wins/ties: 9/10

## I. Token Reduction

| Scenario | Auto token reduction | Fixed SDI token reduction |
| -------- | -------------------- | ------------------------- |
| Realistic open loop | -224.9% | -224.9% |
| Pinned early fact | 69.0% | 69.0% |
| Long filler constraint | 76.8% | 76.8% |
| Open loop continuation | -155.1% | -155.1% |
| Conflicting state | -181.8% | -181.8% |
| Tool output preservation | -62.9% | -268.6% |
| Project handoff | -116.3% | -116.3% |
| Debug root cause | -114.9% | -114.9% |
| Tool exactness | -78.2% | -300.8% |
| Constraint contradiction | -173.1% | -173.1% |

Interpretation: Auto reduced packet bloat on tiny tool-output cases by selecting recent_only/no_packet, but still chooses SDI for small high-risk open-loop/conflict cases where quality beats token size.

## J. Swap Behavior

- Swap stable: True
- Max swap delta: 0 MB
- Observed swap: about 454MB during run, below 1GB hard stop

## K. Tool-Output Result

- Phase 26N tool-output SDI score: 0.35
- Phase 26O existing tool-output auto score: 0.5
- Phase 26O new tool-output auto score: 0.85

Status: improved by policy selection; exact tool-output packet itself still needs stronger model or stricter output contract

## L. Failures / Limitations

| Scenario | Auto policy | Auto score | Best baseline | Best score | Missing | Note |
| -------- | ----------- | ---------- | ------------- | ---------- | ------- | ---- |
| Long filler constraint | sdi_packet | 0.750 | auto | 0.750 | run001.csv | required facts still missing |
| Tool output preservation | no_packet | 0.500 | auto | 0.500 | 8192, /home/matthew-villnave/profiling/run42, /tmp/mem_profile_f5e6d7c8.json, COMPLETED | required facts still missing |
| Project handoff | sdi_packet | 0.125 | no_packet | 0.250 | /home/matthew-villnave/orion, c0309aa, queue item 17 processed before item 16, preserve FIFO, do not run migration scripts against production | auto under best baseline |
| Tool exactness | recent_only | 0.850 | auto | 0.850 | OK | required facts still missing |

Key limitation: auto improves policy selection, but exact tool-output packet mode is still not solved for qwen2.5:0.5b. The policy routes around it when context is tiny.

## M. Verdict

PASS_PHASE26O_AUTO_POLICY_EVAL_WITH_TOOL_PACKET_LIMITATION

Supporting verdicts:

- PASS_PHASE26O_AUTO_POLICY_EVAL
- PASS_AUTO_POLICY_REDUCES_BLOAT
- PASS_AUTO_BEATS_BASELINES
- PASS_TOOL_OUTPUT_PRESERVATION_IMPROVED
- PARTIAL_AUTO_POLICY_MIXED

## N. Recommended Next Phase

Phase 26P - package standalone SDI runtime as reproducible demo with policy thresholds documented, while separately tightening exact tool-output scoring/format.

## O. Models / Sidecars / F32 Refs Staged?

No.

## P. Secrets Detected?

No real secrets detected. Fixtures use fake data only.

## Q. Tags Touched?

No.
