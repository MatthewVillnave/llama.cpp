# Phase 26Q: Exact Tool-Output Packet Refinement

## A. Branch

`experimental/prt-phase19a-alt-sidecar-backed`

## B. Current HEAD

`a45bf353c1c55dbce2c47fe680e0341abb62ce96`

## C. Phase 26O/26P Limitation

Phase 26O showed that general packet mode remained weak for exact tool-output extraction on `qwen2.5:0.5b`.

- `5_tool_result_preservation`: fixed `sdi_packet` scored 0.350 and omitted required tool details.
- `8_tool_output_exactness`: fixed `sdi_packet` scored 0.700 and was competitive but not clearly better than raw/recent context.
- Phase 26P documented this as the main demo limitation.

The packet generally preserved the facts, but the small model did not reliably extract exact fields unless the tool facts were presented in a rigid, copy-oriented shape.

## D. Exact Tool-Output Format

Added `[EXACT_TOOL_OUTPUTS]` near the top of compact/full packets when tool-output facts are detected:

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

Rules added:

- preserve exact strings, paths, commits/hashes, numbers, units, statuses, and errors
- write `not provided` when a field is absent
- for exact tool-output questions, copy values from `[EXACT_TOOL_OUTPUTS]`
- ignore conflicting/distractor tool facts outside `[EXACT_TOOL_OUTPUTS]`
- auto policy selects `sdi_packet` for exact tool-output questions

## E. Scenarios Tested

Focused rerun with Ollama HTTP API, `qwen2.5:0.5b`, temperature 0, bounded output, and memory guard active:

- `5_tool_result_preservation`
- `8_tool_output_exactness`
- `11_exact_path_recall`
- `12_numeric_metric_recall`
- `13_error_status_recall`

Generated outputs were written to `$PRT_SCRATCH/phase26q`, not the repo.

## F. Score Table

| Scenario | no_packet | recent_only | sdi_packet | auto | Auto policy |
| -------- | --------- | ----------- | ---------- | ---- | ----------- |
| 5_tool_result_preservation | 0.500 | 0.500 | 0.350 | 0.350 | sdi_packet |
| 8_tool_output_exactness | 0.700 | 0.700 | 0.700 | 0.700 | sdi_packet |
| 11_exact_path_recall | 0.100 | 0.600 | 0.850 | 0.850 | sdi_packet |
| 12_numeric_metric_recall | 0.925 | 0.925 | 0.850 | 0.850 | sdi_packet |
| 13_error_status_recall | 0.550 | 0.550 | 1.000 | 1.000 | sdi_packet |

Focused averages:

- all 5 focused scenarios, fixed `sdi_packet`: 0.750
- all 5 focused scenarios, `auto`: 0.750
- 3 new exact-output scenarios, fixed `sdi_packet`: 0.900
- 3 new exact-output scenarios, `auto`: 0.900
- 3 new exact-output scenarios, `no_packet`: 0.525
- 3 new exact-output scenarios, `recent_only`: 0.692

## G. Exact Path/Hash/Number Preservation

Result:

- Exact path recall: PASS, copied `/tmp/vega/build/final-report.json` without the distractor path.
- Numeric metric recall: PASS, copied `packet_bytes`, `18432`, and `bytes`.
- Error/status recall: PASS, copied `upload-artifact --file /tmp/raven/release.zip`, `FAILED`, `E_CHECKSUM_MISMATCH expected=abc123 actual=def456`, and `exit code 42`.
- Existing hash/path scenario: stable at 0.700 in the final rerun; the answer preserved path, hash, size, and commit.

## H. Hallucination Behavior

The 3 new exact-output `sdi_packet` runs had no forbidden path/hash/number hits.

The refinement specifically fixed the prior distractor problem in `13_error_status_recall`: the final packet-mode answer did not include the unrelated `Status: OK` verification line.

## I. Token Overhead

Exact mode intentionally adds overhead on small contexts:

| Scenario | sdi_packet token reduction |
| -------- | -------------------------- |
| 11_exact_path_recall | -600.0% |
| 12_numeric_metric_recall | -427.6% |
| 13_error_status_recall | -726.9% |

This is acceptable only for exact tool-output risk cases. It should not be generalized to tiny self-contained prompts.

## J. Swap Behavior

Swap remained stable:

- observed per-run swap delta: 0MB
- local inference stayed under the Phase 26Q safety threshold
- no 7B/14B tests were run

## K. Verdict

`PASS_PHASE26Q_EXACT_TOOL_OUTPUT_REFINEMENT`

`PASS_TOOL_OUTPUT_EXACTNESS_IMPROVED`

With limitation: the older broad `5_tool_result_preservation` scenario remains weak on `qwen2.5:0.5b`, so this is not a claim that exact packet mode solves all tool-output extraction.

## L. Recommended Next Phase

Phase 26R: real-task eval beyond fixtures.

Keep exact-tool mode gated to tool-output questions. Do not expand it to all small-context prompts because overhead is high.

## M. Models/Sidecars/F32 Refs Staged?

No.

## N. Secrets Detected?

No real secrets detected in staged Phase 26Q files. Fixture values are fake.

## O. Tags Touched?

No.
