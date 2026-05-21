# Phase 27H-C: 7B WS-6144 Extension Probe

## A. Branch

`experimental/prt-phase19a-alt-sidecar-backed`

## B. Current HEAD

`ceb9fa14f8f295e1f1babb2a61623a01dd323f49`

## C. Phase 27H-B Summary

Phase 27H-B completed as `PASS_BOUNDED_7B_CLIFF_PROBE`.

Tested bounded 7B working-set sizes:

| Working set | Context | Score | Swap delta |
|---:|---:|---:|---:|
| WS-512 | 2048 | 1.0 | 0.0 GB |
| WS-1024 | 2048 | 1.0 | 0.0 GB |
| WS-2048 | 4096 | 1.0 | 0.0 GB |
| WS-4096 | 4096 | 1.0 | 0.0 GB |

Phase 27H-B did not find a cliff through WS-4096. That result does not imply broad 7B validation, general 7B safety, production readiness, or speedup.

## D. Preflight RAM/Swap

Preflight passed.

- Model availability: `qwen2.5:7b` present in Ollama
- Branch: `experimental/prt-phase19a-alt-sidecar-backed`
- Preflight RAM available: 7.48 GB
- Preflight swap used: 0.405 GB
- Guard thresholds: stop if swap used > 1.0 GB or available RAM < 3.0 GB
- Disk: no low-disk blocker observed
- Stale process check: no stale llama/python workload blocker before the controlled WS-6144 attempt

## E. WS-6144 Result

`FAIL_WS6144_CLIFF`

The WS-6144 extension probe did not produce a sane bounded answer. The controlled auto-policy run at `num_ctx=8192` returned a non-zero Ollama/curl error path with no response text captured. A follow-up connected request also cleared without a captured model response. Because output sanity and score could not be established, the result is classified as a failure/pressure point rather than a pass.

## F. Context Used

- Working-set target: WS-6144
- Estimated input tokens: ~6170
- Prompt size: 24683 chars
- Context used: `num_ctx=8192`
- Reason: WS-6144 cannot safely fit inside `num_ctx=4096`; `num_ctx=8192` was used only for this WS-6144 probe.

## G. Policy Used

- Policy: auto only
- Comparison policy: not run
- Additional working-set sizes: not run

## H. Score/Output Sanity

- Output sanity: failed / not established
- Score: 0.0 by classification, because no valid response text was captured
- Timeout/cap behavior: bounded request exited through an error path with blank stderr; no runaway output observed
- Abort criteria triggered: output sanity failure / timeout-error behavior

## I. Swap Behavior

Swap remained stable during the attempted run.

| Point | Swap used |
|---|---:|
| Preflight | 0.405 GB |
| Controlled run before | 0.367 GB |
| Controlled run after | 0.367 GB |
| Delta | 0.0 GB |
| Later recovery check | 0.367 GB |

Swap did not spike above the 250 MB delta tripwire.

## J. RAM Behavior

RAM remained above the guard threshold.

| Point | Available RAM |
|---|---:|
| Preflight | 7.48 GB |
| Controlled run before | 7.35 GB |
| Controlled run after | 7.20 GB |
| Later recovery check | 7.10 GB |

Available RAM did not drop below the 3.0 GB tripwire.

## K. Cliff Point If Any

WS-6144 is the first observed pressure/failure point in this bounded 7B working-set sequence.

The failure mode was not swap exhaustion. It was bounded generation failure/error behavior at `num_ctx=8192` with no sane answer captured.

## L. Interpretation

`FAIL_WS6144_CLIFF`

Bounded 7B probing passed through WS-4096 in Phase 27H-B, but the WS-6144 extension did not produce a valid answer under the strict guard. This should be treated as the current cliff/pressure zone for the tested setup.

## M. Allowed Claim

Bounded 7B working-set probing passed through WS-4096, but WS-6144 did not complete with a sane captured output under the strict guard.

## N. Forbidden Claims

Do not claim:

- broad 7B validation
- 7B is generally safe
- speedup
- long-context solved
- 14B support
- production readiness
- KV cache solved
- weight-residency solved

## O. Recommended Next Phase

Stop 7B working-set expansion here and document WS-6144 as the pressure/cliff zone.

Recommended next: return to KV memory mapping design or formalize the bounded 7B summary with WS-6144 marked as the first failed extension point.

## P. Models/Sidecars/F32 Refs Staged?

No.

Only this Markdown report and its companion JSON file should be staged for commit.

## Q. Secrets Detected?

No secrets detected in the report content.

## R. Tags Touched?

No tags touched.
