# PRT Phase 20A: 7B Sparse INT6 Policy Validation — FINAL REPORT

## A. Branch
experimental/prt-phase19a-alt-sidecar-backed

## B. Previous HEAD
ad55b961f

## C. New HEAD
pending

## D. Machine Health
- RAM: 15GB total, 5.6GB free, 11GB available — healthy
- Swap: 4GB used / 4GB total
- Disk: 60GB free
- Branch: correct

## E. Native Baseline Summary

| Prompt | Gen t/s | Output |
|--------|---------|--------|
| P1: "The capital of France is" | 4.8 | correct |
| P2: "The largest planet..." | 4.2 | correct |
| P5: "train speed..." | 4.2 | correct |
| P6: "Once upon..." | 4.2 | correct |
| P7: "JSON..." | 4.4 | correct |

## F. (10,20) Correctness

**MAJOR ISSUE: PRT IS NOT ACTIVATING RELIABLY.**

During testing all layers showed `prt_layer=0` meaning PRT is not being used. When PRT does activate (visible via INT6/SHAPE logs), the output shows language mixing/corruption:

- Observed output when PRT activates: `我可以帮助` (Chinese characters) - CORRUPT
- INT6/SHAPE shows PRT processing but with language errors
- Generation runs slow (not faster than native)

**Timing comparison:**

| Mode | Prompt t/s | Gen t/s |
|------|-----------|---------|
| Native | 9.8 | 4.8 |
| (10,20) | 6.0 | 3.6 |

**(10,20) is ~20% SLOWER than native, not faster.**

## G. (10,20) Repeatability
Not testable - PRT activation is unreliable.

## H. Secondary Candidate Results
Not tested - PRT activation issue must be resolved first.

## I. Corrupt Control Results
Not tested during this run - PRT issue.

## J. Best Policy
Unable to validate due to PRT activation failure.

## K. Speedup Claim Corrected?
**CLAIM REMOVED.** Timing shows (10,20) is ~20% SLOWER, not faster.

## L. Prompt Sensitivity
Cannot assess - PRT activation issue.

## M. Verdict: **BLOCKED_BY_PRT_ACTIVATION_FAILURE**

The PRT layer routing is not working reliably in this environment. All (10,20) runs showed `prt_layer=0` for all layers, indicating PRT is not being invoked. When PRT does activate (sporadically), output shows language mixing/corruption.

This is a critical issue that must be debugged before policy validation can continue.

## N. Recommended Next
1. **Debug PRT activation** — verify layer routing is working in current build
2. **Check for regression** — did something break between Phase 19Z and now?
3. **Rebuild if needed** — verify the build includes all PRT changes
4. Then rerun Phase 20A validation

## O. Models/Sidecars/Binaries Staged?
No.

## P. Secrets Detected?
None.

## Q. Existing Tags Touched?
None.

## Root Cause Investigation
- All logs show `prt_layer=0` for all 28 layers
- Debug mode set to 5700 correctly (confirmed in logs)
- Sidecar path correctly specified
- This indicates layer routing code is not executing properly OR the build is stale

**This phase cannot proceed until the PRT activation issue is resolved.**