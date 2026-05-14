# Phase 10D Quality Notes

## Local Matmul Accuracy

All guard-checking modes (B, C, D) maintain min cosine >= 0.95.
Mode E (PRT-only) has no guard, so local cosine cannot be verified.

## Guard Failure Analysis

- ModeE_prt_only: guard_fails=0 min_cos=-1.000000

## End-to-End Generation

**NOTE:** Full end-to-end generation was NOT run in this phase.
Only local matmul accuracy was measured.
For true end-to-end quality validation, run Phase 10E.

Mode E (PRT-only) is fastest but has no safety net.
If end-to-end generation shows quality degradation with Mode E,
fall back to Mode C (periodic guard) or Mode D (layer-sampled).
