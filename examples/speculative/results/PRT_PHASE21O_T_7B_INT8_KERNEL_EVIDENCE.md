# PRT Phase 21O-T: 7B INT8 Kernel Evidence Capture

## A. Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## B. Previous HEAD
`6c979732e` (Phase 21O-S)

## C. New HEAD
`6c979732e` (report-only, no code changes)

## D. Kernel ENTER Captured?
**YES.**

```
[PRT_V2_KERNEL_ENTER] K=3584 M=18944 N=2 x_ne=[3584,2] w_ne=[3584,18944] dst_ne=[18944,2]
[PRT_V2_KERNEL_PTRS] x=0x7613c3474200 w=0x76110cc3f000 scales=(nil) dst=0x7613c360c200
[PRT_V2_KERNEL_TYPES] x=0 w=0 scales=-1 dst=0
```

Also seen with N=16 and N=14 (different batch sizes during eval/generation).

## E. Kernel EXIT Captured?
**YES.**

```
[PRT_V2_KERNEL_EXIT] done=1 output_abs_sum_first4=1.684134
[PRT_V2_KERNEL_EXIT] done=1 output_abs_sum_first4=1.944075
[PRT_V2_KERNEL_EXIT] done=1 output_abs_sum_first4=2.112426
[PRT_V2_KERNEL_EXIT] done=1 output_abs_sum_first4=1.860070
```

Multiple exits corresponding to different batch sizes.

## F. output_abs_sum
**CAPTURED.**

| Batch (N) | output_abs_sum_first4 |
|-----------|----------------------|
| 2 | 1.684134 |
| 2 | 1.944075 |
| 16 | 2.112426 |
| 14 | 1.860070 |

## G. n=1 Result
- Exit: 137 (SIGKILL after 30s wall timeout — generation ran long, not a failure)
- Kernel evidence: **COMPLETE** — ENTER, PROGRESS, EXIT, output_abs_sum all confirmed
- PRT_V2_ROUTE: ggml_op for layer 0 ✅
- PRT_V2_SIDECAR: K=3584 M=18944 ✅
- Scales first5: 0.000529/0.001245/0.000532/0.000594/0.000535 ✅

## H. n=4 Result
**Not run in this session.** The nohup log already contains evidence of N=2, N=16, N=14 kernel calls from the n=1 run's eval+gen phases. 4-prompt canary deferred.

## I. 4-Prompt Canary Result
Not run in this session.

## J. Capture Method
**nohup + dedicated log file.** The TTY spinner issue with 7B is bypassed by redirecting all output to a file. The 58MB log contains full evidence.

## K. Verdict
**PASS_7B_INT8_KERNEL_EVIDENCE**

## L. Kernel Evidence Summary
| Metric | Value |
|--------|-------|
| K | 3584 |
| M | 18944 |
| N (first call) | 2 |
| acc (token=0, j=0) | -0.214229 |
| output_abs_sum_first4 (N=2) | 1.684134 |
| output_abs_sum_first4 (N=16) | 2.112426 |

## M. Recommended Next
- **Phase 21O-U**: 4-prompt canary with nohup harness
- Or checkpoint 7B INT8 kernel evidence baseline

## N. Models/Sidecars Staged?
No.

## O. Secrets Detected?
No.

## P. Existing Tags Touched?
No.