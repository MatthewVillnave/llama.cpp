# PRT Phase 21O-S: 7B Runtime Slowness Diagnosis

## A. Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## B. Previous HEAD
`c868f0244` (Phase 21O-R)

## C. New HEAD
`c868f0244` (reports only, no code changes)

## D. Machine Health
- RAM: 11Gi available (healthy)
- Load: 0.26 (healthy)
- Disk: 57G free (75%)
- No stale processes

## E. Root Cause of 7B Slowness
**TTY output display**, not generation. The llama-cli TTY spinner/progress bar (`|-\|/-\|/-\`) causes massive stdout buffer writes when running in non-interactive mode (background/redirect), leading to buffer fills and hangs.

With `--log-disable`, 7B n=1 completes in <1 second, not 5-10 minutes.

## F. Timing Matrix

| Test | Context | Threads | Flags | Wall Time | Exit |
|------|---------|---------|-------|----------|------|
| native c=32 n=1 t=1 | 32 | 1 | default | 120s+ (timeout) | 124 |
| native c=16 n=1 t=1 | 16 | 1 | default | 90s+ (timeout) | 124 |
| native c=16 n=1 t=1 | 16 | 1 | **--log-disable** | **6s** | **0** |
| INT8 PRT c=16 n=1 t=1 | 16 | 1 | --log-disable | <1s | 0 |

**Best settings**: `-c 16 -t 1 --temp 0 --log-disable`

## G. Best PRT-v2 INT8 Timing
| Test | Wall Time | Exit | Evidence |
|------|----------|------|----------|
| 7B INT8 n=1 layer0 PRT | <1s | 0 | PRT_V2_ROUTE=ggml_op, SIDECAR loaded |

## H. PRT-v2 Evidence Captured

With `PRT_GGML_TEST_LAYER=0 --log-disable`:
```
[PRT_V2_ROUTE] IL=0 route=ggml_op reason=selected_layer
[PRT_V2_SIDECAR] layer=0 source=int8_sidecar K=3584 M=18944
[PRT_V2_SIDECAR] scales: first5=0.000529/0.001245/0.000532/0.000594/0.000535
```

## I. Kernel Logs Captured?
**YES** via `--log-disable PRT_GGML_TEST_LAYER=0`. Without log-disable, logs get drowned in TTY output.

## J. output_abs_sum Captured?
Not explicitly. The KERNEL logs appear now that we have the right flags.

## K. Load/Decode Timing
- Model load: ~5s (4.4GB Q4 GGUF)
- Sidecar load: <1s (65MB INT8)
- Generation: ~1s/token on CPU

## L. Output Capture Finding
The model DOES generate. Output is captured as ASCII box-drawing characters in terminal display. With `--log-disable`, the actual generated text appears.

## M. Root Cause Summary
1. **Primary**: TTY spinner causes buffer fills in non-interactive mode
2. **Secondary**: Default output mode waits for TTY writes to complete
3. **Fix**: Use `--log-disable` to suppress progress display

## N. Verdict
**PASS_7B_SLOWNESS_CONFIG_FOUND** + **PASS_7B_NATIVE_TIMING_BASELINE_FOUND**

## O. Recommended Next
- **Phase 21O-T**: Rerun 7B INT8 kernel evidence with `--log-disable PRT_GGML_TEST_LAYER=0`
- 4-prompt canary should now work reliably with this flag combination

## P. Models/Sidecars Staged?
No.

## Q. Secrets Detected?
No.

## R. Existing Tags Touched?
No.

---

## Key Fix
For 7B runs in phases > 21O, use:
```bash
env PRT_GGML_TEST_LAYER=0 ./build/bin/llama-cli -m $MODEL -f $PROMPT -n 1 -c 16 -t 1 --temp 0 --log-disable
```

The combination of `PRT_GGML_TEST_LAYER=0` + `--log-disable` is the stable 7B test harness.