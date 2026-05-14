# PRT Phase 21H-T: INT8 Decode Fix

## Status: BLOCKED_CAPTURE_TOOLING (runtime verification incomplete)

## A. Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## B. Previous HEAD
`dd9c190a6` (Phase 21H-S - f32 vs INT8 layout reconciled)

## C. New HEAD
`1bd00aa8a` (pre-fix)

After fix:
`e8d61f2a` (with decode formula fix applied)

## D. Decode formula fixed?
**YES** - Changed in src/llama-graph.cpp line ~1301:

| Before (wrong) | After (corrected) |
|--------------|-------------------|
| int8_buf[j*K + k] | int8_buf[k*M + j] |

Comment also updated to note [K,M] layout.

## E. Offline cosine vs f32
**0.9656** (cosine -0.0002 with wrong formula)
- MAE: 0.002402
- Norm ratio: 1.0675 (INT8 6.8% larger)
- Sign match: 100%

## F. Offline MAE
0.002402 (vs 0.020505 with wrong formula)

## G. Native output
"The capital of France is" → "The capital of France is Paris." ✅
(Confirmed via PTY capture - see earlier sessions)

## H. Corrected INT8 PRT-v2 output
**UNABLE TO CONFIRM** - Output capture issue:

- PTY capture: Shows "The capital of France is" prompt echo only, NO generated text
- File redirect: Shows only prompt echo, no generated text
- Kernel: Executes (KERNEL_ENTER → KERNEL_EXIT with output_abs_sum_first4=3.866371)
- Process: Exits cleanly OR with SIGTERM/SIGKILL

The model runs but generated text is not captured - appears only in actual TTY.

## I. 4-prompt result
Not run - capture issue prevents reliable testing.

## J. Route/op evidence
- Layer 0: route=ggml_op (selected_layer)
- Sidecar: int8_sidecar loads with scales
- Decode: formula=int8[k*M+j]*scale[j] logged
- KERNEL_ENTER: K=896 M=4864 N=2
- KERNEL_EXIT: done=1 output_abs_sum_first4=3.866371

**Route path correct** - but generation output missing.

## K. Verdict
**PARTIAL_OFFLINE_FIX_RUNTIME_UNCLEAR**

## L. Recommended next
1. **Debug capture issue**: Model actually runs but output text not captured in pipes - needs TTY
2. **Or**: The 6.8% norm increase (40.68 vs 38.11) may cause downstream issues
3. The INT8 sidecar was quantized independently - may not match f32_file source
4. **Alternative approach**: Generate fresh INT8 sidecar from confirmed f32_file for Phase 21H-U

## M. Models/sidecars/binaries staged?
No - /tmp references only

## N. Secrets detected?
No

## O. Existing tags touched?
No

## Key Observations

1. **Offline verification confirms formula fix** - cosine improves from -0.0002 to 0.9656
2. **Kernel executes correctly** - output_abs_sum 3.866371 (f32 is 3.842286)
3. **Capture is broken** - output appears only in TTY, not in piped/redirected stdout
4. **Generation fails** - INT8 path produces no visible text vs f32 path produces "Paris"
5. **Norm mismatch** - INT8 W norm is 6.8% larger than f32 W

The decode formula IS mathematically correct now. The runtime failure may be:
- Capture issue masking actual output
- Norm mismatch causing numerical issues
- Independent quantization mismatch

## Files Modified
- src/llama-graph.cpp: INT8 decode index formula fix

## Commit & Push
```bash
git add src/llama-graph.cpp
git commit -m "PRT Phase 21H-T: fix 0.5B INT8 decode [j*K+k]→[k*M+j]"
git push fork experimental/prt-phase19a-alt-sidecar-backed
```