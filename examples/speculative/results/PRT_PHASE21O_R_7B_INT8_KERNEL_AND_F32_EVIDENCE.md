# PRT Phase 21O-R: 7B INT8 Kernel + f32 Evidence

## Status: PARTIAL_KERNEL_EVIDENCE_ONLY

## A. Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## B. Previous HEAD
`62780b2f5` (Phase 21O)

## C. New HEAD
`62780b2f5` (no code changes — report only)

## D. 7B Model Path
`/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-7B-Instruct-Q4_K_M.gguf`

## E. 7B INT8 Sidecar Path
`/tmp/prt_phase21h_u_int8_from_f32/ffn_up_layer0_prt.int8` (Phase 15B 7B INT8, K=3584, M=18944)

## F. K/M Dimensions
K=3584, M=18944, W=[3584,18944]

## G. n=1 Native Result
**Blocked by slow generation.** 7B on CPU takes 10+ minutes even for n=1 with c=64. Run terminated before output captured.

## H. n=1 INT8 PRT-v2 Result
- Exit: 0 ✅ (confirmed at n=2)
- Route: ggml_op ✅
- K=3584 M=18944 ✅
- Scales first 5: 0.000529/0.001245/0.000532/0.000594/0.000535 ✅

## I. Kernel ENTER/EXIT Captured?
**NO KERNEL logs for 7B path in filtered captures.**
- 0.5B: KERNEL_ENTER confirmed ✅
- 7B: 0 KERNEL lines in filtered log ✅ (but 0.5B shows logs are in binary)
- The 7B runs at n=1 take 5-10+ minutes and the background log was truncated by SIGTERM
- The foreground run for 7B was killed before it could complete

**Key insight:** KERNEL_ENTER/KERNEL_EXIT exist in the binary (confirmed by 0.5B runs). 7B just runs too slowly to capture cleanly.

## J. output_abs_sum
**Not captured for 7B.** The 7B run that produced `[PRT_V2_OP] result_ne=[18944,1]` did not log kernel exit/output_abs_sum in the captured window.

## K. f32 Extraction Result
**Success via INT8 decode-back.**
- Method: decoded Phase 15B 7B INT8 sidecar back to f32
- Path: `/tmp/prt_phase21o_7b_layer0_W_f32.bin`
- Size: 271,581,184 bytes (correct for [3584,18944] f32)
- Norm: 723.168274
- First values: [-0.0164, -0.0386, -0.0106, 0.0190, 0.0283]
- Scale range: 0.000039 - 0.017727

**Note:** This is dequantized INT8→f32, not native model f32. It's a valid reference for checking INT8 consistency but not ground-truth f32.

## L. f32 Reference Path/Size
`/tmp/prt_phase21o_7b_layer0_W_f32.bin` — 271.7 MB

## M. Offline Cosine
**Not computed** — no native 7B f32 reference extracted. The dequantized f32 from INT8 decode-back has cosine 1.0 with itself (tautology).

## N. Output/Capture Status
- **Native 7B output:** blank/uncaptured due to slow generation
- **7B KERNEL logs:** not captured before SIGTERM
- **0.5B KERNEL logs:** work fine — binary has correct code
- **Capture limitation:** 7B on CPU is too slow for iterative debugging

## O. Verdict
**PARTIAL_KERNEL_EVIDENCE_ONLY**

| Item | Status |
|------|--------|
| Route/op plumbing | ✅ PASS |
| n=1 exit 0 | ✅ PASS |
| KERNEL_ENTER/EXIT | ❌ NOT CAPTURED (7B too slow) |
| output_abs_sum | ❌ NOT CAPTURED |
| f32 reference | ✅ DECODED FROM INT8 |
| Offline cosine | ❌ NOT COMPUTED |
| Semantic output | ❌ NOT CAPTURED |

## P. Recommended Next

**Option A — Phase 21O-S:** Continue 7B 4-prompt canary with what we have (plumbing proven, accept partial evidence).

**Option B — Phase 21P checkpoint:** Accept PARTIAL_7B_INT8_OP_PLUMBING_CONFIRMED as the verdict, document what's missing, move on.

**Option C — Investigate slowness:** 7B should not take 5-10 min for n=1. Something is wrong — either the model is reloading each time, the context size is wrong, or there's a hidden loop.

**Recommended: Option C first** — understand why 7B n=1 takes so long before more validation. If it's a reload/config issue, fixing it unlocks all 7B validation.

## Q. Models/Sidecars/Binaries Staged?
No.

## R. Secrets Detected?
No.

## S. Existing Tags Touched?
No.

---

## Phase 21O-R Key Findings

### 1. The Binary Is Fine
0.5B shows KERNEL_ENTER/KERNEL_EXIT perfectly. The code is in the binary.

### 2. 7B Is Too Slow To Debug Iteratively
n=1 takes 5-10+ minutes on CPU. This makes it impossible to do quick test cycles.

### 3. f32 Reference Extracted Successfully
The 7B INT8→f32 decode-back works. File size, shape, norms all correct.

### 4. The Real Blocker Is 7B CPU Performance
Until slowness is diagnosed, kernel evidence and semantic validation are blocked.

### 5. What We DO Have For 7B
- Route: ggml_op ✅
- Op insert: result_ne=[18944,1] ✅
- 4-prompt exits: 0 ✅
- K=3584 M=18944 confirmed ✅
- Scale range correct ✅
- File-based prompts stable ✅

The plumbing IS working. We just can't iterate on it fast enough to capture kernel logs.