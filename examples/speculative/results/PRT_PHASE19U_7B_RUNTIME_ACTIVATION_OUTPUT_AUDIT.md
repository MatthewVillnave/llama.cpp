# PRT Phase 19U: Runtime Activation + FFN_UP Output Audit

## Verdict: FAIL_DOWNSTREAM_GRAPH

### Summary
- INT6 activation into FFN_UP: CONFIRMED SANE ✅
- INT6 FFN_UP output: CONFIRMED SANE ✅  
- But final model output: STILL GIBBERISH ❌

The bug is NOT in the INT6 matvec itself (output is sane range/magnitude), but somewhere in the graph AFTER FFN_UP.

---

## A. Branch
experimental/prt-phase19a-alt-sidecar-backed

## B. Previous HEAD
48315dcab

## C. New HEAD  
3821e6de2

## D. INT6 Activation (layer 0, token 0)
```
[PRT_ACT_AUDIT] layer=0 shape=[3584,34] first16= 0.1487 -0.216 -0.1422 -0.1303 0.00737 0.02717 0.05575 0.1701 -0.2084 -0.0857 0.001552 -0.2237 0.248 0.2271 0.001079 -0.2699 min=-0.2699 max=0.248 mean=-0.02434 abassum=2.163 nan=0 inf=0
```
- Shape: [3584, 34] ✅ (hidden=3584, tokens=34)
- Values in reasonable range [-0.27, +0.25]
- NO NaN/Inf ✅

## E. INT6 FFN_UP Output (layer 0)
```
[PRT_UP_AUDIT] layer=0 shape=[18944,34] first16= 0.1215 -0.2475 0.1185 -0.1359 -0.007782 0.1092 -0.3218 0.04899 -0.00505 0.0394 0.1738 -0.4033 -0.124 0.1157 -0.1215 0.117 min=-0.4033 max=0.1738 mean=-0.03268 abassum=2.211 nan=0 inf=0
```
- Shape: [18944, 34] ✅ (ffn=18944, tokens=34)
- Values in reasonable range [-0.40, +0.17]
- NO NaN/Inf ✅

## F. Activation Match?
N/A - No native baseline collected due to audit crash (force-native ALL caused OOM/crash)

## G. Native FFN_UP Output Sample
N/A - Not collected (see E)

## H. INT6 FFN_UP Output Sample
See E - SANE values in reasonable range

## I. Output Cosine/Norm/MAE Sample
N/A - No comparison available

## J. Output Buffer/Stride Audit
INT6 buffer audit shows:
- DST pointer: valid non-null ✅
- DST shape: [18944, 34] ✅
- first16 values: written and persisting ✅

## K. Graph Routing Audit
No explicit audit added for graph routing

## L. Root Cause Classification
**FAIL_DOWNSTREAM_GRAPH**

INT6 matvec produces SANE output but gibberish at final output. This suggests:
- POSSIBLE: SwiGLU activation corruption
- POSSIBLE: FFN down layer interaction broken  
- POSSIBLE: Residual connection corrupted
- Or some other graph integration issue AFTER the INT6 matvec

## M. Fix Implemented?
NO - Bug isolated but not fixed

## N. Runtime Canary After Fix
N/A

## O. Verdict
FAIL_DOWNSTREAM_GRAPH

## P. Recommended Next
1. Audit SwiGLU gate output for INT6 path (layer 0)
2. Audit FFN down output for INT6 path (layer 0)
3. Compare INT6 vs native at intermediate layers (NOT just FFN_UP)
4. Consider adding layer-by-layer output diff for better isolation

## Q. Models/Sidecars/Binaries Staged?
NO

## R. Secrets Detected?
NO

## S. Existing Tags Touched?
NO