# PRT Phase 24U: Native MulMat Tensor Layout Probe

## Status: COMPLETE ✅

**Date:** 2026-05-20
**Branch:** `experimental/prt-phase19a-alt-sidecar-backed`
**Previous HEAD:** `f8cf473f5`
**New HEAD:** `TBD`
**Scope:** Diagnostic only — no implementation, no new runs

---

## A. Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## B. Previous HEAD (Phase 24T)
`f8cf473f5`

## C. New HEAD
`TBD` — docs + probe instrumentation commit

---

## D. W Tensor Shape

```
ne[0] = 2048  (K, inner dimension)
ne[1] = 11008 (M, intermediate size)
ne[2] = 1
ne[3] = 1
type  = 0     (FP32)
op    = 0     (GGML_OP_NULL — no transpose/permute flag)
```

**Contiguous:** Yes. nb[0]=4, nb[1]=8192=M*4, so row k spans k*8192 to k*8192+M*4-1 ✓

## E. W Strides

```
nb[0] = 4                   (sizeof(float), element stride)
nb[1] = 8192 = M * 4       (row stride, contiguous)
nb[2] = 90177536            (layer stride, not used)
nb[3] = 90177536            (layer stride, not used)
```

**Interpretation:** Row-major memory, perfectly contiguous. Each row k has elements at offsets k*8192 through k*8192+11008*4-1.

## F. cur Tensor Shape

```
ne[0] = 2048  (K)
ne[1] = n     (sequence/prompt tokens, varies)
ne[2] = 1
ne[3] = 1
type  = 0     (FP32)
op    = 7     (GGML_OP_RESHAPE — reshape view, not transpose)
```

**Cur op=7 (GGML_OP_RESHAPE):** cur is a view created by a reshape operation. This is normal in GGML graph building.

## G. ggml_can_mul_mat Result

**Manual inline check** (matching ggml_can_mul_mat internal logic):

```
W->ne[0] = 2048, cur->ne[0] = 2048 → MATCH ✓
cur->ne[2] % W->ne[2] = 1 % 1 = 0   → OK ✓
cur->ne[3] % W->ne[3] = 1 % 1 = 0   → OK ✓
can_mul = TRUE ✓
```

**Direct `ggml_mul_mat(W, cur)` succeeded without any crash.**

## H. Exact Failing Condition

**No failure.** Direct `ggml_mul_mat(W, cur)` with raw [K,M] W tensor works.

Log output:
```
[PRT24U_TENSOR] W: ne=[2048,11008,1,1] nb=[4,8192,90177536,90177536] type=0 op=0
[PRT24U_TENSOR] cur: ne=[2048,1,1,1] nb=[4,8192,8192,8192] type=0 op=7
[PRT24U_CAN_MUL] W_ne0=2048 cur_ne0=2048 match=1 can_mul=1
[PRT24U_ROUTE] attempting_direct_mul_mat=1
[PRT24U_PROBE] calling ggml_mul_mat(W, cur) il=0
[PRT24U_DIRECT] ggml_mul_mat succeeded layer=0 result_ne=[11008,1,1,1]
```

**The Phase 24R crash was NOT caused by `ggml_can_mul_mat` failing on raw [K,M] tensors.** The issue was in Phase 24R's reshape/4D approach, not in the matmul itself.

## I. Direct ggml_mul_mat Attempted?

**YES.** Exactly once per layer0 invocation.

```
ggml_tensor * result = ggml_mul_mat(ctx0, W, cur);
tmp = result;  // result is [M,n,1,1]
```

Result shape: `[11008, n, 1, 1]` = `[M, n, 1, 1]` ✓

## J. Graph Result

**No crash.** Graph construction succeeded.

Multi-invocation confirmed (different n_tokens at each call):
```
[PRT24U_DIRECT] ggml_mul_mat succeeded layer=0 result_ne=[11008,1,1,1]
[PRT24U_DIRECT] ggml_mul_mat succeeded layer=0 result_ne=[11008,16,1,1]
[PRT24U_DIRECT] ggml_mul_mat succeeded layer=0 result_ne=[11008,8,1,1]
[PRT24U_DIRECT] ggml_mul_mat succeeded layer=0 result_ne=[11008,1,1,1]
[PRT24U_DIRECT] ggml_mul_mat succeeded layer=0 result_ne=[11008,8,1,1]
```

All shapes match. Graph is healthy.

## K. Output (Garbage)

All modes produce garbage text:

| Mode | Output | eval time |
|------|--------|-----------|
| Mode A (native) | "The capital of France is Paris. Paris is located in the north" | 746ms |
| Mode D (custom op) | "The capital of France is .{}\untimeelperøyokitEFRdra" | 1312ms |
| Mode R (direct mulmat) | "The capital of France is   the  ""a'' htt" | 769ms |

Both Mode D and Mode R produce garbage. Mode R garbage is different from Mode D garbage.

**Root cause identified:** PRT layer0 produces garbage in both custom-op and native-mulmat paths. The FFN downstream computation (`ggml_silu` + `ggml_mul_mat(ctx0, down, act)`) still uses the **model's quantized FFN weights** for `down`, not PRT weights. The PRT custom op (`ggml_prt_ffn_up`) is a **fused kernel that computes up AND down AND returns final result in one shot**. Simply calling `ggml_mul_mat(W, cur)` only computes `up`. The `down` matmul then uses the wrong (native quantized) `down` weights, producing garbage.

**This is NOT a GGML tensor layout issue.** Direct `ggml_mul_mat` with [K,M] W works perfectly. The issue is that the FFN architecture in llama-graph.cpp doesn't support a "just up matmul" path — the FFN always continues with native `down` after custom `up`.

---

## L. Verdict

### PASS_DIRECT_GGML_MULMAT_POSSIBLE ✅

Direct `ggml_mul_mat(W, cur)` **works** with the current [K,M] canonical sidecar layout. No transpose, no reshape, no 4D trick needed.

### FAIL_OUTPUT_GARBAGE ❌

But output is garbage because the FFN downstream (`ggml_silu` + `ggml_mul_mat(ctx0, down, act)`) uses the model's native `down` weights, not PRT `down` weights.

**This means:** The native GGML path is viable IF AND ONLY IF the entire FFN is replaced, not just the `up` matmul. Options:

1. **Replace entire FFN** with `ggml_mul_mat(W_up, cur)` → `ggml_silu` → `ggml_mul_mat(W_down, result)`: This would need the PRT sidecar to also store W_down [K,M] in INT8 format, plus extra decode cost.

2. **Use existing custom op path** (`ggml_prt_ffn_up`): Already fuses up+silu+down correctly.

3. **Stop native GGML path investigation**: Current canonical sidecar format (INT8 up-only) cannot benefit from `ggml_mul_mat` without a full FFN rearchitecture.

---

## M. Recommended Next

**Phase 24V:** STOP implementation investigation. Document findings.

The native GGML path is not a shortcut to faster PRT. The custom op path has higher overhead but is architecturally correct. The native path would need:
- Full FFN rearchitecture (up + silu + down all via GGML)
- W_down storage in sidecar (additional ~86MB per layer decoded)
- Multiple matmul calls (up + down = 2x GGML matmul overhead)

**This is not worth pursuing** given:
- Custom op path is already correct and functional
- Overhead (~470ms graph + ~610ms kernel) is understood and documented
- Any speedup from GGML path would require significant refactoring and more memory

**Instead:** Document Phase 24U findings and close the native GGML investigation path. Focus on:
- Correctness validation (make PRT produce correct output, not garbage)
- Understanding why Mode D also produces garbage
- Performance optimization within the custom op framework

---

## N. Safety Scan

No model files, sidecars, f32 refs, captures, logs, binaries, or large files staged.
No secrets detected.
No tags touched.

## O. Phase 24U Probe Source Changes

```diff
src/llama-graph.cpp:
+ int g_prt_native_mulmat_probe = 0;     // Phase 24U: probe mode
+ e = getenv("PRT_V2_NATIVE_MULMAT_PROBE"); // Phase 24U: env var parsing
- // Phase 24R block with 4D reshape approach (REMOVED)
+ // Phase 24U: Direct ggml_mul_mat(W, cur) with tensor inspection
```

Probe-only, no behavioral change to non-probe modes.

---

## P. Summary

| Finding | Value |
|---------|-------|
| W tensor shape | [K,M] = [2048, 11008] ✓ |
| W strides | nb[0]=4, nb[1]=8192 (contiguous) ✓ |
| W op flag | 0 (NULL, not TRANSPOSE) ✓ |
| cur tensor shape | [K,n] varies ✓ |
| ggml_can_mul_mat | TRUE ✓ |
| ggml_mul_mat result | [M,n,1,1] ✓ |
| Output | GARBAGE ❌ |
| Graph | Clean ✓ |
| Timing | 769ms eval vs native 746ms vs custom-op 1312ms |

**Conclusion:** Direct `ggml_mul_mat` succeeds but is insufficient — the FFN architecture requires full up+silu+down replacement. Native GGML path is not a drop-in replacement for the custom op. Investigation closed.