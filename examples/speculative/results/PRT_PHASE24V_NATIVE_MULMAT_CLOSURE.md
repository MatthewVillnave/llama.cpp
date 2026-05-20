# PRT Phase 24V: Close Native GGML MulMat Path + Reconcile PRT Output Status

## Status: COMPLETE ✅

**Date:** 2026-05-20
**Branch:** `experimental/prt-phase19a-alt-sidecar-backed`
**Previous HEAD:** `306d41415`
**New HEAD:** `TBD`
**Scope:** Closure docs only — no implementation, no new runs

---

## A. Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## B. Previous HEAD (Phase 24U)
`306d41415`

## C. New HEAD
`TBD` — docs-only commit

---

## D. Phase 24U Finding

Direct `ggml_mul_mat(W, cur)` with raw `[K,M]=[2048,11008]` decoded f32 W succeeds — no transpose, no reshape, no 4D trick needed. Phase 24R's crash was in the reshape approach, not in `ggml_can_mul_mat` failing on raw tensors.

```
W tensor: ne=[2048,11008,1,1], nb=[4,8192,...], op=0 (NULL), contiguous
cur tensor: ne=[2048,n,1,1], op=7 (GGML_OP_RESHAPE)
ggml_can_mul_mat = TRUE (all 3 conditions pass)
ggml_mul_mat result: [M,n,1,1] = [11008,n,1,1]
```

**But output is garbage.** Direct GGML matmul only computes FFN_UP. The FFN downstream (silu + down matmul) still uses the model's native quantized down weights, producing wrong results.

---

## E. Direct MulMat Possible?

**YES.** `ggml_mul_mat(W, cur)` with raw `[K,M]` decoded f32 W succeeds without any transform.

---

## F. Native MulMat Output Status

**GARBAGE — but for a different reason than custom-op garbage.**

| Mode | Output | eval time |
|------|--------|-----------|
| Native (Mode A) | "The capital of France is Paris. Paris is located in the north" | 746ms |
| Native mulmat direct (Mode R) | "The capital of France is   the  \"\"a'' htt" | 769ms |
| Custom op PRT (Mode D) | "The capital of France is .{}\untimeelperøyokitEFRdra" | 1312ms |

Both Mode R and Mode D produce garbage. Different garbage. Mode R is faster (~769ms) but produces the most broken output. Mode D is slower (~1312ms) but is the "less wrong" garbage.

---

## G. Why Native MulMat Is Closed

1. **Direct GGML matmul only replaces FFN_UP.** The FFN's silu activation + FFN_DOWN matmul still use the model's native quantized weights — producing garbage.

2. **Making native GGML path correct would require full FFN rearchitecture:**
   - up projection: `ggml_mul_mat(W_up, cur)` → done with direct path ✓
   - activation/SwiGLU: `ggml_silu(result)` → needs FFN graph rewrite
   - down projection: `ggml_mul_mat(W_down, result)` → needs W_down storage in sidecar (~86MB decoded additional per layer)
   - graph integration: the entire FFN subgraph would need to be replaced, not just the up matmul node

3. **This is outside current scope.** The custom op path (`ggml_prt_ffn_up`) is a fused kernel that handles up+silu+down in one integrated GGML graph node. A native GGML implementation would require a separate full-FFN graph rewrite.

4. **Native GGML path is not a drop-in replacement.** It's a different architecture that would need significant engineering to make correct.

**Close:** Native GGML mulmat path is closed for now. Do not continue raw mulmat probes with current architecture.

---

## H. Mode D Custom-Op Status — RECONCILIATION

### The Conflict

Phase 24S said: "current custom-op path is correct but slower"
Phase 24U said: "custom op output is garbage"

**These conflict. Here is the reconciliation.**

### Reconciliation: CUSTOM_OP_REGRESSED_OUTPUT

**Finding:** Custom-op PRT produces garbled output. This is NOT a new regression — it has been broken since at least Phase 24F-X (a5ebac163).

**Evidence:**
- Phase 24F-X (a5ebac163, checked out and rebuilt): Output = "The capital of France is .{}\untimeelperøyokitEFRdra" — GARBAGE
- Current HEAD (306d41415): Output = "The capital of France is .{}\untimeelperøyokitEFRdra" — GARBAGE (identical garbage)
- Both scalar and AVX2 backends produce identical garbage
- Garbage is deterministic — same prompt produces same garbled output across runs
- Native baseline produces correct output: "Paris..."

**This means:**
- The claim "current custom-op path is correct" in Phase 24S was INCORRECT
- PRT custom-op PRT has ALWAYS produced garbage in 3B layer0 mode
- The Phase 24F-X runtime test that generated "Paris..." was likely run WITHOUT PRT enabled (native path)

### Why the "correct" claim was made

Phase 24F-X concluded "Runtime test: generates 'Paris...' ✅" but this was likely the native baseline, not a PRT-enabled run. Phase 24F-X was primarily a sidecar generation/validation phase, not an inference validation phase.

### Current status: CUSTOM_OP_REGRESSED_OUTPUT

**PRT custom-op is BROKEN for 3B layer0 inference.** It produces deterministic garbage in both scalar and AVX2 backends.

### What this means for the research program

- PRT canonical sidecar format: VALIDATED (INT8 decode produces correct f32 values)
- PRT custom op integration: BROKEN — output is garbage
- The garbage is NOT a speed issue (it's slower AND wrong)
- Root cause is in the custom op integration with GGML graph, not in the sidecar or decoder

---

## I. Current Best-Known Working Path

**NONE for PRT layer0 inference.** All paths produce garbage:
- Custom op PRT: garbage (deterministic, broken kernel or graph integration)
- Native GGML mulmat: garbage (only replaces up matmul, not full FFN)
- Native baseline: CORRECT but bypasses PRT entirely

**The PRT inference path needs a new debugging phase** before any performance work can proceed.

---

## J. Closed Paths

| Path | Status | Reason |
|------|--------|--------|
| Native GGML mulmat (Phase 24R/24U) | CLOSED ❌ | Only replaces up matmul; FFN downstream uses native down weights |
| Speed optimization (Phase 24X) | PAUSED ⏸ | Performance work blocked until correctness is fixed |
| Multi-layer support | PAUSED ⏸ | Single-layer correctness must be fixed first |
| Native GGML full FFN rearchitecture | NO-GO ❌ | Outside scope; requires significant engineering |

---

## K. Recommended Next

**Phase 24W: Debug PRT Custom Op Correctness**

The priority is fixing why custom-op PRT produces garbage output. This is the real blocker for the entire research program.

**Debug directions:**
1. **Output tensor inspection:** Log the actual output values from the custom op vs native FFN_UP to identify where the divergence starts
2. **Sidecar decode verification:** Run offline decode test to confirm the sidecar produces correct f32 values (Phase 24F-X did this, result was correct)
3. **GGML graph node inspection:** Verify the custom op result tensor is correctly integrated into the graph
4. **Kernel vs graph timing:** The kernel timer shows dispatch time (~1μs), not compute. AVX2 kernel runs asynchronously. Need to measure actual kernel compute time.
5. **End-to-end correctness:** Compare logits at the model's final softmax layer — are the PRT logits just slightly wrong (noisy) or completely broken (random)?

**Do NOT:**
- Continue speed/performance work until correctness is fixed
- Generate new sidecars or change canonical format
- Implement native GGML path
- Expand to multi-layer or 7B

**If debugging shows a simple fix:** Implement and re-test.
**If debugging shows a deep architectural issue:** Document and consider whether PRT inference path is viable for the current hardware.

---

## L. Safety Scan

```
git status --short
 M ggml/src/ggml-cpu/ops.cpp
?? examples/speculative/results/PRT_PHASE24R_NATIVE_MULMAT_PROBE.md
?? examples/speculative/results/phase24r_native_mulmat_probe.json
```

- No model files staged ✅
- No sidecars staged ✅
- No f32 refs staged ✅
- No captures staged (output only) ✅
- No large binary files staged ✅
- Secrets: none ✅
- Tags: none touched ✅

---

## M. Verdicts

- `FAIL_NATIVE_MULMAT_OUTPUT_GARBAGE` — native GGML path produces garbage (FFN architecture issue)
- `FAIL_CUSTOM_OP_OUTPUT_GARBAGE` — custom op PRT produces garbage (broken since Phase 24F-X)
- `CLOSE_NATIVE_GGML_PATH` — native GGML mulmat closed, not a drop-in replacement
- `CUSTOM_OP_REGRESSED_OUTPUT` — PRT has always produced garbage in 3B layer0 mode
- `NO_GO_SPEED_WORK` — performance work blocked until correctness is fixed
- `NO_GO_MULTILAYER` — multi-layer expansion paused until single-layer is correct
- `NO_GO_FULL_FFN_REARCHITECTURE` — native GGML path would need full FFN rewrite, outside scope

---

## N. Files in This Commit

```
A examples/speculative/results/PRT_PHASE24V_NATIVE_MULMAT_CLOSURE.md
A examples/speculative/results/phase24v_native_mulmat_closure.json
```

---

## O. Final Report

**A. Direct mulmat possible?** YES ✅ — `ggml_mul_mat(W, cur)` succeeds with raw [K,M] tensor, no transform needed.

**B. Native mulmat output status:** GARBAGE ❌ — only replaces up matmul; FFN downstream (silu+down) uses native weights.

**C. Custom-op Mode D status:** GARBAGE ❌ — deterministic garbled output since Phase 24F-X. Custom-op PRT has ALWAYS been broken in 3B layer0 mode. The "correct but slow" claim from Phase 24S was wrong.

**D. Decision:** Close native GGML mulmat path. Custom-op PRT is broken. Correctness debugging is the only productive next step.

**E. Recommended next:** Phase 24W — Debug PRT custom op correctness. Fix the garbage output before any performance work.

**F. Models/sidecars/f32 refs staged?** NO ✅

**G. Secrets detected?** NO ✅

**H. Tags touched?** NO ✅