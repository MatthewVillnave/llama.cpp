# PRT Phase 22H: INT8 ggml-native AVX2 Backend Baseline Checkpoint

## Branch
experimental/prt-phase19a-alt-sidecar-backed

## Previous HEAD
39111f369 (Phase 22G-R)

## New HEAD
3a8c9f12b (Phase 22H checkpoint)

---

# 1. Executive Summary

INT8 sidecar path now routes through ggml-native GGML_OP_PRT_FFN_UP to ops.cpp AVX2 backend:
- 0.5B INT8 AVX2 runtime is repeat-stable
- 7B INT8 AVX2 backend reachability is confirmed
- 7B same-input/call numeric sanity supported by second-call exact abs_sum match
- **This is a backend baseline, not a production/speed/semantic baseline**

---

# 2. Frozen Technical Baseline

| Parameter | Value |
|-----------|-------|
| **Branch** | experimental/prt-phase19a-alt-sidecar-backed |
| **HEAD** | 3a8c9f12b |
| **Op** | GGML_OP_PRT_FFN_UP |
| **Backend** | ops.cpp scalar/AVX2 |
| **Env** | PRT_V2_AVX2=1 for AVX2 |
| **Layer** | layer0 only |
| **Sidecar type** | INT8 decoded to f32 W |
| **0.5B dimensions** | K=896, M=4864 |
| **7B dimensions** | K=3584, M=18944 |
| **Prompt method** | file prompts only |
| **Scratch** | /media/matthew-villnave/VL_usb/prt_scratch |

---

# 3. 0.5B Evidence

## Regenerated Valid INT8 Sidecar
- Path: `sidecars/prt_phase22e_05b_int8_from_f32/ffn_up_layer0_prt.int8`
- Size: 4,377,600 bytes (expected: 896*4864 + 4864*4)
- Offline cosine vs f32: **0.99996627**

## Scalar Path
- `[PRT_V2_PATH] mode=ggml_native_op` ✅
- `[PRT_V2_BACKEND] scalar` ✅
- Output_abs_sum: 17.721163, 15.601974

## AVX2 Path
- `[PRT_V2_PATH] mode=ggml_native_op` ✅
- `[PRT_V2_BACKEND] avx2` ✅
- Output_abs_sum: 31.598730, 18.259108

## Repeat Validation
- Scalar deterministic across 3 runs (exact match)
- AVX2 deterministic across 3 runs (exact match)

## n=8 Sanity
- All 8 tokens used AVX2, no fallback ✅

## Output Valid
- "The capital of France is" ✅

---

# 4. 7B Evidence

## INT8 Sidecar
- Size: 67,971,072 bytes ✅ (3584*18944 + 18944*4)

## Scalar ggml-native Route
- `[PRT_V2_PATH] mode=ggml_native_op` ✅
- `[PRT_V2_BACKEND] scalar` ✅

## AVX2 ggml-native Route
- `[PRT_V2_PATH] mode=ggml_native_op` ✅
- `[PRT_V2_BACKEND] avx2` ✅

## No Inline Fallback Observed
- Both scalar and AVX2 route through ggml-native op path

## Kernel Evidence
- K=3584, M=18944
- Kernel ENTER/EXIT captured
- Scalar kernel time: ~1201ms
- AVX2 kernel time: below ms logging resolution

## Phase 22G-R Same-Call Sanity
- Second-call abs4 scalar: **19.584993**
- Second-call abs4 AVX2: **19.584993** (exact match)
- This confirms AVX2 produces identical output for same input state

---

# 5. Correct Interpretation

**DO:**
- State that 0.5B INT8 AVX2 runtime path is repeat-stable
- State that 7B INT8 AVX2 backend reachability is confirmed
- State that 7B same-input numeric sanity has supporting evidence
- State that inline fallback was avoided for tested INT8 ggml-native paths
- State this is a backend baseline checkpoint

**DO NOT:**
- Claim end-to-end speedup
- Claim production readiness
- Claim multi-layer support
- Claim all-layer support
- Claim INT6 support
- Claim 14B support
- Claim broad semantic equivalence
- Claim exact/token match
- Claim universal scalar/AVX2 equivalence across mismatched invocations

---

# 6. Allowed Claims

✅ INT8 decoded-f32 path reaches ggml-native PRT op
✅ 0.5B INT8 ggml-native AVX2 backend is repeat-stable
✅ 7B INT8 ggml-native AVX2 backend is reachable
✅ 7B same-call numeric sanity has supporting evidence
✅ Inline fallback was avoided for tested INT8 ggml-native paths
✅ This is a backend baseline checkpoint

---

# 7. Forbidden Claims

❌ End-to-end speedup
❌ Production readiness
❌ Multi-layer support
❌ All-layer support
❌ 7B INT6 support
❌ 14B support
❌ Broad semantic equivalence
❌ Exact/token match
❌ Universal scalar/AVX2 equivalence across mismatched invocations

---

# 8. Recommended Next

**Phase 22I** — Add microsecond timing instrumentation for AVX2/scalar kernels and repeat 0.5B + 7B timing.

Rationale: AVX2 currently logs 0ms due to ms-resolution limits. Clean microsecond timing is needed before any speed claims can be made.

---

# 9. Safety Scan

| Check | Status |
|-------|--------|
| No model files staged | ✅ |
| No sidecars staged | ✅ |
| No f32 refs staged | ✅ |
| No captures staged | ✅ |
| No prompt files staged | ✅ |
| No binaries staged | ✅ |
| No huge logs staged | ✅ |
| No secrets | ✅ |
| Scratch drive used | ✅ |
| System disk healthy | ✅ (58G free) |
| Scratch disk healthy | ✅ (51G free) |

---

**Checkpoint Verdict:** `PASS_INT8_GGML_NATIVE_AVX2_BACKEND_BASELINE_CHECKPOINT`
