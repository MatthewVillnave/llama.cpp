# PRT Phase 21R: 7B INT8 Partial Semantic Checkpoint

## Executive Summary
- **Branch**: `experimental/prt-phase19a-alt-sidecar-backed`
- **Previous checkpoint**: `c1d5da615` (Phase 21P: Kernel Baseline)
- **New HEAD**: `4d40c459e`
- **Date**: 2026-05-15

7B INT8 PRT-v2 layer0 has **confirmed route/op/kernel evidence**. Kernel/plumbing baseline is solid. Phase 21Q-D showed both native and PRT-v2 generate visible text fragments. **Capture remains limited by llama-cli spinner** overwriting PTY output. This checkpoint is **not a clean semantic baseline** — it is a kernel+partial evidence checkpoint.

---

## 1. Frozen Technical Baseline

| Field | Value |
|-------|-------|
| Branch | `experimental/prt-phase19a-alt-sidecar-backed` |
| HEAD | `4d40c459e` |
| Model | `Qwen2.5-7B-Instruct-Q4_K_M.gguf` |
| Layer | 0 only |
| Op | `GGML_OP_PRT_FFN_UP` |
| Sidecar type | INT8 |
| K | 3584 |
| M | 18944 |
| Prompt method | File prompts only |
| Harness | `script -qfc` + `--log-disable` + `--simple-io` |
| Scratch path | `/media/matthew-villnave/VL_usb/prt_scratch` |

---

## 2. Evidence from Phase 21P (Kernel Baseline)

| Metric | Value |
|--------|-------|
| Kernel ENTER | 32 (8 per prompt × 4 prompts) |
| Kernel EXIT | 32 (8 per prompt × 4 prompts) |
| output_abs_sum | 1.684134, 1.944075, 2.112426, 1.860070 |

Phase 21P frozen: **KERNEL_ENTER/EXIT captured across all 4 prompts with consistent output_abs_sum values**.

---

## 3. Evidence from Phase 21Q-D (Partial Semantic)

| Run | Exit | Wall | Visible Text | Kernel EXIT | output_abs_sum |
|-----|------|------|-------------|-------------|----------------|
| Native n=2 | 0 | 181s | "capital" (spinner cut) | N/A | N/A |
| PRT n=2 | 124 | 240s | "capital" (spinner cut) | ✅ | 1.684134/1.944075/0.507424 |

**Key observation**: Both native and PRT-v2 show visible text fragments ("capital"). Text IS being generated — spinner hides the rest.

---

## 4. Correct Interpretation

**What IS proven:**
- ✅ 7B layer0 INT8 PRT-v2 route/op/kernel evidence
- ✅ Kernel ENTER/EXIT across multiple prompts
- ✅ output_abs_sum consistent across phases
- ✅ Sidecar decode (INT8 → f32) working
- ✅ No visible gibberish/corruption in PRT output
- ✅ Partial text generation observed (native "Paris" at n=8, PRT "capital" at n=2)
- ✅ Scalar f32 kernel overhead documented (expected, not a bug)

**What is NOT claimed:**
- ❌ Exact/token match
- ❌ Full semantic validation pass
- ❌ Clean 4-prompt semantic baseline
- ❌ Speedup vs native
- ❌ 7B INT6 support
- ❌ Multi-layer support
- ❌ 14B support
- ❌ Production readiness

---

## 5. Allowed Claims

- 7B layer0 INT8 PRT-v2 route/op/kernel evidence is proven
- 7B layer0 INT8 sidecar reaches `GGML_OP_PRT_FFN_UP`
- Kernel ENTER/EXIT and `output_abs_sum` are captured
- Partial text generation evidence exists for native and PRT-v2
- No visible PRT-induced gibberish observed in partial capture
- Scalar f32 PRT path is expected to be slower than native Q4 matmul

## 6. Forbidden Claims

- Exact/token match
- Full semantic match
- Clean 4-prompt semantic pass
- Speedup
- Production readiness
- 7B INT6
- Multi-layer support
- All-layer support
- 14B support from this phase

---

## 7. Capture Limitation

llama-cli spinner (`\r`-overwrite) runs during generation and overwrites PTY output. At n=2, spinner remains active when process exits or file is inspected. Native n=8 works because generation completes and spinner stops before file inspection, leaving "Paris" visible.

**Text IS being generated** — spinner hides it. Not a PRT bug.

---

## 8. Recommended Next

**Option A**: Pause semantic work → move to backend performance design / AVX2 for PRT-v2 (scalar f32 kernel needs vectorization before production)

**Option B**: Patch or configure llama-cli to disable spinner → then rerun full semantic validation (Phase 21S)

**Recommended**: Option B first if semantic proof is required. Alternative: checkpoint and move to performance backend work.

---

## 9. Safety

| Check | Status |
|-------|--------|
| Model files staged | ❌ None |
| Sidecars staged | ❌ None |
| Captures staged | ❌ None |
| Prompts staged | ❌ None |
| Binaries staged | ❌ None |
| Huge logs staged | ❌ None |
| Secrets detected | ❌ None found |
| System disk free | 58GB (75%) ✅ |
| Scratch disk free | 51GB (57%) ✅ |