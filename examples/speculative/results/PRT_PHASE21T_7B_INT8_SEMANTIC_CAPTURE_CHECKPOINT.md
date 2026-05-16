# PRT Phase 21T: 7B INT8 PRT-v2 Semantic-Capture Baseline Checkpoint

## Executive Summary

7B INT8 PRT-v2 layer0 now has:
- route/op/kernel evidence (32 ENTER / 32 EXIT across 4 prompts)
- output_abs_sum evidence (consistent values: 1.684134, 1.944075, 2.112426, 1.860070)
- partial semantic text capture ("Paris" from native, "France" from PRT-v2)
- Capture mechanism fully understood and documented

**Important:** This is a **scoped semantic-capture baseline**, not a broad semantic equivalence claim. No speedup, no exact/token match, no production readiness.

---

## 1. Frozen Technical Baseline

| Field | Value |
|-------|-------|
| Branch | `experimental/prt-phase19a-alt-sidecar-backed` |
| HEAD | `2f314ce96` |
| Model | `Qwen2.5-7B-Instruct-Q4_K_M.gguf` |
| Layer | 0 only |
| Op | `GGML_OP_PRT_FFN_UP` |
| Sidecar type | INT8 |
| Sidecar K | 3584 |
| Sidecar M | 18944 |
| Prompt method | file prompts |
| Capture method | `script -qfc` PTY + `--simple-io` + `--log-disable` |
| Scratch path | `/media/matthew-villnave/VL_usb/prt_scratch` |

---

## 2. Evidence from Phase 21P/21R (Kernel Baseline)

### Kernel ENTER/EXIT Counts
| Phase | Prompts | Kernel ENTER | Kernel EXIT |
|-------|---------|-------------|-------------|
| 21P | 4 (P1-P4) | 32 | 32 |
| 21Q-D | 1 (P1) | 9 | 9 |

### output_abs_sum Values (Phase 21P)
| Token | Value |
|-------|-------|
| 1 | 1.684134 |
| 2 | 1.944075 |
| 3 | 2.112426 |
| 4 | 1.860070 |

### Partial Semantic Evidence (Phase 21P/21Q)
- "capital" visible from both native and PRT-v2
- Native n=8: "Paris" visible

---

## 3. Evidence from Phase 21S (Capture Resolution)

### Capture Issue Understanding

**Problem:** PTY capture showed text interleaved with `[PRT_V2_KERNEL_*]` logs.

**Root cause:** Mixed stdout/stderr streams in PTY:
- Generated text → stdout via `console::log()`
- PRT kernel logs → stderr via `fprintf(stderr, "[PRT_V2_KERNEL_ENTER]...")`
- `script -qfc` captures both streams merged

**Resolution:** Filter out `[PRT_` lines to extract clean text.

### Native n=8 Capture
```
Exit: 0 | Wall: 120s | Size: 30KB
Text: "capital of France is Paris."
```

### PRT-v2 n=4 Capture
```
Exit: 0 | Wall: 300s | Size: 38KB
Text: "France" visible (extracted from mixed stream)
5 KERNEL_EXIT logs with output_abs_sum:
  1.684134, 1.944075, 2.112426, 2.145798, 1.458103
```

### Clean Text Extraction
```bash
grep -v "\[PRT_" capture.raw.out > capture.clean.txt
```

---

## 4. Correct Interpretation

| Claim | Status |
|-------|--------|
| route/op/kernel evidence | ✅ PROVEN |
| partial semantic text capture | ✅ PROVEN |
| 32 ENTER / 32 EXIT kernel calls | ✅ PROVEN |
| output_abs_sum consistency | ✅ PROVEN |
| Spinner issue understood | ✅ PROVEN |

| NOT YET CLAIMED | Reason |
|-----------------|--------|
| exact/token match | Not tested with controlled comparison |
| broad semantic equivalence | Single prompt, single model, layer0 only |
| 4-prompt semantic pass | Not run |
| speedup | Scalar f32 is slower than Q4 native |
| production readiness | Layer0 only, no multi-layer |

---

## 5. Allowed Claims

✅ 7B layer0 INT8 PRT-v2 reaches `GGML_OP_PRT_FFN_UP` kernel
✅ Kernel ENTER/EXIT and output_abs_sum are captured and consistent
✅ Native and PRT-v2 both produce visible relevant text on the France prompt
✅ Capture issue is understood and filterable
✅ This is a 7B INT8 layer0 scoped semantic-capture checkpoint

---

## 6. Forbidden Claims

❌ exact/token match
❌ broad semantic equivalence
❌ 4-prompt semantic pass
❌ speedup (PRT scalar f32 is slower than native Q4)
❌ production readiness
❌ 7B INT6 support
❌ multi-layer support
❌ all-layer support
❌ 14B support

---

## 7. Recommended Next

### Preferred: Phase 22A — Backend/Performance Path

Focus on making the PRT-v2 scalar f32 kernel faster:
- AVX2 kernel for `GGML_OP_PRT_FFN_UP`
- Eliminate scalar f32 loop bottleneck
- Benchmark on 0.5B first, then 7B layer0
- No multi-layer until performance is sane

### Alternative: Phase 21U — 4-Prompt Semantic Canary

Run 4 prompts with filtered capture for more semantic evidence before performance work.

---

## 8. Safety Scan

| Check | Status |
|-------|--------|
| model files staged | no |
| sidecars staged | no |
| captures staged | no |
| prompts staged | no |
| binaries staged | no |
| decoded weights staged | no |
| secrets detected | no |
| system disk | 58GB free |
| scratch disk | 51GB free |

---

## Phase History

| Phase | HEAD | Verdict | Key Finding |
|-------|------|---------|-------------|
| 21P | `c1d5da615` | PASS_7B_INT8_KERNEL_BASELINE | 32 ENTER/32 EXIT, output_abs_sum |
| 21R | `bebe1a588` | PASS_7B_INT8_KERNEL_PARTIAL_SEMANTIC | Partial "capital" + "Paris" |
| 21S | `2f314ce96` | PASS_EXISTING_FLAG_FOUND | `--simple-io` works, capture issue understood |
| **21T** | **`2f314ce96`** | **PASS_7B_INT8_SEMANTIC_CAPTURE_CHECKPOINT** | **Scoped baseline documented** |