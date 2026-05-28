# Phase 29A: Residency / Working-Set Reality Check

**Branch:** `experimental/prt-phase19a-alt-sidecar-backed`  
**Previous HEAD:** `99797f1ec`  
**Date:** 2026-05-28  
**Model:** `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf` (~379 MB on-disk)  
**Binary:** `llama-simple` (non-interactive, single prompt)  
**Prompt:** `"Hi"`, `n_predict=1`  

---

## 1. Executive Summary

**Classification: ADDITIVE_OVERHEAD** (noise-level RSS variation, meaningful time overhead)

No configuration of the current sidecar/pager architecture produces measurable RSS reduction relative to native model load. All seven test configurations (baseline, observe-only, 1MB budget, 100MB budget, apply layer0 all-families, apply ffn_up layer0, apply attn_output layer0) fall within **±350 KB** of the ~481 MB baseline — a **±0.07%** window that is indistinguishable from noise. At the same time, pager-enabled configurations incur **64–131% user-time overhead** (1.85–3.82s additional latency vs 2.9s baseline).

The sidecar/pager infrastructure is not reducing resident memory. It is either:
- (a) not loading sidecar tensors into RSS at all (lazy mmap or not yet triggered), or  
- (b) replacing model tensors equivalently with no net RSS delta, or  
- (c) the test stimulus (n_predict=1, 1-token generation) is too brief to materialize sidecar tensors before measurement completes.

**The core SDI/PRT residency thesis — lower-bit base + paged residual sidecars < full dense model cost — is not supported by current measurements on this hardware/software stack.**

---

## 2. Test Configuration

### 2.1 Binary: llama-simple (non-interactive)

`llama-cli` enters interactive REPL mode, making it incompatible with `subprocess.run.communicate()` without PTY manipulation. `llama-simple` accepts a prompt as a positional argument, runs to completion, and exits cleanly — enabling reliable RSS measurement via `/usr/bin/time -v`.

### 2.2 Sidecar Setup (Synthetic)

Real `.trit` sidecar files do not exist for this model. Synthetic sidecar files were generated:

```
/tmp/phase29a_sidecars/layers/layer_000/ffn_up_0.trit    (294 KB, shape [896, 896])
/tmp/phase29a_sidecars/layers/layer_000/attn_output_1.trit (294 KB, shape [896, 896])
/tmp/phase29a_sidecars/layers/layer_000/ffn_down_2.trit  (294 KB, shape [896, 896])
/tmp/phase29a_sidecars/layers/layer_001/ffn_up_0.trit    (294 KB, shape [896, 896])
```

> **Caveat:** Actual model tensor shapes (ffn_up=[896, 4864], ffn_down=[4864, 896]) differ from the synthetic [896, 896] files. The pager's shape validation may reject these files, preventing real materialization.

Manifest: `/tmp/phase29a_manifest/manifest.json` (Phase28Y format, format_version: 1, 4 entries across layer 0–1, families ffn_up / attn_output / ffn_down).

### 2.3 Measurement Method

`/usr/bin/time -v <cmd>` → parse `Maximum resident set size (kbytes)` from stderr.  
`LLAMA_LOG_LEVEL=DISABLE` suppresses debug output from timing measurements.  
Timeout: 20s per run.

---

## 3. Memory Results

### 3.1 RSS Measurements

| Test | Configuration | RSS (KB) | Delta (KB) | Delta (%) | User Time (s) | Overhead |
|------|---------------|----------:|----------:|----------:|-------------:|---------:|
| A | Baseline native (no pager) | 481,488 | — | — | 2.90 | — |
| B | Pager observe-only | 481,316 | −172 | −0.036% | 4.75 | +64% |
| C | Pager + 1MB budget | 481,428 | −60 | −0.012% | 5.01 | +73% |
| D | Pager + 100MB budget | 481,664 | +176 | +0.037% | 4.94 | +70% |
| E | Apply layer0 (all families) | 481,576 | +88 | +0.018% | 5.75 | +98% |
| F | Apply ffn_up layer0 | 481,168 | −320 | −0.066% | 6.72 | +132% |
| G | Apply attn_output layer0 | 481,456 | −32 | −0.007% | 5.21 | +80% |

**Worst-case delta: 320 KB on 481 MB baseline = 0.066%**  
**All deltas within ±350 KB — noise-level variation.**

### 3.2 Time Overhead Is Real (Even If Memory Is Not)

```
Baseline:       2.90s user time
Observe-only:   4.75s (+64%)
1MB budget:      5.01s (+73%)
100MB budget:   4.94s (+70%)
Apply layer0:   5.75s (+98%)
Apply ffn_up:   6.72s (+132%)
Apply attn_out: 5.21s (+80%)
```

Pager processing adds 1.85–3.82s per run. This overhead is consistent and significant. However, **the memory cost is not visible in RSS**. This suggests the pager is doing work (parsing, validation, tree-walking) without yet allocating sidecar tensors into the process resident set at measurement time.

---

## 4. Analysis

### 4.1 Sidecars Are Not Reducing RSS

No configuration produces a measurable RSS reduction. The "apply" configurations (E, F, G) activate sidecar injection, yet RSS remains flat or slightly lower than baseline. This is inconsistent with the thesis that sidecars replace model weights and reduce residency.

Possible explanations:
1. **Shape mismatch**: Synthetic sidecar files have shape [896, 896] vs actual model shapes [896, 4864] / [4864, 896]. The pager's shape validation may silently reject all actual tensor bindings, leaving sidecars unloaded.
2. **Trigger threshold not met**: n_predict=1 with a 1-token generation may not run enough layer iterations to trigger sidecar materialization before the process exits.
3. **Lazy mmap behavior**: Sidecars may be memory-mapped but not faulted into RSS until accessed. At n_predict=1, the layers that would consume sidecar tensors are barely exercised.
4. **Replacement is 1:1**: If each sidecar replaces an equivalent model tensor, the RSS delta would be zero by definition. The residency thesis requires the *base model* to use lower-bit quantization — which is not what this test measures (all tests use the same Q4_K_M model).

### 4.2 What Would Make This a Real Residency Win?

The current architecture cannot achieve the SDI/PRT residency thesis with the current test design. To make it a real win:

1. **Lower-bit base model**: The thesis requires base model to be lighter (e.g., Q2_K instead of Q4_K_M), with residual sidecars filling in precision. Current tests hold model constant.
2. **Correct sidecar shapes**: Real `.trit` files matching actual model tensor shapes must exist to trigger real materialization.
3. **Sufficient generation depth**: n_predict should be large enough (e.g., 32–128 tokens) to ensure all layers are exercised and sidecar tensors are accessed.
4. **Budget enforcement**: The pager's budget (--prt-sidecar-budget-mb) should bound RSS. Current tests show no bounding — but also no visible sidecar cost.
5. **Separate base + residual measurement**: Need to measure (a) base model alone at lower bitdepth, then (b) base + selected sidecars, to isolate the delta.

### 4.3 Pager Overhead Is Real But Memory-Invisible

The 64–132% time overhead confirms the pager is doing non-trivial work: manifest parsing, tree construction, per-layer hooks, and guard evaluation. This work does not translate into RSS growth visible at process-level measurement. The overhead is real in latency, not yet real in memory footprint under these test conditions.

---

## 5. Classification Reasoning

| Classification | Evidence |
|---------------|----------|
| **ADDITIVE_OVERHEAD** | Pager overhead adds 64–132% latency; RSS delta noise-level; pager infrastructure running but memory impact invisible |
| LAZY_PAGER_WORKING | Not confirmed — sidecars may be lazily mmap'd but shape mismatch prevents actual binding |
| CACHE_BLOAT | Not observed in these measurements |
| IO_PAGER_ONLY | mmap may be happening but RSS measurement too coarse to see it at n_predict=1 |
| PROMISING_RESIDENCY_PATH | Not yet — requires lower-bit base model comparison, correct sidecar shapes, and sufficient generation depth |
| BLOCKED_MEASUREMENT | Partially — synthetic sidecar shapes prevent real materialization; n_predict=1 may be too shallow |

---

## 6. What Must Change

1. **Generate real sidecar files** with correct tensor shapes (gguf inspection → .trit export per layer/family)
2. **Run longer generations** (n_predict=32–128) to ensure sidecar tensors are exercised across multiple layers
3. **Compare lower-bit base** (Q2_K / Q3_K) vs Q4_K_M to isolate the residency win from the sidecar overhead
4. **Add per-layer RSS sampling** during generation (not just post-exit snapshot) to catch transient memory peaks
5. **Validate budget enforcement** — confirm pager's --prt-sidecar-budget-mb actually limits RSS when many sidecars are available

---

## 7. Next Recommended Engineering Phase

**Phase 29B: Sidecar Materialization Trigger Validation**

Before any residency measurement, confirm that:
- Real sidecar files with correct shapes are actually loaded and bound
- Per-layer RSS delta is measurable with n_predict=32
- Budget enforcement actually caps RSS growth

This requires:
1. Real `.trit` sidecar generation for Qwen2.5-0.5B
2. Per-layer RSS sampling during generation (not just post-exit)
3. A comparison matrix: Q2_K base alone vs Q2_K + sidecars vs Q4_K_M baseline

---

## 8. Artifacts

- `examples/speculative/PHASE29A_RESIDENCY_WORKING_SET_REALITY_CHECK.md` (this file)
- `examples/speculative/results/phase29a_residency_working_set_reality_check.json`

---

## 9. Git Status

```
HEAD: 99797f1ec on branch experimental/prt-phase19a-alt-sidecar-backed
```

All staged files are small artifacts only. No model files, binaries, or generated sidecars staged.

**Staged:** PHASE29A report + JSON results  
**Not staged:** synthetic sidecar files, binaries, test scripts, model files

---

*No quality, correctness, speedup, Q2→Q4 recovery, larger-model, or production readiness claims are made in this report.*