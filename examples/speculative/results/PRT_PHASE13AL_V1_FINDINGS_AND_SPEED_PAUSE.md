# PRT Phase 13AL — v1 Findings and Speed-Path Pause

**Date:** 2026-05-07  
**Verdict:** PRT_V1_QUALITY_VALIDATED_SPEED_NEGATIVE  
**Branch:** `experimental/prt-phase13-model-generalization`  
**Git:** `771a32035`

---

## Verdict: PRT_V1_QUALITY_VALIDATED_SPEED_NEGATIVE

**PRT v1 successfully demonstrates clean active FFN_UP replacement on Qwen2.5-0.5B and Qwen2.5-3B with preserved quality on tested prompt suites. PRT v1 does not currently beat native llama.cpp speed. The sparse ternary float32 sidecar custom-op path is speed-negative versus native llama.cpp. The old Phase 12 speedup (~1.8×) was NOT reproduced under the clean Phase 13 llama-cli stack. Future speed work should move to a new architecture path (packed/lower-precision sidecars, native ggml backend integration, or speculative/batch use case) rather than continuing to tune the current float32 sidecar custom-op path.**

---

## Executive Summary

### What PRT v1 Demonstrates
- **Clean active FFN_UP replacement** on Qwen2.5-0.5B and Qwen2.5-3B
- **Quality preserved** on tested prompt suites — 8/8 semantic matches on 3B, 100% match on 0.5B
- **Replacement is real** — 272 PRT calls confirmed active, AVX2 path verified, 0 fallback calls
- **mmap sidecar loading** removes the sidecar load overhead (4542ms → 0.14ms), reducing wall time by 46%

### What PRT v1 Does Not Do
- **Does not beat native speed** — PRT is ~2.5× slower wall and ~0.41× generation throughput on 3B
- **Phase 12 speedup (~1.8×) was not reproduced** — the old result was specific to its binary/measurement setup
- **Current float32 sidecar custom-op path is speed-negative** — the memory bandwidth and compute overhead of sparse ternary float32 matmul exceeds the savings from bypassing the native FFN_UP

### What This Means
The PRT v1 float32 sidecar custom-op approach is **exhausted as a speed solution**. The quality prototype is complete. Future speed work requires a different architecture — either packed/quantized sidecar representation, native ggml backend integration, or a different use case (speculative/batch verification).

---

## Proven Quality/Correctness Milestones

### 0.5B — Clean Quality Validation
| Field | Value |
|-------|-------|
| Phase | 13S |
| Tag | `PRT_PHASE13S_05B_CLEAN_QUALITY_CHECKPOINT` |
| Commit | `43390ca72` |
| Result | CLEAN_QUALITY_VALIDATED |
| Semantic match rate | 100% |
| Replacement evidence | Confirmed |
| Quality degradations | 0 |
| AVX2 path | Active |

### 0.5B — Post-Fix Validation
| Field | Value |
|-------|-------|
| Phase | 13AB |
| Tag | `PRT_PHASE13AB_05B_POST_FIX_CHECKPOINT` |
| Commit | `d44c57cfd` |
| Result | POST_FIX_QUALITY_VALIDATED |
| Details | 8/8 semantic matches, 0 degradations, self-contained evidence |
| After | AVX2 indexing bug fix |

### 3B — Full Quality Validation
| Field | Value |
|-------|-------|
| Phase | 13AE |
| Tag | `PRT_PHASE13AE_3B_QUALITY_CHECKPOINT` |
| Commit | `19491b7bc` |
| Result | FULL_QUALITY_VALIDATED |
| Prompt suite | 8 prompts, varied types |
| Semantic matches | **8/8** |
| Quality degradations | **0** |
| Repetition/collapse | **0** |
| Sidecars loaded | **36/36** |
| AVX2 replacement | **Active** |
| Fallback calls | **0** |
| Generation clean | ✅ |
| llama_batch active | ✅ |
| Replacement confirmed | ✅ |

---

## Proven Engineering Milestones

| Milestone | Phase | Description | Verified |
|-----------|-------|-------------|---------|
| Dynamic shape support | 13B / 13C | Arbitrary Qwen2.5 configurations (0.5B/1.5B/3B/7B) | ✅ |
| Sidecar generation | 13A / 13AE | Clean pipeline for both 0.5B and 3B | ✅ |
| llama-cli integration | 13 / ongoing | `--prt-mode`, `--prt-force-native`, `--prt-sidecar-dir`, `--prt-sidecar-mmap`, `--prt-log-file`, `--prt-log-level` | ✅ |
| PTY argv runner | 13O | Safe CLI invocation with clean JSON output routing | ✅ |
| AVX2 activation | 13X | `LLAMA_PRT_AVX2` CMake option, `__AVX2__` verified, ymm registers in disasm | ✅ |
| AVX2 indexing fix | 13Y | k loop indexed [k][j] → [j][k] fix, post-fix replacement confirmed | ✅ |
| mmap sidecar loading | 13AG | 4542ms → 0.14ms, 46% wall improvement | ✅ |
| Replacement evidence | 13V / ongoing | 272 calls (34 layers × 8 batches), 0 callback_overwrites, 0 fallback | ✅ |
| Phase 12 audit | 13AJ / 13AK | Speedup NOT reproduced, KV cache ruled out, binary difference identified | ✅ |

---

## Speed Findings

### 0.5B — Phase 13AA
| Metric | Native | PRT | Ratio |
|--------|--------|-----|-------|
| Wall time | 0.625s | 1.001s | **1.60× slower** |
| Gen tok/s | 94.86 | 49.54 | **1.92× slower** |

### 3B — Phase 13AF / 13AK
| Metric | Native | PRT | Ratio |
|--------|--------|-----|-------|
| Wall time (clean) | 2.131s | 5.284s | **2.48× slower** |
| Wall time (+ KV cache) | 2.129s | 5.346s | **2.51× slower** |
| Gen tok/s (clean) | 20.83 | 8.60 | **0.41×** |
| Gen tok/s (+ KV cache) | 20.50 | 8.44 | **0.41×** |

**KV cache effect:** Adding `--cache-type-k q8_0 --cache-type-v f16` changes timing by <2%. KV cache is NOT the cause of the Phase 12 speedup. Speedup was NOT reproduced with or without KV cache.

### mmap Effect — Phase 13AG (3B)
| Field | Before mmap | After mmap | Change |
|-------|------------|------------|--------|
| Sidecar load | 4542.6ms | 0.14ms | **-99.97%** |
| PRT wall | 9.884s | 5.300s | **-46.4%** |
| Gen tok/s | 8.6 | 8.5 | **unchanged** |

**Finding:** mmap eliminates sidecar load overhead. Wall improved 46%. Gen tok/s unchanged because the kernel (compute) is the bottleneck, not I/O.

### Warm Harness — Phase 13AI
**Finding:** Warm harness provides no meaningful wall improvement over mmap alone. mmap already provides available warm behavior via OS page cache. PRT overhead is compute-bound, not I/O-bound.

### Kernel/Layout — Phase 13AH
**Finding:** AVX2 kernel is optimal for [N][M] row-major layout. Current implementation is BLAS-equivalent. Strided W access is inherent to row-major layout. No quick gain from kernel optimization within the current float32 custom-op path.

---

## Phase 12 Speedup Status

| Field | Value |
|-------|-------|
| Claimed speedup | ~1.82× |
| Source | Phase 12E / `PRT_PHASE12E_L11_L15_CHECKPOINT` / `bca5a1f32` |
| Binary used | `llama-prt-posix` (custom) |
| Reproduction attempted | YES (Phase 13AK) |
| KV cache tested | YES — adding `--cache-type-k q8_0 --cache-type-v f16` has <2% effect |
| Speedup reproduced | **NO** |
| Status | **NOT_REPRODUCED_IN_CURRENT_CLEAN_STACK** |

**Explanation:** Phase 12's 1.82× speedup was specific to the `llama-prt-posix` binary and measurement setup — not a general property of PRT. The most likely cause is the binary difference (custom vs clean llama-cli). The result is superseded by Phase 13AK and should NOT be used as a current claim. If mentioned, describe as historical/contextual only.

---

## Allowed Claims

The following claims are supported by the evidence collected across Phase 13:

1. ✅ **PRT v1 active path preserves clean generation quality** on Qwen2.5-0.5B and Qwen2.5-3B in tested prompt suites
2. ✅ **Qwen2.5-3B active PRT passed 8-prompt clean quality validation** — 8/8 semantic matches, 0 quality degradations, 36/36 sidecars loaded, AVX2 active, fallback calls 0
3. ✅ **mmap sidecar loading reduced 3B PRT wall time substantially** — from 4542ms to 0.14ms sidecar load, 46% wall improvement
4. ✅ **Current PRT v1 float32 sidecar custom-op path is speed-negative** versus native llama.cpp on both 0.5B and 3B
5. ✅ **Phase 12's ~1.8× speedup was NOT reproduced** under current clean llama-cli stack (Phase 13AK)
6. ✅ **AVX2 custom op path verified active** on Phase 13 stack

---

## Forbidden Claims

The following claims are NOT supported by the evidence and must not be made:

- ❌ **NO native speedup claim** for any model
- ❌ **NO production readiness** 
- ❌ **NO universal model support**
- ❌ **NO larger-than-3B success**
- ❌ **NO exact equivalence** beyond tested prompts
- ❌ **NO claim that Phase 12 speedup still applies** to current stack
- ❌ **NO claim that latency overhead is solved**
- ❌ **NO claim that Phase 12 speedup is falsified** — it is simply not reproduced (context-specific)

---

## Recommended Future Paths

### Priority 1: Packed / Lower-Precision Sidecars
**Next phase: Phase 14A**

**Rationale:** float32 sidecars are too memory-heavy (~90MB per layer for 3B = 3.1GB total). The current sparse ternary float32 matmul reads the entire sidecar from memory per call (34 layers × 8 calls = 272 sidecar reads per inference). int8/int4/ternary packed representation would reduce memory bandwidth dramatically, potentially changing the speed equation to positive.

**Key question:** Can a packed sidecar representation (int8, int4, or ternary) achieve quality parity while enabling a kernel that reads compressed data natively?

### Priority 2: Native ggml/Backend Integration
**Next phase: Phase 14B**

**Rationale:** Current custom-op path cannot beat highly optimized native ggml kernels. The custom op callback introduces dispatch overhead and prevents ggml's optimizer from handling fusion/scheduling. Implementing PRT as a real ggml backend op would eliminate this overhead.

**Key question:** Can PRT computation be expressed as a ggml op that ggml's optimizer can fuse into the computation graph?

### Priority 3: Speculative / Batch Verification Use Case
**Next phase: Phase 14C**

**Rationale:** PRT may be more useful where verification batches amortize memory reads. In speculative decoding, a small "draft" model generates candidate tokens that a larger "verifier" model accepts/rejects. PRT could serve as the verifier-side approximation — replacing expensive FFN_UP with a fast sparse approximation that is acceptably accurate for high acceptance rates.

**Key question:** Does PRT's quality profile match the requirements for speculative verification acceptance?

### Priority 4: Larger Model Research Only After Representation Changes
**Rationale:** Do not assume current float32 sidecar path improves with scale. If the 3B result is ~2.5× slower, 7B/14B may be similarly or more degraded. Test larger models only with a new speed-oriented representation (packed sidecars or native ggml op).

### Priority 5: Publish / Freeze Quality Result
**Rationale:** PRT v1 is a validated active replacement/quality prototype, not a production speed path. The primary durable contribution is the quality validation (8/8 matches, 0 degradations). Consider publishing this as the main PRT contribution and treating speed as a future architecture path.

---

## Recommended Next Phase

**Phase 13AM: Final archive and tag for Phase 13 branch, then Phase 14A: Packed/Quantized Sidecar Design**

**Reasoning:** Current float32 sidecar custom-op path is exhausted as a speed solution. The quality validation is complete. The next productive step is a new architecture path that addresses the memory/compute bottleneck.

**What NOT to do:** Do not continue tuning the current float32 AVX2 custom-op path. Do not run more speed benchmarks on current stack without representation changes. Do not attempt larger model runs on the current path.

---

## v1 Summary

| Dimension | Status | Evidence |
|-----------|--------|----------|
| **Quality (0.5B)** | ✅ VALIDATED | 100% semantic match, 0 degradations |
| **Quality (3B)** | ✅ VALIDATED | 8/8 matches, 0 degradations, 36/36 loaded |
| **Replacement** | ✅ CONFIRMED | 272 calls, AVX2 active, 0 fallback |
| **Speed (0.5B)** | ❌ NEGATIVE | 1.60× wall slower, 1.92× gen slower |
| **Speed (3B)** | ❌ NEGATIVE | 2.48× wall slower, 0.41× gen throughput |
| **Phase 12 speedup** | ❌ NOT_REPRODUCED | Superseded by Phase 13AK |
| **v1 status** | **Quality prototype only** | NOT a production speed solution |

---

## Safety

| Check | Result |
|-------|--------|
| Models/sidecars/binaries staged | NO ✅ |
| Secrets detected | NO ✅ |
| Tags modified | NO ✅ |
| Docs only committed | YES ✅ |

**Tag:** `PRT_PHASE13AL_V1_FINDINGS_FREEZE`