# PRT Phase 13AM — Final Archive

**Date:** 2026-05-07  
**Verdict:** PRT_V1_QUALITY_VALIDATED_SPEED_NEGATIVE  
**Branch:** `experimental/prt-phase13-model-generalization`  
**Final commit:** `0de6bddf0`  
**Tag:** `PRT_PHASE13_FINAL_ARCHIVE_V1`

---

## Final Verdict: PRT_V1_QUALITY_VALIDATED_SPEED_NEGATIVE

**PRT v1 successfully demonstrates clean active FFN_UP replacement on CPU for Qwen2.5-0.5B and Qwen2.5-3B.** The system supports dynamic model shapes, real sidecar loading, AVX2 custom-op execution, clean output routing, and zero-fallback active replacement. Quality is validated on tested 0.5B and 3B suites.

**However, the current float32 sidecar custom-op implementation is speed-negative versus native llama.cpp.** PRT v1 should be preserved as a quality/correctness prototype, not a production speed path.

**Future speed work requires a new architecture** — packed/quantized sidecars, native ggml/backend integration, or speculative/batch verification use case.

---

## Summary

PRT v1 demonstrates that a sparse ternary float32 sidecar can replace the FFN_UP computation of a transformer layer with quality preserved. The AVX2 kernel is correct and efficient for the [N][M] row-major layout. Sidecar generation, loading, mmap optimization, and clean llama-cli integration all work.

The core problem is that the float32 sidecar approach is memory-bandwidth bound: each PRT call must read the entire ~90MB sidecar for a single layer, and 272 such reads occur per inference. The native FFN_UP path is highly optimized and does not have this overhead. PRT's custom-op dispatch and per-call sidecar read overhead exceeds the savings from the sparse ternary approximation.

**Bottom line: The quality prototype is complete and valid. The speed path requires a fundamentally different approach.**

---

## Phase 13 Milestone Chain

| Phase | Result | Commit | Tag |
|-------|--------|--------|-----|
| 13S | 0.5B clean quality checkpoint | `f64ba88ab` | `PRT_PHASE13S_05B_CLEAN_QUALITY_CHECKPOINT` |
| 13T | Isolate 0.5B timing components | `fcdccb289` | — |
| 13U | Ablate logging overhead | `b99539503` | — |
| 13V | Inspect runtime kernel bottleneck | `2fccd83ba` | — |
| 13W | Probe cold/warm custom-op penalty | `d94310f53` | — |
| 13X | Enable and verify AVX2 custom op path | `3d6111eab` | — |
| 13Y | Verify AVX2 indexing fix at runtime | `73b545091` | — |
| 13Z | Verify post-fix 0.5B clean quality | `0889ca068` | — |
| 13Z-R | Rerun post-fix 0.5B quality | `a090c5d04` | — |
| 13AA | Rebaseline post-fix 0.5B timing | `9a72dd479` | — |
| 13AB | Freeze post-fix 0.5B checkpoint | `d44c57cfd` | `PRT_PHASE13AB_05B_POST_FIX_CHECKPOINT` |
| 13AC | Generate and validate 3B sidecars | `2ca3798b9` | — |
| 13AC | Fix 3B shape summary logging | `191a492da` | — |
| 13AD | Run 3B runtime canary | `dd06ef77b` | — |
| 13AE | Run full 3B quality validation | `20d7911ed` | — |
| 13AE | Freeze 3B quality checkpoint | `19491b7bc` | `PRT_PHASE13AE_3B_QUALITY_CHECKPOINT` |
| 13AF | Profile 3B latency overhead | `e206f81fb` | — |
| 13AG | mmap sidecar load optimization | `3f29401a0` | — |
| 13AH | Profile and optimize 3B kernel/layout | `f4162016f` | — |
| 13AI | Measure in-process warm harness | `1b4bfdea8` | — |
| 13AJ | Speed discrepancy audit across phases | `da208ec61` | — |
| 13AK | Phase 12 speed reproduction | `771a32035` | — |
| 13AL | Freeze v1 findings and pause speed path | `0de6bddf0` | — |
| **13AM** | **Final archive** | `0de6bddf0` | **`PRT_PHASE13_FINAL_ARCHIVE_V1`** |

---

## Proven Engineering Capabilities

| Capability | Verified | Notes |
|-----------|---------|-------|
| Dynamic shape support | ✅ | Arbitrary Qwen2.5 configurations (0.5B/1.5B/3B/7B) |
| Qwen2.5-0.5B sidecars | ✅ | Full 0.5B sidecar generation and validation |
| Qwen2.5-3B sidecars | ✅ | Full 3B sidecar generation and validation, 36 layers |
| Sidecar validation | ✅ | Quality validation pipeline for both model sizes |
| Clean llama-cli integration | ✅ | `--prt-mode`, `--prt-force-native`, `--prt-sidecar-dir`, `--prt-sidecar-mmap`, `--prt-log-file`, `--prt-log-level` |
| PTY argv-safe runner | ✅ | `phase13o_pty_argv_runner.py` — safe CLI invocation with clean JSON output |
| `--prt-log-file` output separation | ✅ | PRT debug routed to file, stdout clean |
| `--prt-log-level` | ✅ | `quiet` / `summary` / `verbose` |
| mmap sidecar loading | ✅ | `--prt-sidecar-mmap` eliminates sidecar load overhead |
| AVX2 compile activation | ✅ | `LLAMA_PRT_AVX2` CMake option, `__AVX2__` defined, ymm registers confirmed |
| AVX2 indexing bug fixed | ✅ | k loop `[k][j]`→`[j][k]` transposition fix in AVX2 kernel |
| Replacement evidence audit | ✅ | 272 calls (34 layers × 8 batches), 0 callback_overwrites, 0 fallback |
| Zero-fallback active execution | ✅ | No identity/native fallback on validated layers |
| mmap sidecar load optimization | ✅ | fread 4542ms → mmap 0.14ms, 99.97% reduction |

---

## Proven Quality Results

### Qwen2.5-0.5B

| Field | Value |
|-------|-------|
| Phase | 13AB |
| Tag | `PRT_PHASE13AB_05B_POST_FIX_CHECKPOINT` |
| Validation type | Post-fix quality and timing |
| Semantic matches | 8/8 |
| Exact matches | 8/8 |
| Quality degradations | **0** |
| Repetition/collapse | **0** |
| AVX2 active | ✅ |
| Fallback calls | **0** |
| Generation clean | ✅ |
| JSON/code valid | ✅ |

### Qwen2.5-3B

| Field | Value |
|-------|-------|
| Phase | 13AE |
| Tag | `PRT_PHASE13AE_3B_QUALITY_CHECKPOINT` |
| Validation type | Full 8-prompt clean quality validation |
| Semantic matches | **8/8** |
| Exact matches | 3/8 |
| Quality degradations | **0** |
| Repetition/collapse | **0** |
| JSON/code valid | ✅ |
| Sidecars loaded | **36/36** |
| AVX2 active | ✅ |
| Fallback calls | **0** |
| Generation clean | ✅ |
| Replacement confirmed | ✅ |

---

## Speed Findings

### Qwen2.5-0.5B

| Metric | Native | PRT | Ratio |
|--------|--------|-----|-------|
| Wall time | 0.625s | 1.001s | **1.60× slower** |
| Gen tok/s | 94.86 | 49.54 | **1.92× slower** |

**Status:** PRT remains speed-negative on 0.5B.

### Qwen2.5-3B

| Metric | Native | PRT | Ratio |
|--------|--------|-----|-------|
| Wall time (clean) | 2.131s | 5.284s | **2.48× slower** |
| Gen tok/s (clean) | 20.83 | 8.60 | **0.41×** |

**Status:** PRT ~2.5× wall slower, ~0.41× gen throughput on 3B.

### mmap Effect (3B)

| Field | Before mmap | After mmap | Change |
|-------|------------|------------|--------|
| Sidecar load | 4542.6ms | 0.14ms | **-99.97%** |
| PRT wall | 9.884s | 5.300s | **-46.4%** |
| Gen tok/s | 8.6 | 8.5 | unchanged |

**Bottleneck:** Per-token custom-op/kernel sidecar compute remains. mmap removed I/O overhead but compute overhead is unchanged.

---

## Phase 12 Speedup Status

| Field | Value |
|-------|-------|
| Claimed speedup | ~1.82× |
| Source | Phase 12E / `PRT_PHASE12E_L11_L15_CHECKPOINT` |
| Binary used | `llama-prt-posix` (custom) |
| Reproduction attempted | YES (Phase 13AK) |
| KV cache tested | YES — <2% effect, ruled out |
| Speedup reproduced | **NO** |
| Status | **NOT_REPRODUCED — retired as current claim** |

**Explanation:** Phase 12's 1.82× speedup was specific to its binary/measurement setup. It is not a general property of PRT and should be described as historical/contextual only if referenced.

---

## Allowed Claims

1. ✅ PRT v1 active path preserves clean generation quality on tested Qwen2.5-0.5B and Qwen2.5-3B prompt suites
2. ✅ Qwen2.5-3B active PRT passed 8-prompt clean quality validation: 8/8 semantic matches, 0 quality degradations, 36/36 sidecars loaded, AVX2 active, 0 fallback
3. ✅ mmap sidecar loading reduced 3B PRT wall time by removing sidecar load overhead (4542ms → 0.14ms)
4. ✅ Current PRT v1 float32 sidecar custom-op path is speed-negative versus native llama.cpp

---

## Forbidden Claims

- ❌ NO native speedup claim for any model
- ❌ NO production readiness
- ❌ NO universal model support
- ❌ NO larger-than-3B success
- ❌ NO exact equivalence beyond tested prompts
- ❌ NO Phase 12 speedup as current claim
- ❌ NO claim that latency overhead is solved

---

## Recommended Next Architecture Paths

### Phase 14A — Packed / Quantized Sidecar Design
**Goal:** Reduce sidecar memory traffic via int8/int4/ternary packed representation.

**Key questions:**
- Can int8/int4 sidecars achieve quality parity?
- Can a packed kernel read compressed data natively?
- How does int8 sidecar memory traffic compare to native Q4_K_M?

**Ideas:**
- int8 sidecars with per-plane scale factors
- int4 sidecars with block quantization
- Packed ternary planes with Huffman encoding
- Residual planes with scale groups
- AVX2/VNNI-friendly layout (column-major or block-interleaved)
- Direct comparison against native Q4_K_M memory traffic

### Phase 14B — Native ggml/Backend PRT Integration
**Goal:** Move PRT from custom callback to real ggml/backend op.

**Key questions:**
- Can PRT be expressed as a ggml op?
- Can ggml's optimizer handle fusion/scheduling for PRT?
- Does native ggml path eliminate custom-op dispatch overhead?

### Phase 14C — Speculative / Batch Verification Use Case
**Goal:** Use PRT where batch verification amortizes memory reads.

**Key questions:**
- Does PRT's quality profile match speculative verification acceptance?
- Can batch verification change the economics of PRT's memory reads?

---

## Final Recommendation

**Archive Phase 13 as complete.** Start Phase 14A (packed/quantized sidecar design) on a new branch focused on changing the memory/compute equation that makes current PRT speed-negative.

**Do not continue tuning the current float32 AVX2 custom-op path.** The kernel is optimal for its layout. The bottleneck is the representation itself — float32 is too memory-heavy to beat native. A packed/quantized sidecar representation is the most direct path to a positive speed result.

---

## Safety

| Check | Result |
|-------|--------|
| Models/sidecars/binaries staged | NO ✅ |
| Secrets detected | NO ✅ |
| Existing tags modified | NO ✅ |
| Docs only committed | YES ✅ |

**Tag:** `PRT_PHASE13_FINAL_ARCHIVE_V1`