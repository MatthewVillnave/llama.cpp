# PRT Phase 13AK — Phase 12 Speedup Reproduction

**Date:** 2026-05-07  
**Verdict:** SPEEDUP_NOT_REPRODUCED  
**Model:** Qwen2.5-3B-Instruct-Q4_K_M.gguf  
**Branch:** `experimental/prt-phase13-model-generalization`  
**Git:** `da208ec61`

---

## Verdict: SPEEDUP_NOT_REPRODUCED

**Phase 12's ~1.82× speedup is NOT reproduced on the current clean stack. Adding Phase 12's KV cache settings (`--cache-type-k q8_0 --cache-type-v f16`) does NOT produce speedup — KV cache has negligible effect on timing. PRT is consistently ~2.5× slower in wall time and ~0.41× generation throughput vs native in both clean and cache configurations. The Phase 12 speedup was specific to its binary/measurement setup and is not a general property of PRT.**

---

## Executive Summary

### What was tested
- **Clean stack (no KV cache):** Native vs PRT L11+L15, 3 runs each
- **Phase 12 cache settings:** Same inference with `--cache-type-k q8_0 --cache-type-v f16`, 5 runs each
- Same model (Qwen2.5-3B-Q4_K_M), same policy (L11+L15), same prompt (`"The capital of France is"`), same n=80, same ctx=256, same threads=4

### What was found

| Mode | Avg Wall | Avg Gen t/s | PRT/Native Wall Ratio |
|------|----------|-------------|----------------------|
| Native clean | 2.131s | 20.83 | 1.00× (baseline) |
| PRT clean | 5.284s | 8.60 | **2.48× SLOWER** |
| Native + KV cache | 2.129s | 20.50 | 1.00× (baseline) |
| PRT + KV cache | 5.346s | 8.44 | **2.51× SLOWER** |

**KV cache effect:** Adding `--cache-type-k q8_0 --cache-type-v f16` changed native wall by -0.1% and PRT wall by +1.2%. No meaningful speedup from KV cache.

### What this means

1. **KV cache is NOT the primary confounder** — Adding KV cache does not produce a speedup in the current clean stack. Phase 12's 1.82× speedup was not caused by the KV cache settings.

2. **Phase 12 speedup is NOT reproduced** — The same model, same policy, same prompt, same KV cache settings, but current clean `llama-cli` binary produces PRT as 2.5× slower, not 1.82× faster.

3. **The binary difference is the likely cause** — `llama-prt-posix` (Phase 12) vs `llama-cli` (Phase 13) must be responsible for the discrepancy. The old binary may have had a different timing measurement, different threading model, different graph compilation, or different batch behavior that made PRT appear faster relative to native.

4. **No speedup found in any configuration tested** — Both clean and KV cache configurations show PRT as speed-negative on the current clean stack.

---

## Setup

### Model and Infrastructure
| Field | Value |
|-------|-------|
| Model | `/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-3B-Instruct-Q4_K_M.gguf` |
| Model shape | layers=36, hidden=2048, ffn=11008 |
| Sidecar directory | `/tmp/prt_sidecars_3b` |
| Sidecar count | 36 (all present) |
| Binary | `./build/bin/llama-cli` (clean Phase 13 stack) |
| Build | `b8781-f4162016f` |

### Common Settings
```
Prompt: "The capital of France is"
-n 80 --temp 0 -c 256 -t 4
--no-display-prompt --single-turn
```

### PRT Settings
```
--prt-mode 5700 --prt-force-native 11,15
--prt-sidecar-dir /tmp/prt_sidecars_3b
--prt-sidecar-mmap
--prt-log-level quiet
--prt-log-file /tmp/phase13ak_prt_<mode>_<N>.log
```

### Phase 12 Cache Settings (tested)
```
--cache-type-k q8_0 --cache-type-v f16
```

---

## Results

### Clean Configuration (No KV Cache) — 3 runs each

| Run | Native Wall | Native Gen | PRT Wall | PRT Gen |
|-----|------------|-----------|----------|---------|
| 1 | 2.186s | 20.9 t/s | 5.247s | 8.6 t/s |
| 2 | 2.079s | 20.8 t/s | 5.280s | 8.6 t/s |
| 3 | 2.129s | 20.8 t/s | 5.326s | 8.6 t/s |
| **Avg** | **2.131s** | **20.83** | **5.284s** | **8.60** |
| **Std** | **0.044s** | **—** | **0.033s** | **—** |

**Clean speed ratio:** PRT is **2.48× slower** in wall time, **0.41×** generation throughput vs native.

### Phase 12 Cache Configuration (`--cache-type-k q8_0 --cache-type-v f16`) — 5 runs each

| Run | Native Wall | Native Gen | PRT Wall | PRT Gen |
|-----|------------|-----------|----------|---------|
| 1 | 2.171s | 20.6 t/s | 5.339s | 8.4 t/s |
| 2 | 2.159s | 20.4 t/s | 5.315s | 8.5 t/s |
| 3 | 2.183s | 20.0 t/s | 5.338s | 8.4 t/s |
| 4 | 2.072s | 20.8 t/s | 5.408s | 8.5 t/s |
| 5 | 2.059s | 20.7 t/s | 5.331s | 8.4 t/s |
| **Avg** | **2.129s** | **20.50** | **5.346s** | **8.44** |
| **Std** | **0.052s** | **—** | **0.033s** | **—** |

**Cache speed ratio:** PRT is **2.51× slower** in wall time, **0.41×** generation throughput vs native.

### KV Cache Effect Analysis

| Comparison | Native Wall Change | Native Gen Change | PRT Wall Change | PRT Gen Change |
|-----------|-------------------|------------------|----------------|----------------|
| Cache vs Clean | -0.1% (2.129 vs 2.131) | -1.6% (20.50 vs 20.83) | +1.2% (5.346 vs 5.284) | -1.9% (8.44 vs 8.60) |

**Finding: KV cache has negligible effect.** It neither helps native nor PRT significantly. The Phase 12 speedup was NOT caused by the KV cache settings.

---

## Side-by-Side Four-Mode Comparison

| Mode | Wall (s) | Gen (t/s) | vs Native Wall | vs Native Gen | Notes |
|------|----------|-----------|----------------|---------------|-------|
| **Native clean** | 2.131 | 20.83 | 1.00× | 1.00× | Baseline |
| **PRT clean** | 5.284 | 8.60 | **2.48× SLOWER** | **0.41×** | Current clean stack |
| **Native + KV cache** | 2.129 | 20.50 | 1.00× | 0.99× | Phase 12 cache settings |
| **PRT + KV cache** | 5.346 | 8.44 | **2.51× SLOWER** | **0.41×** | Phase 12 cache settings |

**Key insight:** PRT is ~2.5× wall slower than native in both configurations. KV cache does not change the speed relationship. Phase 12's 1.82× speedup is not reproduced under any configuration tested.

---

## Interpretation

### Did `--cache-type-k q8_0 --cache-type-v f16` explain the discrepancy?

**No.** KV cache has negligible effect on timing:
- Native wall: 2.131s → 2.129s (–0.1%)
- PRT wall: 5.284s → 5.346s (+1.2%)
- The KV cache confounder hypothesis is ruled out as the primary cause.

### Was the old ~1.82× speedup reproduced?

**No.** In both clean and KV cache configurations, PRT is 2.48–2.51× SLOWER, not faster. The Phase 12 speedup is not a general property of PRT — it was specific to Phase 12's measurement setup.

### What explains the Phase 12 speedup vs current speed-negative result?

**The binary difference is the most likely cause.** Phase 12 used `llama-prt-posix` (custom binary built May 5). Phase 13 uses `llama-cli` (clean build). The old binary may have had:
- Different timing measurement (wall clock vs user time vs something else)
- Different threading model (single-threaded vs multi-threaded differently)
- Different batch behavior (processed tokens differently)
- Different graph compilation (different llama.cpp version or flags)
- Different native baseline (native mode may have been slower in old binary for unrelated reasons)

### What should we say about Phase 12 speedup going forward?

**Phase 12 speedup is context-specific.** It was measured in a specific binary/measurement setup that is not reproducible with the current clean stack. It is neither falsified nor universally valid — it simply cannot be replicated with the current clean `llama-cli` binary.

### Is current Phase 13 speed-negative still the main claim?

**Yes.** Current clean stack (llama-cli) consistently shows:
- PRT wall: ~5.3s vs native: ~2.1s (2.5× slower)
- PRT gen: ~8.5 t/s vs native: ~20.7 t/s (0.41×)

This is the reliable, reproducible claim for the current stack.

---

## Claims Discipline Update

### Phase 12 Speedup Status
**NOT_REPRODUCED** — Phase 12's 1.82× speedup cannot be replicated on current clean stack with identical model, policy, and KV cache settings. The result is specific to Phase 12's binary/measurement setup.

### Allowed Claims
- ✅ PRT is **2.5× wall slower** / **0.41× gen throughput** on current clean stack (Qwen2.5-3B, fresh inference)
- ✅ Phase 12's 1.82× speedup **NOT reproduced** on current clean stack
- ✅ KV cache (q8_0/f16) has **negligible effect** on PRT vs native timing on this model/prompt
- ✅ Quality validated for 0.5B and 3B
- ✅ mmap sidecar loading confirmed working

### Forbidden Claims
- ❌ No PRT speedup claim for 3B or larger models
- ❌ No production readiness
- ❌ No claim that Phase 12 speedup applies to current stack
- ❌ No claim that KV cache produces speedup
- ❌ No universal speedup characterization

---

## Recommended Next

**PAUSE_SPEED_PATH_AND_FREEZE — Current clean stack is speed-negative.** Engineering should pivot to one of:

1. **Packed/lower-precision sidecars** — Reduce sidecar memory bandwidth requirements (FP16→INT8 or ternary) to change the speed equation
2. **Native ggml backend integration** — Move PRT computation into the ggml compute graph to eliminate custom-op dispatch overhead
3. **Speculative decoding or batch verification use case** — PRT as an auxiliary accelerator for specific inference patterns rather than a general-speedup path
4. **Quality-only publication** — Freeze and publish the 0.5B + 3B quality validation as the primary PRT result, without speed claims

**Phase 12 speedup is retired as a current-stack claim.** The quality results (0.5B + 3B validated) remain valid and are the primary durable contribution.

---

## Safety

| Check | Result |
|-------|--------|
| Models/sidecars/binaries staged | NO ✅ |
| Secrets detected | NO ✅ |
| Tags modified | NO ✅ |
| Docs only committed | YES ✅ |

**Tag:** `PRT_PHASE13AK_PHASE12_SPEED_REPRODUCTION`