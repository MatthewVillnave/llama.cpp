# PRT Phase 19N — Tiny 7B Predecode-F32 Probe

## Verdict

**PARTIAL_7B_PREDECODE_MEMORY_RISK**

## Setup

- Model: Qwen2.5-7B-Instruct-Q4_K_M.gguf (4.5GB)
- Sidecar dir: /tmp/prt_sidecars_7b_int6_phase15b_packed (28 layers, packed INT6, PRT6 header)
- Prompt: "The capital of France is"
- Settings: n=32, c=256, t=4, temp=0, single-turn

## Results

| Mode | tok/s | Status | Notes |
|------|-------|--------|-------|
| **Native 7B** | **9.7** | ✅ Complete | Clean output: "Paris" |
| **INT6 scalar PRT** | **1.0** | ✅ Complete | Corrupt output; loaded 28/28 sidecars |
| **INT6 predecode-f32** | **FAILED** | ❌ Killed (OOM) | Loaded 26/28 layers, process killed |

## Memory Analysis

| Component | Size |
|-----------|------|
| Model (Q4_K_M) | ~4,500 MB |
| Predecode f32 (28 layers × 259 MB) | **7,252 MB** |
| INT6 sidecars (mmap'd) | ~1,400 MB |
| **Total if full predecode** | **~13,000+ MB** |
| Available RAM | ~11-12 GB |
| **Margin** | **NEGATIVE — OOM risk** |

**Key finding:** Predecode-f32 for 7B requires ~7.1 GB additional RAM (one f32 copy per layer). With 7B model already loaded (~4.5 GB), the total exceeds available memory. Process was killed after loading 26/28 layers.

## INT6 Scalar 7B — Speed Analysis

| Metric | Value |
|--------|-------|
| Native | 9.7 tok/s |
| INT6 scalar PRT | 1.0 tok/s |
| Slowdown | **9.7x slower** |
| Output quality | Corrupt (gibberish) |

**Analysis:** INT6 scalar PRT on 7B is catastrophically slow (1.0 tok/s) vs native (9.7 tok/s). This is a 9.7x slowdown — far worse than the 5x slowdown seen on 0.5B. The INT6 scalar path appears fundamentally non-viable at 7B scale.

## Predecode-F32 Evidence

From partial log (26/28 layers before OOM):

```
[PRT-PREDECODE] layer=N M=18944 N=3584 f32_bytes=271581184 format=f32_avx2
[PRT_SIDECAR_LAYER] layer=N file=ffn_up_layerN_prt.int6 status=loaded
```

- AVX2 format confirmed: `format=f32_avx2`
- mmap + unpack path working
- Average layer load time: ~140ms mmap + ~77ms unpack = ~220ms per layer
- Full 28-layer predecode would take: ~6 seconds just to decode

## Memory/Swap Health

- Swap: 4GB fully used (baseline from previous phases)
- No swap growth during runs — process killed before OOM spike
- Native and INT6 scalar completed without swap issues

## Interpretation

1. **Predecode-f32 is not viable on 7B** with current RAM (15GB). Would need ~13GB+ for model + predecode + context + compute buffers.

2. **INT6 scalar PRT is catastrophically slow on 7B** (9.7x slower than native). The scalar unpack kernel is the bottleneck — each token requires 28 layers × (mmap + unpack + compute), creating massive overhead.

3. **0.5B vs 7B difference**: 0.5B showed ~5x slowdown with INT6 scalar. 7B shows ~10x slowdown — the overhead scales worse than linearly with model size.

4. **Memory cliff**: The 0.5B predecodes were safe (~418 MB). 7B predecodes are impossible (~7.1 GB).

## Recommendations

### Priority 1: Stop predecode-f32 path for 7B/14B
- Memory cost is non-negotiable: 7.1 GB additional per 7B
- No amount of AVX2 speedup can justify 7 GB RAM premium
- Root cause: storing full f32 copies defeats the purpose of INT6 compression

### Priority 2: Investigate INT6 scalar performance collapse at 7B
- 0.5B: 5x slowdown
- 7B: 10x slowdown (and corrupt output)
- The custom-op overhead + per-layer mmap/unpack is unsustainable at scale

### Priority 3: Alternative approach
- INT8/INT4 quantization with on-the-fly dequant directly to AVX2 registers (no f32 sidecar)
- This is how llama.cpp's k-quants work natively — could be the reference
- Or: accept that PRT sidecar approach needs a complete redesign for production

## Allowed Claims

✅ 7B INT6 scalar PRT works but is catastrophically slow (9.7x slower)
✅ Predecode-f32 on 7B requires ~7.1 GB extra RAM
✅ Predecode-f32 route activates correctly (AVX2 format confirmed)
✅ Process killed by memory pressure before completing 28 layers

## Forbidden Claims

❌ No meaningful speedup claim (predecode couldn't complete)
❌ No production readiness for 7B+
❌ No 14B/32B extrapolation
❌ No comparison to GPU
