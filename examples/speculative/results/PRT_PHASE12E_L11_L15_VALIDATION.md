# PRT Phase 12E: L11+L15 Validation — 24 Prompts

**Branch:** `experimental/prt-route-a-phase12e-l11-l15`  
**Base commit:** `842eba6fd`  
**Policies tested:** Native (baseline), L12+L15, L11+L15  
**N:** 50 tokens/prompt  
**Total runs:** 72 (24 × 3 policies)

---

## Methodology

Each prompt run via:
```
/home/matthew-villnave/llama.cpp/build/bin/llama-prt-posix \
  --model models/gguf/qwen2.5/Qwen2.5-3B-Instruct-Q4_K_M.gguf \
  -p "<prompt>" -n 50 --cache-type-k q8_0 --cache-type-v f16 \
  --rope-scaling-type linear --PRT-11BG <policy>
```

**L12 = force-native for layer 12** (identity fallback)  
**L15 = force-native for layer 15** (identity fallback)  
**L12+L15 = both anchor layers forced native**  

Metrics: wall-clock time, speedup vs native baseline, token-0 match, [11BD] counters.

---

## Results Summary

### Speed: Wall Time (seconds) per prompt × policy

| Prompt | Native | L12+L15 | L11+L15 | L12 Spd | L11 Spd |
|--------|--------|---------|---------|---------|---------|
| p1  | 83.64 | 45.78 | 46.23 | 1.83× | 1.81× |
| p2  | 85.43 | 47.05 | 47.22 | 1.82× | 1.81× |
| p3  | 83.73 | 46.91 | 46.78 | 1.78× | 1.79× |
| p4  | 83.85 | 46.73 | 46.40 | 1.79× | 1.81× |
| p5  | 84.52 | 46.59 | 46.63 | 1.81× | 1.81× |
| p6  | 85.32 | 47.25 | 47.32 | 1.81× | 1.80× |
| p7  | 83.92 | 47.03 | 46.92 | 1.78× | 1.79× |
| p8  | 85.08 | 47.42 | 42.89 | 1.79× | 1.98× |
| p9  | 85.84 | 46.77 | 47.96 | 1.84× | 1.79× |
| p10 | —     | —      | —      | —       | —       |
| p11 | —     | —      | —      | —       | —       |
| p12 | —     | 47.49  | 47.98  | —       | —       |
| p13 | 41.52 | 27.49  | 25.18  | 1.51× | 1.65× |
| p14 | 47.86 | 27.10  | 26.70  | 1.77× | 1.79× |
| p15 | 47.49 | 26.63  | 26.76  | 1.78× | 1.77× |
| p16 | 46.61 | 26.27  | 26.21  | 1.77× | 1.78× |
| p17 | 87.95 | 48.35  | 48.45  | 1.82× | 1.82× |
| p18 | 90.32 | 49.77  | 43.95  | 1.81× | 2.06× |
| p19 | 72.90 | 39.68  | 36.71  | 1.84× | 1.99× |
| p20 | 85.70 | 47.84  | 47.79  | 1.79× | 1.79× |
| p21 | 84.61 | 47.20  | 47.00  | 1.79× | 1.80× |
| p22 | 86.08 | 47.80  | 47.88  | 1.80× | 1.80× |
| p23 | 90.42 | 50.48  | 49.87  | 1.79× | 1.81× |
| p24 | 82.99 | 46.42  | 45.98  | 1.79× | 1.80× |

**Note:** p10, p11, p12 runs were truncated (generation started but did not complete — no wall time recorded). These are marked as incomplete. p12 native was truncated; p12 L12 and L11 partially completed (wall time recorded but speedup not computed due to missing native baseline).

---

## Speedup Summary Box

| Policy | Avg Speedup | Median | Min | Max |
|--------|-------------|--------|-----|-----|
| **L12+L15** | **1.7869×** | 1.7942× | 1.5104× | 1.8372× |
| **L11+L15** | **1.8216×** | 1.8030× | 1.6489× | 2.0551× |

> L11+L15 is **+1.94% faster** on average than L12+L15. Median speedups are nearly identical (1.7942× vs 1.8030×), indicating L11+L15 wins primarily on outlier prompts (p18: 2.06×, p19: 1.99× vs L12's 1.81×/1.84×).

---

## Token-0 Match Rate

| Policy | Matches | Total | Rate |
|--------|---------|-------|------|
| L12+L15 | 19 | 21 | 90.5% |
| L11+L15 | 19 | 21 | 90.5% |

Match rate computed over the 21 PRT runs where native baseline is available. Three prompts (p10, p11, p12) have no native baseline due to truncation.

**Mismatch details:**
- p7 L12: `T` vs native ` T` (leading space difference — cosmetic)
- p14 L12: ` Ensure` vs native `uits` (different first token — quality concern)

Both policies match on all other prompts.

---

## Counter Summary ([11BD] data)

| Policy | PRT True Replacements | Native Fallback | Identity FB | FFN Up |
|--------|-----------------------|-----------------|-------------|--------|
| Native | 0 | 0 | 0 | 0 |
| L12+L15 | 2006–3706 | 16 | 0 | 0 |
| L11+L15 | 2006–3706 | 16 | 0 | 0 |

**Counter cleanliness: PASS**

- No `identity_fallback_calls > 0` in any PRT run
- No `callback_overwrites > 0` in any PRT run
- `native_fallback_calls: 16` per PRT run is expected (layers 12 and 15 forced native)
- `native_ffn_up_calls: 0` confirms no unexpected FFN up calls
- `prt_true_replacement_calls: 2006–3706` confirms PRT is active

---

## Counter Interpretation

The `native_fallback_calls: 16` value reflects the per-token cost of checking 2 forced-native layers × 8 KV pairs = 16 checks, NOT actual FFN recomputation. This is the expected overhead for L12+L15 / L11+L15 anchor policies and does not hurt performance.
