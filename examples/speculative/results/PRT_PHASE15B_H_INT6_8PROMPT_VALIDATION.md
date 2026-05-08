# PRT Phase 15B-H — Packed INT6 8-Prompt Validation

## Verdict

**PASS_INT6_8PROMPT_QUALITY** ✅

---

## Context

Phase 15B-G validated that packed INT6 sidecars load and run end-to-end on a single tiny prompt ("The capital of France is Paris."). One prompt is not enough to establish broader quality or timing profile.

This phase runs the full 8-prompt suite comparing native vs packed INT6 PRT to determine:
1. Does INT6 preserve output quality across diverse prompts?
2. What is the timing profile vs native?
3. Is broader INT6 validation warranted?

INT8 remains the validated runtime path. INT6 remains experimental.

---

## Prompt Suite

| # | Category | Prompt |
|---|----------|--------|
| 1 | Simple factual | "The capital of France is" |
| 2 | Short story/prose | "Once upon a time in a" |
| 3 | Code generation | "Write a Python function to compute fibonacci numbers and explain it" |
| 4 | JSON structured output | "Return this exact JSON: {\"name\": \"test\", \"value\": 42}" |
| 5 | Basic reasoning | "Explain CPU inference in two sentences" |
| 6 | Technical instruction | "The fastest way to sort a list in Python is" |
| 7 | Summarization/factual | "In two sentences, explain what RAM does in a computer" |
| 8 | Edge/stability | "AI is going to" |

Settings: n=80, temp=0, c=512, t=4, --no-display-prompt, --single-turn

---

## Quality Results

| P | Match | Native t/s | INT6 t/s | Ratio | Sidecars | Output |
|---|-------|------------|----------|-------|----------|--------|
| 1 | EXACT | 9.6 | 9.7 | 1.010 | 28/28 | "The capital of France is Paris." |
| 2 | EXACT | 8.5 | 8.4 | 0.988 | 28/28 | "Once upon a time in a far-off land..." |
| 3 | EXACT | 8.4 | 8.4 | 1.000 | 28/28 | "Certainly! Below is a Python function..." |
| 4 | EXACT | 8.8 | 8.9 | 1.011 | 28/28 | "```json ..." |
| 5 | EXACT | 8.4 | 8.4 | 1.000 | 28/28 | "CPU inference refers to the process..." |
| 6 | EXACT | 8.2 | 8.2 | 1.000 | 28/28 | "The fastest way to sort a list..." |
| 7 | EXACT | 8.4 | 8.2 | 0.976 | 28/28 | "RAM (Random Access Memory) temporarily..." |
| 8 | EXACT | 8.2 | 8.1 | 0.988 | 28/28 | "AI is going to continue transforming..." |

### Aggregate

| Metric | Value |
|--------|-------|
| Native completed | 8/8 |
| INT6 completed | 8/8 |
| **Exact matches** | **8/8** |
| Semantic matches | 0/8 |
| Quality degradations | **0** |
| JSON validity (P4) | Both native and INT6 produced valid JSON blocks |
| Code plausibility (P3) | Both produced plausible Python function with explanation |
| Collapse/repetition | **0** |
| Stdout contamination | **None** |

---

## Timing Results

| Metric | Native | INT6 |
|--------|--------|------|
| Avg tok/s | 8.562 | 8.537 |
| Median tok/s | 8.400 | 8.400 |
| Min tok/s | 8.2 | 8.1 |
| Max tok/s | 9.6 | 9.7 |

**INT6/Native token rate ratio: 0.997 (avg), 1.000 (median)**

Token generation rate is essentially equivalent between native and INT6. The ratio spans 0.976–1.011 across prompts, with no systematic slow-down.

### Wall-clock timing (diagnostic only)

| P | Native wall (s) | INT6 wall (s) | Ratio |
|---|----------------|---------------|-------|
| 1 | 4.22 | 5.62 | 1.332 |
| 2 | 12.97 | 14.98 | 1.155 |
| 3 | 13.15 | 14.98 | 1.139 |
| 4 | 6.06 | 7.98 | 1.317 |
| 5 | 10.56 | 12.54 | 1.188 |
| 6 | 13.72 | 15.43 | 1.124 |
| 7 | 9.00 | 10.76 | 1.195 |
| 8 | 13.52 | 15.28 | 1.130 |

Average wall ratio: ~1.19× (INT6 is ~19% slower on wall clock in this setup).

**Timing note:** Token rate is essentially identical. Wall-clock difference is partly attributable to INT6 unpack overhead at startup (28 layers × ~50MB of sidecar data unpacked before first token). The generation phase itself shows near-parity token rate (0.997×). **No speed claim is made from this data.**

---

## Runtime Evidence

| Check | Result |
|-------|--------|
| Sidecars loaded 28/28 | ✅ 8/8 runs confirmed "Loaded 28/28 sidecars" in PTY output |
| Format=int6 logged | ✅ 8/8 runs confirmed "[PRT_FORMAT] sidecar_format=int6" |
| Fallback only 11,15 | ✅ `--prt-force-native 11,15` set in all INT6 runs |
| No unintended fallback | ✅ All 28 layers had sidecars loaded |
| Clean stdout | ✅ No path fragments, no debug contamination, no errors |
| All exit code 0 | ✅ Native and INT6 completed for all 8 prompts |

Log evidence sample (P1):
```
[PRT_FORMAT] sidecar_format=int6 scale_scheme=per_row
[PRT-FORMAT] INT6 sidecar set: layer=0 M=18944 N=3584 format=int6 per_row
...
[PRT-FORMAT] INT6 sidecar set: layer=27 M=18944 N=3584 format=int6 per_row
[PRT] Loaded 28/28 sidecars from /tmp/prt_sidecars_7b_int6_phase15b_packed
```

---

## Interpretation

**Does INT6 preserve quality across 8 prompts?** ✅ YES — 8/8 exact matches, 0 quality degradations, JSON validity preserved, code plausibility confirmed.

**Is runtime stability acceptable?** ✅ YES — 28/28 sidecars loaded for all runs, no crashes, no errors, clean stdout.

**Is timing acceptable or clearly slower?** MIXED — Token rate is essentially 1:1 (0.997× avg, 1.000× median). Wall clock is ~19% slower in this setup, partly attributed to unpack overhead. **This is not a speed claim** — timing is diagnostic and setup-specific.

**Is broader INT6 testing justified?** YES — Quality is preserved across diverse prompts (factual, prose, code, JSON, reasoning). Token rate is equivalent. Only wall-clock overhead needs further characterization before any speed claim.

**Should next phase be longer-gen INT6, optimization, provenance logging, or stop INT6?** 

Recommended: **Phase 15B-I: INT6 longer-generation smoke** — Run n=160–320 generation to check if quality holds on longer outputs. This is the natural next step since the 8-prompt × 80-token test is still relatively short.

---

## Allowed Claims

- ✅ INT6 passed 8-prompt validation with 8/8 exact matches
- ✅ INT6 token rate is essentially equivalent to native (0.997× avg, 1.000× median)
- ✅ INT6 remains experimental
- ✅ INT8 remains the validated runtime path
- ✅ Timing is diagnostic; wall-clock ~19% slower in this setup, no speed claim
- ✅ 28/28 sidecars loaded, format=int6 confirmed, no stdout contamination

## Forbidden Claims

- ❌ INT6 validated for production use
- ❌ Universal speedup demonstrated
- ❌ INT6 replaces INT8
- ❌ Full long-context stability established
- ❌ Larger-than-7B support validated
- ❌ GPU comparison or cross-platform speed claim

---

## Recommended Next Phase

**Phase 15B-I: INT6 longer-generation smoke** — Run n=160–320 token generation on 3–4 diverse prompts to check whether quality holds over longer outputs. This bridges the gap between the 80-token validation and the longer-context tests already done for INT8.

---

## Summary

| Metric | Value |
|--------|-------|
| Prompts tested | 8 |
| Exact matches | 8/8 |
| Quality degradations | 0 |
| Token rate ratio (INT6/Nat) | 0.997× avg, 1.000× median |
| Wall ratio | ~1.19× (diagnostic only) |
| Sidecars loaded | 28/28 (8/8 runs) |
| Format confirmed | int6 (8/8 runs) |
| Verdict | **PASS_INT6_8PROMPT_QUALITY** |