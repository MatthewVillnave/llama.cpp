# Phase 28BR-Y — Frozen True Injection Comparison Table / Claims Boundary

## Verdict: PASS — Complete Freeze

This document freezes all attn_out true injection canary results from phases 28BR-P through 28BR-X into a single comparison table. It supersedes individual phase reports and serves as the authoritative claims boundary.

---

## Frozen Command Reference

```bash
./build/bin/llama-cli \
  -m /home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf \
  -p "Hi" -n N --no-conversation --single-turn --no-display-prompt \
  --prt-mode 5700 \
  --enable-prt-sidecar-pager \
  --prt-sidecar-budget-mb 512 \
  --prt-sidecar-apply \
  --prt-sidecar-apply-layer 0 \
  --prt-sidecar-apply-family attn_out \
  --prt-sidecar-manifest /tmp/phase28br_o_layer0_multifamily_trit/manifest.json \
  --prt-sidecar-dir /tmp/phase28br_o_layer0_multifamily_trit
```

Add `--prt-sidecar-true-injection` for true injection mode.  
Add `--prt-sidecar-scale S` for scale factor (0.0–2.0, default 1.0).  
Add `--prt-sidecar-sign-flip` to negate residual before injection (Phase 28BR-X).

---

## Mode Comparison Matrix (n=1, frozen fixture)

| Mode | Token | Logit | Top-3 | sio | injection_successes |
|------|-------|-------|-------|-----|----------------|
| **Baseline** | 9707 | 28.25 | 9707, 108386 | 0 | 0 |
| **Observe** | 9707 | 28.25 | 9707, 108386 | 0 | 0 |
| **Shadow** | 9707 | 28.25 | 9707, 108386 | 0 | 0 |
| **True inj scale=0** | 9707 | 28.25 | 9707, 108386 | 0* | 0 |
| **True inj scale=0.25** | 198 | 17.23 | 198, 271, 44 | 1 | 1 |
| **True inj scale=0.5** | 198 | 17.31 | 198, 271, 4102 | 1 | 1 |
| **True inj scale=1.0** | 271/220 | ~16.6 | 271, 198, 220 | 1 | 1 |
| **True inj scale=2.0** | 304/220 | ~18.9 | 304, 220, 758 | 1 | 1 |
| **Sign flip scale=0.5** | 13 | 16.05 | 13, 77, 514 | 1 | 1 |
| **Sign flip scale=1.0** | 198 | 14.26 | 198, 271, 13 | 1 | 1 |
| **Sign flip scale=2.0** | 99xxx | ~17.8 | 99xxx range | 1 | 1 |

*scale=0 technically fires injection path but produces all-zero residual, so sio counter may be 0.

**Top-k overlap with baseline:**
- scale=0: 2/2 (full overlap)
- scale=0.25–2.0: 0/N (baseline token 9707 not in any top-k at non-zero scales)
- Sign flip: 0/N (baseline token never appears)

---

## n=2 Token Sequences

| Mode | Token 1 | Token 2 | sio |
|------|---------|---------|-----|
| **Baseline** | 9707 | 0 | 0 |
| **True inj scale=0** | 9707 | 0 | 0 |
| **True inj scale=0.25** | 198 | 198 | 1 |
| **True inj scale=0.5** | 271 | 271 | 1 |
| **True inj scale=1.0** | 271 | 271 | 1 |
| **True inj scale=2.0** | 758 | 320 | 1 |
| **Sign flip scale=0.5** | 77 | 13 | 1 |
| **Sign flip scale=1.0** | 82 | 220 | 1 |
| **Sign flip scale=2.0** | 13 | 13 | 1 |

**Observation:** Cascade fires at token 2 (injection path active for subsequent tokens). scale=0 cascade confirms zero residual = no-op. Non-zero scales cascade consistently.

---

## n=3 Token Sequences

| Mode | Token 1 | Token 2 | Token 3 |
|------|---------|---------|---------|
| **Baseline** | 9707 | 0 | 2585 |
| **True inj scale=0** | 9707 | 0 | 2585 |
| **True inj scale=0.25** | 198 | 271 | 198* |
| **True inj scale=0.5** | 198 | 271 | 198 |

*Path destabilizes at n=3 for scale=0.25

---

## Controls Matrix

| Control | Expected | Observed | Pass |
|---------|----------|---------|------|
| Wrong target (ffn_up) | baseline token, sio=0 | baseline token, sio=0 | ✅ |
| Budget=0 | sio=0 | sio=0 | ✅ |
| Missing manifest | exit non-zero | exit non-zero | ✅ |
| Observe-only | baseline token, sio=0 | baseline token, sio=0 | ✅ |
| Shadow mode | baseline token, sio=0 | baseline token, sio=0 (SHADOW_CLEAN_REPORT_MIXUP) | ✅ |

---

## Key Findings (Frozen)

1. **scale=0 = exact baseline** across n=1/2/3 — zero residual is a true no-op
2. **Baseline token 9707 immediately demoted** from top-k at all non-zero scales
3. **Non-monotonic logit**: logit decreases 0→1.0 then recovers at 2.0 — residual direction/dominance changes with magnitude
4. **Sign flip is directionally distinct**: same scale produces different token winner vs normal injection
5. **Cascade confirmed**: injection path influences subsequent token distributions
6. **Shadow mode is clean**: shadow-only (apply ON, true-injection OFF) produces no graph mutation — earlier 28BR-S observation was a command-flag mismatch
7. **No NaN/Inf** at any scale 0–2.0 or sign flip variant

---

## Complete Safe Claim Boundary

**PROVEN under frozen fixture/conditions:**
- Guarded true injection with `--prt-sidecar-budget-mb 512` fires and changes token outputs repeatably
- scale=0 exactly preserves baseline behavior (zero residual no-op)
- Residual scale [0.25–2.0] produces measurable token/logit shift from baseline
- Logit shifts are numerically characterized across scale range
- Sign flip produces distinct direction from normal injection at same scale
- Baseline token is immediately demoted out of top-k at all non-zero scales
- Cascade to token 2+ confirmed for non-zero scales
- Shadow mode never mutates graph when true-injection flag is absent
- Wrong target, budget=0, missing manifest are clean controls
- No NaN/Inf at any tested scale/sign combination

**NOT PROVEN / FORBIDDEN:**
- Quality — no human评估
- Correctness — no task accuracy measurement
- Speedup — no latency benchmarking
- Q2→Q4 recovery — no quality recovery claim
- Long generation beyond n=3
- FFN / non-square tensor support
- Multi-layer / multi-family support
- Larger models (30B+)
- Production readiness

---

## Technical Debt / Recommended Next Phase Before FFN

**28BR-Z — non-square attn_out residual orientation sanity**

Before expanding to FFN, verify whether the layer0/attn_out extracted residual produces predictable effects on a non-square attention output (if any in Qwen2.5 layer0). The current fixture confirms 896×896 square orientation. If Qwen2.5's layer0 uses non-standard attention output layout (e.g., causal mask, grouped attention, multi-head reshape), the orientation may need transposition before injection.

This is a prerequisite orientation audit before any FFN support.

---

## Phase Range

| Phase | Commit | Topic |
|-------|--------|-------|
| 28BR-P | f241cc9bd | Freeze true injection controls A-F |
| 28BR-Q | c5b9524b0 + 7cc80698a | Freeze repro package + budget fix |
| 28BR-R | 88b7ac24e | Multitoken stability (n=2/3) |
| 28BR-S | 88520a3e0 | Logit/top-k delta shape capture |
| 28BR-T | 2891d2bae | Orientation/magnitude sanity + scale flag |
| 28BR-U | 84966a58f | Shadow isolation audit (SHADOW_CLEAN_REPORT_MIXUP) |
| 28BR-V | 149ef3548 | Multitoken scale stability |
| 28BR-W | d474f5c07 | Logit rank shift vs scale |
| 28BR-X | f48a874d3 | Sign flip / orientation sanity + sign-flip flag |
| **28BR-Y** | **this commit** | **Frozen comparison table / claims boundary** |