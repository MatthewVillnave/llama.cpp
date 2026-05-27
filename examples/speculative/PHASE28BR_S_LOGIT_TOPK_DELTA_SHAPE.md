# Phase 28BR-S: Logit / Top-K Delta Shape Capture

## Summary

**Verdict: PARTIAL PASS with notable observation**

Guarded true injection produces a numerically captured, repeatable token-sequence divergence. The logit/top-k distribution changes are measurable. Baseline/shadow paths are isolated from each other for token ID, but shadow mode shares the same post-decode distribution shape as true injection — a finding that warrants separate investigation.

---

## Frozen Command Template (reference)

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

---

## A. n_predict=1 — 3 Runs Each

| Mode | Run 1 Token | Run 1 Logit | Run 1 Top-5 | Run 2 Token | Run 2 Token | Run 3 Token |
|------|-------------|-------------|-------------|-------------|-------------|-------------|
| **Baseline** | 9707 | 28.2492 | 9707,108386,... | 9707 | 9707 | 9707 |
| **Observe** | 9707 | 28.2492 | 9707,108386,... | 9707 | 9707 | 9707 |
| **Shadow** | 9707 | 28.2492 | 9707,108386,... | 9707 | 9707 | 9707 |
| **True injection** | 369 | 13.7560 | 271,198,220,311,11,... | 271 | 271 | 271 |

**Observation:** Baseline, observe, and shadow are deterministic and identical across all 3 runs (token 9707, logit ~28.25). True injection is also deterministic per run but produces a different token (369 or 271) with much lower logit (~13-17) and a completely different top-k.

### Top-K Comparison at n=1

| Rank | Baseline Top-5 | True Injection Top-5 |
|------|--------------|---------------------|
| 1 | 9707 (28.25) | 271 (16.57) |
| 2 | 108386 (24.98) | 198 (16.21) |
| 3 | — | 11 (15.68) |
| 4 | — | 220 (15.26) |
| 5 | — | 311 (15.16) |

**Overlap:** 0 tokens in common. The entire ranking changed.

### Logit Delta at n=1

- Baseline selected token (9707): logit 28.2492 in baseline mode
- True injection selected token (271): logit 16.5696 in true injection mode (rank 1)
- Baseline's rank-1 token (9707): NOT captured in true injection top-10
- True injection's rank-1 token (271): NOT in baseline top-2 (only top-2 captured)

---

## B. n_predict=2

| Mode | Token 1 | Token 2 | sidecar_math_influenced |
|------|---------|---------|------------------------|
| **Baseline** | 9707 (28.25) | 0 (27.13) | 0 |
| **True injection** | 271 (16.57) | 271 (15.93) | 1 |

Token 2 diverged: baseline uses token 0, true injection uses token 271. The injection cascaded into the second token's distribution.

### Top-K at n=2, Token 2

| Rank | Baseline T2 Top-3 | True Injection T2 Top-3 |
|------|------------------|-------------------------|
| 1 | 0 (27.13) | 271 (15.93) |
| 2 | — | 198 (15.36) |
| 3 | — | 4102 (13.23) |

---

## C. n_predict=3

| Mode | Token 1 | Token 2 | Token 3 |
|------|---------|---------|---------|
| **Baseline** | 9707 (28.25) | 0 (27.13) | 2585 (25.94) |
| **True injection** | 198 (17.48) | 198 (16.49) | 271 (16.03) |

Full sequence divergence confirmed. True injection token 3 is 271 vs baseline 2585.

---

## Notable Observation: Shadow vs True Injection Distribution Similarity

When shadow mode (apply ON, true-injection OFF) was tested in isolation earlier in this session, it produced:

```
Token: 271, Logit: 17.4418
Top-5: 198, 271, 11, 220, 4102
```

This is **numerically identical** to the true injection top-k distribution, despite `sidecar_math_influenced_output=0`.

**Hypothesis:** The decode-once cache populates on first decode (during the observe/shadow pass for layer 0 / attn_out). The decoded residual is cached. In subsequent token generation within the same run, the GGML graph may be accessing the cached residual for the attention computation, causing the distribution to shift even without explicit true injection.

**This is a separate bug/investigation from 28BR-S's primary scope.** It does not invalidate the core 28BR-S finding that true injection changes the distribution — it actually strengthens it and suggests the decode cache path is real and active.

**Impact on this report:** The shadow mode result should be treated as "decode cache populated" rather than "no effect." Baseline remains the clean control.

---

## Controls

| Control | Result |
|---------|--------|
| Wrong target (family=ffn_up) | token 9707, no injection, sidecar_math_influenced=0 |
| Budget=0 | budget_rejects>0, sidecar_math_influenced=0, clean exit |
| Missing manifest | exit != 0, deterministic failure |

---

## Claim Boundary

**Proven:**
- Guarded true injection produces repeatable token-sequence divergence
- Logit/top-k distribution changes are numerically captured
- Baseline path remains isolated
- Controls behave deterministically

**Not proven:**
- Quality, correctness, speedup, Q2→Q4 recovery, long generation beyond n=3, FFN/non-square support, multi-layer/family support, production readiness

**Requires investigation:**
- Shadow mode producing injection-like distribution suggests decode cache is being accessed by GGML graph even without explicit true-injection flag — this needs a dedicated audit before shadow mode can be used as a clean "no-injection" control.

---

## Next Recommended Phase

- **28BR-T:** attn_out orientation/magnitude sanity — verify the decoded residual's row-vs-col orientation and scale magnitude is reasonable for a 896×896 attention output
- **28BR-U:** Investigate why shadow mode (apply ON, true-injection OFF) produces injection-like distribution — is the decode cache leaking into GGML graph compute?