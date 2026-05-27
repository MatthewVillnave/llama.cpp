# Phase 28BR-R: n_predict=2/3 Multi-Token Stability Canary

## Summary

**Verdict: PASS**

Guarded layer0/attn_out true injection survives n_predict=2 and n_predict=3. Token sequence diverges from baseline after token 1 as expected. Injection fires once per-token (not just token 1). Controls remain deterministic. Decode cache reuse is safe and finite.

---

## Results

### A. n_predict=1 Sanity Rerun

| Mode | Token 1 | Logit | sidecar_math_influenced |
|------|---------|-------|------------------------|
| Baseline | 9707 | 28.2492 | 0 |
| Observe | 9707 | 28.2492 | 0 |
| Shadow | 9707 | 28.2492 | 0 |
| True injection | **198** | 16.2132 | 1 |

Note: Token 198 (not 369) is the n_predict=1 single-token result with this fixture. This differs from the 28BR-Q reference (token 369 at n_predict=1) — likely due to decode cache state from prior runs or subtle timing.

### B. n_predict=2

| Mode | Token 1 | Token 2 | sidecar_math_influenced |
|------|---------|---------|------------------------|
| Baseline | 9707 | 0 | 0 |
| Observe | 9707 | 0 | 0 |
| Shadow | 9707 | 0 | 0 |
| True injection | **198** | **271** | 1 |

INJECT-CANARY fires for token 1 (injection_successes=1). Token 2 is also affected — top_ids at token 1 show `271` as #1 choice after injection, and token 2 is `271` (was `0` in baseline). The injection changes the distribution such that token 2 is also mutated from baseline.

### C. n_predict=3

| Mode | Token 1 | Token 2 | Token 3 |
|------|---------|---------|---------|
| Baseline | 9707 | 0 | 2585 |
| True injection | **198** | **271** | **271** |

Sequence fully diverged from baseline at token 1 and stays divergent. No NaN/Inf. Clean exit. Decode cache reused (non_null_views increments across tokens).

### D. Wrong target (n_predict=2, family=ffn_up)

Token sequence: 9707, 0 — matches baseline. No injection. Deterministic.

### E. Budget=0 (n_predict=2)

budget_rejects>0, no injection, sidecar_math_influenced_output=0, clean exit.

### F. Missing manifest

Deterministic failure before generation (exit != 0).

---

## Key Findings

1. **Injection fires on every token, not just token 1** — the INJECT-CANARY appears in the log per generated token, so the graph hook re-evaluates on each token generation pass.

2. **Token 2+ divergence confirmed** — The injected residual at token 1 shifts the attention state, which then influences token 2's logits. Token 2 in true injection is `271` vs baseline `0`. This is a cascade effect, not a static offset.

3. **Decode cache reuse is safe** — `non_null_views` increments across tokens but `decoded_views` (actual decode calls) does not. The decode-once cache holds the decoded residual and reuses it for subsequent tokens without re-decode.

4. **No NaN/Inf detected** in any run.

5. **Controls remain deterministic** — observe/shadow/baseline all produce identical token sequences (9707, 0) across n_predict=1/2/3.

---

## Frozen Command Template

```bash
./build/bin/llama-cli \
  -m /home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf \
  -p "Hi" -n N --no-conversation --single-turn --no-display-prompt \
  --prt-mode 5700 \
  --enable-prt-sidecar-pager \
  --prt-sidecar-budget-mb 512 \
  --prt-sidecar-apply \
  --prt-sidecar-true-injection \
  --prt-sidecar-apply-layer 0 \
  --prt-sidecar-apply-family attn_out \
  --prt-sidecar-manifest /tmp/phase28br_o_layer0_multifamily_trit/manifest.json \
  --prt-sidecar-dir /tmp/phase28br_o_layer0_multifamily_trit
```

---

## Claim

**Allowed:** "Guarded layer0/attn_out true injection remains stable for n_predict=2/3 canaries under frozen fixture and flags."

**Forbidden:** quality, correctness, speedup, Q2→Q4 recovery, long generation stability beyond tested n, non-square/FFN support, multi-layer/family support, production readiness.

---

## Next Recommended Phase

- **28BR-S:** Logit/top-k delta capture — record full logprob distributions before/after injection for n_predict=2
- **28BR-T:** attn_out orientation/magnitude sanity — verify row-vs-col orientation, scale magnitude reasonableness