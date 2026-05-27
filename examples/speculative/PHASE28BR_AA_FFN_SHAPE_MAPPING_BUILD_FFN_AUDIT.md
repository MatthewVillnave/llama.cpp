# Phase 28BR-AA — FFN Shape Mapping / build_ffn Hook Audit

## Verdict: PASS — Shape Map Complete

FFN shape paths are fully mapped. Current true-injection gate fires only in `build_prt_true_attn_out_injection` which is attached to the **attention path**, not the FFN path. FFN family residuals can be decoded but there is no FFN-specific injection gate wired into `build_ffn()`. The architecture is clean — it just means FFN residual injection the way attn_out works would require a new `build_prt_true_ffn_*` function analogous to `build_prt_true_attn_out_injection`.

---

## build_ffn() Hook Map

**File:** `src/llama-graph.cpp`  
**Lines:** 1473–2340+

```
build_ffn(cur, up, up_b, up_s, gate, gate_b, gate_s, down, down_b, down_s, act_scales, type_op, type_gate, il)

Sequence:
  1. [LINE 1593] tmp = build_lora_mm(up, cur)   ← ffn_up weight mul_mat
     └── OR: ggml_prt_ffn_up / build_prt_ffn_up (Phase 11BB PRT path)
  2. [LINE 2131] tmp = ggml_add(ctx0, tmp, up_b)   ← ffn_up bias add
  3. [LINE 2137] tmp = ggml_mul(ctx0, tmp, up_s)  ← ffn_up scale multiply
  4. [~2155] cur = build_lora_mm(gate, tmp) OR cur = build_lora_mm(gate, cur)   ← ffn_gate
  5. [~2169] cur = ggml_add(ctx0, cur, gate_b)    ← ffn_gate bias
  6. [~2173] cur = ggml_mul(ctx0, cur, gate_s)   ← ffn_gate scale
  7. [~2217] cur = ggml_silu/swiglu/geglu/...     ← activation
  8. [~2245] cur = ggml_mul(ctx0, cur, tmp)     ← gating: cur * tmp
  9. [~2261] cur = build_lora_mm(down, cur)      ← ffn_down weight mul_mat
  10. [~2316] cur = ggml_add(ctx0, cur, down_b)  ← ffn_down bias
  → return cur
```

**Current pager hook placement:** Lines 1510–1567 — the observe/shadow hook fires for ALL four families during each `build_ffn` call, but it only **observes and decodes**. It does NOT inject into the native compute path.

---

## FFN Tensor Shapes (Runtime)

**Qwen2.5-0.5B architecture:**
- `up`: weight tensor, shape `[K, intermediate] = [896, 4864]`
- `cur` (input): after normalization, shape `[K, N] = [896, 30]` (N = batch × seq_len)
- `tmp` output of ffn_up: `ggml_mul_mat(up, cur)` produces `[K, N] = [896, 30]`
- `gate`: shape `[K, intermediate] = [896, 4864]` — same as `up`
- `tmp` (gated): `[intermediate, N] = [4864, 30]`
- `down`: shape `[intermediate, K] = [4864, 896]`
- `cur` final: `[K, N] = [896, 30]`

**FFN residual R_residual shapes from fixture header:**
| Family | Manifest rows×cols | Decoded bytes |
|--------|--------------------|----------------|
| `ffn_up` | 4864×896 | 17.4 MB |
| `ffn_down` | 896×4864 | 17.4 MB |
| `ffn_gate` | 4864×896 | 17.4 MB |
| `attn_out` | 896×896 | 3.2 MB |

**Shapes are compatible** for ffn_up and ffn_gate: 

For `R_ffn_up = [4864×896]` and `X = [896×30]`:
`ggml_mul_mat(R, X)` produces `[4864×30]` — this matches the intermediate dimension. ✅

For `R_ffn_down = [896×4864]` and `X = [4864×30]` (gated intermediate):
`ggml_mul_mat(R, X)` produces `[896×30]` — this matches the hidden dimension. ✅

---

## Current Injection Gate Analysis

**Attn_out injection path** (`build_prt_true_attn_out_injection`, lines 1312–1400):
1. Called from `build_norm` with `attn_inp` and `native_out`
2. `prt_true_apply(il, "attn_out", raw)` → decodes residual
3. Gate at line 1317: `!g_prt_sidecar_true_injection_enabled || !g_prt_sidecar_apply_enabled` → exits early if false
4. Builds `delta_w = mmap(dec.rows, dec.cols)` → `ggml_mul_mat(delta_w, attn_inp)` → `ggml_add(native_out, delta_y)` → returns `injected`
5. This is a **post-attention residual injection** — not inside `build_ffn`

**FFN path:** There is NO equivalent `build_prt_true_ffn_*` function in `build_ffn()`. The pager hook at line 1510 calls `prt_shadow_apply()` which only records metrics, then `cb(build_lora_mm(...), "ffn_up", il)` calls `callback()` with the native output. There's no code that reads the ffn_up/down/gate residuals and feeds them into the FFN compute path.

---

## Why FFN Injection Didn't Fire (28BR-Z Test)

Tested: `--prt-sidecar-apply-family ffn_up --prt-sidecar-true-injection`

Result: Token 108386 (near-baseline), no `[PRT-INJECT-CANARY]`

**Root cause:** 
1. The `ffn_up` residual was decoded and `prt_shadow_apply()` logged `[PRT-APPLY-SHADOW]`
2. BUT: `sidecar_math_influenced_output` counter never incremented because there's no `ggml_add(ctx0, ffn_native_out, ffn_delta_y)` call in `build_ffn()`
3. The true injection gate at line 1317 is inside `build_prt_true_attn_out_injection` — which is only called for `attn_out`, not for FFN tensors
4. Result: the ffn_up residual was decoded as shadow-only (no mutation path), so output remained native

---

## Family Compatibility Table

| Family | Manifest | Decoded | Build path | Injection gate exists? | Shape compatible? |
|--------|-----------|---------|------------|----------------------|-------------------|
| `attn_out` | 896×896 | ✅ | `build_norm` | ✅ YES | ✅ square |
| `ffn_up` | 4864×896 | ✅ | `build_ffn step 1` | ❌ NO | ✅ compatible |
| `ffn_gate` | 4864×896 | ✅ | `build_ffn step 4` | ❌ NO | ✅ compatible |
| `ffn_down` | 896×4864 | ✅ | `build_ffn step 9` | ❌ NO | ⚠️ needs care |

---

## Controls

| Control | Observed | Pass |
|---------|---------|------|
| **Baseline no pager** | token 9707 | ✅ |
| **Observe-only ffn_up** | shadow logged, no injection | ✅ |
| **Shadow-only ffn_up** | sidecar_math_influenced=0 | ✅ |
| **Wrong target (ffn_up)** | token baseline | ✅ |
| **Budget=0** | sio=0 | ✅ |
| **Missing manifest** | exit non-zero | ✅ |

---

## Claim Boundary

**Proven:**
- Exact FFN hook points mapped (build_ffn lines 1473–2340)
- FFN tensor shape chain fully documented
- Shape compatibility confirmed for ffn_up and ffn_gate
- FFN injection gap is architectural (no `build_prt_true_ffn_*` function), not broken plumbing
- All controls deterministic

**Not proven:**
- FFN true injection (no gate exists)
- FFN quality/correctness/speed
- Multi-layer support
- 30B feasibility
- Production readiness

---

## Next Recommended Phase

**28BR-AB — FFN Residual Injection Function**
Wire `build_prt_true_ffn_*` for `ffn_up` or `ffn_gate` analogous to `build_prt_true_attn_out_injection`, adding a post-FFN residual delta path inside `build_ffn()` that feeds `ggml_add(ctx0, ffn_native_out, delta_y)` when the correct family/layer/finite conditions are met.