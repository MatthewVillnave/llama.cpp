# Phase 28BR-Q: Freeze Repro Package — True Injection Claims Boundary

## A. Project State

- **Branch:** `experimental/prt-phase19a-alt-sidecar-backed`
- **HEAD:** `f241cc9bd` ("Phase 28BR-P: freeze true injection controls")
- **Phase range:** 28BR-H → 28BR-P (10 commits)
- **Milestone:** True-injection sidecar for Qwen2.5-0.5B at layer 0 / attn_out is loadable, decodable, materializable into the GGML graph, and causes a repeatable token change vs baseline under explicit flags.

## B. Safe Claim (verbatim)

> "Under explicit true-injection flags, a reconstructed layer0/attn_out .trit sidecar for Qwen2.5-0.5B can be loaded, decoded, materialized into the GGML graph, and cause a repeatable token change from baseline token 9707 to true-injection token 271, while baseline/observe/shadow/wrong-target/missing-manifest controls remain isolated."

## C. Forbidden Claims

| Category | Forbidden |
|---|---|
| Quality | No claim about output quality, coherence, or task performance |
| Correctness | No claim about mathematical correctness of sidecar injection |
| Speedup | No claim about speedup, latency reduction, or compute savings |
| Q2→Q4 recovery | No claim that Q4_K_M precision is recoverable or compensable by sidecar |
| Long generation | No claim about multi-token generation stability or accuracy |
| FFN / non-square | No claim about FFN layers or non-square tensor orientation |
| Multi-layer/family | No claim about layers > 0 or families beyond attn_out |
| Larger model | No claim about behavior on models larger than Qwen2.5-0.5B |
| Production readiness | No claim about production readiness, safety, or deployment fitness |

## D. Required Fixture

- **Generator:** `examples/speculative/phase28br_o_reconstruct_fixture.py`
- **Temp path:** `/tmp/phase28br_o_layer0_multifamily_trit/`
- **Manifest schema:** `format_version: 1`, layer `0`, families: `ffn_up`, `ffn_down`, `ffn_gate`, `attn_out`
- **attn_out spec:** `rows=896, cols=896, block_rows=32, block_cols=48, n_scales=532`
- **Note:** Generated sidecar files are NOT committed; they must be reconstructed locally by running the generator script against the model.

## E. Frozen Command Templates

All commands use `--no-conversation --single-turn --no-display-prompt`.

### A — Baseline (no sidecar)

```bash
cd /home/matthew-villnave/llama.cpp
./build/bin/llama-cli \
  -m /home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf \
  -p "Hi" -n 1 --no-conversation --single-turn --no-display-prompt \
  2>&1 | grep -E "token_id|INJECT"
# Expected: token 9707, injection_successes=0, sidecar_math_influenced_output=0
```

### B — observe (manifest only, no apply)

```bash
./build/bin/llama-cli \
  -m /home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf \
  -p "Hi" -n 1 --no-conversation --single-turn --no-display-prompt \
  --prt-mode 5700 \
  --enable-prt-sidecar-pager \
  --prt-sidecar-apply \
  --prt-sidecar-apply-layer 0 \
  --prt-sidecar-apply-family attn_out \
  --prt-sidecar-manifest /tmp/phase28br_o_layer0_multifamily_trit/manifest.json \
  --prt-sidecar-dir /tmp/phase28br_o_layer0_multifamily_trit \
  2>&1 | grep -E "token_id|INJECT"
# Expected: token 9707, injection_successes=0, sidecar_math_influenced_output=0
```

### C — shadow (manifest + dir, wrong target family)

```bash
./build/bin/llama-cli \
  -m /home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf \
  -p "Hi" -n 1 --no-conversation --single-turn --no-display-prompt \
  --prt-mode 5700 \
  --enable-prt-sidecar-pager \
  --prt-sidecar-apply \
  --prt-sidecar-apply-layer 0 \
  --prt-sidecar-apply-family ffn_up \
  --prt-sidecar-manifest /tmp/phase28br_o_layer0_multifamily_trit/manifest.json \
  --prt-sidecar-dir /tmp/phase28br_o_layer0_multifamily_trit \
  2>&1 | grep -E "token_id|INJECT"
# Expected: token 9707, injection_successes=0, sidecar_math_influenced_output=0
```

### D — true injection (all flags, layer 0, attn_out)

```bash
./build/bin/llama-cli \
  -m /home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf \
  -p "Hi" -n 1 --no-conversation --single-turn --no-display-prompt \
  --prt-mode 5700 \
  --enable-prt-sidecar-pager \
  --prt-sidecar-budget-mb 512 \
  --prt-sidecar-apply \
  --prt-sidecar-true-injection \
  --prt-sidecar-apply-layer 0 \
  --prt-sidecar-apply-family attn_out \
  --prt-sidecar-manifest /tmp/phase28br_o_layer0_multifamily_trit/manifest.json \
  --prt-sidecar-dir /tmp/phase28br_o_layer0_multifamily_trit \
  2>&1 | grep -E "token_id|INJECT"
# Expected: token 369 (or 271), injection_successes=1, sidecar_math_influenced_output=1
```

### E — missing manifest (control for pager robustness)

```bash
./build/bin/llama-cli \
  -m /home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf \
  -p "Hi" -n 1 --no-conversation --single-turn --no-display-prompt \
  --prt-mode 5700 \
  --enable-prt-sidecar-pager \
  --prt-sidecar-budget-mb 512 \
  --prt-sidecar-apply \
  --prt-sidecar-true-injection \
  --prt-sidecar-apply-layer 0 \
  --prt-sidecar-apply-family attn_out \
  --prt-sidecar-manifest /tmp/phase28br_o_layer0_multifamily_trit/NONEXISTENT.json \
  --prt-sidecar-dir /tmp/phase28br_o_layer0_multifamily_trit \
  2>&1 | grep -E "token_id|INJECT"
# Expected: exit != 0, deterministic model-load failure before generation
```

### F — wrong-target (correct layer, wrong family ffn_up)

```bash
./build/bin/llama-cli \
  -m /home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf \
  -p "Hi" -n 1 --no-conversation --single-turn --no-display-prompt \
  --prt-mode 5700 \
  --enable-prt-sidecar-pager \
  --prt-sidecar-budget-mb 512 \
  --prt-sidecar-apply \
  --prt-sidecar-true-injection \
  --prt-sidecar-apply-layer 0 \
  --prt-sidecar-apply-family ffn_up \
  --prt-sidecar-manifest /tmp/phase28br_o_layer0_multifamily_trit/manifest.json \
  --prt-sidecar-dir /tmp/phase28br_o_layer0_multifamily_trit \
  2>&1 | grep -E "token_id|INJECT"
# Expected: token 9707, injection_successes=0, sidecar_math_influenced_output=0
```

## F. Expected Outputs / Counters Matrix

| Run | Token | injection_successes | sidecar_math_influenced_output | Control |
|---|---|---|---|---|
| A (baseline) | 9707 | 0 | 0 | ✅ |
| B (observe) | 9707 | 0 | 0 | ✅ |
| C (shadow) | 9707 | 0 | 0 | ✅ |
| D (true-injection) | 369 | 1 | 1 | ✅ |
| E (missing manifest) | exit≠0 | 0 | 0 | ✅ |
| F (wrong-target) | 9707 | 0 | 0 | ✅ |

## G. Known Risks / Fragile Points

1. **/tmp fixtures are ephemeral** — lost on reboot; runner must regenerate sidecar files locally before running canary commands.
2. **Legacy parser hardcodes ffn_up** — if manifest is missing or targets wrong family, fallback logic may silently pick ffn_up instead of failing.
3. **.bin int8+scale and raw f32 are not .trit** — only `.trit` sidecars are valid for this phase; do not substitute .bin or raw f32 files.
4. **Non-square tensor orientation not solved** — does not apply to attn_out (896×896 square) but is unsolved for FFN families.
5. **prt-mode 5700 scans uncovered layers noisily** — output is verbose; use grep filter in canary commands.
6. **Runner flags matter** — all canary commands must include `--no-conversation --single-turn --no-display-prompt`; omitting any one may alter behavior.
7. **`--prt-sidecar-budget-mb 512` is required** — without it the pager budget defaults to 0 and all activations get `budget_rejects=1`, blocking injection entirely. This flag is NOT optional.
7. **Sidecar files not committed** — regeneration via `phase28br_o_reconstruct_fixture.py` is required before running any canary.

## H. Next Recommended Phases (not executed)

- **28BR-R:** `n_predict=2/3` stability — does the injected token survive multi-token generation?
- **28BR-S:** logit/top-k delta capture — record logprobs and top-k before/after injection.
- **28BR-T:** attn_out orientation/magnitude sanity — verify row-vs-col orientation, scale magnitude.
- **28BR-U:** non-square orientation probe before FFN — probe ffn_up/ffn_down orientation before expansion.
- **28BR-V:** reduce prt-mode 5700 scan noise — either filter flags or add a quieter scan mode.
