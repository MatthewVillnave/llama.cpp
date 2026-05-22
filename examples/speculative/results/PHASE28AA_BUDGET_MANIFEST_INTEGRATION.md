# Phase 28AA: Budget Calculator + Manifest Integration

## Verdict: PASS_PHASE28AA_BUDGET_MANIFEST_INTEGRATION ✅

## Summary
Extended `prt_residual_budget.py` with a manifest mode that reads residual sidecar manifests, evaluates all five budget policies, and produces SAFE/UNSAFE output under configurable RAM/KV constraints. Added `--self-test` for offline validation.

---

## A. Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## B. Current HEAD
`ab5a318f9`

## C. Budget Calculator Updates

### C1. Added `--manifest` mode

New CLI arguments (all `--manifest`-mode only):
```
--manifest PATH              Manifest JSON path
--policy POLICY              base_only|mlp_all|attention_partial|all_validated|budget_greedy|manual
--manual-tensors KEYS        Comma-separated "layer.tensor_family" strings
--residual-budget-mb MB      Max residual budget for budget_greedy policy
--ram-gb GB                  Available RAM (default 16.0)
--context-size TOKENS        KV context length (default 1024)
--kv-bytes-per-token BYTES   KV storage per token (default 2048)
--runtime-buffer-mb MB       Runtime buffer (default 1024)
--os-headroom-mb MB          OS headroom (default 2048)
--out-json PATH              Output JSON report
```

### C2. Policy selection implemented

| Policy | Behavior |
|--------|----------|
| `base_only` | No residuals — Q2 base only |
| `mlp_all` | FFN_UP + FFN_DOWN + FFN_GATE |
| `attention_partial` | attn_q + attn_output |
| `all_validated` | All 5 validated families |
| `budget_greedy` | Ranked by `score_per_byte` or `delta_cos/byte_size`, fill until budget |
| `manual` | User-specified `layer.tensor_family` keys |

### C3. Budget greedy ranking

Ranking score computed as:
```
score = score_per_byte  (if available in validation_metrics)
      OR delta_cosine / byte_size  (fallback)
```
Higher score first. Adds tensors greedily until `residual-budget-mb` limit.

### C4. `--self-test` mode

Runs 17 self-tests without any files:
- `base_only` selects 0 tensors
- `mlp_all` selects MLP families only
- `attention_partial` selects attn families only
- `all_validated` selects all 5 families
- `budget_greedy` respects residual budget limit
- `manual` selects specific tensors correctly
- `sum_residual_bytes` correctness
- `estimate_q2_base_bytes` positive
- `compute_manifest_budget` SAFE/UNSAFE correctness
- `format_manifest_result` output content

### C5. Original parametric mode preserved

Old CLI still works:
```bash
python3 prt_residual_budget.py \
  --param-count 7000000000 --base-bits 2 --residual-bits 1 \
  --residual-layer-count 28 --total-layer-count 28 \
  --residual-fraction 0.85 --context-size 2048
```

---

## D. Fixture Manifest

**Path:** `examples/speculative/fixtures/prt_residual_manifest_budget_fixture.json`

- 10 tensor entries across 2 layers
- All 5 validated families present: ffn_up, ffn_down, ffn_gate, attn_q, attn_output
- `score_per_byte` present in each validation_metrics
- `byte_size` fields populated
- All data is fake/synthetic — no real model data

---

## E. Policy Results

Fixture: 2 layers, 10 tensors total (~918 KB total residual)

| Policy | Tensors | Residual | SAFE |
|--------|---------|----------|------|
| `base_only` | 0 | 0 bytes | ✅ |
| `mlp_all` | 6 | 786 KB | ✅ |
| `attention_partial` | 4 | 131 KB | ✅ |
| `all_validated` | 10 | 918 KB | ✅ |
| `budget_greedy` | 10 | 918 KB | ✅ |

Note: All SAFE on fixture because the fixture is tiny (~2.4 MB base estimate). On a real 7B model with Q2 base (~3 GB) + full MLP residuals (~680 MB), results would differ. The fixture demonstrates the policy routing logic, not actual memory economics.

---

## F. Budget Greedy Result

With `--residual-budget-mb 1024` (1 GB budget):
- Selected all 10 tensors totaling 918 KB (well within 1 GB budget)
- Ranking by `score_per_byte`: attn_q and attn_output rank highest (small byte size, strong delta_cos)
- FFN tensors fill remaining budget
- Output shows full tensor list selected

---

## G. SAFE/UNSAFE Behavior

- `safe = total_estimated_bytes < ram_budget_bytes`
- Total = base + residual + KV + runtime_buffer + os_headroom
- With 16 GB RAM, 1 GB KV, 1 GB buffer, 2 GB headroom → ~3.2 GB estimated → ~13 GB remaining → SAFE
- With tighter budgets (e.g., 8 GB RAM, 4 GB context), the math tightens
- The fixture is tiny, so even `all_validated` is far under limit

---

## H. Limitations

1. **Q2 base estimate is rough** — `estimate_q2_base_bytes` uses shape sums × 0.25 bytes/param. Real Q2_K encoding varies.
2. **Fixture is small** — policy behavior is correct but memory totals don't reflect real 7B/30B scale.
3. **No real model integration** — budget is computed from manifest data only.
4. **KV estimate is rough** — `kv_bytes_per_token` is a fixed estimate; real KV depends on model architecture.
5. **No runtime PRT** — only offline budget estimation.

---

## I. Recommended Next Phase

**Phase 28AB — Real 7B Manifest Scan + Budget Calculation**

1. Run real 7B tensor extraction (FFN_UP only, all 28 layers) to generate a real manifest
2. Compute actual Q2 base bytes for 7B from GGUF metadata
3. Run budget policies against real 7B data
4. Compare Q2 alone vs Q2 + selective residuals vs Q4 baseline

This closes the loop between the synthetic tooling and real model budget reality.

---

## J. Models/Sidecars/F32 Refs Staged?
**NO.** No model files, no sidecars, no f32 refs staged.

## K. Secrets Detected?
None.

## L. Tags Touched?
None.

---

## Files Committed
- `examples/speculative/prt_residual_budget.py` — updated with manifest mode + self-test
- `examples/speculative/fixtures/prt_residual_manifest_budget_fixture.json` — budget fixture
- `examples/speculative/results/PHASE28AA_BUDGET_MANIFEST_INTEGRATION.md` — this report
- `examples/speculative/results/phase28aa_budget_manifest_integration.json` — structured results