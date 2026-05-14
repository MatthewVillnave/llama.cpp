# Phase 10E-4: Generation Quality Notes

## Test: Layer0 PRT Generation

### Run Configuration
- Model: Qwen2.5-3B-Instruct-Q4_K_M.gguf
- Prompt: "The future of AI is"
- Tokens: 20
- Temperature: 0 (deterministic)
- Seed: 42
- Layer scope: 0 (layer0 only with PRT, other layers standard FFN)

### Results

| Metric | Result |
|--------|--------|
| Ran | YES |
| Crashed | NO |
| Coherent | YES |
| Repetition loops | NO |
| Replacement count | 21 |
| Fallback count | 0 |
| Custom op firing | 21× at layer 0 |

The PRT custom op executes 21 times (once per token for the prompt decode + each new token). Each execution processes nelem=11088 (prompt) or 11008 (decode). No fallbacks — PRT runs successfully for every layer0 FFN_up pass.

### Output Analysis

Generation produces coherent text. The PRT operator modifies layer0 FFN output:
- Layer 0 uses PRT (X @ |W|) instead of standard matmul (X @ W_signed)
- Layers 1–35 remain unchanged (standard FFN matmul)
- The modification is confined to layer0 only — minimal scope

### Quality Assessment

| Aspect | Status |
|--------|--------|
| Visible degradation | NONE |
| Repetition loops | NONE |
| Acceptance rate | Cannot evaluate (single run) |
| Latency impact | Minimal (layer0 only, 1/36 layers) |
| Fragile layers touched | NONE (only ffn_up at layer 0) |

**Conclusion: Generation quality maintained with layer0 PRT active.**

Note: Generation quality with all-layer PRT (layers 0–35) cannot be assessed yet, as the harness only loads layer0 sidecar by default.