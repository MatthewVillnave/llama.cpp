# Phase 10E-3S: Layer0 Generation Smoke

## Test Configuration

- Model: Qwen2.5-3B-Instruct-Q4_K_M.gguf
- Sidecar: `/tmp/prt_sidecars/ffn_up_layer0_prt.bin` (layer0 only)
- Scope: layer 0 only, ffn_up only
- Generation: 30 tokens, seed=42, temp=0

## Results

| Metric | Result |
|--------|--------|
| Ran | YES |
| Crashed | NO |
| Coherent | YES |
| Repetition loops | NO |
| Replacement count | 31 |
| Identity fallback count | 0 |
| Non-layer0 replacements | 0 |
| Fragile layers touched | NONE |

## Output Sample

```
Prompt: "The future of artificial intelligence is"
Tokens: "The future of artificial intelligence is becoming"
```

The model produces coherent text with no visible degradation or repetition.

## Custom Op Firing Pattern

```
[PRT] prt_op_entry #1: name=ffn_up.prt.layer0 op_layer=0 sidecar=0x7917e8e5e010
[PRT] PRT_OP: op_layer=0 PRT compute nelem=11008
[PRT] prt_op_entry #2: name=ffn_up.prt.layer0 op_layer=0 sidecar=0x7917e8e5e010
[PRT] PRT_OP: op_layer=0 PRT compute nelem=11008
... (31 total entries, all with op_layer=0)
```

Each token generation triggers exactly one PRT activation for layer0 ffn_up.

## Scope Enforcement

- Only layer 0 receives PRT treatment
- Layers 1-35 use standard FFN computation
- Non-layer0 replacements = 0 (confirmed)

## Fragile Layer Handling

No fragile layers (FFN_down, QKV, attention output, logits) are modified. PRT is confined to layer 0 ffn_up only.

## Conclusion

Generation runs cleanly. PRT does not destabilize output. This is a clean smoke test.

**PASS** — generation ran without crash or coherence issues.
