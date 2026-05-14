# Phase 10E-7 Quality Notes

## Baseline Output
Prompt: "Write one sentence about CPUs." | n_predict=3 | seed=42 | temp=0
```
CPUs,
```
Coherent English. Expected.

## PRT Output
Same prompt/settings:
```
amup/prt_sidecars/ffn_up_layer35_prt.bin
.Act/prt_sidecars/ffn_up_layer35_prt.bin
 includedsidecars/ffn_up_layer35_prt.bin
```

## Quality Assessment

### CRITICAL FAILURE: PRT output is garbage
The PRT-generated tokens decode to path-like strings instead of coherent text. This is NOT expected behavior.

Baseline: coherent English ("CPUs,")
PRT: incoherent garbage strings

### Possible Causes
1. **Token decoding issue**: `llama_token_to_piece` may be producing bytes that, when concatenated, form path-like strings — suggesting memory corruption or buffer overread in the PRT output path
2. **PRT modifies downstream activations**: The PRT (X @ |W|) produces different activation values that cascade through the network, potentially causing the model to sample very different tokens
3. **Memory corruption**: The path strings ('amup/prt_sidecars/...') suggest the output buffer is being partially overwritten with sidecar path bytes

### This is NOT normal PRT behavior
Standard PRT should produce different but still coherent text. It should not produce garbage or buffer overflow artifacts.

## Verdict Impact
- PRT coherent: NO
- Visible degradation: YES (output is completely incoherent)
- Repetition loops: NO (but output is gibberish, not repetition)