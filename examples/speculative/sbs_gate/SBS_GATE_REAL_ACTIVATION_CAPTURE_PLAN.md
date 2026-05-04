# SBS-Gate Real Activation Capture Plan

## Goal

Capture real FFN hidden activations from a live llama.cpp inference run, then run oracle block-skip analysis on that real data. This is the Mode B follow-up to the synthetic oracle probe.

---

## What Needs to Be Captured

Per selected layer and token, capture:

| Field | Type | Description |
|-------|------|-------------|
| `layer_id` | int | Which transformer layer (0-indexed) |
| `token_id` | int | Which generated token (0-indexed) |
| `prompt_id` | int | Which prompt from the suite |
| `h_hidden` | float32[] | The FFN hidden vector = silu(gate) * up — shape [ffn_dim] |
| `out_full` | float32[] | Full W_down @ h_hidden — shape [hidden_dim] — the reference |
| `token_text` | str | The decoded token string |
| `prompt_text` | str | The full prompt (truncated) |

The `out_full` is the ground truth. After capture, we run oracle analysis on each captured sample.

---

## Minimal Instrumentation: What to Modify

### llama.cpp: new eval callback hook

Add an optional eval callback to `llama_decode_internal` or `llama_build_graph` that fires after each layer's FFN computation and exposes:

```c
// In llama.h
typedef void (*llama_ffn_act_callback)(void *user_data,
    int layer, int token_idx,
    const float *hidden, size_t hidden_len,  // silu(gate)*up output
    const float *out_full, size_t out_len);  // W_down @ hidden

struct llama_context_params {
    // ... existing fields ...
    llama_ffn_act_callback ffn_act_callback;
    void *ffn_act_user_data;
};
```

The callback fires after the down projection completes for each layer, capturing the hidden vector and full output.

### New CLI flag

```bash
--ffn-act-capture layer=N   # capture FFN activations for layer N
--ffn-act-capture layer=all  # capture all layers (expensive)
--ffn-act-capture outdir=/path/to/dump  # where to write captures
```

### Capture format

JSONL file, one JSON object per captured token:

```json
{
  "layer": 15,
  "token_idx": 3,
  "prompt_idx": 0,
  "h_hidden": [0.0, -0.001, 0.3, ...],   // length ffn_dim
  "out_full": [0.01, -0.2, ...],           // length hidden_dim
  "token_text": " machine",
  "prompt_text": "The old machine began to hum when"
}
```

File naming: `ffn_activations_layer15.jsonl`

### Why this is minimal

- One new callback type
- One new CLI flag pair
- No changes to GGML ops or weight layout
- No changes to PRT code
- Runs on existing models with no conversion needed
- The callback is opt-in; normal operation is unchanged

---

## What to Capture (Practical Scope)

To get meaningful oracle results, we need:

- **Prompts:** 24 prompts from the Phase 12A suite (same suite as PRT validation)
- **Layers to instrument:** Layer 15 (mid-layer) and Layer 25 (near-end) — representative of different depth positions
- **Tokens to capture:** First 20 generated tokens per prompt (captures early-token behavior where FFN is most active)
- **Total estimated captures:** 24 prompts × 2 layers × 20 tokens = 960 samples

This is a manageable size (~100MB JSONL) and gives solid oracle data.

---

## Running the Capture

```bash
cd /home/matthew-villnave/llama.cpp

# Capture layer 15
./build/bin/llama-prt-posix \
  -m /home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-3B-Instruct-Q4_K_M.gguf \
  --ffn-act-capture layer=15 \
  --ffn-act-capture outdir=/tmp/sbs_activations \
  -p "Once upon a time in a" -n 20 --prt-mode 0 --seed 42 -t 8

# (repeat for each of 24 prompts)
# (repeat for layer 25)
```

A harness script would batch this across all 24 prompts.

---

## Oracle Analysis on Real Activations

Once captures are collected, run Mode B of the oracle probe:

```bash
python3 examples/speculative/sbs_gate/sbs_gate_oracle_probe.py \
  --mode real \
  --activation-dir /tmp/sbs_activations \
  --block-size 256 \
  --threshold-cos 0.995 \
  --threshold-l2 0.02 \
  --out results/sbs_gate_oracle_real.json
```

Mode B loads captured `h_hidden` and `out_full` from JSONL, runs oracle block-drop analysis per sample, and computes aggregate statistics.

---

## Instrumentation Difficulty Assessment

| Step | Difficulty | Notes |
|------|-----------|-------|
| Add callback struct + user_data field | Easy | Add 2 fields to llama_context_params struct |
| Fire callback after down projection | Easy-Medium | Insert one callback call in the forward loop |
| Add --ffn-act-capture CLI flag | Easy | Pattern matches existing --logit-bias style flags |
| Write JSONL capture files | Easy | fopen/fprintf in the callback |
| Run batch capture harness | Medium | 48 runs (24 prompts × 2 layers), ~30 min |
| Implement Mode B oracle analysis | Easy | Same oracle logic as Mode A, reading real data |

Total difficulty: **Low-Medium**. The instrumentation is isolated and non-invasive. It can coexist with PRT enabled or disabled.

---

## If Instrumentation Is Too Invasive

If the llama.cpp forward pass is too opaque to instrument cleanly, an alternative is:

**GGUF tensor dump approach:**
1. Modify the sidecar extraction tool (`llama-prt-ffn-up-extract`) to also dump the down projection weights for specific layers
2. Run forward pass with normal llama-cli (no PRT) and log the hidden vectors via the existing eval callback (if available)
3. Combine with saved W_down slices to compute oracle skip rates

This is less direct but avoids modifying the GGML compute graph.

---

## Next Steps

1. **Mode A (synthetic) probe:** Must show ≥ 30% skip rate before proceeding
2. **Instrument llama.cpp:** Add ffn_act_callback + CLI flag
3. **Capture activations:** Run 48 inference passes, collect ~960 samples
4. **Mode B oracle analysis:** Run oracle on real data
5. **Decision:** If real oracle shows ≥ 20% skip rate → proceed to gate bootstrapping. If < 10% → falsified.

---

*Plan prepared by ELVIS for The ForgeHQ / Matthew Villnave*
