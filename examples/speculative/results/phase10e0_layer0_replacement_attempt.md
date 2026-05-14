# results/phase10e0_layer0_replacement_attempt.md
# Phase 10E-0 Layer 0 Replacement Attempt

## Implementation

### Code Location
`examples/speculative/phase10e0_layer0_replacement.cpp`

### Approach
Using `ggml_backend_sched_eval_callback` to intercept ffn_up tensor during compute.

### Steps:
1. Load PRT sidecars from `/tmp/prt_sidecars/` 
2. Create llama context with eval callback
3. During decode, callback fires for each graph tensor
4. For ffn_up tensor at layer 0:
   - Get input activation (src[1])
   - Compute PRT(mul) using sidecar |W|
   - Write PRT output to tensor buffer

### LAYER_SCOPE = 0
Only layer 0 ffn_up is replaced, all others bypassed.

### Code Summary

```c
// Config
static const int LAYER_SCOPE = 0;

// PRT matmul (magnitude-only threshold)
static void matmul_prt(const float * X, float * Y, int batch, int M, int N) {
    for thresholds T_HIGH_0=2.0, T_HIGH_1=0.5, T_HIGH_2=0.1:
    sum += x * W where |x| > threshold
}

// Callback
static bool prt_eval_callback(t, ask, user_data) {
    if (ask) return true for blk.0.ffn_up;
    // else: get src1, compute PRT, write to dst
}
```

### Compilation

```bash
g++ phase10e0_layer0_replacement.cpp \
    -I. -I./ggml/include -I./include \
    -Lbuild/bin -l:libllama.so.0 -l:libggml-base.so \
    -o build/bin/llama-phase10e0-layer0
```

Compiled successfully.

---

## Test Run

### Command
```bash
./llama-phase10e0-layer0 \
    -m models/gguf/qwen2.5/Qwen2.5-3B-Instruct-Q4_K_M.gguf \
    -p "Hello" -n 3
```

### Output
- Generation: works (coherent output)
- PRT replacements: 0
- Callback: NOT FIRING

---

## Why It Didn't Work

The callback isn't receiving events. This may require:
1. Different API to set callback (common_init_from_params vs llama_init_from_model)
2. Batch parameter tuning
3. Or: the scheduler isn't using the callback path

---

## Additional Details

### Sidecar Format
- File: `/tmp/prt_sidecars/ffn_up_layer{n}.prt_bin`
- Shape: {2048, 11008} = M=2048 (input), N=11008 (output)
- Type: float32 magnitude |W|
- Built in Phase 10A

### Dimensions Match
- Qwen2.5-3B: hidden=2048, ffn=11008
- All 28 layer sidecars loaded
- Size per sidecar: 90,177,536 bytes (2048*11008*4)

### PRT Correctness
- Computes magnitude-only threshold matmul
- Compare with float: cosine > 0.95
- Replace float output with PRT output