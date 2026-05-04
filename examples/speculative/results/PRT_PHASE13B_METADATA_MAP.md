# PRT Phase 13B: Metadata Map

**Date:** 2026-05-03

## How llama.cpp Stores Model Dimensions

### Layer Count
```cpp
// Public API (llama.h:572):
int32_t llama_model_n_layer(const struct llama_model * model);

// Internal (llama-model.h):
llama_hparams hparams;  // in struct llama_model
// hparams.n_layer → uint32_t block_count
// Loaded from: LLM_KV_BLOCK_COUNT via ml.get_key()
```

### Hidden Dimension
```cpp
// From hparams:
hparams.n_embd   // embedding length
hparams.n_embd_out_impl  // output embedding length

// From tensor shapes:
// blk.{L}.ffn_up.weight: [n_embd, n_ff] — Qwen2: [896, 4864] or [2048, 11008]
// blk.{L}.ffn_down.weight: [n_ff, n_embd] — Qwen2: [4864, 896] or [11008, 2048]
// blk.{L}.attn_q.weight: [n_embd, n_embd]
```

### FFN Dimension (n_ff)
```cpp
// Not directly in hparams — must read from tensor shape
// FFN_UP: first dimension of ffn_up.weight tensor = n_ff
// e.g., for Qwen2.5-3B: ffn_up.weight shape = [2048, 11008] → n_ff = 11008
// e.g., for Qwen2.5-0.5B: ffn_up.weight shape = [896, 4864] → n_ff = 4864

// Read from first layer's ffn_up tensor:
struct ggml_tensor * t = get_tensor(model, "blk.0.ffn_up.weight");
int n_ff = t->ne[0];   // first dim = n_ff
int n_embd = t->ne[1]; // second dim = n_embd (same as hparams.n_embd)
```

### Relevant API Functions

| Function | File | Purpose |
|----------|------|---------|
| `llama_model_n_layer()` | llama.h:572 | Get layer count from model |
| `llama_get_model(ctx)` | llama.h | Get model from context |
| `hparams.n_layer` | llama-model.h | Internal layer count |
| `hparams.n_embd` | llama-hparams.h | Hidden dimension |
| Tensor `->ne[0]` | ggml.h | First tensor dimension (n_ff for ffn_up) |
| Tensor `->ne[1]` | ggml.h | Second tensor dimension (n_embd for ffn_up) |

### How the PRT Loader Currently Gets Dimensions

```cpp
// From phase10e0_layer0_replacement.cpp callback:
int M = (int)src1->ne[0];  // ← FFN dim from tensor (correct!)
int N = (int)t->ne[0];    // ← hidden dim from tensor (correct!)
```

The runtime PRT compute already uses dynamic tensor dimensions! Only the **sidecar metadata initialization** and **TOTAL_LAYERS** use hardcoded constants.

### Dynamic Sidecar Struct (Current)

```cpp
// Sidecar struct — currently hardcoded:
struct Sidecar { int layer; float * data; int M, N; };
// M = hidden_dim, N = ffn_dim (or vice versa — check transposes)

// Currently hardcoded as:
g_sidecars[layer] = {layer, data, 2048, 11008};

// Should be:
int n_ff = tensor->ne[0];
int n_embd = tensor->ne[1];
g_sidecars[layer] = {layer, data, n_embd, n_ff};
```

### Sidecar File Naming Convention

Current convention (hardcoded path):
```
/tmp/prt_sidecars/ffn_up_layer{L}_prt.bin
```

Dynamic naming should include dims to prevent mismatches:
```
/tmp/prt_sidecars/{model_name}/ffn_up_layer{L}_M{hidden}_N{ffn}_prt.bin
```

### Expected Sidecar Byte Size

```cpp
expected_bytes = hidden_dim * ffn_dim * sizeof(float);  // 4 bytes per float
// Qwen2.5-3B: 2048 * 11008 * 4 = 90,113,024 bytes (~90MB)
// Qwen2.5-0.5B: 896 * 4864 * 4 = 17,432,576 bytes (~17.4MB)
```

### Validation Flow

```cpp
// In load_all_sidecars():
int expected_layers = llama_model_n_layer(model);
int expected_ffn = tensor->ne[0];  // from ffn_up tensor
int expected_hidden = tensor->ne[1];

// Check file size matches:
off_t file_size = stat(file_path).st_size;
off_t expected_size = (off_t)expected_hidden * expected_ffn * sizeof(float);
if (file_size != expected_size) {
    fprintf(stderr, "[PRT ERROR] Sidecar %s: got %ld bytes, expected %ld\n",
            file_path, file_size, expected_size);
    return false;
}
```