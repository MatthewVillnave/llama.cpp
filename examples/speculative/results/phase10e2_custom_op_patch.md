# results/phase10e2_custom_op_patch.md
# Phase 10E-2: Custom Op Patch Summary

## Files Modified

### 1. src/llama-graph.cpp

**Added at top (after includes):**
```cpp
// PRT Phase 10E-2: custom op state — non-static so it's exported from libllama.so for harness access
int g_prt_ffn_up_custom_op_count = 0;
int g_prt_ffn_up_replacement_count = 0;
int g_prt_ffn_up_layer_last = -1;
const char * g_prt_ffn_up_name_last = nullptr;

// PRT Phase 10E-2 smoke test custom op:
// Fills output with zeros. Downstream gets zeroed FFN input.
static void prt_ffn_up_smoke_op(
        struct ggml_tensor * dst,
        const struct ggml_tensor * a,
        int ith, int nth, void * userdata) {
    (void)userdata;
    (void)a;
    if (ith != 0) return;
    
    const int64_t n = (int64_t)ggml_nelements(dst);
    float * dst_data = (float *)dst->data;
    
    for (int64_t i = 0; i < n; i++) {
        dst_data[i] = 0.0f;
    }
    
    g_prt_ffn_up_custom_op_count++;
    g_prt_ffn_up_replacement_count++;
    fprintf(stderr, "  [PRT] CUSTOM_OP ffn_up zero-fill: layer=%d count=%d nelem=%lld\n",
            g_prt_ffn_up_layer_last, g_prt_ffn_up_custom_op_count, (long long)n);
}
```

**Modified build_ffn() around line 1127:**
```cpp
ggml_tensor * tmp;
if (up) {
    g_prt_ffn_up_layer_last = il;
    ggml_tensor * matmul_result = build_lora_mm(up, cur);
    
    if (il == 0) {
        // Phase 10E-2 smoke test: use ggml_map_custom1_inplace
        tmp = ggml_map_custom1_inplace(ctx0, matmul_result, prt_ffn_up_smoke_op, 1, nullptr);
        ggml_set_name(tmp, "blk.0.ffn_up.prt_smoke");
        fprintf(stderr, "  [PRT] GRAPH_SUBSTITUTION: layer=0 using ggml_map_custom1_inplace\n");
    } else {
        tmp = matmul_result;
    }
} else {
    tmp = cur;
}
```

### 2. src/llama.cpp

**Added at end (before final curly):**
```cpp
extern "C" LLAMA_API int llama_get_prt_replacement_count(void);
int llama_get_prt_replacement_count(void) {
    extern int g_prt_ffn_up_custom_op_count;
    return g_prt_ffn_up_custom_op_count;
}
```

### 3. include/llama.h

**Added before #ifdef __cplusplus:**
```cpp
// PRT Phase 10E-2: replacement count accessor
LLAMA_API int llama_get_prt_replacement_count(void);
```

### 4. examples/speculative/phase10e0_layer0_replacement.cpp

**Modified to use the exported getter:**
```cpp
// Changed from: g_prt_state.total_replacements
// To: llama_get_prt_replacement_count()
fprintf(stderr, "Total PRT replacements: %d\n", llama_get_prt_replacement_count());
```

---

## Build Commands

```bash
# Rebuild library
cd build && cmake --build . --target llama -j4

# Rebuild harness
g++ -std=c++17 -O2 -I../examples/speculative -I./ggml/include -I../include -I../ggml/include -I./common \
    ../examples/speculative/phase10e0_layer0_replacement.cpp ./common/libcommon.a \
    -L./bin -l:libllama.so.0 -l:libggml-base.so.0 -Wl,-rpath,./bin \
    -o ./bin/llama-phase10e0-layer0
```

---

## Test Run

```bash
LD_LIBRARY_PATH=build/bin ./llama-phase10e0-layer0 \
    -m /home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-3B-Instruct-Q4_K_M.gguf \
    -p "Hello" -n 5
```

**Expected output:**
```
[PRT] GRAPH_SUBSTITUTION: layer=0 using ggml_map_custom1_inplace
  [PRT] CUSTOM_OP ffn_up zero-fill: layer=35 count=1 nelem=11008
Promp decoded (replacements: 1)
  [PRT] CUSTOM_OP ffn_up zero-fill: layer=35 count=2 nelem=11008
  0: '
  [PRT] CUSTOM_OP ffn_up zero-fill: layer=35 count=3 nelem=11008
  ...
Total PRT replacements: <N>
```

---

## Verification

| Check | Status |
|-------|--------|
| Library compiles | ✅ |
| Custom op function compiles | ✅ |
| Harness compiles | ✅ |
| Custom op fires | ✅ (count > 0) |
| Replacement count accessible | ✅ (llama_get_prt_replacement_count) |
| Graph integration works | ✅ |
| Generation runs | ✅ |
| Output garbage (expected) | ✅ (zeroed FFN) |