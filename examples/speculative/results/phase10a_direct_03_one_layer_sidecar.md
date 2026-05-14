# Phase 10A Direct — One-Layer Sidecar Build

**Timestamp:** 2026-04-30 09:05 EDT
**Status:** IN PROGRESS

## Approach
Extract ffn_up layer 0 from GGUF, dequantize, apply PRT_3P, save sidecar.

## Strategy
Write C++ extraction tool: `llama_prt_sidecar_extract.cpp`
- Links against llama.cpp build
- Uses `llama_model_get_tensor()` 
- Extracts `blk.0.ffn_up`
- Dequantizes Q4_K → float
- Applies PRT_3P ternary
- Outputs binary sidecar

## PRT_3P Config (from Phase 8)
```
thresholds: >2.0, 0.5-2.0, 0.1-0.5
3 planes: magnitude-split
```

## Build Command
```
cd build && cmake .. -DLLAMA_SERVER_VERBOSE=ON 2>/dev/null
```

## Testing
Offline accuracy test with random X{batch, 2048}