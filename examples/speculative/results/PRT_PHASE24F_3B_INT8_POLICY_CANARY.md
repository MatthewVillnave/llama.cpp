# PRT Phase 24F: 3B INT8 Policy Canary - PARTIAL

## Date
2026-05-19

## Branch
experimental/prt-phase19a-alt-sidecar-backed

## HEAD
942d873be (parent commit: R2/R3 isolation)

## Status: BLOCKED_3B_EXTRACTION_TOOLING

### What Passed
- Preflight: clean memory/disk ✅
- Native 3B bounded runner: ✅ ("Paris" generated)
- Shape detection: K=2048, M=11008 ✅

### What Blocked
- 3B INT8 sidecar generation
- Runtime canary test

### Blocker
No working offline GGUF tensor extraction path for 3B layer0 FFN_UP.
- Runtime extraction (PRT_GGML_TEST_LAYER=0) is circular because it expects a sidecar file to exist
- Offline extraction tools found but fail to compile/link (API changes in llama.cpp)

### Source Changes
Added 3B M detection in llama-graph.cpp:
```cpp
const int M = (K == 3584) ? 18944 : (K == 2048) ? 11008 : 4864;
```
This enables shape detection for 3B (K=2048→M=11008).

### Model
/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-3B-Instruct-Q4_K_M.gguf

### Shape
- K=2048 (hidden_size)
- M=11008 (intermediate_size)
- layers=36
- f32 ref size: 90,177,536 bytes
- INT8 sidecar size: ~22,588,416 bytes

### Native Test Result
```
prompt: "The capital of France is"
output: "Paris. Paris is located in the north..."
exit: 0
```

### Required Next
Phase 24F-X: Find working offline GGUF extractor that made 0.5B/7B sidecars.

---
Verdict: BLOCKED_3B_EXTRACTION_TOOLING