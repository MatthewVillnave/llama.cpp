# PRT Phase 24F-X: 3B Extraction Tooling

## Date
2026-05-20

## Branch
experimental/prt-phase19a-alt-sidecar-backed

## HEAD
e02214740

## Status: BLOCKED_NO_EXTRACTION_TOOLING

### What Passed
- Native 3B bounded runner works (generates "Paris...")
- 3B shape confirmed: K=2048, M=11008
- 3B M detection added to source

### What Blocked
- 3B f32 extraction from GGUF
- 3B INT8 sidecar generation

### Root Cause
No working offline GGUF→f32→INT8 extractor for Qwen2.5-3B Q4_K_M.

### Search Results
Found prior extraction tools:
- Phase15B-D: exists with Q4_K dequantization logic
- phase15b_int8_sidecar_regen.cpp
- tools/prt-ffn-up-extract.cpp

Attempted but blocked:
- Python transformers: no GGUF write support
- Custom C++ tool: GGUF/GGML API complexity (context types incompatible)
- Build tools: no direct tensor dump

### Available Artifacts (wrong sizes for 3B)
- 0.5B INT8 sidecar: ~4.4MB
- 7B INT8 sidecar: ~68MB
- 3B target: ~22.5MB

### Report
Verdict: BLOCKED_NO_EXTRACTION_TOOLING

Recommended next:
1. Study Phase15B-D extraction logic more thoroughly
2. OR adapt llama.cpp quantize flow
3. OR use external conversion (safetensors, etc)
