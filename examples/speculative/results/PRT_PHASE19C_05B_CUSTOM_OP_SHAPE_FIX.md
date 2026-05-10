# PRT Phase 19C: 0.5B Custom Op Shape Fix

## Status: PARTIAL FIX

**Root Cause Found:**
The sidecar M/K values from header were swapped in `build_prt_ffn_up()`:
- Header: M=4864 (FFN), K=896 (hidden)
- Code used: g_prt_sidecar_M = 4864 → hidden (WRONG!)
- Code used: g_prt_sidecar_N = 896 → ffn (WRONG!)

**Fix Applied:**
```cpp
// In prt_graph_replace.h, build_prt_ffn_up():
// OLD (WRONG):
int hidden = g_prt_sidecar_M[layer_id];   // 4864
int ffn    = g_prt_sidecar_N[layer_id];   // 896

// NEW (CORRECT):
int hidden = g_prt_sidecar_N[layer_id];   // 896  
int ffn    = g_prt_sidecar_M[layer_id];   // 4864
```

**Results:**

✅ Shape fix VERIFIED:
- PRT result: `ne=[4864,34]` (CORRECT!)
- Gate output: `ne=[4864,34]` (CORRECT!)
- swiglu_split shape check PASSES

✅ PRT compute path activated for all 24 layers:
- `[PRT_COMPUTE] layer=N mode=int6 hit=1` for each layer

❌ BUT: Custom op returns all zeros (`out_sum=0.0000`)
- Kernel produces no output, causing generation to hang forever
- This is a SEPARATE bug in INT6 unpack/dequantization

## Files Modified
- `examples/speculative/prt_graph_replace.h` — M/K swap fix

## Test Commands
```bash
cd /home/matthew-villnave/llama.cpp
./build/bin/llama-cli -m models/gguf/qwen2.5/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf \
  -p "The capital of France is" -n 32 --temp 0 -c 256 -t 4 --single-turn \
  --prt-mode 5700 --prt-sidecar-dir /tmp/prt_phase19b --prt-sidecar-format int6
```

## Recommended Next Steps

1. Debug why INT6 kernel returns zeros:
   - Check INT6 unpack/dequantization logic
   - Verify scale data is loaded correctly
   - Test with float32 sidecar as baseline

2. If INT6 kernel cannot be fixed quickly:
   - Use float32 sidecars instead of INT6
   - The shape fix works for ANY format

3. Finalize for Phase 19D:
   - Fix the kernel zeros issue
   - Then test with proper validation prompt set

## Verdict
- **PASS_05B_PRT_SHAPE_FIX**: Shape issue fixed, PRT path works
- **FAIL_05B_PRT_KERNEL_ZEROS**: INT6 kernel returns zeros, blocks generation

*Date: 2026-05-10*
*Commit: a8ce1a1d5 + M/K swap*