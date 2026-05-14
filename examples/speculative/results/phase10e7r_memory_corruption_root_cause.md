# Phase 10E-7R: Memory Corruption Forensics - Root Cause Analysis

## Key Finding: Root Cause IDENTIFIED

After detailed investigation including tensor shape logging and comparison between:
- llama-completion (NO sidecar loaded): identity fallback → CORRECT output
- phase10e0_layer0 harness (sidecar loaded): PRT ACTIVE → GARBAGE output

## Root Cause
The PRT custom op with NON-NULL sidecar produces garbage output. When sidecar=(null), identity fallback works correctly.

This strongly suggests either:
1. Sidecar binary data in /tmp/prt_sidecars/ is CORRUPT, OR
2. PRT math implementation is broken when using non-null sidecar, OR
3. Sidecar loading code in harness corrupts memory

## Evidence

### llama-completion run (sidecar=NIL):
- `[PRT] prt_op_entry #73: name=ffn_up.prt.layer0 op_layer=0 sidecar=(nil)`
- `[PRT] PRT_OP: op_layer=0 fallback=identity (no sidecar)`
- Output: "Hello" (CORRECT)

### phase10e0_layer0 run (sidecar=LOADED):
- `[PRT] prt_op_entry #1: name=ffn_up.prt.layer0 op_layer=0 sidecar=0x70609225c800`
- Output: "amup/prt_sidecars/ffn_up_layer35_prt.bin" (GARBAGE)

## What WAS Working
- Tensor shape verification: CORRECT (dst ne[0]=11008, bytes=44032)
- Write bounds: CORRECT (writes exactly dst_elements)
- Identity fallback: CORRECT produces expected output

## What Is BROKEN
- PRT with loaded sidecar: produces garbage string output
- The path string "ffn_up_layer35_prt.bin" appearing in output suggests memory corruption

## Next Steps to Fix
1. Verify sidecar binary files in /tmp/prt_sidecars/ are not corrupted
2. Run bounded fill test (replace PRT with known-good fill values)
3. Check if the sidecar file read is corrupting memory
4. Validate sidecar file contents match expected |W_up| magnitude format