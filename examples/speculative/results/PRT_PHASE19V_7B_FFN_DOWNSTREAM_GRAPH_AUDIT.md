# PRT Phase 19V: FFN Downstream Graph Audit — Interim Report

## Verdict: INCOMPLETE_AUDIT

### Summary
Phase 19V adds audits at each FFN stage (gate, FFN_UP, SwiGLU, FFN_down) in native path for comparison. However, audits are not firing properly during testing.

### What Was Added
- Gate projection audit after `build_lora_mm(gate, cur)`
- FFN_UP output audit (already in Phase 19U)
- SwiGLU result audit after `ggml_swiglu_split`
- FFN_down output audit after `build_lora_mm(down, cur)`

### INT6 Path Test (with Force-native 11,15)
- Output for "The capital of France is": **GIBBERISH** ("azi.fullNamewed...")
- Model speed: 1.1 t/s (much slower than native 4.7 t/s)

Key logs from INT6 run:
```
[PRT_PRE-AUDIT] scales=0x56b845008610 scale[0]=0.002169 scale[1]=0.005101
[PRT_PRE-AUDIT] direct_mmap_scale[0]=0.002169 direct_mmap_scale[1]=0.005101
[PRT_UP_AUDIT] layer=0 shape=[18944,34] first16= 0.1215 -0.2475 0.1185 -0.1359...
```

### Native Path Test
Could not get native baseline audit logs due to:
- Empty sidecar directory not triggering PRT sidecar loading
- Force-native ALL leads to different execution path
- Need better test configuration to isolate native baseline

### Code Available
- Added audit hooks in `src/llama-graph.cpp` at key FFN stages
- Added output audits in custom op in `prt_graph_replace.h`

### Next Steps Needed
1. Get native baseline FFN values working
2. Verify gate projection output in both paths has matching shape
3. Verify SwiGLU input/output shapes match
4. Compare INT6 gate vs native gate values
5. Identify exact corruption point

### Branch
experimental/prt-phase19a-alt-sidecar-backed

### Latest Commit  
5b5ab9f2f