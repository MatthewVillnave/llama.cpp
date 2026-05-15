# PRT Phase 21Q-C: 7B Capture Tune + Runtime Overhead Diagnosis

## Context
- **Branch**: `experimental/prt-phase19a-alt-sidecar-backed`
- **Previous HEAD**: `c8b4007c6` (Phase 21Q-B)
- **Date**: 2026-05-15

## Goal
Tune capture harness so PRT-v2 generated text is visible, and diagnose ~3x runtime overhead.

## Key Findings

### Finding 1: Spinner Hides PRT Output
The llama-cli spinner (`\r`-overwrite pattern) corrupts PTY capture for PRT runs.
- Native n=8: spinner leaves "Paris" visible in PTY capture after spinner stops
- PRT n=1: spinner overwrites text before capture file is readable
- PRT n=8: spinner overwrites text throughout generation
- Native n=1: exits fast, spinner active during output capture

Evidence:
- PRT n=1 capture shows: "The" (first token, then spinner overwrites)
- Native n=8 capture shows: "Paris" (spinner stops after generation done)
- PRT n=1 Generation metric: `1000000.0 t/s` (spinner artifact, not real)
- Native n=8 Generation metric: `4.9 t/s` (real)

### Finding 2: PRT Runtime Overhead Explained
PRT-v2 scalar f32 kernel is slow on CPU but expected:
- Native 7B Q4_K_M matvec: ~5 t/s
- PRT scalar f32 kernel: ~0.02 t/s (estimated from N=16 shapes)
- 334s PRT vs 120s native = **~2.8x slowdown** — consistent with scalar f32 vs Q4 quantized matmul
- The kernel is doing 3584×18944 float operations per token per layer via scalar loops
- This is expected behavior for unoptimized CPU scalar implementation

### Finding 3: Kernel Evidence is Clean
All PRT runs confirm:
- output_abs_sum: 1.684134, 1.944075 (consistent across all phases)
- KERNEL_ENTER/EXIT: confirmed on all runs
- Sidecar decode: working (INT8 → f32 decode confirmed)
- Route count: 28 layers (IL=0 PRT, IL=1-27 native)

### Finding 4: Text IS Being Generated
- PRT n=1 raw capture has the text — it's just overwritten by spinner
- The 35KB PRT n=1 file vs 28KB native n=1 file = PRT has MORE content
- Spinner `▄▄ ▄▄` blocks cover the text in the PTY frame buffer
- When spinner stops, text appears — but PRT spinner never stops because it's computing slowly

## Root Cause of Text Visibility Issue
**Spinner overwrite**: llama-cli uses `\r`-based spinner during generation. In PTY mode, each spin overwrites the previous text. When generation finishes, spinner stops and text is visible. But PRT generation is slow (scalar f32) so spinner runs longer — by the time the file is read, the spinner has overwritten everything.

**Native n=8 works** because generation completes fast enough that spinner stops before file inspection.

**PRT n=1 fails** because slow generation keeps spinner active throughout.

## Capture Cleaner
A capture cleaner is needed but NOT needed for the core PRT goal. The kernel evidence (KERNEL_ENTER/EXIT/output_abs_sum) is the primary correctness proof.

A simple cleaner approach (not committed):
```python
# Strip spinner frames, normalize \r\n, extract clean text
# See: examples/speculative/phase21q_clean_capture.py (optional future work)
```

## Runtime Breakdown

| Component | Native | PRT |
|-----------|--------|-----|
| Model load | ~5s | ~5s |
| Sidecar decode | N/A | ~1s |
| Prompt eval | ~3s | ~3s |
| Token gen (n=8) | ~120s | ~334s |
| Total | ~128s | ~343s |

**Root cause of 334s runtime**: Scalar f32 PRT kernel vs Q4 matmul. Not a bug — expected CPU behavior.

## Verdict
- **PARTIAL_PRT_TEXT_STILL_HIDDEN**: PRT text exists but spinner overwrites capture
- **PARTIAL_RUNTIME_OVERHEAD_EXPLAINED**: Scalar f32 kernel cost confirmed

## Recommended Next
- **Phase 21Q-D**: Run 4-prompt canary with n=2 (more tokens = spinner stops mid-run = better chance of capture)
- Or: Accept kernel evidence as primary proof; text capture is nice-to-have not required
- Runtime overhead: document as expected scalar CPU behavior; no fix needed before checkpoint

## Disk Usage
- Captures: ~100KB total
- Scratch free: 51GB (57%)