# PRT_PHASE10E7_CAPTURED_CANARY_VERDICT

## Test Configuration
- Prompt: "Write one sentence about CPUs."
- n_predict: 3
- temp: 0, seed: 42, CPU inference
- Baseline: llama-completion (standard float path)
- PRT: llama-phase10e0-layer0 (all 36 layers with |W| sidecars)

---

## Final Response Format

1. **n_predict tested:** 3

2. **Baseline completed:** YES

3. **PRT completed:** YES

4. **Outputs captured:** YES (baseline only — PRT output is garbage)

5. **Replacement count:** 144 (36 layers × 4 calls)

6. **Fallback count:** 0

7. **Wrong-sidecar count:** 0

8. **Fragile layers touched:** NO

9. **Quality:**
   - **baseline output:** "CPUs," (coherent English)
   - **PRT output:** "amup/prt_sidecars/ffn_up_layer35_prt.bin\n.Act/prt_sidecars/ffn_up_layer35_prt.bin\n includedsidecars/ffn_up_layer35_prt.bin" (GARBAGE — path fragments in output buffer)
   - **PRT coherent:** NO
   - **visible degradation:** YES (complete garbage)
   - **repetition loops:** NO

10. **Timing:**
    - **baseline latency:** 1.92s
    - **PRT latency:** 20.76s
    - **baseline tok/s:** 1.56
    - **PRT tok/s:** 0.14
    - **wall-clock speedup:** 10.8x SLOWER

11. **Verdict:** **FAIL**

12. **Is Phase 10F broader benchmark allowed:** **NO**

---

## Verdict Reasoning: FAIL

### Why FAIL

1. **Output is complete garbage** — the PRT output strings contain sidecar path fragments (`ffn_up_layer35_prt.bin`), which is a strong indicator of **memory corruption** in the PRT output path

2. **Not expected PRT behavior** — PRT should produce different but coherent text. This produces gibberish with path fragments.

3. **Slowdown is extreme** — 10.8x slower (20.76s vs 1.92s for 3 tokens). Even accounting for model load, this is dominated by PRT overhead plus potentially a corruption-retry loop.

### What's Working
- All 36 sidecars load correctly
- Zero fallbacks
- Zero wrong-sidecar events
- No crash

### What's Broken
- **Memory corruption**: Output buffer `buf[256]` is being overwritten with path-like bytes
- **Output quality**: Garbage instead of coherent text
- The path fragment "ffn_up_layer35_prt.bin" appearing in output is the smoking gun

### Required Fix Before Phase 10F
1. **Investigate memory corruption**: Examine PRT custom op output buffer allocation and writes
2. **Check `ggml_backend_tensor_set`**: Is it writing to the correct tensor memory?
3. **Verify stack variable safety**: Is `buf[256]` on the stack being corrupted by adjacent memory writes?
4. **Add bounds checking**: Verify `prt_output.data()` size before writing to tensor
5. **Isolate the corruption source**: Run with ASAN/UBSAN if available

### Not a MAYBE
This is not MAYBE because the output is not "degraded" — it is completely broken. The presence of path fragments in the output buffer is a definitive memory corruption indicator.