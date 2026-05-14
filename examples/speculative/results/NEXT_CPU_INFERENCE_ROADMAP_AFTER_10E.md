# NEXT_CPU_INFERENCE_ROADMAP_AFTER_10E.md

## CPU Inference Strategy After Phase 10E

### Phase 10E Outcome

- **PRT active generation:** ARCHIVED / BLOCKED
- **ggml_map_custom2 integration:** UNRESOLVED memory corruption
- **PRT_3P standalone kernel:** VALIDATED (offline)

---

## Recommended Roadmap

### A. Keep SpecBenchCPU as the Validated CPU Win ✅

**Status:** Continue as primary CPU speculative decoding path

SpecBenchCPU (speculative decoding harness) is validated and produces correct output. This remains the CPU-side acceleration win.

**Actions:**
- Continue using SpecBenchCPU for CPU inference speedup
- Keep it as the reference implementation for CPU-side wins
- Do not replace with PRT until PRT integration is fixed

---

### B. Keep PRT_3P as Standalone Kernel/Storage Research ✅

**Status:** Continue as research/exploration

PRT_3P (offline sidecar construction + standalone PRT kernel) is validated and clean. Continue this line of research.

**Actions:**
- Continue PRT_3P sidecar construction pipeline
- Continue standalone kernel validation
- Publish/present PRT_3P as a validated offline kernel research result
- Do NOT claim end-to-end PRT works

---

### C. Do NOT Continue llama.cpp Custom-Op Integration Unless...

**Status:** BLOCKED

Do not continue the current llama.cpp `ggml_map_custom2` integration path unless you are deliberately starting a **new low-level ggml backend investigation**.

**If continuing:**
- Accept that this is a new research thread, not a continuation
- Start from scratch with fresh git state
- Budget significant time for ggml internals debugging
- Consider whether this is the best use of time vs. other paths

**If pausing:**
- Archive the current branch
- Return to it only when ready to invest in ggml internals deep-dive
- Do not let this block other PRT research paths

---

### D. If PRT Resumes Later — Preferred Implementation Route

**Status:** Future consideration

If PRT active generation research resumes, prefer a **cleaner implementation route** over the current `ggml_map_custom2` path:

**Option 1: External Sidecar Engine**
- Compute PRT outside llama.cpp in a separate process or thread
- Feed PRT-corrected output back into llama.cpp via a clean interface
- Avoids ggml custom op entirely

**Option 2: Lower-Level ggml Backend Investigation**
- Deep-dive into ggml tensor lifecycle and buffer management
- Understand why ggml_map_custom2 corrupts memory when sidecar is loaded
- Fix at the ggml level (requires significant time investment)

**Option 3: Alternative Graph Integration**
- Use ggml hooks that don't involve custom ops
- Patch the graph at a different level (e.g., after matmul, before activation)
- Requires llama.cpp graph architecture knowledge

**Option 4: Fork llama.cpp**
- Fork llama.cpp and modify the FFN computation directly
- Replace the FFN up-projection with PRT-corrected computation
- Most invasive but most control

---

## Summary Decision Table

| Path | Status | Recommendation |
|------|--------|----------------|
| SpecBenchCPU | ✅ VALIDATED | Continue as primary CPU win |
| PRT_3P standalone | ✅ VALIDATED | Continue as research |
| llama.cpp ggml_map_custom2 | ❌ BLOCKED | Do not continue unless deliberately starting new investigation |
| External sidecar engine | 🔄 FUTURE | Preferred restart route for PRT |
| Lower-level ggml debug | 🔄 FUTURE | If ggml internals investment justified |

## Immediate Actions

1. ✅ Archive Phase 10E branch
2. ✅ Mark PRT active generation as BLOCKED
3. ✅ Continue SpecBenchCPU as primary path
4. ✅ Continue PRT_3P as research
5. ⏸️ Decide later: whether to invest in ggml internals or pursue external engine