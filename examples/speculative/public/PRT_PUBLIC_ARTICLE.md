# PRT Phase 14: Packed Sidecars Made the Replacement Path Viable

## Opening

I've been testing whether CPU inference can be improved by changing the compute/weight path — not by forcing dense GPU-shaped inference onto CPUs, but by finding a better representation for the same computation.

PRT is an experimental active replacement path for FFN_UP in llama.cpp. Instead of relying on the model's stored float32 FFN_UP weights, PRT loads sidecar files with alternative weight representations and routes the relevant computation through those sidecars at runtime.

This is a lab result, not a production release. What follows is what the measurements showed.

---

## What Failed First

Phase 13 proved the active replacement path worked in principle:

- PRT replaced FFN_UP in the ggml graph and generated correct outputs
- Output quality was preserved on tested prompts
- Clean llama-cli integration worked
- No dirty harness contamination

But float32 sidecars were too memory-heavy. Memory bandwidth became the bottleneck, and throughput fell materially below native ggml. The idea was not rejected — the representation was.

---

## The Phase 14 Pivot

The fix was to switch from float32 to packed INT8 sidecars:

- **~4× smaller** per layer (int8 weights + per-row scales, combined in one file)
- Less memory traffic per token
- Same active replacement path
- Same llama-cli frontend

The hypothesis: if the bottleneck was representation weight, not replacement logic, a smaller representation should recover throughput while keeping quality.

---

## Results

| Model | Quality | Throughput |
|-------|---------|------------|
| Qwen2.5-0.5B (Q4_K_M) | 8/8 exact matches | INT8 ~1.81× faster than float32 PRT |
| Qwen2.5-3B (Q4_K_M) | 8/8 semantic matches, 0 degradations | **1.000×** native avg (10 independent runs) |
| Qwen2.5-7B (Q4_K_M) | 8/8 exact or semantic matches, 0 degradations | **0.993×** native avg (full 8-prompt suite) |

On the measured CPU setup:
- 3B: native 21.0 t/s → INT8 PRT 21.0 t/s, 1.000× repeatably
- 7B: native 8.75 t/s → INT8 PRT 8.69 t/s, 0.993× avg, 0.989× median

---

## Why This Matters

CPU inference bottlenecks are often memory-traffic problems, not compute problems. Float32 sidecars were too heavy — they worked correctly but the data movement cost dominated. INT8 sidecars preserved the same replacement logic but reduced the memory footprint enough to recover native-class throughput.

This suggests that representation design matters as much as replacement logic. The active replacement path is viable. The float32 representation was not.

---

## What This Does Not Mean

- ❌ Not production-ready
- ❌ Not a universal speedup
- ❌ Not a GPU comparison
- ❌ Not all tasks or prompt types
- ❌ Not larger than 7B (untested)
- ❌ Not deployment-ready

Results are scoped to: measured Qwen2.5 Q4_K_M models, measured CPU setup, measured prompt suites.

---

## Next Work

- **Longer-context / larger-n stability** — Tests whether native-parity holds at production-scale generation lengths
- **INT4 sidecar prototype** — Could halve memory traffic again; requires offline parity validation first
- **Native ggml/backend integration** — Removes custom callback limitations; highest architectural value
- **Optimization toward native-beating throughput** — Highest ceiling; depends on clean 7B validation as baseline
- **Broader benchmark suites** — More prompt types and model configurations

---

## Closing

The active replacement path survived measurement. The float32 representation did not. Packed sidecars made the idea viable.

*All measurements are on the measured CPU setup for Qwen2.5 Q4_K_M only. This is a research checkpoint, not a production claim.*
