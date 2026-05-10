# PRT Phase 16 — Public X Thread

**Thread tone:** Clear, confident, technically grounded. No hype, no GPU comparison, no production claims.

---

## Tweet 1 (Hook)
CPU inference doesn't have to mean "run the full dense GPU-shaped transformer badly."

There's another approach: change *what* gets computed and *how* weights are represented — rather than brute-forcing GPU-shaped math on CPU hardware.

---

## Tweet 2 (SDI concept)
This idea has a name: **Sub-Dense Inference (SDI)**.

It's the philosophy that CPU-native inference viability comes from smarter computation and representation — not from better hardware.

The goal isn't to beat GPUs. It's to make CPU-only systems genuinely useful for local AI.

---

## Tweet 3 (PRT is the proof path)
**PRT** (Progressive Residual / Packed-sidecar path) is the concrete proof path for SDI in llama.cpp.

Instead of loading full-weight FFN layers on every forward pass, PRT uses pre-computed compressed sidecar files at startup — and swaps them into the kernel at generation time.

---

## Tweet 4 (The progression)
The path to 14B went through failures:

- **float32 sidecars:** preserved behavior but were too heavy (memory traffic dominated)
- **INT4 per-row:** too lossy — offline parity failed consistently
- **INT8 packed sidecars (Phase 14):** recovered near-native throughput on 0.5B, 3B, 7B
- **INT6 packed sidecars (Phase 15):** compressed further while preserving quality

Each step taught us something about the representation bottleneck.

---

## Tweet 5 (14B milestone)
**Phase 16 brought PRT to Qwen2.5-14B Q4_K_M on a CPU-only machine.**

Testing:
- Tiny canary (8 tokens) ✅
- 8-prompt quality validation ✅
- Longer-generation (n=320 prose) ✅
- Larger-context (c=2048 factual retrieval) ✅

All tested behaviors preserved. No collapses, no repetitions.

---

## Tweet 6 (What the numbers mean)
**14B INT6 on CPU-only hardware:**
- Generation throughput: ~0.977–1.009× native in the tested suite
- Memory: stable throughout (12GB available, no OOM)
- Sidecars: 40/40 loaded via mmap

The generation rate feels the same as native to a user. That's the point.

---

## Tweet 7 (Careful framing)
**What this IS:**
- A credible experimental SDI path
- Tested behavior preserved on Qwen2.5-14B in Matt's measured CPU setup
- Near-native generation throughput in the tested prompt suite

**What this is NOT:**
- Production ready
- GPU comparison
- Universal claim across all models/prompts
- A drop-in replacement for llama.cpp

---

## Tweet 8 (Why it matters)
On GPU-less hardware, local inference often means painful tradeoffs: tiny context, quantized to death, slow generation.

PRT suggests that changing the *representation* — not just the quantization depth — can recover near-native behavior while using less memory bandwidth.

That makes local inference genuinely more usable on the hardware people already have.

---

## Tweet 9 (What remains)
Open problems:
- Backend/ggml integration (no hardcoded model sizes)
- Broader benchmark suite
- Repeatability validation from clean state
- Production hardening
- Larger model feasibility (30B+)

The pipeline is validated at 14B. The engineering is not done.

---

## Tweet 10 (Closing)
**PRT is the proof path. SDI is the philosophy.**

The result: sidecar-backed compressed FFN replacement works at 14B scale on CPU-only hardware — carefully scoped, experimentally validated, not production-ready, but real.

If you're interested in CPU-native inference research, PRT is worth watching.

*(Thread link: PRT_PHASE16M_14B_INT6_EXPERIMENTAL_CHECKPOINT)*

---

## One-paragraph summary for copy-paste
> PRT is now a credible experimental SDI path: sidecar-backed compressed FFN replacement preserved tested behavior up to Qwen2.5-14B Q4_K_M on a CPU-only machine, with near-native generation throughput in the measured prompt suite. This suggests CPU-only inference may be more viable than the usual dense-transformer-on-CPU framing implies — if the runtime and model representation are designed around CPU constraints. The pipeline remains experimental and scoped to tested prompts/hardware.
