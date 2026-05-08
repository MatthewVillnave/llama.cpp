# Public Claims Boundary

## Safe Public Wording

Use these phrases — they are consistent with the measured evidence:

- ✅ "on the measured CPU setup"
- ✅ "in tested prompt suites"
- ✅ "near-native throughput"
- ✅ "research checkpoint"
- ✅ "not production-ready"
- ✅ "preserved tested output quality"
- ✅ "measured Qwen2.5 Q4_K_M only"
- ✅ "Packed INT8 sidecars let PRT..."
- ✅ "The active replacement path survived measurement."
- ✅ "Float32 sidecars were too memory-heavy."

## Avoid These Phrases

These are not supported by evidence and would constitute overclaiming:

- ❌ "PRT beats llama.cpp"
- ❌ "Solves CPU inference"
- ❌ "Production-ready"
- ❌ "Universal speedup"
- ❌ "Works on all models"
- ❌ "GPU comparable"
- ❌ "Lossless"
- ❌ "Zero degradation everywhere"
- ❌ "Exacts match in all cases"
- ❌ "Faster than native in all cases"
- ❌ "Works at any model size"
- ❌ "Deployment-ready"

## Approved Short Claim (copy-paste ready)

> **"Packed INT8 sidecars let PRT preserve tested output quality while recovering near-native llama.cpp throughput on Qwen2.5-3B and 7B in my measured CPU setup."**

## What to Say Instead of Overclaiming

| Instead of... | Say... |
|--------------|--------|
| "PRT is faster" | "Near-native throughput was measured" |
| "It works" | "Tested quality was preserved in measured suites" |
| "Universal" | "On Qwen2.5 Q4_K_M in my CPU setup" |
| "Production-ready" | "Research checkpoint only" |
| "Solves CPU inference" | "Representation design matters as much as replacement logic" |

## Claim Categories

### Allowed with scope qualifier
- Near-native throughput — must say "on measured CPU setup"
- Tested quality preserved — must say "in measured prompt suites"
- Native parity at 3B — must say "in 10 independent runs, measured setup only"
- 7B near-native — must say "0.993× avg in 8-prompt suite, measured setup only"

### Forbidden unconditionally
- Any claim without a scope qualifier
- Any production or deployment claim
- Any GPU comparison
- Any claim about model sizes not tested
- Any claim about task types not tested
