# PRT Phase 16B — 14B Feasibility Canary

## Verdict

**BLOCKED_MODEL_MISSING** ⛔

---

## Context

Phase 16A recommended 14B feasibility canary as the next step to test whether the PRT sidecar pipeline scales to a 14B-class model — directly testing Matt's CPU inference / SDI goal.

---

## Preflight Results

### Machine Health
| Field | Value |
|-------|-------|
| RAM total | 15 GB |
| RAM available | 11 GB |
| RAM used | 4.2 GB |
| Swap total | 4 GB |
| Swap free | **640 KB** (fully used) |
| Disk free | 126 GB ✅ |
| Ollama daemon | Running (minimal RAM, 13 MB) |
| openclaw-gateway | 1.3 GB RSS |

**Note:** Swap is fully allocated but 11 GB RAM is available. Sufficient for the initial model check.

### Model Check

| Field | Value |
|-------|-------|
| **Expected path** | `/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-14B-Instruct-Q4_K_M.gguf` |
| **Status** | **MISSING** |
| **Available disk** | 126 GB ✅ |

---

## Blocked — Model Not Downloaded

The 14B model has not been downloaded yet. This phase cannot proceed until:

1. Matt downloads `Qwen2.5-14B-Instruct-Q4_K_M.gguf` from HuggingFace
2. Saves to `/home/matthew-villnave/models/gguf/qwen2.5/`
3. Confirms the file exists and has a valid SHA256

### Estimated Download Size
- `Qwen2.5-14B-Instruct-Q4_K_M.gguf` ≈ **9 GB**

### Expected Next Steps
Once the model is downloaded:
1. Verify SHA256
2. Inspect metadata (layer count, hidden size, FFN dimensions)
3. Estimate sidecar sizes for INT8 and INT6
4. Run native tiny canary (`n=8, c=128`)
5. Generate one-layer sidecar probe
6. Generate INT6 sidecars (safer path given RAM constraints)
7. Run INT6 tiny PRT canary

---

## What This Phase Would Test

If the model were present, the canary would test:
- Can Qwen2.5-14B load on 15GB RAM machine without OOM?
- Can the PRT sidecar pipeline scale to 14B dimensions?
- Is INT6 sidecar approach feasible for larger models?
- Is RAM sufficient for INT6 sidecars (~13 GB total estimate)?
- Is INT8 sidecar approach too tight on current hardware?

---

## Recommendations

1. **Download the model** — Matt needs to acquire `Qwen2.5-14B-Instruct-Q4_K_M.gguf`
2. **After download:** Rerun Phase 16B from preflight
3. **If RAM becomes a concern:** Consider stopping Ollama daemon before running (`sudo systemctl stop ollama` or `pkill -f "ollama runner"`)

---

**Test environment:**
- Branch: `experimental/prt-phase14a-packed-sidecars`
- Commit: `e4e2ec1fc71699ff230a64d29725301fb4bdb6a5`
- Model expected at: `/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-14B-Instruct-Q4_K_M.gguf`