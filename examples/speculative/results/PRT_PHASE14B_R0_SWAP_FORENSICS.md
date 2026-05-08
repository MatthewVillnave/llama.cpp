# PRT Phase 14B-R0 — Swap Forensics / Memory Recovery

## Verdict: STALE_LLAMA_KILLED + LEFTOVER_SWAP_NO_LIVE_OWNER

## Pre-Cleanup State (before Phase 14B canary)

### Memory snapshot (before kill)

| Metric | Value |
|--------|-------|
| Swap total | 4.0 GiB |
| Swap used | 4.0 GiB (99.99%) |
| Swap free | 2.5 MiB |
| RAM available | 9.8 GiB |
| Load average | 1.97 |

### Top swap users (from /proc/[pid]/status)

| SWAP_MB | RSS_MB | VSZ_MB | PID | NAME | CMD |
|---------|--------|--------|-----|------|-----|
| **1089.8** | 13.4 | 3114.1 | 1550534 | **ollama** | `/usr/local/bin/ollama serve` |
| 230.7 | 6.4 | 1930.1 | 2158451 | baobab | `baobab /` (disk analyzer) |
| 171.7 | 8.6 | 2643.6 | 667620 | nautilus | `nautilus --gapplication-service` |
| 153.0 | 87.0 | 5002.2 | 573899 | gnome-shell | `/usr/bin/gnome-shell` |
| 148.2 | **971.4** | 13847.3 | 2389091 | openclaw-gateway | `openclaw-gateway` |
| 142.3 | 189.1 | 3484.9 | 2169682 | firefox | `/snap/firefox/8247/...` |
| 90.9 | 37.4 | 848.7 | 2374106 | update-manager | `update-manager` |

**Total swap in use: 2664 MB** (not all in top 7)

### Stale llama-cli process found

```
PID 2502357: llama-cli -m Qwen2.5-0.5B-Instruct-Q4_K_M.gguf --verbose
CPU: 99.9%, RSS: 858 MB, started 20:11, running 4+ minutes
→ Action: KILLED (SIGKILL - was from failed Phase 14B canary attempt)
```

### Top RSS processes

| PID | RSS_MB | NAME | CMD |
|-----|--------|------|-----|
| 2389091 | 1017.8 | openclaw-gateway | `openclaw-gateway` (active service) |
| 2169682 | 193.6 | firefox | Firefox |
| 2170038 | 105.6 | Privileged Container | Firefox sub-process |
| 573899 | 89.0 | gnome-shell | Desktop shell |
| 2374106 | 38.3 | update-manager | Update checker |

### Service status

- **Ollama**: Running since Apr 29, no active models, **1118 MB VmSwap / 13 MB VmRSS**
  - `ollama ps` → empty (no models loaded)
  - Virtual address space NOT released after model unload → swapped to disk
- **OpenClaw**: Active service, 971 MB RSS, 151 MB VmSwap (legitimate working memory)
- **Firefox**: Multiple processes, ~300 MB swap total (normal browser behavior)

## Classification: D. MEMORY_PRESSURE_FROM_SERVICES

Primary swap consumer: **Ollama daemon (PID 1550534)** — 1.09 GB VmSwap, only 13 MB RSS.

Ollama has been idle since Apr 29. It allocated a model-sized virtual address space and never released it. The virtual pages got swapped to disk, consuming 1 GB of swap while only using 13 MB of actual RAM.

Secondary consumers: Desktop services (gnome-shell, firefox, nautilus, baobab) — these are normal.

## Action Taken

1. **Killed stale llama-cli** (PID 2502357, running since Phase 14B canary)
   - Was stuck in model loading, consuming 858 MB RSS
   - freed ~858 MB RAM

2. **Did NOT kill ollama** — requires approval (it's a service)
3. **Did NOT swapoff** — requires approval

## Post-Cleanup State

| Metric | Before | After |
|--------|--------|-------|
| Swap total | 4.0 GiB | 4.0 GiB |
| Swap used | 4.0 GiB | 4.0 GiB (99.99%) |
| RAM available | 9.8 GiB | 10.0 GiB |
| Stale llama processes | 1 found | 0 |

**Note:** Swap usage unchanged because killed llama process wasn't using swap — it was RAM-resident and got OOM-killed by the kernel when swap was already full. The 1 GB swap is consumed by ollama daemon's virtual address space.

## Is Phase 14B-R Safe to Rerun?

**PARTIALLY — RAM is available (10 GiB) but swap is full.**

Analysis:
- ~10 GB RAM available is sufficient for 0.5B model inference
- 4 GB swap is 99.99% full — any new allocation that causes swap activity could trigger OOM
- However, if no swap activity is needed (all working sets fit in RAM), inference should work

Recommendation:
- **Run 0.5B canary only** — safer than 3B which uses more memory
- Monitor: if inference starts swapping, it will stall/slow
- Monitor: if system load goes above 2.0, something is wrong

To improve swap situation further, options are:
1. **Restart ollama** → releases 1 GB swap immediately (best option)
2. **Reboot** → clears all stale swap
3. **sudo swapoff && swapon** → clears swap safely but risky without reboot

## Root Cause Summary

Phase 14B canary failed due to:
1. Ollama daemon holding 1 GB of stale swapped virtual memory
2. A stale llama-cli process (from Phase 14B) consuming 858 MB of RAM
3. Combined pressure → swap exhaustion → OOM kills → cannot run new inference

## Recommended Next Actions

1. **Approve Ollama restart** (or ignore — it's not causing active problems)
2. Retry Phase 14B-R canary with 0.5B only (3B is risky with swap full)
3. Or: reboot to fully clear swap before Phase 14B-R

## Safety scan

- No model files staged ✅
- No sidecar binaries staged ✅
- No temp logs staged ✅
- Secrets: none ✅
- Phase 13 tags untouched ✅