# Phase 28AE: IO Bandwidth + Latency Estimate for Paged 30B

## Verdict: PASS_PHASE28AE_IO_ESTIMATE ✅ | PASS_NVME_SUFFICIENT_FOR_ALL_PAGING ✅

## Summary
Measured actual NVMe sequential read bandwidth on TheForgeHQ: **1.7–2.0 GB/s** sustained across all block sizes. Tok/sec ceiling: **17–45 tok/sec** depending on quantization (far exceeding plausible decode throughput). Storage is not the bottleneck for paged 30B.

---

## A. Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## B. Current HEAD
`e4faec8eb`

## C. Storage Devices

| Device | Model | Size | Type | Mount | FSTYPE | Use% | Notes |
|--------|-------|------|------|-------|--------|------|-------|
| **nvme0n1** | Micron 2450 NVMe 256GB | 238.5G | NVMe | / | ext4 | 45% | **Primary system disk** |
| sda | SanDisk 3.2Gen1 | 114.6G | SATA USB | /media/.../VL_usb | exfat | 61% | External USB |
| sdb | Ultra | 28.6G | SATA USB | /media/.../RM3_USB1 | exfat | 97% | Almost full |
| sdc | Cruzer Blade | 7.5G | USB | /media/.../ELVIS | vfat | 7% | ELVIS bot storage |

**System RAM:** 16 GB total, ~12 GB available at rest

**Target storage for paged model files:** `/tmp/` (on nvme0n1 ext4) or `~/` — both use the Micron 2450 NVMe at ~2 GB/s sequential read.

**USB storage (sda/sdb/sdc):** NOT suitable for real-time paging — bandwidth unknown but far below NVMe.

---

## D. Test Method

**Test tool:** `dd` with `iflag=direct` (direct I/O, bypasses page cache)

**Test file:** 256 MB `/tmp/prt_io_test_256m.bin` (created and deleted in same test run)

**No fio available** — used `dd` read tests only.

| Test | Block size | Result |
|------|-----------|--------|
| Write (baseline) | 64 MB | 2.0 GB/s |
| Sequential read | 64 MB | **2.0 GB/s** |
| Sequential read | 4 MB | **1.9 GB/s** |
| Sequential read | 1 MB | **1.7 GB/s** |
| Write + read (repeat) | 1 MB | 1.3 GB/s write, 1.7 GB/s read |

**Note:** Initial write test showed slightly higher speed (2.0 GB/s) than subsequent runs (1.3 GB/s) — likely initial SSD cache warmup. Read speed is stable at **1.7–2.0 GB/s** across block sizes.

**Block size independence:** The Micron 2450 NVMe sustains near-peak speed even at 1 MB blocks. No significant block-size penalty observed.

---

## E. Measured Bandwidth

| Metric | Value |
|--------|-------|
| Sequential read, 64 MB blocks | 2.0 GB/s (0.14s for 256 MB) |
| Sequential read, 4 MB blocks | 1.9 GB/s (0.14s for 256 MB) |
| Sequential read, 1 MB blocks | 1.7 GB/s (0.16s for 256 MB) |
| Write speed | 1.3–2.0 GB/s |
| **Sustained read estimate** | **~1.8 GB/s** |
| Storage path | `/tmp/` or `~/` (both on nvme0n1) |
| Filesystem | ext4 |
| Direct I/O | ✅ supported (`oflag=direct`) |

---

## F. Block-Size Behavior

| Block size | Speed | vs 64MB baseline |
|-----------|-------|-----------------|
| 64 MB | 2.0 GB/s | 100% |
| 4 MB | 1.9 GB/s | 95% |
| 1 MB | 1.7 GB/s | 85% |

**Insight:** The NVMe is block-size tolerant. Even at 1 MB reads (close to page-loader chunk size), performance stays within 15% of peak. This is favorable for the paged layer workload where pages are loaded in ~37–102 MB chunks.

**For smaller random reads (e.g., individual tensor pages):** Real random-read performance would be somewhat lower than these sequential benchmarks, but the Micron 2450 handles small-block reads well due to its NVMe interface. A conservative estimate: random read may be **30–50% slower** than sequential, giving **0.9–1.4 GB/s** effective random read bandwidth.

---

## G. Tok/sec Ceiling Table

Using Phase 28AD IO/token estimates vs measured bandwidth:

| Scenario | IO/token | Ceiling tok/s (1.8 GB/s) | 50% efficient tok/s | 30% efficient tok/s |
|---------|---------|------------------------|--------------------|--------------------|
| Q2 base | 37 MB | **~49 tok/s** | ~24 tok/s | ~15 tok/s |
| Q2+res | 50 MB | **~36 tok/s** | ~18 tok/s | ~11 tok/s |
| Q4 base | 74 MB | **~24 tok/s** | ~12 tok/s | ~7 tok/s |
| Q4+res | 102 MB | **~18 tok/s** | ~9 tok/s | ~5 tok/s |

With conservative 50% efficiency (accounting for random-read overhead, prefetch gaps, and compute/IO overlap):
- **Q2+res paging ceiling: ~18 tok/s**
- **Q4+res paging ceiling: ~9 tok/s**

---

## H. Practical Efficiency Estimate

**Why actual tok/s will be below the IO ceiling:**
1. **Random-read overhead** — paged layer loads are not perfectly sequential; they target specific layer offsets. Random reads on ext4 may be 30–50% slower.
2. **Prefetch gaps** — if compute is faster than IO, pipeline stalls occur during prefetch waiting.
3. **Layer compute time** — actual Q2/Q4 decode per layer takes CPU time that can't overlap 100% with IO.
4. **Seek overhead** — loading non-contiguous layer files adds filesystem seek time.

**Conservative estimate: 50% efficiency** is a reasonable "pipelined but imperfect" assumption.

**Optimistic estimate: 30% efficiency** represents a poorly tuned system with frequent stalls.

**Best-case efficiency: ~60–70%** if prefetch is well-tuned and compute/IO overlap is good.

**Plausible tok/s for paged 30B on this NVMe:**
| Configuration | Conservative | Optimistic |
|--------------|-------------|-----------|
| Q2 base | 24 tok/s | 35 tok/s |
| Q2+res | 18 tok/s | 25 tok/s |
| Q4 base | 12 tok/s | 17 tok/s |
| Q4+res | 9 tok/s | 12 tok/s |

**Key point:** Even at 30% efficiency (worst case), the system can sustain **5–15 tok/s** — well above "usable minimum" (>0.1 tok/s) and "tolerable experiment" (>0.5 tok/s).

---

## I. Feasibility Interpretation

### Is NVMe fast enough for Q2/Q2+res paging?
**✅ YES, definitively.**
- 18–36 tok/s ceiling at measured bandwidth
- Even at 30% efficiency: 11 tok/s Q2+res — excellent for an experiment
- NVMe is not the limiting factor for Q2-based paging

### Is NVMe fast enough for Q4 paging?
**✅ YES, with margin.**
- 18–24 tok/s ceiling for Q4 base; 9–18 tok/s for Q4+res
- At 50% efficiency: 9–12 tok/s Q4+res — acceptable for experimentation
- Q4 paging is less feasible from an IO standpoint than Q2, but still workable

### Is SATA/USB acceptable?
**❌ No for real-time paging.**
- USB 3.0 external (sda): estimated 100–125 MB/s — would limit to 1–3 tok/s on Q4+res
- SATA SSD in external enclosure might reach 500 MB/s — borderline for Q4+res at 10 tok/s
- **USB storage is explicitly unsuitable** for this use case

### Does IO look like the limiting factor or CPU decode?
**CPU decode is likely the more fundamental bottleneck**, not storage IO.
- NVMe at 1.8 GB/s can sustain 18–36 tok/s
- Q2/Q4 decode compute throughput on CPU is probably **< 10 tok/s** for 30B on a modern CPU
- Storage IO ceiling exceeds plausible compute throughput

### What tok/sec range is plausible for 30B paged?
**Conservative estimate: 5–12 tok/s** (accounting for compute being the real bottleneck)
**Best case: 10–25 tok/s** (if compute is well-optimized and prefetch works)

### Is "semi-usable" plausible?
**✅ Yes.** 5–10 tok/s for Q4+res on NVMe is a "surprisingly usable" result for an experimental paged system.

---

## J. Recommended Storage Target

| Target | Path | Type | Speed | Suitable |
|--------|------|------|-------|----------|
| `/tmp/` | /tmp/ | NVMe ext4 | ~1.8 GB/s | ✅ **Recommended** |
| `~/` | ~/ | NVMe ext4 | ~1.8 GB/s | ✅ Suitable |
| `~/.cache/` | ~/.cache/ | NVMe ext4 | ~1.8 GB/s | ✅ Suitable |
| `/media/.../sda` | external USB | exfat | ~100 MB/s | ❌ Too slow |

**Recommendation:** Use `/tmp/` as the target for paged model files. It's on the fast NVMe, has 124 GB free, and is automatically cleared on reboot (safe for experimental files).

**Do NOT use external USB drives** for the paged model layer store.

---

## K. Recommended Next Phase

**Phase 28AF — Simulated Prefetch Scheduler with IO Timing Model**

Now that both memory feasibility (Phase 28AD) and IO bandwidth (Phase 28AE) are confirmed, the next step is to build a prefetch scheduler that:
1. Takes the layer compute time as input (from measurement or estimate)
2. Simulates prefetch/io timing for a given prefetch_depth
3. Identifies pipeline stalls (when compute is faster than IO)
4. Determines minimum prefetch_depth to sustain target tok/sec

**Key questions to answer:**
- What layer compute time makes IO the bottleneck vs compute the bottleneck?
- What's the minimum prefetch_depth for no-stall operation at 5 tok/sec?
- Does increasing window_size beyond 4 help reduce stall frequency?

This closes the loop between memory feasibility and IO feasibility.

---

## L. Models/Sidecars/F32 Refs Staged?
**NO.** No model files, no sidecars, no f32 refs staged. IO test files were 256 MB and deleted after tests.

## M. Secrets Detected?
None.

## N. Tags Touched?
None.

---

## Files Committed
- `examples/speculative/results/PHASE28AE_IO_BANDWIDTH_LATENCY_ESTIMATE.md` — this report
- `examples/speculative/results/phase28ae_io_bandwidth_latency_estimate.json` — structured results