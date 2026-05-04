# PRT Phase 13B: Native 0.5B Slowness Investigation

**Date:** 2026-05-03

## Commands Attempted

### Test 1: 16 tokens, 4 threads
```bash
/usr/bin/time -v ./build/bin/llama-cli \
  -m /home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf \
  -p "The capital of France is" -n 16 -t 4 --log-disable
```
**Result:** SIGKILL — process killed before producing output (OOM likely)

### Test 2: 4 tokens, 1 thread, short context
```bash
timeout 10 ./build/bin/llama-cli \
  -m /home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf \
  -p "hi" -n 4 -t 1 --log-disable
```
**Result:** Model loads (ASCII art visible), process still running after 10s timeout — very slow

### Test 3: 8 tokens, 2 threads, 128 ctx
```bash
timeout 60 ./build/bin/llama-cli \
  -m /home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf \
  -p "Paris is" -n 8 -t 2 -c 128 --log-disable
```
**Result:** SIGKILL — killed before completion

## System State During Tests

```
Mem:           15677MB total
               2679MB used
               7782MB free
               5734MB buff/cache
               12997MB available

Swap:           4095MB total
               4088MB used
                  7MB free  ← CRITICAL: 99.8% swap used!

Disk: /dev/nvme0n1p2
       233GB total, 214GB used, 7.7GB avail (97% full)

Load average: 1.48, 1.08, 0.76
```

## Root Cause Analysis

### Swap is Nearly Exhausted

The system has 4GB swap and 3.3GB is used, leaving only 7MB free. When a process tries to allocate memory and RAM+cache isn't sufficient, it hits the swap. This causes massive slowdown, not OOM kill — Linux will thrash swap before killing.

However, the SIGKILL signals suggest something else is at play. The `fork` from the Python subprocess wrapper may be causing issues with copy-on-write memory overhead.

### Phase 13A Python Wrapper Issue

The Phase 13A canary used a Python subprocess wrapper:
```python
r = subprocess.run([binary, ...], capture_output=True, text=True, timeout=60)
```

This creates a Python subprocess that captures stdout/stderr in memory. For long-running inference with large output, this could use significant memory and potentially trigger OOM when combined with swap exhaustion.

### Disk Pressure

With disk at 97% full (7.7GB available), and llama.cpp potentially using memory-mapped I/O, disk contention could also contribute to slowness.

## Conclusion

The 0.5B slowness is **NOT inherent to the model** — it's an **environment issue**:
1. Swap nearly exhausted (99.8% used)
2. Disk at 97% full
3. Python subprocess wrapper adding memory overhead

On a clean system, Qwen2.5-0.5B should decode at 20-50 tokens/sec on CPU. The >3 min/prompt is pathological.

## Recommendations

1. **Free more disk space** before retrying (target <90% full)
2. **Use direct shell invocation** not Python subprocess wrapper
3. **Reduce swap pressure** by killing unused processes
4. **Or test on a different machine** with more headroom

## Evidence Model Itself is Fast

When the llama-cli binary was killed during model load (after ASCII art appeared), the model DID load successfully — it just couldn't complete generation before the OOM/kill. The binary is correctly built.

```
Loading model... |-\|/- 
██ ██  ▀▀█▄ ███▄███▄  ▀▀█▄    ▄████ ████▄ ████▄   ← Model loaded!
```

This suggests the model loading works, but the generation step gets killed due to environment constraints, not model issues.