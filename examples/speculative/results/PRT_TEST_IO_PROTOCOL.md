# PRT Test I/O Protocol

## Problem
llama-cli writes ANSI spinner/animation noise to stdout even in non-interactive mode with `--no-display-prompt`. This creates massive stdout files (500MB–2GB per run) that fill disk during PRT test suites.

## Solution
For all PRT test runs, redirect stdout to `/dev/null`. Capture stderr only.

## Protocol

### Standard PRT Test Run
```
./build/bin/llama-cli \
  -m /path/to/model.gguf \
  -p "PROMPT" \
  -n 16 \
  --temp 0 \
  -c 256 \
  -t 4 \
  --no-display-prompt \
  > /dev/null \
  2> /tmp/<test>_p<N>_<mode>_stderr.txt
```

### What goes where
- **stdout** → `/dev/null` (spinner noise only)
- **stderr** → captured file (quality text, PRT_SHAPE, sidecar load, counters, timing)

### Validation after each run
```bash
ls -lh /tmp/<test>_p<N>_*_stderr.txt
```
- If stderr > 20MB → stop and diagnose
- If any stdout file created → investigate (shouldn't happen)

### Before/after every suite
```bash
df -h /
```
Never let disk go above 85% during testing.

## What to capture from stderr
- PRT_SHAPE (n_layer, M, N)
- Sidecars loaded count
- Force-native layers
- Quality text (actual generated output)
- tok/s (if available)
- PRT debug logs

## When stdout capture IS needed
Only for tiny single-prompt debug probes. Run, extract, delete immediately:
```bash
./build/bin/llama-cli ... > /tmp/debug_probe.txt 2>/dev/null
# extract text
strings /tmp/debug_probe.txt | grep -v "^>\|^\[" | ...
rm -f /tmp/debug_probe.txt
```

## Never do
- Never save llama-cli stdout during PRT test suites
- Never run with stdout going to a file without size monitoring
- Never let /tmp test output exceed 100MB total per test session

## Max sizes
| File type | Max size |
|-----------|----------|
| stderr capture per run | 20MB |
| Total /tmp per test session | 100MB |
| Individual stdout (debug only) | 10MB |

## Quick reference: before/after template
```bash
# Before
df -h /

# Run tests with > /dev/null throughout

# After
df -h /
find /tmp -size +10M -name "*.txt" -o -name "*.out" 2>/dev/null
```