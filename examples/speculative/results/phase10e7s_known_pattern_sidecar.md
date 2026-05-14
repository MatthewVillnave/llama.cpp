# Phase 10E-7S: Known Pattern Sidecar

## Status
NOT YET RUN — pending build fix.

## Planned Test
Create a tiny synthetic sidecar file with known constant values (e.g., all 1.0f or sequential 0.001*i pattern), run standalone PRT compute, and verify exact expected output.

## Purpose
Isolates whether corruption is in file loading vs compute indexing. If compute is correct, output will match expected values exactly.

## Configuration
- Sidecar: synthetic file with planes all set to 0.1f
- Activation X: all 1.0f (2048 elements)
- Expected Y: for each j, Y[j] = sum_k 1.0 * 0.1 = 2048 * 0.1 = 204.8 (since all k pass T2)
- Compare actual vs expected within tolerance

## Status
Awaiting build fix and execution.