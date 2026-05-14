# Phase 10E-7S: Custom Op Sidecar-Read-Only Test

## Status
NOT YET RUN — pending bounded fill test and build fix.

## What This Test Proves
Read sidecar data but do NOT use it for PRT computation. Write bounded fill instead.

Steps inside custom op:
1. Read sidecar pointer (no dereference needed)
2. Log sidecar values (read-only)
3. Write bounded fill to dst

## Why This Test Matters
If path fragments still appear here, the sidecar loading or metadata storage is corrupting memory (struct aliasing, pointer corruption).

If path fragments disappear, the corruption is specifically in the PRT computation loops (using sidecar weights to compute output).

## Expected Outcome
- With sidecar read + bounded fill: determines if sidecar loading corrupts
- Will help isolate: loader corruption vs compute corruption