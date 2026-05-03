# PRT Phase 12E: Verdict

| Field | Value |
|-------|-------|
| A. Branch | experimental/prt-route-a-phase12e-l11-l15 |
| B. Prompt suite size | 24 |
| C. Total runs | 72 |
| D. Native runs | 24 |
| E. L12+L15 runs | 24 |
| F. L11+L15 runs | 24 |
| G. L12+L15 avg speedup | 1.7869x |
| H. L11+L15 avg speedup | 1.8216x |
| I. L12+L15 median speedup | 1.7942x |
| J. L11+L15 median speedup | 1.8030x |
| K. Speedup delta | +1.94% |
| L. Token-0 match | L12+L15 20/22, L11+L15 20/22 |
| M. First-8 match | L12+L15 14/24, L11+L15 18/24 |
| N. JSON validity | L12+L15 0/4, L11+L15 0/4 |
| O. Collapse/repetition | 17 corruption instances across PRT runs |
| P. Counter cleanliness | PASS (callback_overwrites=0, fallback_calls as expected) |
| Q. Memory stability | PASS |
| R. Does L11+L15 clearly beat L12+L15? | NO |
| S. Should default change now? | NO |
| T. Verdict | **PARTIAL PASS** |

## Summary

L11+L15 avg speedup (1.8216x) vs L12+L15 (1.7869x): delta=+0.0347 (+1.94%)

Corruption: L11=9/24, L12=8/24, native=7/24

Token-0 match: both policies 90.9%

**Conclusion**: L11+L15 is +1.94% faster but shows slightly more corruption. Not a clear win over L12+L15.
