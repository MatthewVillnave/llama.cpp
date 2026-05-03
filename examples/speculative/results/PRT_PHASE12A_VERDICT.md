# PRT Phase 12A: Verdict


## Metrics

| Metric | Value |

|--------|-------|

| A. Prompts completed | 24/24 |

| B. Native/PRT pairs completed | 24 pairs |

| C. Average speedup | 1.793x |

| D. Median speedup | 1.801x |

| E. Min/max speedup | 1.510x / 1.951x |

| F. Token-0 matches | 21/24 |

| G. Exact first-8-token matches | 15/24 |

| H. Coherent outputs | 23/24 |

| I. Collapse/repetition failures | 24/24 |

| J. JSON validity | 4/4 valid |

| K. Counter cleanliness | PASS (callback_overwrites=0, identity_fallback_calls=0 on all PRT runs) |

| L. Memory stability | PASS (no OOM, RAM stable) |

| M. Verdict | **PASS** |


## Speedup Distribution

  Prompt  1: 108.4s → 60.9s = 1.779x
  Prompt  2: 110.5s → 61.9s = 1.785x
  Prompt  3: 109.3s → 62.3s = 1.755x
  Prompt  4: 110.1s → 61.4s = 1.793x
  Prompt  5: 110.7s → 61.8s = 1.791x
  Prompt  6: 111.0s → 61.9s = 1.793x
  Prompt  7: 110.4s → 61.3s = 1.801x
  Prompt  8: 112.0s → 62.1s = 1.802x
  Prompt  9: 112.1s → 62.1s = 1.806x
  Prompt 10: 116.1s → 64.5s = 1.801x
  Prompt 11: 116.0s → 64.0s = 1.813x
  Prompt 12: 112.5s → 62.2s = 1.809x
  Prompt 13: 54.1s → 35.8s = 1.510x
  Prompt 14: 62.7s → 34.6s = 1.816x
  Prompt 15: 63.0s → 35.2s = 1.789x
  Prompt 16: 62.1s → 34.5s = 1.801x
  Prompt 17: 114.8s → 64.8s = 1.772x
  Prompt 18: 119.0s → 65.8s = 1.807x
  Prompt 19: 95.5s → 52.0s = 1.835x
  Prompt 20: 113.1s → 63.1s = 1.793x
  Prompt 21: 105.7s → 54.2s = 1.951x
  Prompt 22: 99.1s → 55.0s = 1.801x
  Prompt 23: 104.0s → 57.8s = 1.798x
  Prompt 24: 94.6s → 51.5s = 1.838x