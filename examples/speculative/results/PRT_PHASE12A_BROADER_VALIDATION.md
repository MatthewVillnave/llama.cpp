# PRT Phase 12A: Broader Validation Results

| # | Prompt | n | Native Time | PRT Time | Speedup | Token 0 Match | First 8 Match | Coherence | Collapse | JSON Valid | Notes |

|---|--------|---|-------------|----------|---------|---------------|---------------|-----------|----------|------------|-------|

| 1 | Once upon a time in a... | 100 | 108.372s | 60.926s | 1.779x | True | True | PASS | PASS | - |  |
| 2 | In a small village near the mountains,... | 100 | 110.522s | 61.917s | 1.785x | True | True | PASS | PASS | - |  |
| 3 | The old machine began to hum when... | 100 | 109.306s | 62.283s | 1.755x | False | False | PASS | PASS | - |  |
| 4 | A scientist opened the notebook and foun... | 100 | 110.091s | 61.4s | 1.793x | True | False | PASS | PASS | - |  |
| 5 | What is the capital of France?... | 100 | 110.701s | 61.825s | 1.791x | True | True | PASS | PASS | - |  |
| 6 | Explain why the sky appears blue.... | 100 | 110.987s | 61.911s | 1.793x | True | True | PASS | PASS | - |  |
| 7 | What causes tides on Earth?... | 100 | 110.394s | 61.308s | 1.801x | True | True | PASS | PASS | - |  |
| 8 | Give a short explanation of photosynthes... | 100 | 111.983s | 62.132s | 1.802x | True | True | PASS | PASS | - |  |
| 9 | Write a Python function to reverse a lis... | 100 | 112.078s | 62.065s | 1.806x | True | True | PASS | PASS | - |  |
| 10 | Write a Python function that checks whet... | 100 | 116.081s | 64.45s | 1.801x | False | False | PASS | PASS | - |  |
| 11 | Explain what a hash map is in simple ter... | 100 | 116.018s | 64.004s | 1.813x | True | True | PASS | PASS | - |  |
| 12 | Write a small JSON parser example in Pyt... | 100 | 112.511s | 62.192s | 1.809x | True | True | PASS | PASS | - |  |
| 13 | Return a JSON object with keys name, sta... | 50 | 54.131s | 35.846s | 1.51x | True | False | PASS | PASS | PASS |  |
| 14 | Return only valid JSON describing three ... | 50 | 62.745s | 34.56s | 1.816x | False | False | PASS | PASS | PASS |  |
| 15 | Create a JSON array of three tasks with ... | 50 | 63.038s | 35.243s | 1.789x | True | True | PASS | PASS | PASS |  |
| 16 | Return a JSON object with a nested addre... | 50 | 62.075s | 34.475s | 1.801x | True | False | PASS | PASS | PASS |  |
| 17 | Answer in exactly three bullet points: w... | 100 | 114.759s | 64.77s | 1.772x | True | True | PASS | PASS | - |  |
| 18 | Summarize this in one sentence: CPUs are... | 100 | 118.999s | 65.846s | 1.807x | True | False | PASS | PASS | - |  |
| 19 | Give five short names for a CPU inferenc... | 100 | 95.465s | 52.025s | 1.835x | True | False | PASS | PASS | - |  |
| 20 | Explain PRT in one paragraph without hyp... | 100 | 113.149s | 63.095s | 1.793x | True | False | PASS | PASS | - |  |
| 21 | Repeat the word forge exactly five times... | 100 | 105.689s | 54.159s | 1.951x | True | True | FAIL | PASS | - |  |
| 22 | Translate 'the machine is awake' into Fr... | 100 | 99.08s | 55.009s | 1.801x | True | True | PASS | PASS | - |  |
| 23 | Classify this sentiment as positive, neu... | 100 | 103.988s | 57.843s | 1.798x | True | True | PASS | PASS | - |  |
| 24 | The company is a large... | 100 | 94.607s | 51.476s | 1.838x | True | True | PASS | PASS | - |  |