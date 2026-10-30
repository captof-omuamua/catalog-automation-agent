# Parity check, FP8 weights (QUANT=fp8): new tagger client vs the tagger project (dev, 2000 products)

| | Tagger project | This client |
|---|---|---|
| Accuracy | 86.4% | 86.2% |
| Auto-accepted at confidence >= 0.72 | 77.1% | 77.5% |

- Same answer as the tagger project: **97.3%** (1947/2000)
- Mean absolute confidence difference: 0.0215
- Failed or unparseable answers: 0
- Time: 46 s (43.9 products/s, 32 parallel requests)
