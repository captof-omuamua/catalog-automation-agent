# Parity check: new tagger client vs the tagger project (dev, 2000 products)

| | Tagger project | This client |
|---|---|---|
| Accuracy | 86.4% | 86.6% |
| Auto-accepted at confidence >= 0.72 | 77.1% | 77.5% |

- Same answer as the tagger project: **99.5%** (1989/2000)
- Mean absolute confidence difference: 0.0055
- Failed or unparseable answers: 0
- Time: 71 s (28.1 products/s, 32 parallel requests)
