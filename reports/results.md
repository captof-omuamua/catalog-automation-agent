# Results

## 1. Filling in attributes (end to end through the agent)

| Run | Products | Model accuracy | Auto-tagged | Accuracy of auto-tagged | Sent to review | Write fidelity | Catalog accuracy after review |
|---|---|---|---|---|---|---|---|
| dev, bf16 | 2000 | 86.3% | 77.2% | 94.9% | 22.8% | 100.0% | 96.1% |
| **test, bf16** | 1610 | 85.7% | 79.2% | 94.6% | 20.8% | 100.0% | 95.7% |
| test, FP8 | 1610 | 84.9% | 79.6% | 94.4% | 20.4% | 100.0% | 95.5% |

## 2. Catalog QC (20% of listings state a wrong value)

| Split | Cut-off | Wrong listings caught | Precision of flags | False alarms on correct listings |
|---|---|---|---|---|
| dev (cut-off chosen here) | 0.86 | 70.0% | 90.1% | 1.8% |
| **test** | 0.86 | 67.8% (208 of 307) | 90.0% | 1.8% |

## 3. Reliability (1,000 products, faults injected, process killed)

- Store: 10% of requests fail (503); 5% of writes are applied but the reply is lost (500)
- Agent killed with SIGKILL after 15 s (431 products written), then the batch re-run
- Outcome: 761 written + 239 waiting for review = 1000; lost: **0**; written twice: **0**
- 52 duplicate writes were stopped by the idempotency key (lost replies retried); 15.3 retries per 100 products
- Passed: **True**

## 4. Throughput and cost (RTX 5090 laptop GPU, 512 px images)

| Weights | What | Parallel requests | Products/s | GPU-hours per 1M | ₹ per 1,000 |
|---|---|---|---|---|---|
| bf16 | raw tagger | 16 | 25.4 | 10.9 | 0.63 |
| bf16 | raw tagger | 32 | 28.0 | 9.9 | 0.57 |
| bf16 | raw tagger | 64 | 30.3 | 9.2 | 0.53 |
| bf16 | full agent | 64 | 30.3 | 9.2 | 0.53 |
| fp8 | raw tagger | 16 | 39.2 | 7.1 | 0.41 |
| fp8 | raw tagger | 32 | 44.1 | 6.3 | 0.36 |
| fp8 | raw tagger | 64 | 47.9 | 5.8 | 0.33 |
| fp8 | full agent | 64 | 45.1 | 6.2 | 0.35 |

Cost = GPU time × $0.60 per GPU-hour (rented 24-32 GB GPU) at ₹95.46/$, the same assumption as the tagger project. Image resize + encode on the CPU: 0.4 ms per product.
