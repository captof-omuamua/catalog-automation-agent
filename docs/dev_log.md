# Dev log

## 2026-10-30: parity check caught a silent 46-point drop

Before building the agent, the new tagger client was run on the 2,000 dev products and
compared with the visual-product-tagging project's saved predictions on the same rows.

| Serve flags | Dev accuracy | Same answers as before | Mean confidence difference |
|---|---|---|---|
| + `--structured-outputs-config '{"backend": "xgrammar", "disable_any_whitespace": true}'` | 39.5% | 41.9% | 0.267 |
| Same flags as the tagger project | **86.6%** (before: 86.4%) | **99.5%** | 0.006 |

The whitespace flag (added because it fixed runaway blank lines for Gemma 4 in Project 1)
forbids the space in `{"value": "X"}`. The model was fine-tuned on outputs with that space,
so banning it pushed every answer onto token paths it never saw in training. There were no
errors or parse failures, only wrong answers. Lesson: a constraint that fixes one model can
silently break a fine-tuned one, and a parity check against known outputs is the cheap way
to catch it. Here the answers are capped at 32 tokens, so the runaway problem can't happen
anyway.

## 2026-10-30: FP8 weights on Blackwell

`QUANT=fp8 scripts/serve_tagger.sh` quantises the bf16 weights to FP8 when vLLM loads them
(CUTLASS FP8 kernels; no calibration data, no nvcc needed).

| | bf16 | FP8 |
|---|---|---|
| Dev accuracy (2,000) | 86.6% | 86.2% (97.3% same answers) |
| Test accuracy (1,610, run once each) | 85.7% | 84.9% |
| Test: auto-tagged / accuracy of auto-tagged | 79.2% / 94.6% | 79.6% / 94.4% |
| Tagger alone, 64 parallel | 30.3 products/s | 47.9 products/s (+58%) |

bf16 stays the default (it is what the accuracy numbers were validated on); FP8 is the
throughput option at a cost of about 1 point of accuracy.

## 2026-10-30: the agent was slower than the tagger; profiling found the checkpoints

With FP8 the full agent ran at 34.7 products/s while the tagger alone did 48. The trace showed
each tag step taking 1.1 s at 64 parallel threads (room for ~58/s), so the time was going
elsewhere: LangGraph saved every thread's full state after every step, through one SQLite
connection (93 MB of checkpoints for 5,000 products).

Fix: `durability="exit"`, which saves a thread only when it finishes or pauses for review.
That is safe here because a product takes about a second (a crash costs at most a re-tag) and
writes are idempotent (a re-tagged product's write is rejected as already applied). Result:
45-46 products/s with FP8 (overhead 28% → 4%), 30.3 with bf16 (no overhead), checkpoints
21 MB. The reliability test was re-run with the new setting and still loses nothing and
writes nothing twice.
