"""Diagnostic before anything else: does the new tagger client reproduce the tagger project?

Runs the 2,000 dev products through `catalog_agent.tagger` and compares with the tagger
project's saved predictions on the same rows: same answers, accuracy, confidence, and the
share auto-accepted at the 0.72 cut-off. Writes reports/parity.md.
"""

import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx

from catalog_agent.config import settings
from catalog_agent.tagger import tag

OLD = Path.home() / "catalog-vlm-cascade/reports/implicitave/val_qwen35_4b_finetuned.json"


async def run(products: list[dict]) -> list:
    sem = asyncio.Semaphore(32)
    async with httpx.AsyncClient() as client:

        async def one(p):
            ask = p["ask"][0]
            async with sem:
                return await tag(
                    client,
                    settings.vllm_url,
                    settings.tagger_model,
                    p,
                    ask["attribute"],
                    ask["options"],
                    p["image_path"],
                    settings.image_size,
                )

        return await asyncio.gather(*(one(p) for p in products))


def main() -> None:
    products = [json.loads(line) for line in open(settings.data_dir / "products_dev.jsonl")]
    gold = {
        json.loads(line)["id"]: json.loads(line)["value"]
        for line in open(settings.data_dir / "gold_dev.jsonl")
    }
    old = {p["id"].split("|")[0]: p for p in json.load(open(OLD))["preds"]}

    t = time.perf_counter()
    tags = asyncio.run(run(products))
    secs = time.perf_counter() - t

    ids = [p["id"] for p in products]
    same = sum(tg.value == old[i].get("pred") for i, tg in zip(ids, tags, strict=True))
    acc_new = sum(tg.value == gold[i] for i, tg in zip(ids, tags, strict=True)) / len(ids)
    acc_old = sum(old[i].get("pred") == gold[i] for i in ids) / len(ids)
    conf_diff = statistics.mean(
        abs(tg.confidence - (old[i].get("conf") or 0)) for i, tg in zip(ids, tags, strict=True)
    )
    acc_new_at = sum(tg.confidence >= settings.accept_threshold for tg in tags) / len(ids)
    cut = settings.accept_threshold
    acc_old_at = sum((old[i].get("conf") or 0) >= cut for i in ids) / len(ids)
    errors = sum(tg.value is None for tg in tags)
    md = f"""# Parity check: new tagger client vs the tagger project (dev, {len(ids)} products)

| | Tagger project | This client |
|---|---|---|
| Accuracy | {100 * acc_old:.1f}% | {100 * acc_new:.1f}% |
| Auto-accepted at confidence >= {cut} | {100 * acc_old_at:.1f}% | {100 * acc_new_at:.1f}% |

- Same answer as the tagger project: **{100 * same / len(ids):.1f}%** ({same}/{len(ids)})
- Mean absolute confidence difference: {conf_diff:.4f}
- Failed or unparseable answers: {errors}
- Time: {secs:.0f} s ({len(ids) / secs:.1f} products/s, 32 parallel requests)
"""
    settings.reports_dir.mkdir(exist_ok=True)
    (settings.reports_dir / "parity.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
