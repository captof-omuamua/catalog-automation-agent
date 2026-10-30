"""Eval 1: the agent fills in attributes for a whole split, end to end.

1. Load the split into the catalog store; run the agent (fill mode).
2. Confident answers are written back automatically; the rest wait for a person.
3. A simulated reviewer answers every waiting product with the gold value, through the same
   resume path a real reviewer uses (this exercises pause/resume at scale).
4. Score from the store's write log: auto-tag rate, accuracy of auto-tagged, review rate,
   write-back fidelity (written value == model's answer), final catalog accuracy.

Run: python scripts/eval_tagging.py --split dev   (then --split test, once)
"""

import argparse
import asyncio
import json
import time

import httpx

from catalog_agent.agent import resume_review, run_batch
from catalog_agent.config import settings
from catalog_agent.evaluate import eval_deps, fresh_dir, load_store, pct, read_jsonl, tag_traces


async def main(split: str, label: str = "") -> dict:
    name = f"tagging_{split}{'_' + label if label else ''}"
    run_dir = fresh_dir(name)
    ckpt = run_dir / "checkpoints.sqlite"
    gold = {g["id"]: g for g in read_jsonl(settings.data_dir / f"gold_{split}.jsonl")}
    async with httpx.AsyncClient(timeout=120) as http:
        await load_store(http, settings.data_dir / f"products_{split}.jsonl")
        deps = eval_deps(http, run_dir)
        products = await deps.connector.fetch_new(limit=100_000)
        t = time.perf_counter()
        s = await run_batch(deps, products, ckpt, concurrency=settings.concurrency)
        agent_s = time.perf_counter() - t

        auto_writes = (await http.get(f"{settings.catalog_url}/_admin/applied")).json()
        waiting = deps.reviews.pending(100_000)
        for item in waiting:  # the simulated reviewer always picks the gold value
            await resume_review(deps, ckpt, item["thread_id"], gold[item["product_id"]]["value"])
        final = (await http.get(f"{settings.catalog_url}/_admin/applied")).json()
        stats = (await http.get(f"{settings.catalog_url}/_admin/stats")).json()

    tags = tag_traces(run_dir)
    n = len(gold)
    auto = {w["product_id"]: w["value"] for w in auto_writes}
    auto_right = sum(v == gold[pid]["value"] for pid, v in auto.items())
    model_right = sum(t["value"] == gold[pid]["value"] for pid, t in tags.items())
    faithful = sum(v == tags[pid]["value"] for pid, v in auto.items())
    final_vals = {w["product_id"]: w["value"] for w in final}
    final_right = sum(final_vals.get(pid) == g["value"] for pid, g in gold.items())
    result = {
        "split": split,
        "products": n,
        "model_accuracy": pct(model_right, n),
        "auto_tagged": pct(len(auto), n),
        "auto_accuracy": pct(auto_right, len(auto)),
        "sent_to_review": pct(len(waiting), n),
        "write_fidelity": pct(faithful, len(auto)),
        "final_catalog_accuracy": pct(final_right, n),
        "failed": s.decisions["failed"],
        "products_written_twice": stats["products_written_twice"],
        "agent_seconds": round(agent_s, 1),
        "products_per_second": round(n / agent_s, 1),
    }
    out = settings.reports_dir / f"{name}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--label", default="", help='e.g. "fp8": saved as tagging_<split>_fp8.json')
    a = ap.parse_args()
    asyncio.run(main(a.split, a.label))
