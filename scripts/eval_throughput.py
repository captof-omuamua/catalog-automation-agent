"""Eval 4: how fast, and what does it cost? Raw tagger vs the full agent.

  raw    the tagger alone on 2,000 load products, at 16 / 32 / 64 parallel requests
  agent  the whole agent (catalog API, image download, routing, checkpoints, writes) on all
         5,000 load products, at the best raw setting
Also profiles one part on the CPU (image resize + encode) before any tuning.
Cost uses the tagger project's assumption: $0.60 per GPU-hour, ₹95.46 per $.

Run once per server setting:  --label bf16   (then restart vLLM with QUANT=fp8: --label fp8)
"""

import argparse
import asyncio
import json
import subprocess
import time

import httpx

from catalog_agent.agent import fetch_products, run_batch
from catalog_agent.config import settings
from catalog_agent.evaluate import eval_deps, fresh_dir, inr_per_1000, load_store, read_jsonl
from catalog_agent.tagger import image_data_url, tag


def gpu_memory_mb() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    ).stdout
    return int(out.strip().splitlines()[0])


async def raw(products: list[dict], concurrency: int) -> float:
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=120) as http:

        async def one(p):
            ask = p["ask"][0]
            async with sem:
                return await tag(
                    http,
                    settings.vllm_url,
                    settings.tagger_model,
                    p,
                    ask["attribute"],
                    ask["options"],
                    p["image_path"],
                    settings.image_size,
                )

        t = time.perf_counter()
        await asyncio.gather(*(one(p) for p in products))
        return len(products) / (time.perf_counter() - t)


async def agent(concurrency: int) -> tuple[float, int]:
    run_dir = fresh_dir("throughput_agent")
    async with httpx.AsyncClient(timeout=120) as http:
        await load_store(http, settings.data_dir / "products_load.jsonl")
        deps = eval_deps(http, run_dir)
        products = await fetch_products(deps)
        t = time.perf_counter()
        s = await run_batch(deps, products, run_dir / "checkpoints.sqlite", concurrency=concurrency)
        return len(products) / (time.perf_counter() - t), s.decisions["failed"]


async def main(label: str) -> None:
    products = read_jsonl(settings.data_dir / "products_load.jsonl")
    t = time.perf_counter()
    for p in products[:200]:
        image_data_url(p["image_path"], settings.image_size)
    encode_ms = 1000 * (time.perf_counter() - t) / 200

    await raw(products[:64], 16)  # warm-up
    rows = []
    for c in (16, 32, 64):
        pps = await raw(products[:2000], c)
        rows.append({"what": "raw tagger", "parallel": c, "products_per_s": round(pps, 1)})
        print(rows[-1], flush=True)
    best = max(rows, key=lambda r: r["products_per_s"])["parallel"]
    mem = gpu_memory_mb()
    pps, failed = await agent(best)
    rows.append(
        {"what": "full agent", "parallel": best, "products_per_s": round(pps, 1), "failed": failed}
    )
    print(rows[-1], flush=True)
    for r in rows:
        r["inr_per_1000"] = round(inr_per_1000(r["products_per_s"]), 2)
        r["gpu_hours_per_million"] = round(1e6 / r["products_per_s"] / 3600, 1)
    result = {
        "label": label,
        "image_encode_ms_per_product": round(encode_ms, 1),
        "gpu_memory_mb_while_serving": mem,
        "rows": rows,
    }
    (settings.reports_dir / f"throughput_{label}.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    asyncio.run(main(ap.parse_args().label))
