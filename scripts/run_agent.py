"""Run the agent over a batch of products from the catalog API.

  python scripts/run_agent.py --products data/products_dev.jsonl --limit 50
  python scripts/run_agent.py --products data/qc_dev.jsonl --mode qc
  python scripts/run_agent.py --products data/products_dev.jsonl --dry-run

--load puts the products into the mock store first (local demo). Without it, the agent reads
whatever the store at AGENT_CATALOG_URL lists as new.
"""

import argparse
import asyncio
import json
import shutil

import httpx

from catalog_agent.agent import Deps, fetch_products, run_batch
from catalog_agent.config import settings
from catalog_agent.connectors import RestConnector
from catalog_agent.notify import notify
from catalog_agent.review import ReviewQueue


async def main(a: argparse.Namespace) -> None:
    async with httpx.AsyncClient(timeout=120) as http:
        if a.products:
            batch = settings.data_dir / "batch.jsonl"
            if a.limit:
                lines = a.products.read_text().splitlines()[: a.limit]
                batch.write_text("\n".join(lines) + "\n")
            else:
                shutil.copyfile(a.products, batch)
            r = await http.post(
                f"{settings.catalog_url}/_admin/load",
                params={"path": f"{a.store_data_dir}/batch.jsonl"},
            )
            r.raise_for_status()
        store = RestConnector(settings.catalog_url, http)
        deps = Deps(
            connector=store,
            http=http,
            reviews=ReviewQueue(a.review_db),
            vllm_url=settings.vllm_url,
            model=settings.tagger_model,
            model_version=settings.model_version,
            image_size=settings.image_size,
            accept_threshold=settings.accept_threshold,
            qc_threshold=settings.qc_threshold,
            trace_path=a.trace,
        )
        products = await fetch_products(deps, a.status)
        summary = await run_batch(
            deps, products, a.checkpoints, mode=a.mode, dry_run=a.dry_run, concurrency=a.concurrency
        )
        await notify(http, summary.text())
        print(
            json.dumps(
                {
                    "decisions": summary.decisions,
                    "writes": summary.writes,
                    "retries": summary.retries,
                    "seconds": round(summary.seconds, 1),
                }
            )
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--products", type=lambda p: settings.data_dir.parent / p)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--mode", default="fill", choices=["fill", "qc"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status", default="new", help='"all" re-checks every product (resume)')
    ap.add_argument("--concurrency", type=int, default=settings.concurrency)
    ap.add_argument(
        "--checkpoints", type=lambda p: settings.data_dir.parent / p, default=settings.checkpoint_db
    )
    ap.add_argument(
        "--review-db", type=lambda p: settings.data_dir.parent / p, default=settings.review_db
    )
    ap.add_argument(
        "--trace",
        type=lambda p: settings.data_dir.parent / p,
        default=settings.data_dir.parent / "logs" / "trace.jsonl",
    )
    ap.add_argument(
        "--store-data-dir",
        default="/app/data",
        help="where the mock store sees ./data (/app/data in Docker)",
    )
    asyncio.run(main(ap.parse_args()))
