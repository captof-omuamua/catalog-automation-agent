"""Shared pieces for the evaluation scripts: fresh state, loading the store, scoring."""

import json
import shutil
from pathlib import Path

import httpx

from catalog_agent.agent import Deps
from catalog_agent.config import ROOT, settings
from catalog_agent.connectors import RestConnector
from catalog_agent.review import ReviewQueue

EVAL_DIR = ROOT / "data" / "eval"
PRICE_USD_PER_HOUR = 0.60  # same assumption as the tagger project: one rented 24-32 GB GPU
USD_TO_INR = 95.46


def fresh_dir(name: str) -> Path:
    """An empty folder for one eval run's checkpoints, review queue and trace."""
    d = EVAL_DIR / name
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    return d


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


async def load_store(http: httpx.AsyncClient, jsonl: Path) -> int:
    """Put these products in the mock store (it sees ./data as /app/data in Docker)."""
    target = settings.data_dir / "batch.jsonl"
    shutil.copyfile(jsonl, target)
    r = await http.post(
        f"{settings.catalog_url}/_admin/load", params={"path": "/app/data/batch.jsonl"}
    )
    r.raise_for_status()
    await http.post(f"{settings.catalog_url}/_admin/faults", json={})
    return r.json()["products"]


def eval_deps(http: httpx.AsyncClient, run_dir: Path, **overrides) -> Deps:
    kw = dict(
        connector=RestConnector(settings.catalog_url, http),
        http=http,
        reviews=ReviewQueue(run_dir / "review.sqlite"),
        vllm_url=settings.vllm_url,
        model=settings.tagger_model,
        model_version=settings.model_version,
        image_size=settings.image_size,
        accept_threshold=settings.accept_threshold,
        qc_threshold=settings.qc_threshold,
        trace_path=run_dir / "trace.jsonl",
    )
    return Deps(**{**kw, **overrides})


def tag_traces(run_dir: Path) -> dict[str, dict]:
    """product id -> the model's answer and confidence, from the run's trace."""
    out = {}
    for row in read_jsonl(run_dir / "trace.jsonl"):
        if row["node"] == "tag" and "value" in row:
            out[row["thread"].split("|")[0]] = row
    return out


def inr_per_1000(products_per_second: float) -> float:
    return 1000 / products_per_second / 3600 * PRICE_USD_PER_HOUR * USD_TO_INR


def pct(a: float, b: float) -> str:
    return f"{100 * a / b:.1f}%" if b else "-"
