"""Build the agent's dependencies from settings, so the CLI, API and MCP server agree."""

import httpx

from catalog_agent.agent import Deps
from catalog_agent.config import ROOT, settings
from catalog_agent.connectors import RestConnector
from catalog_agent.review import ReviewQueue


def make_deps(http: httpx.AsyncClient, connector=None) -> Deps:
    return Deps(
        connector=connector or RestConnector(settings.catalog_url, http),
        http=http,
        reviews=ReviewQueue(settings.review_db),
        vllm_url=settings.vllm_url,
        model=settings.tagger_model,
        model_version=settings.model_version,
        image_size=settings.image_size,
        accept_threshold=settings.accept_threshold,
        qc_threshold=settings.qc_threshold,
        trace_path=ROOT / "logs" / "trace.jsonl",
    )
