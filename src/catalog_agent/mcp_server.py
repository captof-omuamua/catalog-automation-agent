"""MCP server: the agent's tools for Claude Desktop / Claude Code (or any MCP client).

  tag_product        ask the vision model for one attribute of one product photo
  list_review_queue  products waiting for a person
  approve_tags       a person's answer for one waiting product; the agent writes it back
  run_catalog        tag every new product in the catalog (dry_run=True writes nothing)

Run over stdio:  catalog-agent-mcp   (installed by `pip install .`)
"""

import httpx
from mcp.server import MCPServer

from catalog_agent.agent import fetch_products, resume_review, run_batch
from catalog_agent.config import settings
from catalog_agent.runtime import make_deps
from catalog_agent.tagger import tag

server = MCPServer(
    name="catalog-agent",
    instructions="Fill in and check product attributes from photos. Unsure answers wait in a "
    "review queue for a person; approve_tags resolves them.",
)
deps_factory = make_deps  # tests swap this for a fake model and store


@server.tool()
async def tag_product(
    title: str, category: str, attribute: str, options: list[str], image_path: str
) -> dict:
    """Ask the vision model for one attribute of one product, e.g. attribute="Sleeve Style",
    options=["Long Sleeve", "Short Sleeve", "Sleeveless"]. Returns the value, the model's
    confidence (0-1) and whether the agent would auto-accept it or ask a person."""
    async with httpx.AsyncClient(timeout=120) as http:
        deps = deps_factory(http)
        product = {"title": title, "category": category}
        t = await tag(
            http,
            deps.vllm_url,
            deps.model,
            product,
            attribute,
            options,
            image_path,
            deps.image_size,
        )
    accept = t.value is not None and t.confidence >= deps.accept_threshold
    return {
        "value": t.value,
        "confidence": t.confidence,
        "decision": "auto-accept" if accept else "needs review",
    }


@server.tool()
async def list_review_queue(limit: int = 20) -> list[dict]:
    """Products the model was unsure about (kind="fill") or listings whose stated value
    disagrees with the photo (kind="qc"), least confident first."""
    async with httpx.AsyncClient() as http:
        items = deps_factory(http).reviews.pending(limit)
    keep = [
        "thread_id",
        "title",
        "attribute",
        "options",
        "kind",
        "predicted",
        "confidence",
        "stated",
    ]
    return [{k: it[k] for k in keep} for it in items]


@server.tool()
async def approve_tags(thread_id: str, value: str) -> dict:
    """Give the answer for one waiting product (thread_id from list_review_queue). The value
    must be one of its options. The agent then writes it to the catalog."""
    async with httpx.AsyncClient(timeout=120) as http:
        deps = deps_factory(http)
        item = deps.reviews.get(thread_id)
        if item is None or item["status"] != "pending":
            return {"error": "not waiting for review"}
        if value not in item["options"]:
            return {"error": f"value must be one of {item['options']}"}
        state = await resume_review(deps, settings.checkpoint_db, thread_id, value)
    return {"thread_id": thread_id, "value": value, "write_status": state.get("write_status")}


@server.tool()
async def run_catalog(mode: str = "fill", dry_run: bool = True) -> dict:
    """Tag every new product in the catalog. mode="fill" fills empty attributes; mode="qc"
    checks stated values against the photos. dry_run=True (the default) writes nothing."""
    async with httpx.AsyncClient(timeout=120) as http:
        deps = deps_factory(http)
        products = await fetch_products(deps)
        s = await run_batch(
            deps,
            products,
            settings.checkpoint_db,
            mode=mode,
            dry_run=dry_run,
            concurrency=settings.concurrency,
        )
    return {"summary": s.text(), "decisions": dict(s.decisions), "writes": dict(s.writes)}


def main() -> None:
    server.run("stdio")


if __name__ == "__main__":
    main()
