"""HTTP API and review page. Run: make api  ->  http://localhost:8080/docs and /review/ui

  POST /webhooks/product-created   one new product (n8n calls this) -> its decision
  POST /runs                       tag every new product in the catalog (background job)
  GET  /runs/{run_id}              the run's summary
  GET  /review                     products waiting for a person
  POST /review/{thread}/approve    a person's answer; the paused agent thread finishes
  GET  /review/ui                  the same queue as a web page

Routes that change data need the header  X-API-Key: <AGENT_API_KEY>.
"""

import asyncio
import html
import uuid
from pathlib import Path

import httpx
from fastapi import Depends, FastAPI, Form, Header, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from pydantic import BaseModel

from catalog_agent.agent import fetch_products, resume_review, run_batch
from catalog_agent.config import settings
from catalog_agent.connectors import PayloadImages
from catalog_agent.notify import notify
from catalog_agent.review import ReviewQueue
from catalog_agent.runtime import make_deps

app = FastAPI(title="catalog-automation-agent", version="0.1.0")
runs: dict[str, dict] = {}
queue = ReviewQueue(settings.review_db)  # read-only views of the review queue


def require_key(x_api_key: str = Header(default="")) -> None:
    if x_api_key != settings.api_key:
        raise HTTPException(401, "missing or wrong X-API-Key")


class Product(BaseModel):
    id: str
    title: str
    category: str
    image_path: str | None = None
    image_url: str | None = None
    ask: list[dict]  # [{"attribute": ..., "options": [...]}]
    attributes: dict = {}


class RunRequest(BaseModel):
    mode: str = "fill"  # fill | qc
    dry_run: bool = False


class Approve(BaseModel):
    value: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/webhooks/product-created", dependencies=[Depends(require_key)])
async def product_created(p: Product, mode: str = "fill") -> dict:
    """Tag one product right away; return what happened to each attribute."""
    async with httpx.AsyncClient(timeout=120) as http:
        deps = make_deps(http)
        deps.connector = PayloadImages(deps.connector, http)  # photo comes with the request
        s = await run_batch(deps, [p.model_dump()], settings.checkpoint_db, mode=mode)
        pending = [r for r in deps.reviews.pending(1000) if r["product_id"] == p.id]
    return {
        "product_id": p.id,
        "decisions": s.decisions,
        "writes": s.writes,
        "needs_review": bool(pending),
        "review": pending,
    }


@app.post("/runs", dependencies=[Depends(require_key)])
async def start_run(req: RunRequest) -> dict:
    run_id = uuid.uuid4().hex[:8]
    runs[run_id] = {"status": "running"}

    async def job() -> None:
        async with httpx.AsyncClient(timeout=120) as http:
            deps = make_deps(http)
            products = await fetch_products(deps)
            s = await run_batch(
                deps,
                products,
                settings.checkpoint_db,
                mode=req.mode,
                dry_run=req.dry_run,
                concurrency=settings.concurrency,
            )
            await notify(http, s.text())
        runs[run_id] = {
            "status": "done",
            "summary": s.text(),
            "decisions": s.decisions,
            "writes": s.writes,
            "retries": s.retries,
        }

    asyncio.create_task(job())
    return {"run_id": run_id}


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict:
    if run_id not in runs:
        raise HTTPException(404, "no such run")
    return runs[run_id]


@app.get("/review")
def review_queue(limit: int = 50) -> list[dict]:
    return queue.pending(limit)


@app.post("/review/{thread}/approve", dependencies=[Depends(require_key)])
async def approve(thread: str, body: Approve) -> dict:
    return await _approve(thread, body.value)


async def _approve(thread: str, value: str) -> dict:
    async with httpx.AsyncClient(timeout=120) as http:
        deps = make_deps(http)
        item = deps.reviews.get(thread)
        if item is None or item["status"] != "pending":
            raise HTTPException(404, "not waiting for review")
        if value not in item["options"]:
            raise HTTPException(400, "value is not one of the allowed options")
        state = await resume_review(deps, settings.checkpoint_db, thread, value)
    return {"thread": thread, "value": value, "write_status": state.get("write_status")}


# --- review page ------------------------------------------------------------------------


@app.get("/review/image/{thread:path}")
def review_image(thread: str) -> FileResponse:
    item = queue.get(thread)
    if not item or not Path(item["image"]).exists():
        raise HTTPException(404, "no image")
    return FileResponse(item["image"])


CSS = """
body { font-family: system-ui; max-width: 860px; margin: 24px auto; padding: 0 16px; color: #222 }
.card { display: flex; gap: 16px; border: 1px solid #ddd; border-radius: 8px; padding: 12px;
        margin: 12px 0 }
img { width: 140px; height: 140px; object-fit: contain; background: #fafafa }
h3 { margin: 0 0 6px; font-size: 16px }
button { margin: 3px; padding: 6px 10px; border: 1px solid #bbb; border-radius: 6px;
         background: #fff; cursor: pointer }
.guess { border-color: #2a78d6; color: #1c5cab }
"""


@app.get("/review/ui", response_class=HTMLResponse)
def review_page() -> str:
    items = queue.pending(20)
    cards = []
    for it in items:
        t = html.escape(it["thread_id"])
        why = (
            f"Listing says <b>{html.escape(it['stated'] or '')}</b>; the model sees "
            f"<b>{html.escape(it['predicted'] or '')}</b>"
            if it["kind"] == "qc"
            else f"Model's guess: <b>{html.escape(it['predicted'] or '')}</b>"
        )
        buttons = "".join(
            f'<button name="value" value="{html.escape(o)}"'
            f"{' class=guess' if o == it['predicted'] else ''}>{html.escape(o)}</button>"
            for o in it["options"]
        )
        cards.append(
            f'<div class=card><img src="/review/image/{t}"><div><h3>{html.escape(it["title"])}'
            f"</h3><p>{html.escape(it['attribute'])} · {why} · confidence "
            f'{it["confidence"]:.2f}</p><form method=post action="/review/ui/{t}">{buttons}'
            f"</form></div></div>"
        )
    body = "".join(cards) or "<p>Nothing to review.</p>"
    return (
        f"<!doctype html><meta charset=utf-8><title>Review queue</title><style>{CSS}</style>"
        f"<h1>Review queue <small>({len(items)} shown)</small></h1>{body}"
    )


@app.post("/review/ui/{thread:path}")
async def review_page_approve(thread: str, value: str = Form(...)) -> RedirectResponse:
    """The page's buttons (same-origin form posts, so no API key; keep the API local)."""
    await _approve(thread, value)
    return RedirectResponse("/review/ui", status_code=303)


@app.post("/mock/slack")
async def mock_slack(body: dict) -> dict:
    """Stand-in for a Slack incoming webhook (same request shape: {"text": ...})."""
    async with httpx.AsyncClient() as http:
        await notify(http, body.get("text", ""), webhook="")
    return {"ok": True}
