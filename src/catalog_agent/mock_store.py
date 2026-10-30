"""A small stand-in for a store's catalog API, with fault injection.

Big catalog teams have their own product APIs, not Shopify; this mock plays that role locally
and lets the reliability test break things on purpose:
  - fail_rate:        the request fails with a 503 before doing anything
  - lost_ack_rate:    a write is applied, then the reply is lost (500) - the dangerous case:
                      a naive client retries and writes twice
Writes carry an idempotency key; a replayed key is answered with 409 and not applied again.

Run: uvicorn catalog_agent.mock_store:app --port 8100
"""

import json
import random
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI(title="mock catalog store")

products: dict[str, dict] = {}  # product id -> product record
applied: dict[str, dict] = {}  # idempotency key -> the write that was applied
write_log: list[dict] = []  # every write request that reached the store, for the tests
faults = {"fail_rate": 0.0, "lost_ack_rate": 0.0}
rng = random.Random(0)


class Write(BaseModel):
    attribute: str
    value: str
    source: str  # "model" or "review"
    idempotency_key: str


class Faults(BaseModel):
    fail_rate: float = 0.0
    lost_ack_rate: float = 0.0
    seed: int = 0


def _maybe_fail() -> None:
    if rng.random() < faults["fail_rate"]:
        raise HTTPException(503, "injected failure")


@app.get("/products")
def list_products(status: str = "new", limit: int = 1000, offset: int = 0) -> list[dict]:
    _maybe_fail()
    rows = [p for p in products.values() if status == "all" or p["status"] == status]
    return rows[offset : offset + limit]


@app.get("/products/{pid}")
def get_product(pid: str) -> dict:
    _maybe_fail()
    if pid not in products:
        raise HTTPException(404, "no such product")
    return products[pid]


@app.get("/images/{pid}")
def get_image(pid: str) -> FileResponse:
    if pid not in products:
        raise HTTPException(404, "no such product")
    return FileResponse(products[pid]["image_path"])


@app.put("/products/{pid}/attributes")
def write_attribute(pid: str, w: Write) -> dict:
    _maybe_fail()
    if pid not in products:
        raise HTTPException(404, "no such product")
    write_log.append({"product_id": pid, **w.model_dump()})
    if w.idempotency_key in applied:
        raise HTTPException(409, "already applied")
    applied[w.idempotency_key] = {"product_id": pid, **w.model_dump()}
    products[pid]["attributes"][w.attribute] = w.value
    products[pid]["status"] = "tagged"
    if rng.random() < faults["lost_ack_rate"]:
        raise HTTPException(500, "injected: applied, but the reply was lost")
    return {"status": "applied"}


# --- admin routes, used by tests and the eval scripts ---------------------------------


@app.post("/_admin/load")
def load(path: str) -> dict:
    """Replace the catalog with the products in a JSONL file."""
    products.clear()
    applied.clear()
    write_log.clear()
    for line in Path(path).read_text().splitlines():
        p = json.loads(line)
        products[p["id"]] = {**p, "status": "new", "attributes": dict(p.get("attributes", {}))}
    return {"products": len(products)}


@app.post("/_admin/faults")
def set_faults(f: Faults) -> dict:
    faults.update(fail_rate=f.fail_rate, lost_ack_rate=f.lost_ack_rate)
    rng.seed(f.seed)
    return faults


@app.get("/_admin/stats")
def stats() -> dict:
    per_key: dict[tuple, int] = {}
    for w in applied.values():
        k = (w["product_id"], w["attribute"])
        per_key[k] = per_key.get(k, 0) + 1
    return {
        "products": len(products),
        "write_requests": len(write_log),
        "applied": len(applied),
        "replays_rejected": len(write_log) - len(applied),
        "products_written_twice": sum(n > 1 for n in per_key.values()),
    }


@app.get("/_admin/applied")
def applied_writes() -> list[dict]:
    return list(applied.values())
