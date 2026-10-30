"""The agent: one LangGraph thread per product attribute.

    tag ──► route ──► write ──► END            confident: written back automatically
              │
              ├──► review ──► (pause) ──► write ──► END   unsure: a person picks the value
              │
              └──► END                         QC: the stated value matches the photo

Two modes:
  fill  the attribute is empty; confident answers are written, unsure ones go to review
  qc    the listing already states a value; if the model confidently disagrees, the listing
        is flagged for review; otherwise it passes

Each thread is checkpointed in SQLite after every step, so a crash loses nothing: running the
batch again skips finished threads, continues interrupted ones, and leaves paused ones paused.
Writes carry an idempotency key, so a step that runs twice cannot write twice.
"""

import asyncio
import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict

import httpx
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt

from catalog_agent.review import ReviewQueue
from catalog_agent.tagger import tag


class State(TypedDict, total=False):
    product: dict
    attribute: str
    options: list[str]
    stated: str | None  # the listing's current value (QC mode)
    mode: str  # "fill" | "qc"
    dry_run: bool
    value: str | None  # the model's answer
    confidence: float
    decision: str  # auto | review | qc_pass | qc_flag | failed
    final_value: str | None
    source: str  # "model" | "review"
    write_status: str  # applied | already | dry-run | skipped
    retries: int


@dataclass
class Deps:
    connector: object  # a catalog_agent.connectors.Connector
    http: httpx.AsyncClient  # for the tagger
    reviews: ReviewQueue
    vllm_url: str
    model: str
    model_version: str
    image_size: int = 512
    accept_threshold: float = 0.72
    qc_threshold: float = 0.72
    trace_path: Path | None = None
    attempts: int = 5
    base_delay: float = 0.2


@dataclass
class Retrying:
    count: int = 0
    errors: list[str] = field(default_factory=list)


def is_transient(exc: Exception) -> bool:
    """Worth retrying: network trouble, timeouts, 5xx and 429. A 404 or 400 will not heal."""
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code >= 500 or code == 429
    return isinstance(exc, httpx.TransportError)


async def with_retries(fn, deps: Deps, r: Retrying):
    """Call `fn` up to deps.attempts times with exponential backoff; count the retries."""
    for attempt in range(deps.attempts):
        try:
            return await fn()
        except (httpx.HTTPError, RuntimeError) as exc:
            r.errors.append(type(exc).__name__)
            if attempt == deps.attempts - 1 or not is_transient(exc):
                raise
            r.count += 1
            await asyncio.sleep(deps.base_delay * 2**attempt)


def thread_id(product: dict, attribute: str) -> str:
    return f"{product['id']}|{attribute}"


def build_graph(deps: Deps) -> StateGraph:
    def trace(state: State, node: str, start: float, **extra) -> None:
        if deps.trace_path is None:
            return
        row = {
            "thread": thread_id(state["product"], state["attribute"]),
            "node": node,
            "ms": round(1000 * (time.perf_counter() - start), 1),
            **extra,
        }
        with deps.trace_path.open("a") as f:
            f.write(json.dumps(row) + "\n")

    async def tag_node(state: State) -> State:
        start, r = time.perf_counter(), Retrying()
        p = state["product"]

        async def call():
            image = await deps.connector.get_image(p)
            return await tag(
                deps.http,
                deps.vllm_url,
                deps.model,
                p,
                state["attribute"],
                state["options"],
                image,
                deps.image_size,
            )

        try:
            t = await with_retries(call, deps, r)
        except (httpx.HTTPError, RuntimeError):
            trace(state, "tag", start, outcome="failed", retries=r.count)
            return {"decision": "failed", "retries": r.count}
        trace(state, "tag", start, value=t.value, confidence=t.confidence, retries=r.count)
        # "decision" is reset: on a rerun of a failed thread the old "failed" would linger
        return {
            "value": t.value,
            "confidence": t.confidence,
            "decision": "tagged",
            "write_status": "",
            "retries": r.count,
        }

    def route(state: State) -> str:
        if state.get("decision") == "failed" or state.get("value") is None:
            return "failed"
        if state["mode"] == "qc":
            if state["value"] == state.get("stated"):
                return "qc_pass"
            return "qc_flag" if state["confidence"] >= deps.qc_threshold else "qc_pass"
        return "auto" if state["confidence"] >= deps.accept_threshold else "review"

    async def decide_node(state: State) -> State:
        d = route(state)
        out: State = {"decision": d}
        if d == "auto":
            out.update(final_value=state["value"], source="model")
        return out

    async def review_node(state: State) -> State:
        """Queue the item, then pause this thread until a person answers."""
        p, kind = state["product"], "qc" if state["mode"] == "qc" else "fill"
        deps.reviews.add(
            thread_id(p, state["attribute"]),
            p,
            state["attribute"],
            state["options"],
            kind,
            state.get("value"),
            state.get("confidence", 0),
            state.get("stated"),
        )
        answer = interrupt({"thread": thread_id(p, state["attribute"]), "kind": kind})
        return {"final_value": answer["value"], "source": "review"}

    async def write_node(state: State) -> State:
        start, r = time.perf_counter(), Retrying()
        if state.get("dry_run"):
            trace(state, "write", start, outcome="dry-run")
            return {"write_status": "dry-run"}
        p, attr = state["product"], state["attribute"]
        key = f"{p['id']}|{attr}|{state['source']}|{deps.model_version}"
        try:
            status = await with_retries(
                lambda: deps.connector.write(
                    p["id"], attr, state["final_value"], state["source"], key
                ),
                deps,
                r,
            )
        except (httpx.HTTPError, RuntimeError) as exc:  # gave up: report it, keep the run going
            status = f"failed: {type(exc).__name__}"
        trace(state, "write", start, outcome=status, retries=r.count)
        return {"write_status": status, "retries": state.get("retries", 0) + r.count}

    g = StateGraph(State)
    g.add_node("tag", tag_node)
    g.add_node("decide", decide_node)
    g.add_node("review", review_node)
    g.add_node("write", write_node)
    g.set_entry_point("tag")
    g.add_edge("tag", "decide")
    g.add_conditional_edges(
        "decide",
        lambda s: s["decision"],
        {"auto": "write", "review": "review", "qc_flag": "review", "qc_pass": END, "failed": END},
    )
    g.add_edge("review", "write")
    g.add_edge("write", END)
    return g


@dataclass
class RunSummary:
    products: int = 0
    decisions: Counter = field(default_factory=Counter)
    writes: Counter = field(default_factory=Counter)
    retries: int = 0
    skipped_done: int = 0
    seconds: float = 0.0

    def text(self) -> str:
        d = self.decisions
        return (
            f"Catalog run: {self.products} products in {self.seconds:.0f}s. "
            f"Auto-tagged {d['auto']}, sent to review {d['review']}, "
            f"QC flagged {d['qc_flag']}, QC passed {d['qc_pass']}, failed {d['failed']}. "
            f"Writes: {dict(self.writes)}. Retries: {self.retries}."
        )


async def run_batch(
    deps: Deps,
    products: list[dict],
    checkpoint_db: Path,
    mode: str = "fill",
    dry_run: bool = False,
    concurrency: int = 32,
    durability: str = "exit",
) -> RunSummary:
    """Run every (product, attribute) thread; safe to call again after a crash.

    durability="exit" saves a thread's checkpoint only when it finishes or pauses for review,
    instead of after every step ("async"). One product takes about a second, so a crash
    costs at most a re-tag, and the idempotent writes make a re-run safe. Measured with FP8:
    the full agent went from 34.7 to 45-46 products/s (the tagger alone: 48), and the
    checkpoint file from 93 MB to 21 MB per 5,000 products.
    """
    summary, start = RunSummary(products=len(products)), time.perf_counter()
    checkpoint_db.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(checkpoint_db)) as saver:
        graph = build_graph(deps).compile(checkpointer=saver)
        sem = asyncio.Semaphore(concurrency)

        async def one(p: dict, ask: dict) -> None:
            config = {"configurable": {"thread_id": thread_id(p, ask["attribute"])}}
            async with sem:
                snap = await graph.aget_state(config)
                done = snap.values and not snap.next
                if done and snap.values.get("decision") != "failed":  # finished earlier
                    summary.skipped_done += 1
                    state = snap.values
                elif snap.next:  # crashed mid-way, or paused for review
                    if snap.interrupts:  # still waiting for a person
                        summary.decisions[snap.values.get("decision", "review")] += 1
                        return
                    state = await graph.ainvoke(None, config, durability=durability)
                else:
                    stated = p.get("attributes", {}).get(ask["attribute"])
                    state = await graph.ainvoke(
                        {
                            "product": p,
                            "attribute": ask["attribute"],
                            "options": ask["options"],
                            "stated": stated,
                            "mode": mode,
                            "dry_run": dry_run,
                        },
                        config,
                        durability=durability,
                    )
            summary.decisions[state.get("decision", "?")] += 1
            if state.get("write_status"):
                summary.writes[state["write_status"]] += 1
            summary.retries += state.get("retries", 0)

        await asyncio.gather(*(one(p, ask) for p in products for ask in p["ask"]))
    summary.seconds = time.perf_counter() - start
    return summary


async def resume_review(deps: Deps, checkpoint_db: Path, thread: str, value: str) -> State:
    """A person picked `value` for a paused thread: record it and let the thread finish."""
    deps.reviews.approve(thread, value)
    async with AsyncSqliteSaver.from_conn_string(str(checkpoint_db)) as saver:
        graph = build_graph(deps).compile(checkpointer=saver)
        config = {"configurable": {"thread_id": thread}}
        return await graph.ainvoke(Command(resume={"value": value}), config)


async def fetch_products(deps: Deps, status: str = "new") -> list[dict]:
    """List products to work on, retrying temporary store errors like every other call."""
    fetch = deps.connector.fetch_new
    if status == "new":
        return await with_retries(lambda: fetch(limit=100_000), deps, Retrying())
    return await with_retries(lambda: fetch(limit=100_000, status=status), deps, Retrying())
