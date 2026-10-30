"""Agent behaviour with a fake vision model and an in-memory store (no GPU, no network)."""

import json
import math

import httpx
import pytest
from PIL import Image

from catalog_agent.agent import Deps, resume_review, run_batch
from catalog_agent.review import ReviewQueue

# product id -> (answer, confidence) the fake model gives
ANSWERS = {"sure": ("Short", 0.95), "unsure": ("Long", 0.40), "wrong_listing": ("Short", 0.9)}


def fake_vllm(request: httpx.Request) -> httpx.Response:
    title = json.loads(request.content)["messages"][0]["content"][1]["text"]
    pid = title.split("Title: ")[1].split("\n")[0]
    value, conf = ANSWERS[pid]
    tokens = [
        {"token": '{"value": "', "logprob": 0.0},
        {"token": value, "logprob": math.log(conf)},
        {"token": '"}', "logprob": 0.0},
    ]
    msg = {"content": json.dumps({"value": value})}
    return httpx.Response(
        200, json={"choices": [{"message": msg, "logprobs": {"content": tokens}}]}
    )


class Crash(Exception):
    """Stands in for the process dying mid-run (not an HTTP error, so it is not retried)."""


class MemoryStore:
    """Connector that records writes; can crash once on a chosen product."""

    def __init__(self, image, crash_on=None):
        self.image, self.crash_on, self.applied = image, crash_on, {}

    async def get_image(self, product):
        return self.image

    async def write(self, pid, attribute, value, source, key):
        if pid == self.crash_on:
            self.crash_on = None
            raise Crash("simulated crash")
        if key in self.applied:
            return "already"
        self.applied[key] = value
        return "applied"


def product(pid, stated=None):
    ask = [{"attribute": "Sleeve", "options": ["Short", "Long"]}]
    return {
        "id": pid,
        "title": pid,
        "category": "Tops",
        "ask": ask,
        "attributes": {"Sleeve": stated} if stated else {},
    }


@pytest.fixture
def setup(tmp_path):
    image = tmp_path / "x.jpg"
    Image.new("RGB", (8, 8)).save(image)

    def make(crash_on=None):
        store = MemoryStore(str(image), crash_on)
        deps = Deps(
            connector=store,
            http=httpx.AsyncClient(transport=httpx.MockTransport(fake_vllm)),
            reviews=ReviewQueue(tmp_path / "review.sqlite"),
            vllm_url="http://vllm",
            model="m",
            model_version="v1",
            base_delay=0,
        )
        return store, deps

    return make, tmp_path / "ckpt.sqlite"


async def test_confident_is_written_unsure_waits_for_a_person(setup):
    make, ckpt = setup
    store, deps = make()
    s = await run_batch(deps, [product("sure"), product("unsure")], ckpt)
    assert s.decisions["auto"] == 1 and s.decisions["review"] == 1
    assert list(store.applied.values()) == ["Short"]
    assert [r["product_id"] for r in deps.reviews.pending()] == ["unsure"]

    state = await resume_review(deps, ckpt, "unsure|Sleeve", "Short")
    assert state["source"] == "review" and state["write_status"] == "applied"
    assert deps.reviews.pending() == []


async def test_crash_then_rerun_resumes_without_writing_twice(setup):
    make, ckpt = setup
    store, deps = make(crash_on="sure")
    with pytest.raises(Crash):
        await run_batch(deps, [product("sure")], ckpt)
    assert store.applied == {}
    s = await run_batch(deps, [product("sure")], ckpt)  # same batch again
    s2 = await run_batch(deps, [product("sure")], ckpt)  # and once more
    assert len(store.applied) == 1 and s.writes["applied"] == 1 and s2.skipped_done == 1


async def test_qc_flags_a_confident_disagreement_and_passes_a_match(setup):
    make, ckpt = setup
    store, deps = make()
    batch = [product("wrong_listing", stated="Long"), product("sure", stated="Short")]
    s = await run_batch(deps, batch, ckpt, mode="qc")
    assert s.decisions["qc_flag"] == 1 and s.decisions["qc_pass"] == 1
    assert deps.reviews.pending()[0]["kind"] == "qc" and store.applied == {}


async def test_dry_run_writes_nothing(setup):
    make, ckpt = setup
    store, deps = make()
    s = await run_batch(deps, [product("sure")], ckpt, dry_run=True)
    assert s.writes["dry-run"] == 1 and store.applied == {}


def test_only_temporary_errors_are_retried():
    from catalog_agent.agent import is_transient

    def status(code):
        req = httpx.Request("GET", "http://x")
        return httpx.HTTPStatusError("e", request=req, response=httpx.Response(code, request=req))

    assert is_transient(status(503)) and is_transient(status(429))
    assert not is_transient(status(404)) and not is_transient(status(409))
    assert is_transient(httpx.ConnectTimeout("t"))


async def test_a_failed_product_is_tried_again_on_the_next_run(setup):
    make, ckpt = setup
    store, deps = make()
    real_get_image = store.get_image
    calls = []

    async def flaky(product):  # the photo is missing the first time (404: not retried)
        calls.append(1)
        if len(calls) == 1:
            req = httpx.Request("GET", "http://store/images/x")
            raise httpx.HTTPStatusError(
                "404", request=req, response=httpx.Response(404, request=req)
            )
        return await real_get_image(product)

    store.get_image = flaky
    s1 = await run_batch(deps, [product("sure")], ckpt)
    s2 = await run_batch(deps, [product("sure")], ckpt)
    assert s1.decisions["failed"] == 1 and s2.decisions["auto"] == 1
    assert len(store.applied) == 1
