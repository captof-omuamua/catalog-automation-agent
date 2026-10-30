"""The MCP tools, called through a real MCP client connected in memory."""

import json

import httpx
from mcp.client import Client

from catalog_agent import mcp_server
from catalog_agent.config import settings

from .test_agent import MemoryStore, fake_vllm, product


async def test_tools_list_and_review_round_trip(tmp_path, monkeypatch):
    from PIL import Image

    from catalog_agent.agent import Deps, run_batch
    from catalog_agent.review import ReviewQueue

    image = tmp_path / "x.jpg"
    Image.new("RGB", (8, 8)).save(image)
    store = MemoryStore(str(image))
    reviews = ReviewQueue(tmp_path / "review.sqlite")
    monkeypatch.setattr(settings, "checkpoint_db", tmp_path / "ckpt.sqlite")

    def fake_deps(http):
        return Deps(
            connector=store,
            http=httpx.AsyncClient(transport=httpx.MockTransport(fake_vllm)),
            reviews=reviews,
            vllm_url="http://vllm",
            model="m",
            model_version="v1",
            base_delay=0,
        )

    monkeypatch.setattr(mcp_server, "deps_factory", fake_deps)
    await run_batch(fake_deps(None), [product("unsure")], settings.checkpoint_db)

    async with Client(mcp_server.server) as client:
        names = {t.name for t in (await client.list_tools()).tools}
        assert names == {"tag_product", "list_review_queue", "approve_tags", "run_catalog"}

        listed = await client.call_tool("list_review_queue", {})
        queue = listed.structured_content["result"]  # a list comes back as structured content
        assert queue[0]["thread_id"] == "unsure|Sleeve"

        bad = await client.call_tool("approve_tags", {"thread_id": "unsure|Sleeve", "value": "XL"})
        assert "must be one of" in bad.content[0].text

        ok = await client.call_tool(
            "approve_tags", {"thread_id": "unsure|Sleeve", "value": "Short"}
        )
        assert json.loads(ok.content[0].text)["write_status"] == "applied"
        assert list(store.applied.values()) == ["Short"]
