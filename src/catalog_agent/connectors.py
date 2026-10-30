"""Where products come from and where answers go. Three connectors, one interface.

  RestConnector    any catalog API with list / image / write routes (big stores' own systems;
                   the mock store in this repo implements it)
  FileConnector    a JSONL file in, a JSONL file of writes out (batch jobs, CSV exports)
  ShopifyConnector Shopify Admin GraphQL: products in, metafields out (small/mid D2C brands)

Every write carries an idempotency key (product + attribute + model version). A write that
was already applied returns "already" instead of applying twice, so retries are safe.
"""

import json
import re
from pathlib import Path
from typing import Protocol

import httpx


class Connector(Protocol):
    async def fetch_new(self, limit: int) -> list[dict]: ...
    async def get_image(self, product: dict) -> bytes | str: ...
    async def write(self, pid: str, attribute: str, value: str, source: str, key: str) -> str: ...


class RestConnector:
    def __init__(self, base_url: str, client: httpx.AsyncClient):
        self.base, self.client = base_url.rstrip("/"), client

    async def fetch_new(self, limit: int = 1000, status: str = "new") -> list[dict]:
        r = await self.client.get(
            f"{self.base}/products", params={"status": status, "limit": limit}
        )
        r.raise_for_status()
        return r.json()

    async def get_image(self, product: dict) -> bytes:
        r = await self.client.get(f"{self.base}/images/{product['id']}")
        r.raise_for_status()
        return r.content

    async def write(self, pid: str, attribute: str, value: str, source: str, key: str) -> str:
        body = {"attribute": attribute, "value": value, "source": source, "idempotency_key": key}
        r = await self.client.put(f"{self.base}/products/{pid}/attributes", json=body)
        if r.status_code == 409:
            return "already"
        r.raise_for_status()
        return "applied"


class PayloadImages:
    """Wraps a connector for webhook calls: the photo comes with the request (a URL or a local
    path) instead of from the catalog API; writes still go to the catalog."""

    def __init__(self, inner, client: httpx.AsyncClient):
        self.inner, self.client = inner, client

    async def fetch_new(self, limit: int = 1000) -> list[dict]:
        return await self.inner.fetch_new(limit)

    async def get_image(self, product: dict) -> bytes | str:
        if product.get("image_url"):
            r = await self.client.get(product["image_url"])
            r.raise_for_status()
            return r.content
        if product.get("image_path"):
            return product["image_path"]
        return await self.inner.get_image(product)

    async def write(self, pid: str, attribute: str, value: str, source: str, key: str) -> str:
        return await self.inner.write(pid, attribute, value, source, key)


class FileConnector:
    def __init__(self, products: Path, writes: Path):
        self.products, self.writes = products, writes

    def _done_keys(self) -> set[str]:
        if not self.writes.exists():
            return set()
        return {json.loads(line)["key"] for line in self.writes.read_text().splitlines()}

    async def fetch_new(self, limit: int = 1000) -> list[dict]:
        rows = [json.loads(line) for line in self.products.read_text().splitlines() if line]
        return rows[:limit]

    async def get_image(self, product: dict) -> str:
        return product["image_path"]

    async def write(self, pid: str, attribute: str, value: str, source: str, key: str) -> str:
        if key in self._done_keys():
            return "already"
        self.writes.parent.mkdir(parents=True, exist_ok=True)
        with self.writes.open("a") as f:
            row = {"product_id": pid, "attribute": attribute, "value": value, "source": source}
            f.write(json.dumps({**row, "key": key}) + "\n")
        return "applied"


PRODUCTS_QUERY = """
query NewProducts($first: Int!, $query: String!) {
  products(first: $first, query: $query) {
    nodes { id title productType featuredMedia { preview { image { url } } } }
  }
}"""

SET_METAFIELD = """
mutation SetAttribute($metafields: [MetafieldsSetInput!]!) {
  metafieldsSet(metafields: $metafields) { userErrors { field message } }
}"""


def metafield_key(attribute: str) -> str:
    """'Sleeve Style' -> 'sleeve_style' (Shopify metafield keys are lowercase snake case)."""
    return re.sub(r"[^a-z0-9]+", "_", attribute.lower()).strip("_")


class ShopifyConnector:
    """Products tagged `needs-attributes` in, one metafield per attribute out.

    Shopify has no notion of our attribute lists, so `taxonomy` maps a product type to the
    attributes to fill and their allowed values. `metafieldsSet` is an upsert, so writing the
    same value twice is harmless; Shopify needs no idempotency key.
    """

    def __init__(
        self,
        shop: str,
        token: str,
        taxonomy: dict,
        client: httpx.AsyncClient,
        api_version: str = "2025-10",
    ):
        self.url = f"https://{shop}/admin/api/{api_version}/graphql.json"
        self.headers = {"X-Shopify-Access-Token": token}
        self.taxonomy, self.client = taxonomy, client

    async def _gql(self, query: str, variables: dict) -> dict:
        r = await self.client.post(
            self.url, headers=self.headers, json={"query": query, "variables": variables}
        )
        r.raise_for_status()
        body = r.json()
        if body.get("errors"):
            raise RuntimeError(f"Shopify GraphQL error: {body['errors']}")
        return body["data"]

    async def fetch_new(self, limit: int = 50) -> list[dict]:
        data = await self._gql(PRODUCTS_QUERY, {"first": limit, "query": "tag:needs-attributes"})
        out = []
        for n in data["products"]["nodes"]:
            media = n.get("featuredMedia") or {}
            image = ((media.get("preview") or {}).get("image") or {}).get("url")
            if not image or n["productType"] not in self.taxonomy:
                continue
            out.append(
                {
                    "id": n["id"],
                    "title": n["title"],
                    "category": n["productType"],
                    "image_url": image,
                    "ask": self.taxonomy[n["productType"]],
                    "attributes": {},
                }
            )
        return out

    async def get_image(self, product: dict) -> bytes:
        r = await self.client.get(product["image_url"])
        r.raise_for_status()
        return r.content

    async def write(self, pid: str, attribute: str, value: str, source: str, key: str) -> str:
        field = {
            "ownerId": pid,
            "namespace": "attributes",
            "key": metafield_key(attribute),
            "type": "single_line_text_field",
            "value": value,
        }
        data = await self._gql(SET_METAFIELD, {"metafields": [field]})
        errors = data["metafieldsSet"]["userErrors"]
        if errors:
            raise RuntimeError(f"Shopify rejected the write: {errors}")
        return "applied"
