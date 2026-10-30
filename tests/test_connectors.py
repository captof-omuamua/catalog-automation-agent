import json

import httpx
import pytest

from catalog_agent.connectors import FileConnector, ShopifyConnector, metafield_key

# A recorded-shape reply from the Shopify Admin GraphQL API (products query)
SHOPIFY_PRODUCTS = {
    "data": {
        "products": {
            "nodes": [
                {
                    "id": "gid://shopify/Product/1",
                    "title": "Raglan Tee",
                    "productType": "Shirts & Tops",
                    "featuredMedia": {"preview": {"image": {"url": "https://cdn/x.jpg"}}},
                },
                {
                    "id": "gid://shopify/Product/2",
                    "title": "No photo",
                    "productType": "Shirts & Tops",
                    "featuredMedia": None,
                },
            ]
        }
    }
}
TAXONOMY = {"Shirts & Tops": [{"attribute": "Sleeve Style", "options": ["Long", "Short"]}]}


def _shopify(handler):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return ShopifyConnector("demo.myshopify.com", "tok", TAXONOMY, client)


async def test_shopify_fetch_maps_products_and_skips_ones_without_a_photo():
    def handler(request):
        assert request.headers["X-Shopify-Access-Token"] == "tok"
        assert json.loads(request.content)["variables"]["query"] == "tag:needs-attributes"
        return httpx.Response(200, json=SHOPIFY_PRODUCTS)

    products = await _shopify(handler).fetch_new()
    assert [p["id"] for p in products] == ["gid://shopify/Product/1"]
    assert products[0]["ask"] == TAXONOMY["Shirts & Tops"]


async def test_shopify_write_sets_a_metafield_and_surfaces_user_errors():
    sent = []

    def ok(request):
        sent.append(json.loads(request.content)["variables"]["metafields"][0])
        return httpx.Response(200, json={"data": {"metafieldsSet": {"userErrors": []}}})

    assert await _shopify(ok).write("gid://1", "Sleeve Style", "Short", "model", "k") == "applied"
    assert sent[0]["key"] == "sleeve_style" and sent[0]["namespace"] == "attributes"

    def bad(request):
        errs = [{"field": ["value"], "message": "too long"}]
        return httpx.Response(200, json={"data": {"metafieldsSet": {"userErrors": errs}}})

    with pytest.raises(RuntimeError):
        await _shopify(bad).write("gid://1", "Sleeve Style", "Short", "model", "k")


async def test_file_connector_writes_each_key_once(tmp_path):
    products = tmp_path / "in.jsonl"
    products.write_text(json.dumps({"id": "A", "image_path": "a.jpg"}) + "\n")
    c = FileConnector(products, tmp_path / "out.jsonl")
    assert await c.write("A", "Sleeve", "Short", "model", "A|Sleeve|v1") == "applied"
    assert await c.write("A", "Sleeve", "Short", "model", "A|Sleeve|v1") == "already"
    assert len((tmp_path / "out.jsonl").read_text().splitlines()) == 1


def test_metafield_key():
    assert metafield_key("Heel Height (in)") == "heel_height_in"
