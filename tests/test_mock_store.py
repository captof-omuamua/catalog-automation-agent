import json

from fastapi.testclient import TestClient

from catalog_agent import mock_store


def _client(tmp_path):
    f = tmp_path / "p.jsonl"
    f.write_text(json.dumps({"id": "A1", "title": "Tee", "image_path": "x.jpg"}))
    c = TestClient(mock_store.app)
    c.post("/_admin/faults", json={})
    c.post("/_admin/load", params={"path": str(f)})
    return c


def _write(c, key="A1|Sleeve|v1"):
    body = {"attribute": "Sleeve", "value": "Short", "source": "model", "idempotency_key": key}
    return c.put("/products/A1/attributes", json=body)


def test_replayed_write_is_rejected_not_applied(tmp_path):
    c = _client(tmp_path)
    assert _write(c).status_code == 200
    assert _write(c).status_code == 409
    s = c.get("/_admin/stats").json()
    assert s["applied"] == 1 and s["replays_rejected"] == 1
    assert c.get("/products/A1").json()["status"] == "tagged"


def test_lost_ack_applies_the_write_but_fails_the_reply(tmp_path):
    c = _client(tmp_path)
    c.post("/_admin/faults", json={"lost_ack_rate": 1.0})
    assert _write(c).status_code == 500
    assert c.get("/_admin/stats").json()["applied"] == 1
