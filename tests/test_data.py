from catalog_agent.data import corrupt, store_record


def _row(i):
    return {
        "id": f"P{i}",
        "domain": "Clothing",
        "category": "Tops",
        "title": "Tee",
        "attribute": "Sleeve",
        "value": "Short",
        "options": ["Short", "Long", "None"],
        "image_path": "x.jpg",
    }


def test_store_record_hides_the_answer():
    rec = store_record(_row(0))
    assert "value" not in rec and rec["attributes"] == {}
    assert rec["ask"] == [{"attribute": "Sleeve", "options": ["Short", "Long", "None"]}]


def test_corruption_is_seeded_about_20_percent_and_always_wrong():
    rows = [_row(i) for i in range(2000)]
    a, b = corrupt(rows, seed=1), corrupt(rows, seed=1)
    assert a == b
    wrong = [(r, s) for r, s, w in a if w]
    assert 0.17 < len(wrong) / len(rows) < 0.23
    assert all(s != r["value"] and s in r["options"] for r, s in wrong)
    assert all(s == r["value"] for r, s, w in a if not w)
