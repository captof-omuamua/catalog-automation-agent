from catalog_agent.review import ReviewQueue
from catalog_agent.sheets import HEADER, approved_rows, push_pending


class FakeWorksheet:
    """The two gspread calls we use, in memory."""

    def __init__(self):
        self.rows = []

    def clear(self):
        self.rows = []

    def update(self, rows):
        self.rows = rows

    def get_all_records(self):
        return [dict(zip(self.rows[0], r, strict=True)) for r in self.rows[1:]]


def test_round_trip_only_accepts_allowed_values(tmp_path):
    q = ReviewQueue(tmp_path / "r.sqlite")
    for pid in ["A", "B"]:
        q.add(
            f"{pid}|Sleeve",
            {"id": pid, "title": pid},
            "Sleeve",
            ["Long", "Short"],
            "fill",
            "Long",
            0.4,
        )
    ws = FakeWorksheet()
    assert push_pending(q, ws) == 2 and ws.rows[0] == HEADER
    ws.rows[1][-1] = "Short"  # a reviewer filled row A
    ws.rows[2][-1] = "Medium"  # not an allowed value: ignored
    assert approved_rows(ws, q) == [("A|Sleeve", "Short")]
