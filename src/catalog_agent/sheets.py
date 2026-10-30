"""Optional: mirror the review queue to a Google Sheet, for teams that review in Sheets.

push_pending   writes every pending item as a row (one row per paused agent thread)
approved_rows  reads back rows where a reviewer filled the "approved_value" column

Needs `pip install .[sheets]`, a Google service account JSON key, and the sheet shared with
that account:  gspread.service_account(filename=...).open_by_key(...).sheet1
"""

HEADER = [
    "thread_id",
    "title",
    "attribute",
    "options",
    "model_guess",
    "confidence",
    "listing_says",
    "approved_value",
]


def push_pending(queue, worksheet) -> int:
    rows = [
        [
            it["thread_id"],
            it["title"],
            it["attribute"],
            " | ".join(it["options"]),
            it["predicted"] or "",
            round(it["confidence"], 3),
            it["stated"] or "",
            "",
        ]
        for it in queue.pending(limit=10_000)
    ]
    worksheet.clear()
    worksheet.update([HEADER, *rows])
    return len(rows)


def approved_rows(worksheet, queue) -> list[tuple[str, str]]:
    """(thread_id, value) for rows a reviewer filled with one of the allowed values."""
    out = []
    for r in worksheet.get_all_records():
        value = str(r.get("approved_value", "")).strip()
        item = queue.get(r["thread_id"])
        if value and item and item["status"] == "pending" and value in item["options"]:
            out.append((r["thread_id"], value))
    return out
