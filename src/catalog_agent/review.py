"""The human review queue: products the model was unsure about, or listings QC flagged.

SQLite by default (one file, no server). Each row is one paused agent thread; approving it
resumes that thread. Adding the same thread twice is a no-op, because a LangGraph node that
pauses runs again from the top when it resumes.
"""

import json
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS review (
  thread_id TEXT PRIMARY KEY,
  product_id TEXT, title TEXT, image TEXT, attribute TEXT, options TEXT,
  kind TEXT,            -- "fill": model unsure | "qc": stated value disagrees with photo
  predicted TEXT, confidence REAL, stated TEXT,
  status TEXT DEFAULT 'pending',   -- pending | approved
  final_value TEXT,
  created TEXT DEFAULT CURRENT_TIMESTAMP
)"""


class ReviewQueue:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute(SCHEMA)

    def add(
        self,
        thread_id: str,
        product: dict,
        attribute: str,
        options: list[str],
        kind: str,
        predicted: str | None,
        confidence: float,
        stated: str | None = None,
    ) -> None:
        self.db.execute(
            "INSERT OR IGNORE INTO review (thread_id, product_id, title, image, attribute, "
            "options, kind, predicted, confidence, stated) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                thread_id,
                product["id"],
                product.get("title", ""),
                product.get("image_path") or product.get("image_url", ""),
                attribute,
                json.dumps(options),
                kind,
                predicted,
                confidence,
                stated,
            ),
        )
        self.db.commit()

    def pending(self, limit: int = 50) -> list[dict]:
        rows = self.db.execute(
            "SELECT * FROM review WHERE status='pending' ORDER BY confidence LIMIT ?", (limit,)
        ).fetchall()
        return [{**dict(r), "options": json.loads(r["options"])} for r in rows]

    def get(self, thread_id: str) -> dict | None:
        r = self.db.execute("SELECT * FROM review WHERE thread_id=?", (thread_id,)).fetchone()
        return {**dict(r), "options": json.loads(r["options"])} if r else None

    def approve(self, thread_id: str, value: str) -> None:
        self.db.execute(
            "UPDATE review SET status='approved', final_value=? WHERE thread_id=?",
            (value, thread_id),
        )
        self.db.commit()

    def counts(self) -> dict:
        rows = self.db.execute("SELECT status, COUNT(*) FROM review GROUP BY status").fetchall()
        return {r[0]: r[1] for r in rows}
