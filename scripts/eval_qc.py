"""Eval 2: catalog QC. Can the agent spot listings whose stated attribute is wrong?

Listings state a value for the attribute; a seeded 20% state a wrong (but allowed) value
(data.corrupt). The agent tags each photo and flags the listing when it confidently picks a
different value: pred != stated and confidence >= QC cut-off.

  --split dev   runs QC with cut-off 0 (every disagreement flagged), then sweeps cut-offs
                offline from the confidences and picks the one with the most catches at
                precision >= 90%. Writes reports/qc_dev.json (incl. the sweep).
  --split test  runs once with the cut-off chosen on dev.
"""

import argparse
import asyncio
import json

import httpx

from catalog_agent.agent import run_batch
from catalog_agent.config import settings
from catalog_agent.evaluate import eval_deps, fresh_dir, load_store, read_jsonl, tag_traces

TARGET_PRECISION = 0.90


def score(rows: list[dict], tags: dict, cutoff: float) -> dict:
    flagged = {
        r["id"]
        for r in rows
        if r["id"] in tags
        and tags[r["id"]]["value"] != r["stated"]
        and tags[r["id"]]["confidence"] >= cutoff
    }
    wrong = {r["id"] for r in rows if r["is_wrong"]}
    clean = len(rows) - len(wrong)
    caught = len(flagged & wrong)
    return {
        "cutoff": cutoff,
        "flagged": len(flagged),
        "catch_rate": round(caught / len(wrong), 4),
        "precision": round(caught / len(flagged), 4) if flagged else None,
        "false_alarm_rate": round(len(flagged - wrong) / clean, 4),
    }


async def run(split: str, cutoff: float) -> dict:
    run_dir = fresh_dir(f"qc_{split}")
    async with httpx.AsyncClient(timeout=120) as http:
        await load_store(http, settings.data_dir / f"qc_{split}.jsonl")
        deps = eval_deps(http, run_dir, qc_threshold=cutoff)
        products = await deps.connector.fetch_new(limit=100_000)
        s = await run_batch(
            deps,
            products,
            run_dir / "checkpoints.sqlite",
            mode="qc",
            concurrency=settings.concurrency,
        )
    rows = read_jsonl(settings.data_dir / f"qc_gold_{split}.jsonl")
    tags = tag_traces(run_dir)
    online = s.decisions["qc_flag"]
    return {"rows": rows, "tags": tags, "online_flags": online}


async def main(split: str, cutoff: float | None) -> None:
    if split == "dev":
        r = await run("dev", 0.0)
        sweep = [score(r["rows"], r["tags"], c / 100) for c in range(0, 100, 2)]
        ok = [s for s in sweep if s["precision"] and s["precision"] >= TARGET_PRECISION]
        best = max(ok, key=lambda s: (s["catch_rate"], -s["cutoff"])) if ok else sweep[-1]
        result = {
            "split": "dev",
            "chosen": best,
            "target_precision": TARGET_PRECISION,
            "online_flags_at_0": r["online_flags"],
            "sweep": sweep,
        }
    else:
        if cutoff is None:
            cutoff = json.loads((settings.reports_dir / "qc_dev.json").read_text())["chosen"][
                "cutoff"
            ]
        r = await run("test", cutoff)
        result = {
            "split": "test",
            **score(r["rows"], r["tags"], cutoff),
            "online_flags": r["online_flags"],
            "wrong_listings": sum(x["is_wrong"] for x in r["rows"]),
            "listings": len(r["rows"]),
        }
    (settings.reports_dir / f"qc_{split}.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "sweep"}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--cutoff", type=float, default=None, help="test only; default: dev's choice")
    a = ap.parse_args()
    asyncio.run(main(a.split, a.cutoff))
