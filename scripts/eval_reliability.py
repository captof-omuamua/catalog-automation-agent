"""Eval 3: does the agent survive failures without losing or duplicating work?

Setup: 1,000 dev products in the mock store, which now
  - fails 10% of requests with a 503 (listing, writes)
  - applies 5% of writes but loses the reply (500): a naive client would write twice
Then the agent (a separate process) is killed with SIGKILL partway through, and the same
batch is run again with --status all (every product is re-checked).

Pass conditions: every product ends auto-written or waiting for review, none is lost, and no
product is written twice.
"""

import asyncio
import json
import signal
import subprocess
import sys
import time

import httpx

from catalog_agent.config import ROOT, settings
from catalog_agent.evaluate import fresh_dir, load_store, read_jsonl
from catalog_agent.review import ReviewQueue

N = 1000
KILL_AFTER_S = 15


def agent_cmd(run_dir, status: str) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts" / "run_agent.py"),
        "--status",
        status,
        "--checkpoints",
        str((run_dir / "checkpoints.sqlite").relative_to(ROOT)),
        "--review-db",
        str((run_dir / "review.sqlite").relative_to(ROOT)),
        "--trace",
        str((run_dir / "trace.jsonl").relative_to(ROOT)),
    ]


async def main() -> None:
    run_dir = fresh_dir("reliability")
    subset = run_dir / "products.jsonl"
    lines = (settings.data_dir / "products_dev.jsonl").read_text().splitlines()[:N]
    subset.write_text("\n".join(lines) + "\n")
    async with httpx.AsyncClient(timeout=60) as http:
        await load_store(http, subset)
        await http.post(
            f"{settings.catalog_url}/_admin/faults",
            json={"fail_rate": 0.10, "lost_ack_rate": 0.05, "seed": 0},
        )

        # Run 1: killed partway (SIGKILL: no clean-up, like a crashed machine)
        log1 = (ROOT / "logs" / "reliability_run1.log").open("w")
        p = subprocess.Popen(agent_cmd(run_dir, "new"), stdout=log1, stderr=subprocess.STDOUT)
        time.sleep(KILL_AFTER_S)
        p.send_signal(signal.SIGKILL)
        p.wait()
        mid = (await http.get(f"{settings.catalog_url}/_admin/stats")).json()

        # Run 2: the same batch again, every product re-checked
        t = time.perf_counter()
        log2 = (ROOT / "logs" / "reliability_run2.log").open("w")
        rc = subprocess.run(agent_cmd(run_dir, "all"), stdout=log2, stderr=subprocess.STDOUT)
        run2_s = time.perf_counter() - t
        final = (await http.get(f"{settings.catalog_url}/_admin/stats")).json()
        applied = (await http.get(f"{settings.catalog_url}/_admin/applied")).json()
        await http.post(f"{settings.catalog_url}/_admin/faults", json={})

    trace = read_jsonl(run_dir / "trace.jsonl")
    retries = sum(r.get("retries", 0) for r in trace)
    written = {w["product_id"] for w in applied}
    waiting = {r["product_id"] for r in ReviewQueue(run_dir / "review.sqlite").pending(100_000)}
    ids = {json.loads(line)["id"] for line in lines}
    result = {
        "products": N,
        "faults": {"fail_rate": 0.10, "lost_ack_rate": 0.05},
        "killed_after_s": KILL_AFTER_S,
        "written_before_kill": mid["applied"],
        "rerun_exit_code": rc.returncode,
        "rerun_seconds": round(run2_s, 1),
        "auto_written": len(written),
        "waiting_for_review": len(waiting),
        "lost": len(ids - written - waiting),
        "written_and_in_review": len(written & waiting),
        "products_written_twice": final["products_written_twice"],
        "replays_rejected": final["replays_rejected"],
        "retries": retries,
        "retries_per_100_products": round(100 * retries / N, 1),
        "passed": not (ids - written - waiting)
        and final["products_written_twice"] == 0
        and not (written & waiting)
        and rc.returncode == 0,
    }
    (settings.reports_dir / "reliability.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
