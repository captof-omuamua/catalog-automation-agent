"""Send a run summary to Slack (incoming webhook) or, with no webhook set, to a local log.

The local log (logs/notifications.jsonl) stands in for Slack in demos and tests, so nothing
depends on a Slack account.
"""

import json
import time

import httpx

from catalog_agent.config import ROOT, settings

LOCAL_LOG = ROOT / "logs" / "notifications.jsonl"


async def notify(http: httpx.AsyncClient, text: str, webhook: str | None = None) -> str:
    webhook = settings.slack_webhook_url if webhook is None else webhook
    if webhook:
        r = await http.post(webhook, json={"text": text})
        r.raise_for_status()
        return "slack"
    LOCAL_LOG.parent.mkdir(exist_ok=True)
    with LOCAL_LOG.open("a") as f:
        f.write(json.dumps({"ts": time.time(), "text": text}) + "\n")
    return "local"
