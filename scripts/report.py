"""Collect every eval's JSON into reports/results.md, and draw the README chart."""

import json

import matplotlib.pyplot as plt

from catalog_agent.config import ROOT

R = ROOT / "reports"


def load(name: str) -> dict:
    return json.loads((R / f"{name}.json").read_text())


def main() -> None:
    tag = {k: load(k) for k in ["tagging_dev", "tagging_test", "tagging_test_fp8"]}
    qc_dev, qc_test, rel = load("qc_dev"), load("qc_test"), load("reliability")
    thr = {k: load(f"throughput_{k}") for k in ["bf16", "fp8"]}

    lines = [
        "# Results",
        "",
        "## 1. Filling in attributes (end to end through the agent)",
        "",
        "| Run | Products | Model accuracy | Auto-tagged | Accuracy of auto-tagged | "
        "Sent to review | Write fidelity | Catalog accuracy after review |",
        "|---|---|---|---|---|---|---|---|",
    ]
    names = {
        "tagging_dev": "dev, bf16",
        "tagging_test": "**test, bf16**",
        "tagging_test_fp8": "test, FP8",
    }
    for k, t in tag.items():
        lines.append(
            f"| {names[k]} | {t['products']} | {t['model_accuracy']} | "
            f"{t['auto_tagged']} | {t['auto_accuracy']} | {t['sent_to_review']} | "
            f"{t['write_fidelity']} | {t['final_catalog_accuracy']} |"
        )
    c = qc_dev["chosen"]
    lines += [
        "",
        "## 2. Catalog QC (20% of listings state a wrong value)",
        "",
        "| Split | Cut-off | Wrong listings caught | Precision of flags | "
        "False alarms on correct listings |",
        "|---|---|---|---|---|",
        f"| dev (cut-off chosen here) | {c['cutoff']} | {100 * c['catch_rate']:.1f}% | "
        f"{100 * c['precision']:.1f}% | {100 * c['false_alarm_rate']:.1f}% |",
        f"| **test** | {qc_test['cutoff']} | {100 * qc_test['catch_rate']:.1f}% "
        f"({round(qc_test['catch_rate'] * qc_test['wrong_listings'])} of "
        f"{qc_test['wrong_listings']}) | {100 * qc_test['precision']:.1f}% | "
        f"{100 * qc_test['false_alarm_rate']:.1f}% |",
    ]
    lines += [
        "",
        "## 3. Reliability (1,000 products, faults injected, process killed)",
        "",
        f"- Store: {int(100 * rel['faults']['fail_rate'])}% of requests fail (503); "
        f"{int(100 * rel['faults']['lost_ack_rate'])}% of writes are applied but the "
        "reply is lost (500)",
        f"- Agent killed with SIGKILL after {rel['killed_after_s']} s "
        f"({rel['written_before_kill']} products written), then the batch re-run",
        f"- Outcome: {rel['auto_written']} written + {rel['waiting_for_review']} waiting "
        f"for review = {rel['auto_written'] + rel['waiting_for_review']}; lost: "
        f"**{rel['lost']}**; written twice: **{rel['products_written_twice']}**",
        f"- {rel['replays_rejected']} duplicate writes were stopped by the idempotency key "
        f"(lost replies retried); {rel['retries_per_100_products']} retries per 100 "
        "products",
        f"- Passed: **{rel['passed']}**",
    ]
    lines += [
        "",
        "## 4. Throughput and cost (RTX 5090 laptop GPU, 512 px images)",
        "",
        "| Weights | What | Parallel requests | Products/s | GPU-hours per 1M | ₹ per 1,000 |",
        "|---|---|---|---|---|---|",
    ]
    for label, t in thr.items():
        for row in t["rows"]:
            lines.append(
                f"| {label} | {row['what']} | {row['parallel']} | "
                f"{row['products_per_s']} | {row['gpu_hours_per_million']} | "
                f"{row['inr_per_1000']:.2f} |"
            )
    lines += [
        "",
        "Cost = GPU time × $0.60 per GPU-hour (rented 24-32 GB GPU) at ₹95.46/$, the "
        "same assumption as the tagger project. Image resize + encode on the CPU: "
        f"{thr['bf16']['image_encode_ms_per_product']} ms per product.",
        "",
    ]
    (R / "results.md").write_text("\n".join(lines))
    print("\n".join(lines))
    chart(thr)


def chart(thr: dict) -> None:
    """Products per second: tagger alone vs the full agent, bf16 vs FP8."""
    ink, muted, surface = "#0b0b0b", "#52514e", "#fcfcfb"
    colors = {"bf16": "#8d8c86", "fp8": "#2a78d6"}  # gray = baseline precision, blue = FP8
    fig, ax = plt.subplots(figsize=(8, 3.2), dpi=200)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    groups = ["Tagger alone (64 parallel)", "Full agent: fetch, tag, route, write"]
    for i, label in enumerate(["bf16", "fp8"]):
        rows = thr[label]["rows"]
        raw = max(r["products_per_s"] for r in rows if r["what"] == "raw tagger")
        agent = next(r["products_per_s"] for r in rows if r["what"] == "full agent")
        for j, v in enumerate([raw, agent]):
            y = j * 1.0 + (0.18 if i == 0 else -0.18)
            ax.barh(y, v, height=0.34, color=colors[label], edgecolor=surface, linewidth=2)
            ax.text(v + 0.6, y, f"{v:.1f}/s  {label.upper()}", va="center", fontsize=8.5, color=ink)
    ax.set_yticks([0, 1], groups, fontsize=9, color=ink)
    ax.invert_yaxis()
    ax.set_xlim(0, 60)
    ax.set_xlabel("Products tagged per second (one RTX 5090 laptop GPU)", fontsize=8.5, color=muted)
    ax.tick_params(axis="x", colors=muted, labelsize=8)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c9c8c1")
    ax.set_title(
        "FP8 weights: +58% throughput, the agent adds almost no overhead",
        loc="left",
        fontsize=11,
        color=ink,
    )
    fig.tight_layout()
    (ROOT / "docs" / "images").mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / "docs" / "images" / "throughput.png", facecolor=surface)


if __name__ == "__main__":
    main()
