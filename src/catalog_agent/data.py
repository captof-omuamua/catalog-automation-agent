"""Product records from ImplicitAVE (Zou et al., ACL Findings 2024), as a store would hold them.

Each ImplicitAVE row is one product with one attribute to fill (e.g. "Sleeve Style") from a
fixed list of values. We turn rows into:
  - store records: what the catalog API serves (title, category, photo, attribute to fill,
    allowed values) - no answer
  - gold records: the right value, kept apart for scoring
  - QC records: listings that already state a value; a seeded 20% state a wrong one

The row loading mirrors the visual-product-tagging project exactly (same order, same filters),
so the 2,000-row dev sample (seed 0) is the same one its cut-offs were chosen on.
"""

import csv
import json
import random
from pathlib import Path

DOMAINS = {  # tsv name -> (domain, options file, training image folder)
    "Clothing": ("Clothing", "Clothing_Shoes_and_Jewelry", "Clothing_Shoes_and_Jewelry"),
    "Footwear": ("Footwear", "Clothing_Shoes_and_Jewelry", "Clothing_Shoes_and_Jewelry"),
    "Jewlery_and_General_Apparel": (
        "Jewelry&GA",
        "Clothing_Shoes_and_Jewelry",
        "Clothing_Shoes_and_Jewelry",
    ),
    "Food": ("Food", "Food", "Grocery_and_Gourmet_Food"),
    "Home": ("Home", "Home", "Home_and_Kitchen"),
}
SIZES = {"dev": 2000, "load": 5000}  # samples drawn from the "val" and "train" splits


def load_rows(root: Path, split: str) -> list[dict]:
    """ImplicitAVE rows of "test", "val" or "train" that can be asked as a closed question."""
    evaluation = root / "repo" / "data" / "evaluation_data"
    opts = {
        name: json.loads((evaluation / f"options_{name}.json").read_text())
        for name in {d[1] for d in DOMAINS.values()}
    }
    rows = []
    for tsv, (domain, opt_file, image_dir) in DOMAINS.items():
        if split == "test":
            path = evaluation / "texts" / f"{tsv}_annotated_final.tsv"
            images = evaluation / "images"
        else:
            path = root / "repo" / "data" / "training_data" / "texts" / tsv / f"{split}.tsv"
            images = root / "train_images" / image_dir
        with path.open(encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                allowed = opts[opt_file].get(r["attribute_names"])
                image = images / r["images"]
                if not allowed or r["attribute_values"] not in allowed or not image.exists():
                    continue
                rows.append(
                    {
                        "id": r["id"],
                        "domain": domain,
                        "category": r["category"],
                        "title": r["texts"].removeprefix("[Title]").strip(),
                        "attribute": r["attribute_names"],
                        "value": r["attribute_values"],
                        "options": allowed,
                        "image_path": str(image),
                    }
                )
    return rows


def sample(rows: list[dict], n: int) -> list[dict]:
    """Same sampling as the tagger project: random.Random(0).sample."""
    return random.Random(0).sample(rows, min(n, len(rows)))


def store_record(row: dict, stated: str | None = None) -> dict:
    return {
        "id": row["id"],
        "title": row["title"],
        "category": row["category"],
        "domain": row["domain"],
        "image_path": row["image_path"],
        "ask": [{"attribute": row["attribute"], "options": row["options"]}],
        "attributes": {row["attribute"]: stated} if stated else {},
    }


def corrupt(rows: list[dict], rate: float = 0.2, seed: int = 0) -> list[tuple[dict, str, bool]]:
    """(row, stated value, is_wrong): a seeded `rate` of rows state another allowed value."""
    rng = random.Random(seed)
    out = []
    for r in rows:
        if rng.random() < rate:
            wrong = rng.choice([v for v in r["options"] if v != r["value"]])
            out.append((r, wrong, True))
        else:
            out.append((r, r["value"], False))
    return out


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def build(root: Path, out: Path) -> dict[str, int]:
    """Write products_/gold_ files for dev, test and load, and qc_ files for dev and test."""
    splits = {
        "dev": sample(load_rows(root, "val"), SIZES["dev"]),
        "test": load_rows(root, "test"),
        "load": sample(load_rows(root, "train"), SIZES["load"]),
    }
    for name, rows in splits.items():
        write_jsonl(out / f"products_{name}.jsonl", [store_record(r) for r in rows])
        write_jsonl(
            out / f"gold_{name}.jsonl",
            [
                {
                    "id": r["id"],
                    "domain": r["domain"],
                    "attribute": r["attribute"],
                    "value": r["value"],
                }
                for r in rows
            ],
        )
    for name in ["dev", "test"]:
        listings = corrupt(splits[name], seed=1 if name == "dev" else 2)
        write_jsonl(out / f"qc_{name}.jsonl", [store_record(r, s) for r, s, _ in listings])
        write_jsonl(
            out / f"qc_gold_{name}.jsonl",
            [
                {
                    "id": r["id"],
                    "attribute": r["attribute"],
                    "value": r["value"],
                    "stated": s,
                    "is_wrong": w,
                }
                for r, s, w in listings
            ],
        )
    return {k: len(v) for k, v in splits.items()}
