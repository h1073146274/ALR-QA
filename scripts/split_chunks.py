#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("input JSON must be an array")
        return [x for x in data if isinstance(x, dict)]
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def pick_doc_id(row: dict[str, Any], field: str) -> str:
    keys = [field] if field else []
    keys += ["doc_id", "document_id", "articleId", "article_id", "paper_id", "source_id", "title", "id"]
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def split_docs(doc_ids: list[str], seed: int) -> tuple[set[str], set[str], set[str]]:
    rng = random.Random(seed)
    ids = list(dict.fromkeys(doc_ids))
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    if n >= 3:
        n_train = max(1, n_train)
        n_val = max(1, n_val)
    train = set(ids[:n_train])
    val = set(ids[n_train : n_train + n_val])
    test = set(ids[n_train + n_val :])
    return train, val, test


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="")
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--name", default="")
    ap.add_argument("--doc_id_field", default="")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.input or not args.out_dir:
        raise SystemExit("--input and --out_dir are required")

    rows = read_records(Path(args.input))
    rows = [r for r in rows if str(r.get("content", r.get("body", r.get("text", ""))) or "").strip()]
    name = args.name or Path(args.input).stem
    for i, row in enumerate(rows):
        row.setdefault("chunk_id", row.get("id", f"{name}_{i}"))
        row["_split_doc_id"] = pick_doc_id(row, args.doc_id_field) or str(row["chunk_id"])

    train_docs, val_docs, test_docs = split_docs([r["_split_doc_id"] for r in rows], args.seed)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        doc_id = row["_split_doc_id"]
        if doc_id in train_docs:
            split = "train"
        elif doc_id in val_docs:
            split = "val"
        else:
            split = "test"
        clean = dict(row)
        clean["split"] = split
        buckets[split].append(clean)

    out_dir = Path(args.out_dir)
    for split in ["train", "val", "test"]:
        write_jsonl(out_dir / f"{name}_{split}.jsonl", buckets[split])

    manifest = {
        "name": name,
        "input": str(Path(args.input).resolve()),
        "seed": args.seed,
        "rows": {k: len(v) for k, v in buckets.items()},
        "docs": {"train": len(train_docs), "val": len(val_docs), "test": len(test_docs)},
    }
    (out_dir / f"{name}_split_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
