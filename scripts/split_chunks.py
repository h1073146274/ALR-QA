#!/usr/bin/env python3

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

TEXT_FIELDS = ["content", "body", "text", "Body", "document_text", "full_text"]
DOC_ID_FIELDS = ["doc_id", "document_id", "articleId", "article_id", "paper_id", "source_id", "id"]
NOISE_LINE_RE = re.compile(r"^\s*(?:page\s+)?\d+\s*$|^\s*[-_=*]{3,}\s*$", re.IGNORECASE)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+")


def read_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("input JSON must be an array")
        return [row for row in data if isinstance(row, dict)]
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def pick_text(row: dict[str, Any], field: str) -> str:
    fields = [field] if field else []
    fields.extend(TEXT_FIELDS)
    for key in fields:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def pick_doc_id(row: dict[str, Any], field: str, fallback: str) -> str:
    fields = [field] if field else []
    fields.extend(DOC_ID_FIELDS)
    for key in fields:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return fallback


def clean_text(text: str) -> str:
    value = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = []
    for paragraph in re.split(r"\n\s*\n", value):
        lines = []
        for line in paragraph.splitlines():
            line = re.sub(r"\s+", " ", line).strip()
            if line and not NOISE_LINE_RE.match(line):
                lines.append(line)
        cleaned = " ".join(lines).strip()
        if cleaned:
            paragraphs.append(cleaned)
    return "\n\n".join(paragraphs)


def clean_documents(rows: list[dict[str, Any]], text_field: str, doc_id_field: str) -> list[dict[str, Any]]:
    documents = []
    seen = set()
    for index, row in enumerate(rows):
        text = clean_text(pick_text(row, text_field))
        if not text:
            continue
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        doc_id = pick_doc_id(row, doc_id_field, digest[:16])
        text_fields = set(TEXT_FIELDS + ([text_field] if text_field else []))
        metadata = {key: value for key, value in row.items() if key not in text_fields}
        documents.append({"doc_id": doc_id, "text": text, "source_index": index, "metadata": metadata})
    return documents


def token_len(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def split_long_unit(text: str, tokenizer: Any, max_tokens: int) -> list[str]:
    if token_len(tokenizer, text) <= max_tokens:
        return [text]
    sentences = [part.strip() for part in SENTENCE_SPLIT_RE.split(text) if part.strip()]
    if len(sentences) <= 1:
        ids = tokenizer.encode(text, add_special_tokens=False)
        return [
            tokenizer.decode(ids[start : start + max_tokens], skip_special_tokens=True).strip()
            for start in range(0, len(ids), max_tokens)
            if ids[start : start + max_tokens]
        ]
    units = []
    current = []
    for sentence in sentences:
        if token_len(tokenizer, sentence) > max_tokens:
            if current:
                units.append(" ".join(current))
                current = []
            units.extend(split_long_unit(sentence, tokenizer, max_tokens))
            continue
        proposal = " ".join(current + [sentence]).strip()
        if current and token_len(tokenizer, proposal) > max_tokens:
            units.append(" ".join(current))
            current = [sentence]
        else:
            current.append(sentence)
    if current:
        units.append(" ".join(current))
    return units


def paragraph_units(text: str, tokenizer: Any, max_tokens: int) -> list[str]:
    units = []
    for paragraph in [value.strip() for value in text.split("\n\n") if value.strip()]:
        units.extend(split_long_unit(paragraph, tokenizer, max_tokens))
    return units


def overlap_tail(text: str, tokenizer: Any, overlap_tokens: int) -> str:
    if overlap_tokens <= 0:
        return ""
    ids = tokenizer.encode(text, add_special_tokens=False)
    return tokenizer.decode(ids[-overlap_tokens:], skip_special_tokens=True).strip()


def segment_document(
    text: str,
    tokenizer: Any,
    min_tokens: int,
    max_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    units = paragraph_units(text, tokenizer, max_tokens)
    chunks = []
    current = ""
    for unit in units:
        proposal = f"{current}\n\n{unit}".strip() if current else unit
        if current and token_len(tokenizer, proposal) > max_tokens:
            chunks.append(current)
            tail = overlap_tail(current, tokenizer, overlap_tokens)
            proposal = f"{tail}\n\n{unit}".strip() if tail else unit
            if token_len(tokenizer, proposal) > max_tokens:
                proposal = unit
            current = proposal
        else:
            current = proposal
    if current:
        chunks.append(current)
    if len(chunks) > 1 and token_len(tokenizer, chunks[-1]) < min_tokens:
        candidate = f"{chunks[-2]}\n\n{chunks[-1]}".strip()
        if token_len(tokenizer, candidate) <= max_tokens:
            chunks[-2] = candidate
            chunks.pop()
    return [chunk for chunk in chunks if min_tokens <= token_len(tokenizer, chunk) <= max_tokens]


def split_doc_ids(doc_ids: list[str], seed: int) -> dict[str, str]:
    ids = list(dict.fromkeys(doc_ids))
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    return {
        doc_id: "train" if i < n_train else "val" if i < n_train + n_val else "test"
        for i, doc_id in enumerate(ids)
    }


def require_int(value: str, name: str) -> int:
    if not str(value).strip():
        raise SystemExit(f"--{name} is required")
    return int(value)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="")
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--name", default="")
    ap.add_argument("--text_field", default="")
    ap.add_argument("--doc_id_field", default="")
    ap.add_argument("--tokenizer_name_or_path", default="")
    ap.add_argument("--min_tokens", default="")
    ap.add_argument("--max_tokens", default="")
    ap.add_argument("--overlap_tokens", default="")
    ap.add_argument("--sample_docs", default="")
    ap.add_argument("--seed", default="")
    args = ap.parse_args()

    required = ["input", "out_dir", "name", "tokenizer_name_or_path"]
    missing = [name for name in required if not str(getattr(args, name)).strip()]
    if missing:
        raise SystemExit("required arguments: " + ", ".join(f"--{name}" for name in missing))

    min_tokens = require_int(args.min_tokens, "min_tokens")
    max_tokens = require_int(args.max_tokens, "max_tokens")
    overlap_tokens = require_int(args.overlap_tokens, "overlap_tokens")
    sample_docs = require_int(args.sample_docs, "sample_docs")
    seed = require_int(args.seed, "seed")
    if not 0 < min_tokens <= max_tokens:
        raise SystemExit("--min_tokens must be positive and no larger than --max_tokens")
    if not 0 <= overlap_tokens < max_tokens:
        raise SystemExit("--overlap_tokens must be non-negative and smaller than --max_tokens")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    documents = clean_documents(read_records(Path(args.input)), args.text_field, args.doc_id_field)
    rng = random.Random(seed)
    if sample_docs > 0 and len(documents) > sample_docs:
        documents = rng.sample(documents, sample_docs)
    split_by_doc = split_doc_ids([doc["doc_id"] for doc in documents], seed)

    buckets = {"train": [], "val": [], "test": []}
    for doc in documents:
        split = split_by_doc[doc["doc_id"]]
        chunks = segment_document(doc["text"], tokenizer, min_tokens, max_tokens, overlap_tokens)
        for index, chunk in enumerate(chunks):
            buckets[split].append(
                {
                    **doc["metadata"],
                    "doc_id": doc["doc_id"],
                    "chunk_id": f"{doc['doc_id']}_{index}",
                    "chunk_index": index,
                    "content": chunk,
                    "split": split,
                }
            )

    out_dir = Path(args.out_dir)
    for split, rows in buckets.items():
        write_jsonl(out_dir / f"{args.name}_{split}.jsonl", rows)
    manifest = {
        "name": args.name,
        "documents": {split: len({row["doc_id"] for row in rows}) for split, rows in buckets.items()},
        "chunks": {split: len(rows) for split, rows in buckets.items()},
        "min_tokens": min_tokens,
        "max_tokens": max_tokens,
        "overlap_tokens": overlap_tokens,
        "sample_docs": sample_docs,
        "seed": seed,
    }
    (out_dir / f"{args.name}_split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
