#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

VIEW_DESCRIPTIONS = {
    "concept": "Ask about the core idea, meaning, or central object.",
    "mechanism": "Ask how something works, unfolds, or is carried out.",
    "condition": "Ask about prerequisites, constraints, scope, or validity conditions.",
    "evidence": "Ask for observations, facts, or direct support located in the Body.",
    "causation": "Ask about drivers, reasons, or factors that lead to an outcome.",
    "comparison": "Ask by contrasting alternatives, baselines, or differences.",
    "impact": "Ask about consequences, implications, roles, or practical impact.",
    "quantitative": "Ask for numbers, metrics, time, stages, or other structured facts answerable from the Body.",
}


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:

    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1e-6)
    return summed / denom


@torch.no_grad()
def encode_texts(
    texts: List[str],
    tokenizer,
    model,
    device: str,
    batch_size: int,
    max_length: int,
    use_bf16: bool,
    normalize: bool,
) -> torch.Tensor:
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        if use_bf16 and device.startswith("cuda"):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(**enc)
        else:
            out = model(**enc)

        pooled = mean_pool(out.last_hidden_state, enc["attention_mask"])

        if normalize:
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)

        all_embs.append(pooled.detach().cpu())

    return torch.cat(all_embs, dim=0)


def _flatten_merged_qa_result(obj: Dict[str, Any]) -> List[dict]:

    chunk_results = obj.get("chunk_results")
    if not isinstance(chunk_results, list):
        return []

    rows: List[dict] = []
    for cr in chunk_results:
        if not isinstance(cr, dict):
            continue
        chunk_id = cr.get("chunk_id")
        chunk_summary = str(cr.get("chunk_summary", "") or "")
        keywords = cr.get("keywords", [])
        source_round = cr.get("_source_round", None)

        detail_by_question: Dict[str, Dict[str, Any]] = {}
        for d in cr.get("all_similarity_details") or []:
            if isinstance(d, dict):
                q = str(d.get("question", "") or "")
                if q and q not in detail_by_question:
                    detail_by_question[q] = d

        for cand in cr.get("candidate_results") or []:
            if not isinstance(cand, dict):
                continue
            for qa in cand.get("qualified_qa_pairs") or []:
                if not isinstance(qa, dict):
                    continue
                meta = qa.get("meta") if isinstance(qa.get("meta"), dict) else {}
                view = str((meta or {}).get("view", "") or qa.get("view", "") or "").strip()
                question = str(qa.get("question", "") or "")
                answer = str(qa.get("answer", qa.get("final_answer", "")) or "")
                detail = detail_by_question.get(question, {})
                chunk_text = str(detail.get("chunk_text", "") or "")

                row = dict(qa)
                row["chunk_id"] = chunk_id
                row["question"] = question
                row["answer"] = answer
                row["final_answer"] = answer
                row["view"] = view
                row["chunk_summary"] = chunk_summary
                row["summary"] = chunk_summary
                row["chunk_anchor_text"] = chunk_text or chunk_summary
                row["paragraph_keywords"] = keywords
                row["keywords"] = keywords
                row["candidate_id"] = qa.get("candidate_id", cand.get("candidate_id", None))
                if source_round is not None:
                    row["_source_round"] = source_round
                rows.append(row)

    return rows


def read_jsonl(path: Path) -> List[dict]:

    with path.open("r", encoding="utf-8") as f:
        pos = f.tell()
        ch = f.read(1)
        while ch and ch.isspace():
            ch = f.read(1)
        f.seek(pos)
        if ch in "[{":
            try:
                obj = json.load(f)
            except json.JSONDecodeError:
                obj = None
                f.seek(pos)
            if isinstance(obj, list):
                return obj
            if isinstance(obj, dict):
                flattened = _flatten_merged_qa_result(obj)
                if flattened:
                    return flattened
                for key in ("data", "rows", "examples"):
                    rows = obj.get(key)
                    if isinstance(rows, list):
                        return rows
                raise ValueError("JSON object must contain a supported merged QA result or data/rows/examples list.")

        rows = []
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                snippet = line[:300]
                raise ValueError(
                    f"Invalid JSON on line {i} in {path} (len={len(line)}). Snippet: {snippet!r}"
                ) from e
        return rows


def _get_field(row: Dict[str, Any], keys: List[str], default: str = "") -> str:
    for k in keys:
        v = row.get(k, None)
        if isinstance(v, str) and v.strip():
            return v
    return default


def _get_meta_field(row: Dict[str, Any], key: str, default: Any = "") -> Any:
    meta = row.get("meta", None)
    if isinstance(meta, dict) and key in meta:
        return meta.get(key, default)
    return default


def _get_nested_field(row: Dict[str, Any], key: str, default: Any = "") -> Any:
    if key in row:
        return row.get(key, default)
    meta = row.get("meta", None)
    if isinstance(meta, dict) and key in meta:
        return meta.get(key, default)
    return default


def _to_list_keywords(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        out = []
        for v in x:
            s = str(v).strip()
            if s:
                out.append(s)
        return out
    s = str(x).strip()
    if not s:
        return []
    parts = s.replace("，", ",").replace("；", ",").replace("、", ",").split(",")
    return [p.strip() for p in parts if p and p.strip()]


def _append_keywords(text: str, kws: List[str]) -> str:
    if not kws:
        return text
    return text.rstrip() + "\n\nKeywords: " + ", ".join(kws)


def _view_text_from_label(label: str) -> str:
    view = str(label or "").strip()
    if not view:
        return ""
    desc = VIEW_DESCRIPTIONS.get(view, "")
    return f"{view}: {desc}" if desc else view


def _canonical_view_mean_text() -> str:
    parts = [_view_text_from_label(v) for v in VIEW_DESCRIPTIONS]
    return "\n".join(p for p in parts if p)


def build_view_texts(
    rows: List[Dict[str, Any]],
    *,
    view_field: str,
    view_source: str,
) -> Tuple[List[str], str]:

    if view_source not in {"field", "canonical_mean", "auto"}:
        raise ValueError(f"Unsupported view_source={view_source}")

    canonical_text = _canonical_view_mean_text()
    raw_labels = [str(_get_nested_field(r, view_field, "") or "").strip() for r in rows]

    if view_source == "canonical_mean":
        return [canonical_text for _ in rows], "canonical_mean"

    if view_source == "field":
        return [_view_text_from_label(v) for v in raw_labels], "field"

    if any(raw_labels):
        return [
            _view_text_from_label(v) if v else canonical_text
            for v in raw_labels
        ], "auto_field_with_canonical_fallback"
    return [canonical_text for _ in rows], "auto_canonical_mean"


def build_anchor_texts(row: Dict[str, Any], append_keywords: bool) -> Tuple[str, str]:
    doc_text = _get_field(
        row,
        ["doc_anchor_text", "article_summary", "doc_summary", "Article Summary", "articleSummary"],
        "",
    )
    if not doc_text:
        doc_text = str(_get_meta_field(row, "abstract", "") or "")
    if not doc_text:
        doc_text = _get_field(row, ["title", "doc_title"], "")

    chunk_text = _get_field(
        row,
        ["chunk_anchor_text", "chunk_summary", "Chunk Summary", "summary"],
        "",
    )
    if not chunk_text:
        chunk_text = _get_field(row, ["body", "chunk_text", "text", "Body", "content"], "")

    if append_keywords:
        doc_kws = _to_list_keywords(_get_field(row, ["doc_keywords", "doc_kw"], _get_meta_field(row, "keywords", "")))
        chunk_kws = _to_list_keywords(_get_field(row, ["chunk_keywords", "chunk_kw", "paragraph_keywords"], ""))
        doc_text = _append_keywords(doc_text, doc_kws)
        chunk_text = _append_keywords(chunk_text, chunk_kws)

    return doc_text or "", chunk_text or ""


def write_jsonl(path: Path, rows: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", type=str, required=True, help="JSONL or JSON array input")
    ap.add_argument("--out_dir", type=str, required=True, help="output directory")
    ap.add_argument(
        "--encoder_name_or_path",
        type=str,
        required=True,
        help="HF encoder model name or local path (must be available on your machine)",
    )
    ap.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_length", type=int, default=256, help="truncate anchors to this length")
    ap.add_argument("--bf16", action="store_true", help="use bf16 autocast on CUDA")
    ap.add_argument("--normalize", action="store_true", help="L2-normalize embeddings")
    ap.add_argument("--view_field", type=str, default="view", help="field name for view anchor text")
    ap.add_argument(
        "--view_source",
        type=str,
        default="auto",
        choices=["auto", "field", "canonical_mean"],
        help=(
            "source for line-aligned view anchors when --encode_view is set. "
            "auto uses row view labels when available, otherwise a shared canonical 8-view text."
        ),
    )
    ap.add_argument(
        "--encode_view",
        action="store_true",
        help="also encode line-aligned view text and write view_anchor_emb.pt/.npy",
    )
    ap.add_argument(
        "--write_indexed_jsonl",
        action="store_true",
        help="also write a new JSONL with doc_anchor_idx/chunk_anchor_idx = line index",
    )
    ap.add_argument(
        "--append_keywords",
        action="store_true",
        help="append doc/chunk keywords into anchor text (optional)",
    )
    args = ap.parse_args()

    in_path = Path(args.in_jsonl)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(in_path)
    n = len(rows)
    if n == 0:
        raise RuntimeError(f"No rows found in {in_path}")


    doc_texts = []
    chunk_texts = []
    for r in rows:
        d, c = build_anchor_texts(r, append_keywords=args.append_keywords)
        doc_texts.append(d)
        chunk_texts.append(c)
    view_source_used = ""
    if args.encode_view:
        view_texts, view_source_used = build_view_texts(
            rows,
            view_field=args.view_field,
            view_source=args.view_source,
        )
    else:
        view_texts = []
    view_prototype_texts = [_view_text_from_label(v) for v in VIEW_DESCRIPTIONS] if args.encode_view else []


    uniq_map: Dict[str, int] = {}
    uniq_texts: List[str] = []
    all_anchor_texts = doc_texts + chunk_texts + view_texts + view_prototype_texts
    for t in all_anchor_texts:
        t = t.strip()
        if t not in uniq_map:
            uniq_map[t] = len(uniq_texts)
            uniq_texts.append(t)


    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, switching to CPU.")
        device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(
        args.encoder_name_or_path,
        use_fast=True,
        trust_remote_code=True,
    )

    model = AutoModel.from_pretrained(
        args.encoder_name_or_path,
        trust_remote_code=True,
    )

    model.eval()
    model.to(device)

    print(f"[INFO] Loaded encoder: {args.encoder_name_or_path}")
    print(f"[INFO] Rows: {n}")
    print(f"[INFO] Unique anchor texts: {len(uniq_texts)}")
    print(f"[INFO] Encoding on {device}, batch={args.batch_size}, max_length={args.max_length}, bf16={args.bf16}, normalize={args.normalize}")


    uniq_emb = encode_texts(
        texts=uniq_texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        use_bf16=args.bf16,
        normalize=args.normalize,
    )


    doc_idx = torch.tensor([uniq_map[t.strip()] for t in doc_texts], dtype=torch.long)
    chunk_idx = torch.tensor([uniq_map[t.strip()] for t in chunk_texts], dtype=torch.long)
    view_idx = torch.tensor([uniq_map[t.strip()] for t in view_texts], dtype=torch.long) if args.encode_view else None
    view_proto_idx = (
        torch.tensor([uniq_map[t.strip()] for t in view_prototype_texts], dtype=torch.long)
        if args.encode_view
        else None
    )

    doc_emb = uniq_emb.index_select(0, doc_idx)
    chunk_emb = uniq_emb.index_select(0, chunk_idx)
    view_emb = uniq_emb.index_select(0, view_idx) if args.encode_view else None
    view_proto_emb = uniq_emb.index_select(0, view_proto_idx) if args.encode_view else None


    doc_pt = out_dir / "doc_anchor_emb.pt"
    chunk_pt = out_dir / "chunk_anchor_emb.pt"
    view_pt = out_dir / "view_anchor_emb.pt"
    view_proto_pt = out_dir / "view_prototype_emb.pt"
    uniq_pt = out_dir / "anchor_text_unique_emb.pt"
    meta_path = out_dir / "anchor_text_unique_meta.json"

    torch.save(doc_emb, doc_pt)
    torch.save(chunk_emb, chunk_pt)
    if view_emb is not None:
        torch.save(view_emb, view_pt)
    if view_proto_emb is not None:
        torch.save(view_proto_emb, view_proto_pt)
    torch.save(uniq_emb, uniq_pt)

    np.save(out_dir / "doc_anchor_emb.npy", doc_emb.numpy())
    np.save(out_dir / "chunk_anchor_emb.npy", chunk_emb.numpy())
    if view_emb is not None:
        np.save(out_dir / "view_anchor_emb.npy", view_emb.numpy())
    if view_proto_emb is not None:
        np.save(out_dir / "view_prototype_emb.npy", view_proto_emb.numpy())
    np.save(out_dir / "anchor_text_unique_emb.npy", uniq_emb.numpy())


    meta = {
        "encoder_name_or_path": args.encoder_name_or_path,
        "max_length": args.max_length,
        "normalize": args.normalize,
        "encode_view": args.encode_view,
        "view_field": args.view_field,
        "view_source": args.view_source,
        "view_source_used": view_source_used,
        "view_names": list(VIEW_DESCRIPTIONS.keys()),
        "view_descriptions": VIEW_DESCRIPTIONS,
        "n_rows": n,
        "n_unique": len(uniq_texts),
        "unique_texts": uniq_texts,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


    if args.write_indexed_jsonl:
        out_jsonl = out_dir / (in_path.stem + ".with_anchor_index.jsonl")
        new_rows = []
        for i, r in enumerate(rows):
            rr = dict(r)
            rr["anchor_row_idx"] = i

            if args.encode_view:
                rr["view_anchor_idx"] = i
                rr["view_anchor_source"] = view_source_used
            new_rows.append(rr)
        write_jsonl(out_jsonl, new_rows)
        print(f"[INFO] Wrote indexed JSONL: {out_jsonl}")

    print("[DONE]")
    print(f"[OUT] {doc_pt}")
    print(f"[OUT] {chunk_pt}")
    if view_emb is not None:
        print(f"[OUT] {view_pt}")
    if view_proto_emb is not None:
        print(f"[OUT] {view_proto_pt}")
    print(f"[OUT] {uniq_pt}")
    print(f"[OUT] {meta_path}")


if __name__ == "__main__":
    main()
