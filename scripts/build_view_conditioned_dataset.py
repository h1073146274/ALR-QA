#!/usr/bin/env python3

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "configs"))

from qa_prompts_updated import (  # noqa: E402
    QAContext,
    build_acceptance_filter_prompt,
    build_answer_prompt_sft,
    build_question_prompt_sft,
    build_teacher_answer_prompt,
    build_teacher_question_prompt,
    build_view_selection_prompt,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    return [str(value).strip()]


def strip_response_header(prompt: str) -> str:
    marker = "\n\n### Response:\n"
    if prompt.endswith(marker):
        return prompt[: -len(marker)].strip()
    return prompt.strip()


def chunk_body(row: dict[str, Any]) -> str:
    for key in ["content", "body", "chunk_text", "text", "Body"]:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def anchors(row: dict[str, Any]) -> dict[str, Any]:
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    doc_summary = row.get("doc_summary") or row.get("article_summary") or meta.get("abstract") or ""
    doc_keywords = row.get("doc_keywords") or meta.get("keywords") or []
    chunk_summary = row.get("chunk_summary") or row.get("summary") or ""
    chunk_keywords = row.get("chunk_keywords") or row.get("paragraph_keywords") or row.get("keywords") or []
    return {
        "doc_summary": str(doc_summary or ""),
        "doc_keywords": as_list(doc_keywords),
        "chunk_summary": str(chunk_summary or ""),
        "chunk_keywords": as_list(chunk_keywords),
    }


def json_from_text(text: str) -> Any:
    value = (text or "").strip()
    if value.startswith("```"):
        value = value.strip("`")
        if value.lower().startswith("json"):
            value = value[4:].strip()
    start = min([i for i in [value.find("["), value.find("{")] if i >= 0], default=0)
    value = value[start:]
    return json.loads(value)


def chat_json(client: Any, model: str, prompt: str, temperature: float, retries: int) -> Any:
    last_error = None
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            return json_from_text(resp.choices[0].message.content or "")
        except Exception as exc:
            last_error = exc
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(str(last_error))


def chat_text(client: Any, model: str, prompt: str, temperature: float, retries: int) -> str:
    last_error = None
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            last_error = exc
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(str(last_error))


def strip_labeled_text(text: str, label: str) -> str:
    value = (text or "").strip()
    if value.startswith("```"):
        value = value.strip("`").strip()
    prefix = f"{label}:"
    if value.lower().startswith(prefix.lower()):
        value = value[len(prefix):].strip()
    return value


def accepted(verdict: dict[str, Any]) -> bool:
    return (
        int(verdict.get("view_alignment", 0) or 0) >= 4
        and int(verdict.get("answerability", 0) or 0) >= 4
        and int(verdict.get("evidence_support", 0) or 0) >= 4
        and int(verdict.get("specificity", 0) or 0) >= 3
        and not bool(verdict.get("external_knowledge", False))
        and bool(verdict.get("accepted", True))
    )


def make_sft_rows(base: dict[str, Any], qa: dict[str, Any], a: dict[str, Any]) -> list[dict[str, Any]]:
    body = chunk_body(base)
    ctx = QAContext(
        body=body,
        article_summary=a["doc_summary"],
        doc_keywords=a["doc_keywords"],
        chunk_summary=a["chunk_summary"],
        chunk_keywords=a["chunk_keywords"],
        title=str(base.get("title", "") or ""),
    )
    view = str(qa.get("view", "") or "")
    q = str(qa.get("question", "") or "").strip()
    ans = str(qa.get("answer", "") or "").strip()
    common = {
        "doc_id": str(base.get("doc_id", base.get("articleId", base.get("document_id", ""))) or ""),
        "chunk_id": str(base.get("chunk_id", base.get("id", "")) or ""),
        "view": view,
        "title": str(base.get("title", "") or ""),
        "doc_anchor_text": a["doc_summary"],
        "chunk_anchor_text": a["chunk_summary"] or body,
        "doc_keywords": a["doc_keywords"],
        "chunk_keywords": a["chunk_keywords"],
        "chunk_text": body,
    }
    return [
        {
            **common,
            "task": "qgen",
            "prompt": strip_response_header(build_question_prompt_sft(ctx=ctx, view=view, existing_questions=[])),
            "response": f"Question: {q}",
        },
        {
            **common,
            "task": "agen",
            "prompt": strip_response_header(build_answer_prompt_sft(ctx=ctx, question=q, view=view)),
            "response": f"Answer: {ans}",
            "question": q,
        },
    ]


def main() -> None:
    if load_dotenv is not None:
        load_dotenv(ROOT / ".env")

    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", default=os.getenv("TRAIN_CHUNKS_JSONL", ""))
    ap.add_argument("--out_dir", default=os.getenv("DATASET_OUT_DIR", ""))
    ap.add_argument("--api_key", default=os.getenv("API_KEY", ""))
    ap.add_argument("--base_url", default=os.getenv("API_BASE_URL", ""))
    ap.add_argument("--model", default=os.getenv("TEACHER_MODEL", ""))
    ap.add_argument("--k", type=int, default=int(os.getenv("TOP_K_VIEWS", "3") or 3))
    ap.add_argument("--max_chunks", type=int, default=int(os.getenv("MAX_CHUNKS", "0") or 0))
    ap.add_argument("--temperature", type=float, default=float(os.getenv("TEACHER_TEMPERATURE", "0.2") or 0.2))
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--skip_filter", action="store_true")
    args = ap.parse_args()

    if OpenAI is None:
        raise SystemExit("openai is not installed")
    if not args.input_jsonl or not args.out_dir or not args.api_key or not args.model:
        raise SystemExit("--input_jsonl, --out_dir, --api_key and --model are required")

    client_kwargs = {"api_key": args.api_key}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = OpenAI(**client_kwargs)

    rows = read_jsonl(Path(args.input_jsonl))
    if args.max_chunks > 0:
        rows = rows[: args.max_chunks]

    raw_rows: list[dict[str, Any]] = []
    accepted_rows: list[dict[str, Any]] = []
    sft_rows: list[dict[str, Any]] = []

    for i, row in enumerate(rows):
        body = chunk_body(row)
        if not body:
            continue
        a = anchors(row)
        prompt = build_view_selection_prompt(
            doc_summary=a["doc_summary"],
            doc_keywords=a["doc_keywords"],
            chunk_summary=a["chunk_summary"],
            chunk_keywords=a["chunk_keywords"],
            chunk_text=body,
            k=args.k,
        )
        selected = chat_json(client, args.model, prompt, args.temperature, args.retries)
        views = selected.get("views", []) if isinstance(selected, dict) else selected
        if not isinstance(views, list):
            views = []
        existing_questions: list[str] = []
        for view in views[: args.k]:
            view = str(view or "").strip()
            if not view:
                continue
            q_prompt = build_teacher_question_prompt(view=view, chunk_text=body, existing_questions=existing_questions)
            question = strip_labeled_text(chat_text(client, args.model, q_prompt, args.temperature, args.retries), "Question")
            if not question:
                continue
            existing_questions.append(question)
            a_prompt = build_teacher_answer_prompt(view=view, question=question, chunk_text=body)
            answer = strip_labeled_text(chat_text(client, args.model, a_prompt, args.temperature, args.retries), "Answer")
            if not answer:
                continue
            qa = {"view": view, "question": question, "answer": answer}
            item = {**qa, "chunk_id": row.get("chunk_id", row.get("id", "")), "row_index": i}
            raw_rows.append(item)
            verdict = {"accepted": True}
            if not args.skip_filter:
                filter_prompt = build_acceptance_filter_prompt(
                    view=str(qa.get("view", "")),
                    question=str(qa.get("question", "")),
                    answer=str(qa.get("answer", "")),
                    chunk_text=body,
                )
                verdict = chat_json(client, args.model, filter_prompt, 0.0, args.retries)
                if not isinstance(verdict, dict):
                    verdict = {"accepted": False, "reason": "invalid verifier json"}
            if accepted(verdict):
                row_out = {**item, "filter": verdict}
                accepted_rows.append(row_out)
                sft_rows.extend(make_sft_rows(row, qa, a))

    out_dir = Path(args.out_dir)
    write_jsonl(out_dir / "teacher_raw.jsonl", raw_rows)
    write_jsonl(out_dir / "view_conditioned_qa.jsonl", accepted_rows)
    write_jsonl(out_dir / "view_conditioned_sft.jsonl", sft_rows)
    print(json.dumps({"raw": len(raw_rows), "accepted": len(accepted_rows), "sft": len(sft_rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
