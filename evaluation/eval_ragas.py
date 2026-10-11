#!/usr/bin/env python3
"""Evaluate only the RAGAS metrics used in the ALR-QA paper: FAI and REL."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from ragas import SingleTurnSample
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, Faithfulness


def read_records(path: str) -> list[dict[str, Any]]:
    text = Path(path).read_text(encoding="utf-8").strip()
    return json.loads(text) if text.startswith("[") else [json.loads(line) for line in text.splitlines() if line.strip()]


def text(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


async def score(metric: Any, sample: SingleTurnSample) -> float:
    result = await metric.ascore(sample)
    return float(result.value if hasattr(result, "value") else result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa_file", required=True)
    parser.add_argument("--chunks_file", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--llm_model", required=True)
    parser.add_argument("--api_base_url", required=True)
    parser.add_argument("--api_key_env", default="OPENAI_API_KEY")
    parser.add_argument("--embedding_model", required=True)
    parser.add_argument("--faithfulness_max_tokens", type=int, default=2048)
    parser.add_argument("--relevancy_max_tokens", type=int, default=512)
    parser.add_argument("--strictness", type=int, default=3)
    args = parser.parse_args()
    api_key = os.getenv(args.api_key_env, "")
    if not api_key:
        raise ValueError(f"Set {args.api_key_env} before running RAGAS evaluation.")

    chunks = {str(row.get("id") or row.get("chunk_id")): text(row, ("content", "chunk_text", "text", "body"))
              for row in read_records(args.chunks_file)}
    samples = []
    for row in read_records(args.qa_file):
        question, answer = text(row, ("question", "query")), text(row, ("answer", "final_answer"))
        context = chunks.get(str(row.get("chunk_id") or ""), "")
        if question and answer and context:
            samples.append((row, SingleTurnSample(user_input=question, response=answer, retrieved_contexts=[context])))
    if not samples:
        raise ValueError("No valid records. QA rows need chunk_id/question/answer and matching source chunks.")

    client = AsyncOpenAI(api_key=api_key, base_url=args.api_base_url)
    embedding = HuggingFaceEmbeddings(model=args.embedding_model)
    faithfulness = Faithfulness(llm=llm_factory(args.llm_model, provider="openai", client=client,
                                                 max_tokens=args.faithfulness_max_tokens, temperature=0))
    relevancy = AnswerRelevancy(llm=llm_factory(args.llm_model, provider="openai", client=client,
                                                max_tokens=args.relevancy_max_tokens, temperature=0),
                                  embeddings=embedding, strictness=args.strictness)

    per_item = []
    for row, sample in samples:
        fai, rel = asyncio.run(score(faithfulness, sample)), asyncio.run(score(relevancy, sample))
        per_item.append({"chunk_id": row.get("chunk_id"), "FAI": fai, "REL": rel})
    summary = {"metrics": {"FAI": sum(x["FAI"] for x in per_item) / len(per_item),
                           "REL": sum(x["REL"] for x in per_item) / len(per_item)}, "count": len(per_item),
               "definitions": {"FAI": "RAGAS Faithfulness.", "REL": "RAGAS Answer Relevancy."}}
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_item.json").write_text(json.dumps(per_item, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
