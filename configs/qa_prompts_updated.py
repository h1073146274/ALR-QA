#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prompt builders for view-conditioned QA generation.

This module follows the prompt flow in ``提示词.docx``:
1. select suitable views for a chunk;
2. generate a question under each selected view and the current chunk;
3. generate the answer from the same view, generated question, and chunk body.

There is intentionally no intermediate evidence extraction step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple


VIEW_DEFS: Dict[str, str] = {
    "concept": "Ask about the core idea, meaning, or central object.",
    "mechanism": "Ask how something works, unfolds, or is carried out.",
    "condition": "Ask about prerequisites, constraints, scope, or validity conditions.",
    "evidence": "Ask for observations, facts, or direct support available in the Body.",
    "causation": "Ask about drivers, reasons, or factors that lead to an outcome.",
    "comparison": "Ask by contrasting alternatives, baselines, or differences.",
    "impact": "Ask about consequences, implications, roles, or practical impact.",
    "quantitative": "Ask for numbers, metrics, time, stages, or other structured facts answerable from the Body.",
}

DEFAULT_APT_TOKEN = "<|APT|>"


@dataclass(frozen=True)
class QAContext:
    body: str
    article_summary: str = ""
    doc_keywords: Sequence[str] = ()
    chunk_summary: str = ""
    chunk_keywords: Sequence[str] = ()
    title: str = ""


def _safe_join(items: Sequence[str] | str, sep: str = ", ") -> str:
    if isinstance(items, str):
        return items.strip()
    return sep.join([str(x).strip() for x in items if str(x).strip()])


def _response_header() -> str:
    return "\n\n### Response:\n"


def _chunk_body_block(ctx: QAContext) -> str:
    return f"[Chunk Body]\n{ctx.body.strip()}"


def _context_blocks(ctx: QAContext, *, include_anchors: bool = True) -> str:
    blocks = []
    if ctx.title.strip():
        blocks.append(f"[Title]\n{ctx.title.strip()}")
    if include_anchors and ctx.article_summary.strip():
        blocks.append(f"[Document Anchor]\nSummary: {ctx.article_summary.strip()}")
    if include_anchors and _safe_join(ctx.doc_keywords):
        blocks.append(f"[Document Keywords]\n{_safe_join(ctx.doc_keywords)}")
    if include_anchors and ctx.chunk_summary.strip():
        blocks.append(f"[Chunk Anchor]\nSummary: {ctx.chunk_summary.strip()}")
    if include_anchors and _safe_join(ctx.chunk_keywords):
        blocks.append(f"[Chunk Keywords]\n{_safe_join(ctx.chunk_keywords)}")
    blocks.append(f"[Chunk Body]\n{ctx.body.strip()}")
    return "\n\n".join(blocks)


def build_question_probe_prompt(
    ctx: Optional[QAContext] = None,
    *,
    article_summary: str = "",
    doc_keywords: Sequence[str] | str = (),
    chunk_summary: str = "",
    chunk_keywords: Sequence[str] | str = (),
    body: str = "",
    view_name: Optional[str] = None,
    view_definition: Optional[str] = None,
    view: Optional[str] = None,
    apt_token: Optional[str] = DEFAULT_APT_TOKEN,
    title: str = "",
    add_task_prefix: bool = True,
    minimal_context: bool = False,
) -> str:
    if ctx is None:
        ctx = QAContext(
            body=body,
            article_summary=article_summary,
            doc_keywords=doc_keywords if not isinstance(doc_keywords, str) else [doc_keywords],
            chunk_summary=chunk_summary,
            chunk_keywords=chunk_keywords if not isinstance(chunk_keywords, str) else [chunk_keywords],
            title=title,
        )
    eff_view = view_name or view or ""
    eff_def = view_definition or VIEW_DEFS.get(eff_view, "")
    task = "Task: Probe whether the chunk can support this questioning view.\n" if add_task_prefix else ""
    return (
        task
        + "Read the chunk and prepare to generate one question under the specified view. "
        + "Do not generate the question yet; use the special token as the routing position.\n\n"
        + f"[View]\nname: {eff_view}\ninstruction: {eff_def}\n\n"
        + _context_blocks(ctx, include_anchors=not minimal_context)
        + f"\n\n{apt_token}"
        + _response_header()
    )


def build_question_prompt_sft(
    ctx: Optional[QAContext] = None,
    *,
    article_summary: str = "",
    doc_keywords: Sequence[str] | str = (),
    chunk_summary: str = "",
    chunk_keywords: Sequence[str] | str = (),
    body: str = "",
    view_name: Optional[str] = None,
    view_definition: Optional[str] = None,
    title: str = "",
    number: int = 1,
    view: Optional[str] = None,
    attempt_round: int = 1,
    add_task_prefix: bool = True,
    existing_questions: Optional[Sequence[str]] = None,
    minimal_context: bool = False,
) -> str:
    if ctx is None:
        ctx = QAContext(
            body=body,
            article_summary=article_summary,
            doc_keywords=doc_keywords if not isinstance(doc_keywords, str) else [doc_keywords],
            chunk_summary=chunk_summary,
            chunk_keywords=chunk_keywords if not isinstance(chunk_keywords, str) else [chunk_keywords],
            title=title,
        )
    eff_view = view_name or view or ""
    eff_def = view_definition or VIEW_DEFS.get(eff_view, "")
    previous = "\n".join(f"- {q}" for q in (existing_questions or []) if str(q).strip())
    previous_text = previous if previous else "None"
    return (
        "Task: Write ONE clear and natural question under the selected view.\n\n"
        "Guidelines:\n"
        "1) Use the selected view to guide the questioning perspective.\n"
        "2) Generate the question from the Chunk Body.\n"
        "3) The question should be specific and answerable from the Chunk Body.\n"
        "4) Avoid duplicating these earlier questions:\n"
        f"{previous_text}\n\n"
        "Return the question in this format:\n"
        "Question: <question>\n\n"
        "[Selected_View]\n"
        f"{eff_view}: {eff_def}\n\n"
        + _chunk_body_block(ctx)
    )


def build_question_generation_prompt(
    text: str,
    abstract: str,
    keywords,
    number: int = 1,
    attempt_round: int = 1,
    view: Optional[str] = None,
    existing_questions: Optional[Sequence[str]] = None,
    *,
    chunk_summary: str = "",
    chunk_keywords=None,
    add_task_prefix: bool = True,
) -> str:
    ctx = QAContext(
        body=text,
        article_summary=abstract,
        doc_keywords=keywords or [],
        chunk_summary=chunk_summary,
        chunk_keywords=chunk_keywords or [],
    )
    return build_question_prompt_sft(
        ctx=ctx,
        view=view,
        number=number,
        attempt_round=attempt_round,
        existing_questions=existing_questions,
        add_task_prefix=add_task_prefix,
    )


def build_answer_prompt_sft(
    ctx: Optional[QAContext] = None,
    *,
    body: str = "",
    article_summary: str = "",
    doc_keywords: Sequence[str] | str = (),
    chunk_summary: str = "",
    chunk_keywords: Sequence[str] | str = (),
    question: str,
    view_name: Optional[str] = None,
    view_definition: Optional[str] = None,
    view: Optional[str] = None,
    title: str = "",
    n_answers: int = 1,
    add_task_prefix: bool = True,
) -> str:
    if ctx is None:
        ctx = QAContext(
            body=body,
            article_summary=article_summary,
            doc_keywords=doc_keywords if not isinstance(doc_keywords, str) else [doc_keywords],
            chunk_summary=chunk_summary,
            chunk_keywords=chunk_keywords if not isinstance(chunk_keywords, str) else [chunk_keywords],
            title=title,
        )
    eff_view = view_name or view or ""
    eff_def = view_definition or VIEW_DEFS.get(eff_view, "")
    return (
        "Task: Answer the question based on the Chunk Body.\n\n"
        "Guidelines:\n"
        "1) Keep the answer concise and directly responsive to the question.\n"
        "2) The answer should be supported by the Chunk Body.\n"
        "3) Do not use information outside the Chunk Body.\n\n"
        "Return the answer in this format:\n"
        "Answer: <answer>\n\n"
        "[Selected_View]\n"
        f"{eff_view}: {eff_def}\n\n"
        "[Question]\n"
        f"{question.strip()}\n\n"
        + _chunk_body_block(ctx)
    )


def build_answer_generation_prompt(question: str, chunk_text: str) -> str:
    return build_answer_prompt_sft(
        question=question,
        ctx=QAContext(body=chunk_text),
    )


def build_view_selection_prompt(
    *,
    doc_summary: str,
    doc_keywords,
    chunk_summary: str,
    chunk_keywords,
    chunk_text: str,
    k: int = 3,
) -> str:
    views = "\n".join(f"- {name}: {desc}" for name, desc in VIEW_DEFS.items())
    return (
        "You are selecting questioning views for controlled long-document QA generation.\n\n"
        f"[Document Anchor]\nSummary: {doc_summary}\nKeywords: {_safe_join(doc_keywords)}\n\n"
        f"[Chunk Anchor]\nSummary: {chunk_summary}\nKeywords: {_safe_join(chunk_keywords)}\n\n"
        f"[View Set]\n{views}\n\n"
        f"[Chunk Body]\n{chunk_text}\n\n"
        f"Select up to {k} views that are semantically suitable for the chunk and directly supported by its content. "
        "If no view is supported, return an empty list.\n\n"
        'Return JSON: {"views": ["..."]}'
    )


def build_teacher_question_prompt(*, view: str, chunk_text: str, existing_questions: Optional[Sequence[str]] = None) -> str:
    return build_question_prompt_sft(
        ctx=QAContext(body=chunk_text),
        view=view,
        existing_questions=existing_questions,
    )


def build_teacher_answer_prompt(*, view: str, question: str, chunk_text: str) -> str:
    return build_answer_prompt_sft(
        ctx=QAContext(body=chunk_text),
        view=view,
        question=question,
    )


def build_acceptance_filter_prompt(*, view: str, question: str, answer: str, chunk_text: str) -> str:
    return (
        "You are verifying whether a generated QA pair is acceptable for training controlled long-document QA generation.\n\n"
        f"[View]\n{view}\n\n"
        f"[Question]\n{question}\n\n"
        f"[Answer]\n{answer}\n\n"
        f"[Chunk Body]\n{chunk_text}\n\n"
        "Evaluate the QA pair using four criteria:\n"
        "1. View alignment: whether the question follows the intended view.\n"
        "2. Answerability: whether the question can be answered from the chunk body.\n"
        "3. Evidence support: whether the answer is fully supported by the chunk body.\n"
        "4. Specificity: whether the question is specific and not overly broad.\n"
        "Use a 1-5 score for each criterion. Also check whether the answer introduces external knowledge.\n\n"
        "Accept the QA pair only if view_alignment >= 4, answerability >= 4, "
        "evidence_support >= 4, specificity >= 3, and external_knowledge = false.\n\n"
        'Return JSON: {"view_alignment": 1-5, "answerability": 1-5, "evidence_support": 1-5, '
        '"specificity": 1-5, "external_knowledge": true or false, "accepted": true or false, "reason": "..."}'
    )
