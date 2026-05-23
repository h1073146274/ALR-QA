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

VIEW_MUTEX_GROUPS: Tuple[Tuple[str, ...], ...] = (
    ("causation", "mechanism", "impact"),
    ("concept", "condition"),
)

VIEW_SYNERGY_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("causation", "impact"),
    ("mechanism", "impact"),
    ("comparison", "quantitative"),
    ("evidence", "concept"),
)

DEFAULT_MID_TOKEN = "<|MID|>"


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
    mid_token: Optional[str] = DEFAULT_MID_TOKEN,
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
        + f"\n\n{mid_token}"
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
    strict_json: bool = False,
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
    previous_block = f"\nAvoid duplicating these already generated questions:\n{previous}\n" if previous else ""
    output_rule = (
        'Return JSON: {"question": "..."}'
        if strict_json
        else "Return only the question text."
    )
    task = "Task: Generate one view-conditioned question.\n" if add_task_prefix else ""
    return (
        task
        + "You are constructing training data for controlled long-document QA generation.\n\n"
        + f"[Selected View]\nname: {eff_view}\ninstruction: {eff_def}\n\n"
        + _context_blocks(ctx, include_anchors=not minimal_context)
        + "\n\nRequirements:\n"
        + "1. Generate exactly one natural question under the selected view.\n"
        + "2. The question must be answerable from the Chunk Body.\n"
        + "3. Do not use external knowledge.\n"
        + "4. Avoid vague, overly broad, duplicate, or multi-part questions.\n"
        + previous_block
        + output_rule
        + _response_header()
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
    evidence: str = "",
    question: str,
    view_name: Optional[str] = None,
    view_definition: Optional[str] = None,
    view: Optional[str] = None,
    title: str = "",
    n_answers: int = 1,
    add_task_prefix: bool = True,
    strict_json: bool = False,
) -> str:
    if ctx is None:
        ctx = QAContext(
            body=body or evidence,
            article_summary=article_summary,
            doc_keywords=doc_keywords if not isinstance(doc_keywords, str) else [doc_keywords],
            chunk_summary=chunk_summary,
            chunk_keywords=chunk_keywords if not isinstance(chunk_keywords, str) else [chunk_keywords],
            title=title,
        )
    eff_view = view_name or view or ""
    eff_def = view_definition or VIEW_DEFS.get(eff_view, "")
    output_rule = (
        'Return JSON: {"answer": "..."}'
        if strict_json
        else "Return only the answer text."
    )
    task = "Task: Generate the grounded answer for the question.\n" if add_task_prefix else ""
    return (
        task
        + "You are constructing training data for controlled long-document QA generation.\n\n"
        + f"[Selected View]\nname: {eff_view}\ninstruction: {eff_def}\n\n"
        + f"[Question]\n{question.strip()}\n\n"
        + _context_blocks(ctx)
        + "\n\nRequirements:\n"
        + "1. Answer strictly using the Chunk Body.\n"
        + "2. Do not use external knowledge.\n"
        + "3. If the Chunk Body does not contain the answer, return: Not stated in the Body.\n"
        + "4. Be concise and directly answer the question.\n"
        + output_rule
        + _response_header()
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


def build_teacher_question_prompt(*, view: str, chunk_text: str) -> str:
    return (
        "You are generating a view-conditioned question for controlled long-document QA generation.\n\n"
        f"[Selected View]\n{view}: {VIEW_DEFS.get(view, '')}\n\n"
        f"[Chunk Body]\n{chunk_text}\n\n"
        "Generate exactly one natural question under the selected view. The question must be answerable from the chunk body. "
        "Do not use external knowledge. Avoid vague, overly broad, duplicate, or multi-part questions.\n\n"
        'Return JSON: {"question": "..."}'
    )


def build_teacher_answer_prompt(*, view: str, question: str, chunk_text: str) -> str:
    return (
        "You are generating a grounded answer for controlled long-document QA generation.\n\n"
        f"[Selected View]\n{view}: {VIEW_DEFS.get(view, '')}\n\n"
        f"[Question]\n{question}\n\n"
        f"[Chunk Body]\n{chunk_text}\n\n"
        "Generate one concise answer using only the chunk body. Do not use external knowledge. "
        "If the chunk body does not contain the answer, return: Not stated in the Body.\n\n"
        'Return JSON: {"answer": "..."}'
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
