#!/usr/bin/env python3
"""Evaluate the four non-RAGAS metrics reported for ALR-QA.

Metrics: QRelScore, Source Hit@5, Diverse@10, and Self-CosSim.  Input QA
records must contain ``chunk_id`` and ``question``; source chunks must contain
an ID and text.  Scores in ``summary.json`` are in [0, 1].
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


EPS = 1e-8


def read_records(path: str) -> list[dict[str, Any]]:
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"{path} must contain a JSON list or JSONL records.")
        return [row for row in data if isinstance(row, dict)]
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def field(row: dict[str, Any], names: Iterable[str]) -> str:
    for name in names:
        value = row.get(name)
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())
    return ""


def chunk_map(rows: list[dict[str, Any]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in rows:
        chunk_id = str(row.get("id") or row.get("chunk_id") or "").strip()
        text = field(row, ("content", "chunk_text", "text", "body"))
        if chunk_id and text:
            result[chunk_id] = text
    return result


class Encoder:
    def __init__(self, model_name: str, device: str, batch_size: int, max_length: int):
        self.device, self.batch_size, self.max_length = device, batch_size, max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()

    @torch.inference_mode()
    def encode(self, texts: list[str], description: str) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        for start in tqdm(range(0, len(texts), self.batch_size), desc=description):
            batch = self.tokenizer(texts[start:start + self.batch_size], padding=True,
                                   truncation=True, max_length=self.max_length,
                                   return_tensors="pt").to(self.device)
            states = self.model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1)
            pooled = (states * mask).sum(1) / mask.sum(1).clamp_min(1)
            parts.append(F.normalize(pooled.float(), p=2, dim=-1).cpu())
        return torch.cat(parts) if parts else torch.empty((0, 0))


class QRelScorer:
    """QRelScore's word-alignment and conditional-likelihood components."""
    def __init__(self, word_model: str, clm_model: str, device: str, word_max_length: int,
                 clm_max_length: int):
        self.device, self.word_max_length, self.clm_max_length = device, word_max_length, clm_max_length
        self.word_tokenizer = AutoTokenizer.from_pretrained(word_model, use_fast=True)
        try:
            self.word_model = AutoModel.from_pretrained(word_model, attn_implementation="eager")
        except TypeError:
            self.word_model = AutoModel.from_pretrained(word_model)
            if hasattr(self.word_model.config, "_attn_implementation"):
                self.word_model.config._attn_implementation = "eager"
        self.word_model = self.word_model.to(device).eval()
        self.clm_tokenizer = AutoTokenizer.from_pretrained(clm_model, use_fast=True, trust_remote_code=True)
        if self.clm_tokenizer.pad_token is None:
            self.clm_tokenizer.pad_token = self.clm_tokenizer.eos_token
        dtype = torch.bfloat16 if device.startswith("cuda") else None
        self.clm = AutoModelForCausalLM.from_pretrained(clm_model, trust_remote_code=True,
                                                        torch_dtype=dtype).to(device).eval()

    @torch.inference_mode()
    def word_alignment(self, questions: list[str], contexts: list[str]) -> list[float]:
        scores: list[float] = []
        for question, context in tqdm(list(zip(questions, contexts)), desc="QRel word alignment"):
            encoded = self.word_tokenizer(question, context, truncation="only_second",
                                          max_length=self.word_max_length,
                                          return_special_tokens_mask=True, return_tensors="pt")
            sequence_ids = encoded.sequence_ids(0)
            special = encoded["special_tokens_mask"][0].tolist()
            q_idx = [i for i, (sid, token) in enumerate(zip(sequence_ids, special)) if sid == 0 and not token]
            c_idx = [i for i, (sid, token) in enumerate(zip(sequence_ids, special)) if sid == 1 and not token]
            if not q_idx or not c_idx:
                scores.append(0.0)
                continue
            inputs = {key: value.to(self.device) for key, value in encoded.items() if key != "special_tokens_mask"}
            output = self.word_model(**inputs, output_hidden_states=True, output_attentions=True, return_dict=True)
            hidden = torch.stack(output.hidden_states[1:])[..., 0, :, :].float()  # [layers, seq, hidden]
            attention = torch.stack(output.attentions)[:, 0].float()               # [layers, heads, seq, seq]
            q_hidden, c_hidden = F.normalize(hidden[:, q_idx], dim=-1), F.normalize(hidden[:, c_idx], dim=-1)
            q_to_c = attention[:, :, q_idx, :][:, :, :, c_idx].max(1).values
            hard = F.one_hot(q_to_c.argmax(-1), num_classes=len(c_idx)).float()
            cosine = q_hidden @ c_hidden.transpose(-1, -2)
            scores.append(float(((cosine * hard).sum(-1).mean() + 1.0).div(2.0).clamp(0, 1).cpu()))
        return scores

    @torch.inference_mode()
    def avg_logp(self, targets: list[str], prefixes: list[str] | None = None) -> list[float]:
        values: list[float] = []
        for i, target in enumerate(tqdm(targets, desc="QRel conditional likelihood")):
            prefix = "" if prefixes is None else prefixes[i]
            target_ids = self.clm_tokenizer.encode(target, add_special_tokens=False,
                                                    truncation=True, max_length=self.clm_max_length)
            if len(target_ids) >= self.clm_max_length:
                target_ids, prefix_ids = target_ids[-self.clm_max_length:], []
            else:
                prefix_ids = self.clm_tokenizer.encode(
                    prefix, add_special_tokens=False, truncation=True,
                    max_length=self.clm_max_length - len(target_ids),
                )
            ids = prefix_ids + target_ids
            if len(ids) < 2 or not target_ids:
                values.append(0.0)
                continue
            input_ids = torch.tensor([ids], device=self.device)
            logits = self.clm(input_ids=input_ids, use_cache=False).logits[0, :-1].float()
            labels = input_ids[0, 1:]
            start = max(len(prefix_ids) - 1, 0)
            token_logp = F.log_softmax(logits, dim=-1).gather(1, labels[:, None]).squeeze(1)
            values.append(float(token_logp[start:].mean().cpu()))
        return values

    def score(self, questions: list[str], contexts: list[str]) -> list[float]:
        word = self.word_alignment(questions, contexts)
        base = self.avg_logp(contexts)
        conditioned = self.avg_logp(contexts, questions)
        # Shifted contexts are the mismatched negatives used to calibrate the likelihood gain.
        offset = 1 if len(contexts) > 1 else 0
        neg_contexts = contexts[offset:] + contexts[:offset]
        neg_base = base[offset:] + base[:offset]
        neg_conditioned = self.avg_logp(neg_contexts, questions) if offset else [0.0] * len(contexts)
        negative_gain = [c - b for b, c in zip(neg_base, neg_conditioned)]
        scale = max(float(np.std(negative_gain)), 1e-3)
        calibrated = [1.0 / (1.0 + math.exp(-((c - b) - n) / scale))
                      for b, c, n in zip(base, conditioned, negative_gain)]
        return [2 * a * b / max(a + b, EPS) for a, b in zip(word, calibrated)]


def pairwise_cosine(embeddings: torch.Tensor) -> float | None:
    if len(embeddings) < 2:
        return None
    sim = F.normalize(embeddings, dim=-1) @ F.normalize(embeddings, dim=-1).T
    return float(sim.triu(1).sum() / (len(embeddings) * (len(embeddings) - 1) / 2))


def diverse_at_ten(embeddings: torch.Tensor, k: int = 10) -> float | None:
    if len(embeddings) < 2:
        return None
    k = min(k, len(embeddings))
    dist = 1 - F.normalize(embeddings, dim=-1) @ F.normalize(embeddings, dim=-1).T
    selected = [int(dist.mean(1).argmax())]
    candidates = set(range(len(embeddings))) - set(selected)
    while len(selected) < k:
        candidate = max(candidates, key=lambda i: float(dist[i, selected].min()))
        selected.append(candidate)
        candidates.remove(candidate)
    similarity = pairwise_cosine(embeddings[selected])
    return None if similarity is None else 1 - similarity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa_file", required=True)
    parser.add_argument("--chunks_file", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--sim_model", required=True, help="Sentence embedding model for D@10 and SelfSim.")
    parser.add_argument("--word_model", required=True, help="Encoder model for QRel word alignment.")
    parser.add_argument("--clm_model", required=True, help="Causal LM for QRel conditional likelihood.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--embed_batch_size", type=int, default=64)
    parser.add_argument("--embed_max_length", type=int, default=128)
    parser.add_argument("--word_max_length", type=int, default=256)
    parser.add_argument("--clm_max_length", type=int, default=256)
    args = parser.parse_args()

    sources = chunk_map(read_records(args.chunks_file))
    rows = []
    for row in read_records(args.qa_file):
        chunk_id = str(row.get("chunk_id") or "")
        question = field(row, ("question", "query"))
        if chunk_id in sources and question:
            rows.append({"chunk_id": chunk_id, "question": question})
    if not rows:
        raise ValueError("No valid records. QA rows need chunk_id/question and chunk IDs must match source chunks.")
    questions, contexts = [r["question"] for r in rows], [sources[r["chunk_id"]] for r in rows]

    encoder = Encoder(args.sim_model, args.device, args.embed_batch_size, args.embed_max_length)
    q_emb = encoder.encode(questions, "Encode questions")
    source_ids, source_texts = list(sources), list(sources.values())
    # Source Hit@5 follows the local evaluation implementation: retrieve the
    # source chunk from the full chunk collection with TF-IDF (unigram/bigram).
    vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))
    source_matrix = vectorizer.fit_transform(source_texts)
    retrieval = (vectorizer.transform(questions) @ source_matrix.T).toarray()
    topk = np.argsort(-retrieval, axis=1)[:, :min(5, len(source_ids))].tolist()
    hit5 = [int(row["chunk_id"] in {source_ids[i] for i in ranks}) for row, ranks in zip(rows, topk)]

    by_chunk: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_chunk[row["chunk_id"]].append(i)
    d10 = [value for ids in by_chunk.values() if (value := diverse_at_ten(q_emb[ids])) is not None]
    qrel = QRelScorer(args.word_model, args.clm_model, args.device, args.word_max_length, args.clm_max_length).score(questions, contexts)

    summary = {
        "metrics": {
            "QRel": float(np.mean(qrel)),
            "Hit@5": float(np.mean(hit5)),
            "D@10": float(np.mean(d10)) if d10 else None,
            "SelfSim": pairwise_cosine(q_emb),
        },
        "count": len(rows),
        "definitions": {
            "QRel": "QRelScore: word-level alignment and calibrated conditional-likelihood gain, combined by harmonic mean.",
            "Hit@5": "Gold source chunk appears in the TF-IDF top five retrieved source chunks.",
            "D@10": "Macro average within source chunk; farthest-first selection of at most 10 questions, reported as one minus mean pairwise cosine.",
            "SelfSim": "Mean pairwise cosine similarity of all generated-question embeddings (lower is better).",
        },
    }
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
