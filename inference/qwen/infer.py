#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NOTE: Universal APT (B-scheme): we add a dedicated <|APT|> token if missing and pool at its position.
generate_qa.py

End-to-end QA synthesis using a trained dual-anchor soft-guidance generator:
  1) Probe-based multi-view routing (APT lookahead) with coverage prior + greedy selection
  2) Question generation under selected views
  3) Answer generation grounded directly on the current chunk body

This script is designed to work with the training script `train_lora_dual_anchor.py`:
- Loads base model + LoRA adapter (saved by Trainer.save_model()).
- Reuses the learned projection heads `anchor_proj_doc` and `anchor_proj_chunk`.
- Reuses the same layer set (quartiles) and doc/chunk layer-weight schedule.

Prompts are built via `qa_prompts_updated.py` (expected in the same folder or on PYTHONPATH).
"""

import argparse
from collections import Counter
import inspect
import json
import math
import os
import random
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import PreTrainedTokenizerBase
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

# -------------------------
# Views (Bloom-inspired information-function taxonomy, fixed 8)
# -------------------------
VIEWS: Dict[str, str] = {
    "concept": "Ask about the core idea, meaning, or central object.",
    "mechanism": "Ask how something works, unfolds, or is carried out.",
    "condition": "Ask about prerequisites, constraints, scope, or validity conditions.",
    "evidence": "Ask for observations, facts, or direct support located in the Body.",
    "causation": "Ask about drivers, reasons, or factors that lead to an outcome.",
    "comparison": "Ask by contrasting alternatives, baselines, or differences.",
    "impact": "Ask about consequences, implications, roles, or practical impact.",
    "quantitative": "Ask for numbers, metrics, time, stages, or other structured facts answerable from the Body.",
}

# -------------------------
# Import prompts
# -------------------------
def _import_prompts():
    import importlib.util
    import sys

    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", ".."))
    candidates = [
        os.path.join(root, "configs", "qa_prompts_updated.py"),
    ]
    for cand in candidates:
        if not os.path.exists(cand):
            continue
        spec = importlib.util.spec_from_file_location("qa_prompts_updated", cand)
        mod = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod

    import qa_prompts_updated as mod  # type: ignore
    return mod


PROMPTS = _import_prompts()

# Keep view/rule definitions aligned with qa_prompts_updated.py only when it
# already uses this canonical 8-view taxonomy.
if hasattr(PROMPTS, "VIEW_DEFS") and isinstance(PROMPTS.VIEW_DEFS, dict):
    prompt_view_defs = dict(PROMPTS.VIEW_DEFS)
    if set(prompt_view_defs) == set(VIEWS):
        VIEWS = prompt_view_defs


def _token_is_single_id(tokenizer: PreTrainedTokenizerBase, s: str) -> bool:
    ids = tokenizer(s, add_special_tokens=False).get("input_ids", [])
    return isinstance(ids, list) and len(ids) == 1


def ensure_apt_token_and_resize(
    tokenizer: PreTrainedTokenizerBase,
    model: nn.Module,
    apt_token: str,
) -> int:
    """
    Universal B-scheme:
      - Ensure `apt_token` exists in tokenizer AND maps to a single token id.
      - If not, add it as an additional special token, then resize model embeddings.
    Returns: apt_token_id
    """
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    base = _unwrap_base_model(model)
    target = getattr(base, "model", base)

    def _resize_to_tokenizer_len() -> None:
        curr_vocab = int(model.get_input_embeddings().weight.shape[0])
        target_vocab = int(len(tokenizer))
        if curr_vocab != target_vocab:
            print(f"[INFO] Resizing token embeddings from {curr_vocab} to {target_vocab} to match tokenizer.")
            if hasattr(target, "resize_token_embeddings"):
                target.resize_token_embeddings(target_vocab)
            else:
                model.resize_token_embeddings(target_vocab)

    tid = tokenizer.convert_tokens_to_ids(apt_token)
    ok_single = _token_is_single_id(tokenizer, apt_token)
    if tid is None or tid == tokenizer.unk_token_id or not ok_single:
        tokenizer.add_special_tokens({"additional_special_tokens": [apt_token]})
        _resize_to_tokenizer_len()
        tid = tokenizer.convert_tokens_to_ids(apt_token)
        if tid is None or tid == tokenizer.unk_token_id or not _token_is_single_id(tokenizer, apt_token):
            raise ValueError(f"Failed to install APT token '{apt_token}' as a single token id.")
    else:
        _resize_to_tokenizer_len()

    if tokenizer.eos_token_id is not None and tid == tokenizer.eos_token_id:
        raise ValueError(f"APT token '{apt_token}' resolves to eos_token_id={tid}. Choose another APT token.")
    if tokenizer.pad_token_id is not None and tid == tokenizer.pad_token_id:
        raise ValueError(f"APT token '{apt_token}' resolves to pad_token_id={tid}. Choose another APT token.")
    return int(tid)


def load_tokenizer_for_adapter(lora_dir: str, base_model: str) -> PreTrainedTokenizerBase:
    """
    Prefer the tokenizer saved alongside the adapter so special tokens and resized
    embeddings stay aligned with training. Fall back to the base model tokenizer.
    """
    try:
        tok = AutoTokenizer.from_pretrained(lora_dir, use_fast=True, trust_remote_code=True)
        print(f"[INFO] Loaded tokenizer from LoRA dir: {lora_dir}")
        return tok
    except Exception as e:
        print(f"[WARN] Failed to load tokenizer from LoRA dir {lora_dir}: {e}")
        print(f"[INFO] Falling back to base model tokenizer: {base_model}")
        return AutoTokenizer.from_pretrained(base_model, use_fast=True, trust_remote_code=True)


# -------------------------
# Model utils (adapted from train_lora_dual_anchor.py)
# -------------------------
def _unwrap_base_model(model: nn.Module) -> nn.Module:
    if hasattr(model, "get_base_model"):
        try:
            return model.get_base_model()
        except Exception:
            pass
    if hasattr(model, "base_model"):
        bm = model.base_model
        if hasattr(bm, "model"):
            return bm.model
        return bm
    return model


def _get_optional_proj(model: nn.Module, name: str) -> Optional[nn.Module]:
    base = _unwrap_base_model(model)
    if hasattr(base, name):
        return getattr(base, name)
    if hasattr(base, "model") and hasattr(base.model, name):
        return getattr(base.model, name)
    if hasattr(model, name):
        return getattr(model, name)
    return None


def _get_anchor_projs(model: nn.Module) -> Tuple[nn.Module, nn.Module]:
    """
    Anchor projection heads may live on the base model or the PEFT wrapper.
    """
    proj_doc = _get_optional_proj(model, "anchor_proj_doc")
    proj_chunk = _get_optional_proj(model, "anchor_proj_chunk")
    if proj_doc is None or proj_chunk is None:
        raise RuntimeError(
            "Loaded adapter does not expose anchor_proj_doc/chunk. "
            "Please use a LoRA directory produced by train_lora_dual_anchor.py (with modules_to_save)."
        )
    return proj_doc, proj_chunk


def _ensure_anchor_projs_from_adapter(model: nn.Module, lora_dir: str, device: torch.device, dtype: torch.dtype) -> None:
    """
    If anchor_proj_* modules are missing, try to load them from adapter_model.safetensors
    and attach to the base model (base.model if present).
    """
    adapter_path = os.path.join(lora_dir, "adapter_model.safetensors")
    if not os.path.exists(adapter_path):
        # If projections already exist on model, keep going.
        _get_anchor_projs(model)
        return

    try:
        from safetensors.torch import load_file as safe_load
    except Exception as e:
        raise RuntimeError("safetensors is required to load anchor_proj_* from adapter_model.safetensors.") from e

    state = safe_load(adapter_path)

    base = _unwrap_base_model(model)
    target = getattr(base, "model", base)

    def _make_linear(w: torch.Tensor) -> nn.Linear:
        out_f, in_f = int(w.shape[0]), int(w.shape[1])
        lin = nn.Linear(in_f, out_f, bias=False)
        lin.weight.data.copy_(w)
        lin.to(device=device, dtype=dtype)
        return lin

    def _find_weight(name: str) -> Optional[torch.Tensor]:
        suffixes = [
            f".{name}.weight",
            f".{name}.modules_to_save.default.weight",
            f"{name}.weight",
            f"{name}.modules_to_save.default.weight",
        ]
        for k, v in state.items():
            if any(k.endswith(suf) for suf in suffixes):
                return v
        return None

    needed = {
        "anchor_proj_doc": _get_optional_proj(model, "anchor_proj_doc") is None,
        "anchor_proj_chunk": _get_optional_proj(model, "anchor_proj_chunk") is None,
        "anchor_proj_view": _get_optional_proj(model, "anchor_proj_view") is None,
        "anchor_proj_route": _get_optional_proj(model, "anchor_proj_route") is None,
    }

    for name, need in needed.items():
        if not need:
            continue
        w = _find_weight(name)
        if w is None:
            continue
        setattr(target, name, _make_linear(w))

    # doc/chunk are mandatory for this script
    _get_anchor_projs(model)


def _load_view_prototypes_for_views(
    lora_dir: str,
    view_names: List[str],
    override_path: str = "",
) -> Optional[torch.Tensor]:
    """
    Load view prototypes saved by training, then align rows to current view_names order.
    """
    vp_path = override_path or os.path.join(lora_dir, "view_prototypes.pt")
    if not os.path.exists(vp_path):
        return None
    obj = torch.load(vp_path, map_location="cpu")
    if isinstance(obj, torch.Tensor):
        if obj.dim() != 2 or obj.size(0) != len(view_names):
            print(f"[WARN] Ignore view prototypes at {vp_path}: shape mismatch {tuple(obj.shape)}")
            return None
        return F.normalize(obj.float(), p=2, dim=-1)
    if not isinstance(obj, dict):
        print(f"[WARN] Ignore view prototypes at {vp_path}: unsupported format")
        return None
    protos = obj.get("prototypes", None)
    names = obj.get("view_names", None)
    if not isinstance(protos, torch.Tensor) or protos.dim() != 2:
        print(f"[WARN] Ignore view prototypes at {vp_path}: invalid tensor payload")
        return None
    if not isinstance(names, list) or len(names) != protos.size(0):
        print(f"[WARN] Ignore view prototypes at {vp_path}: invalid view_names payload")
        return None

    name_to_idx = {str(n): i for i, n in enumerate(names)}
    missing = [v for v in view_names if v not in name_to_idx]
    if missing:
        print(f"[WARN] Ignore view prototypes at {vp_path}: missing views {missing}")
        return None

    aligned = torch.stack([protos[name_to_idx[v]].float() for v in view_names], dim=0)
    return F.normalize(aligned, p=2, dim=-1)


def find_transformer_layers(model: nn.Module) -> List[nn.Module]:
    base = _unwrap_base_model(model)
    cfg = getattr(base, "config", None)
    num_layers = None
    for k in ["num_hidden_layers", "n_layer", "num_layers", "n_layers"]:
        if cfg is not None and hasattr(cfg, k):
            num_layers = int(getattr(cfg, k))
            break
    if num_layers is None:
        raise ValueError("Cannot infer num_layers from config.")
    candidates: List[Tuple[str, nn.ModuleList]] = []
    for name, mod in base.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) == num_layers:
            candidates.append((name, mod))
    if not candidates:
        raise ValueError(f"Cannot locate transformer layers: no nn.ModuleList len={num_layers}.")
    # pick best-looking
    def score(modlist: nn.ModuleList) -> int:
        m0 = modlist[0]
        s = 0
        for attr in ["self_attn", "attn", "mlp", "feed_forward", "input_layernorm", "post_attention_layernorm"]:
            if hasattr(m0, attr):
                s += 1
        cn = m0.__class__.__name__.lower()
        if "decoder" in cn or "layer" in cn or "block" in cn:
            s += 2
        return s
    candidates.sort(key=lambda x: score(x[1]), reverse=True)
    best_name, best_list = candidates[0]
    print(f"[INFO] Found transformer layers at: {best_name} (len={len(best_list)})")
    return list(best_list)


def pick_layers(num_layers: int) -> List[int]:
    return sorted(set([num_layers // 4, num_layers // 2, (3 * num_layers) // 4, num_layers - 1]))


def make_layer_weights(k: int) -> Tuple[List[float], List[float]]:
    if k == 4:
        return [1.0, 0.7, 0.4, 0.2], [0.2, 0.4, 0.7, 1.0]
    doc, chunk = [], []
    for i in range(k):
        t = i / max(k - 1, 1)
        doc.append(1.0 * (1 - t) + 0.2 * t)
        chunk.append(0.2 * (1 - t) + 1.0 * t)
    return doc, chunk


class PooledLayerHook:
    def __init__(self, layers: List[nn.Module], layer_ids: List[int]):
        self.layers = layers
        self.layer_ids = layer_ids
        self.handles: List[Any] = []
        self.pool_idx: Optional[torch.Tensor] = None  # [B]
        self.cache: Dict[int, torch.Tensor] = {}

    def clear(self):
        self.cache = {}

    def _make_hook(self, layer_id: int):
        def hook_fn(module, inp, out):
            hs = out[0] if isinstance(out, (tuple, list)) else out  # [B,T,H]
            if self.pool_idx is None:
                return
            B = hs.size(0)
            idx = self.pool_idx
            z = hs[torch.arange(B, device=hs.device), idx]  # [B,H]
            self.cache[layer_id] = z
        return hook_fn

    def register(self):
        for lid in self.layer_ids:
            self.handles.append(self.layers[lid].register_forward_hook(self._make_hook(lid)))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


# -------------------------
# Data I/O
# -------------------------
def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    Read either JSONL (one object per line) or JSON array (.json).
    """
    with open(path, "r", encoding="utf-8") as f:
        # Peek first non-whitespace char
        pos = f.tell()
        ch = f.read(1)
        while ch and ch.isspace():
            ch = f.read(1)
        f.seek(pos)
        if ch == "[":
            obj = json.load(f)
            if isinstance(obj, list):
                return obj
            raise ValueError("JSON file must contain a list of objects.")

        rows: List[Dict[str, Any]] = []
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            rows.append(json.loads(ln))
        return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            yield json.loads(ln)


def read_last_jsonl_row(path: str) -> Optional[Dict[str, Any]]:
    """
    Read the last non-empty JSON object from a jsonl file by scanning from file tail.
    This is used by resume logic to continue after the last successfully written chunk
    instead of re-walking failed chunks from the beginning of the input file.
    """
    if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        pos = f.tell()
        buf = b""
        while pos > 0:
            step = min(8192, pos)
            pos -= step
            f.seek(pos)
            buf = f.read(step) + buf
            lines = buf.splitlines()
            for raw in reversed(lines):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    return json.loads(raw.decode("utf-8"))
                except Exception:
                    continue
        return None


def _get_field(row: Dict[str, Any], keys: List[str], default: str = "") -> str:
    for k in keys:
        v = row.get(k, None)
        if isinstance(v, str) and v.strip():
            return v
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
    parts = re.split(r"[,\n;/|，；、]+", s)
    return [p.strip() for p in parts if p and p.strip()]


def _get_meta_field(row: Dict[str, Any], key: str, default: Any = "") -> Any:
    meta = row.get("meta", None)
    if isinstance(meta, dict) and key in meta:
        return meta.get(key, default)
    return default


def _make_prompt_ctx(ctx: Dict[str, Any]):
    cls = getattr(PROMPTS, "QAContext", None)
    if cls is None:
        return ctx
    try:
        return cls(
            body=ctx["body"],
            article_summary=ctx["article_summary"],
            doc_keywords=ctx["doc_keywords"],
            chunk_summary=ctx["chunk_summary"],
            chunk_keywords=ctx["chunk_keywords"],
        )
    except Exception:
        return ctx


def _build_probe_prompt(ctx: Dict[str, Any], view_name: str, apt_token: str) -> str:
    fn = getattr(PROMPTS, "build_question_probe_prompt")
    sig = inspect.signature(fn)
    if "ctx" in sig.parameters:
        return fn(ctx=_make_prompt_ctx(ctx), view=view_name, apt_token=apt_token)
    # backward-compatible fallback
    return fn(
        article_summary=ctx["article_summary"],
        doc_keywords=ctx["doc_keywords"],
        chunk_summary=ctx["chunk_summary"],
        chunk_keywords=ctx["chunk_keywords"],
        body=ctx["body"],
        view_name=view_name,
        view_definition=VIEWS[view_name],
        apt_token=apt_token,
        title=ctx["title"],
    )


def _build_question_prompt(ctx: Dict[str, Any], view_name: str, n_questions: int, existing_questions: List[str]) -> str:
    if hasattr(PROMPTS, "build_question_prompt_sft"):
        fn = getattr(PROMPTS, "build_question_prompt_sft")
        sig = inspect.signature(fn)
        if "ctx" in sig.parameters:
            return fn(
                ctx=_make_prompt_ctx(ctx),
                # v2 prompt outputs ONE question only
                number=1,
                view=view_name,
                existing_questions=existing_questions,
            )
    if hasattr(PROMPTS, "build_question_generation_prompt"):
        # fallback (older VRQA prompt): still request 1 question
        return PROMPTS.build_question_generation_prompt(
            text=ctx["body"],
            abstract=ctx["article_summary"],
            keywords=ctx["doc_keywords"],
            number=1,
            view=view_name,
            existing_questions=existing_questions,
            chunk_summary=ctx["chunk_summary"],
            chunk_keywords=ctx["chunk_keywords"],
        )
    raise RuntimeError("Prompt module lacks question prompt builder.")


def _build_answer_prompt(ctx: Dict[str, Any], question: str) -> str:
    if hasattr(PROMPTS, "build_answer_prompt_sft"):
        fn = getattr(PROMPTS, "build_answer_prompt_sft")
        sig = inspect.signature(fn)
        if "ctx" in sig.parameters:
            qa_ctx = _make_prompt_ctx(
                {
                    **ctx,
                    # Ground the answer on the whole chunk body; do not leak candidate answers.
                    "body": ctx["body"],
                }
            )
            return fn(question=question, ctx=qa_ctx, n_answers=1)
    if hasattr(PROMPTS, "build_answer_generation_prompt"):
        return PROMPTS.build_answer_generation_prompt(
            question=question,
            chunk_text=ctx["body"],
        )
    raise RuntimeError("Prompt module lacks answer prompt builder.")


def _extract_json_array(text: str) -> List[str]:
    t = text.strip()
    if not t:
        return []
    # direct parse
    try:
        obj = json.loads(t)
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()]
    except Exception:
        pass
    # fenced code blocks
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", t, flags=re.I)
    if m:
        try:
            obj = json.loads(m.group(1).strip())
            if isinstance(obj, list):
                return [str(x).strip() for x in obj if str(x).strip()]
        except Exception:
            pass
    # bracket fallback
    m2 = re.search(r"\[[\s\S]*\]", t)
    if m2:
        try:
            obj = json.loads(m2.group(0))
            if isinstance(obj, list):
                return [str(x).strip() for x in obj if str(x).strip()]
        except Exception:
            pass
    return []


def _normalize_question(s: str) -> str:
    q = (s or "").strip()
    q = re.sub(r"^(?:Question\s*:\s*|问题\s*[:：]\s*)", "", q, flags=re.IGNORECASE).strip()
    q = q.strip('"').strip("'").strip()
    return q


def _first_question(s: str) -> str:
    t = (s or "").strip()
    if not t:
        return ""
    # grab first sentence ending with ? or ？
    m = re.search(r"(.+?[\\?？])", t)
    if m:
        return m.group(1).strip()
    return t


_STOPWORDS = {
    "the","a","an","and","or","to","of","in","on","for","with","at","by","from","as","is","are","was","were",
    "be","been","being","this","that","these","those","it","its","into","about","over","under","after","before",
    "between","during","without","within","what","which","who","whom","why","how","when","where"
}

_QUESTION_NUMERIC_CUE_RE = re.compile(
    r"\b(?:how many|how much|what percentage|what percent|what year|which year|when|rank|ranking|"
    r"total|number of|amount of|tonnage|temperature|duration|count|share)\b",
    re.IGNORECASE,
)


def _normalize_num_token(tok: str) -> str:
    return tok.replace(",", "").replace("%", "").strip().lower()


def _extract_num_tokens(text: str) -> List[str]:
    toks = re.findall(r"\b\d[\d,]*(?:\.\d+)?%?\b", text or "")
    return [_normalize_num_token(t) for t in toks if _normalize_num_token(t)]


def _content_tokens(text: str) -> List[str]:
    return [t for t in re.findall(r"[A-Za-z0-9]+", (text or "").lower()) if t not in _STOPWORDS]


def _normalize_chunk_text(text: str) -> str:
    """
    Normalize chunk/body text before prompt construction.
    This removes noisy hard line breaks from raw content while preserving
    readable sentence spacing for downstream sentence splitting.
    """
    t = (text or "").strip()
    if not t:
        return ""
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"\s*\n\s*", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def _question_is_near_duplicate(q: str, seen: List[str]) -> bool:
    q_norm = re.sub(r"[^a-z0-9 ]+", " ", (q or "").lower())
    q_toks = set(_content_tokens(q_norm))
    if not q_toks:
        return False
    q_nums = set(_extract_num_tokens(q))
    for s in seen:
        s_norm = re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower())
        s_toks = set(_content_tokens(s_norm))
        if not s_toks:
            continue
        inter = len(q_toks & s_toks)
        union = max(1, len(q_toks | s_toks))
        jaccard = inter / union
        containment = inter / max(1, min(len(q_toks), len(s_toks)))
        same_nums = q_nums == set(_extract_num_tokens(s))
        if containment >= 0.75:
            return True
        if jaccard >= 0.72:
            return True
        if same_nums and jaccard >= 0.5:
            return True
    return False


def _answer_overlap_score(ans: str, body: str) -> int:
    if not ans or not body:
        return 0
    btoks = set(re.findall(r"[A-Za-z0-9]+", body.lower()))
    atoks = [t for t in re.findall(r"[A-Za-z0-9]+", ans.lower()) if t not in _STOPWORDS]
    if not atoks:
        return 0
    return sum(1 for t in atoks if t in btoks)


def _answer_has_numeric_hallucination(answer: str, body: str) -> bool:
    a_nums = set(_extract_num_tokens(answer))
    if not a_nums:
        return False
    b_nums = set(_extract_num_tokens(body))
    return any(n not in b_nums for n in a_nums)


def _clean_answer_surface(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    t = re.sub(r"([(\[])\s+", r"\1", t)
    t = re.sub(r"\s+([)\]])", r"\1", t)
    # Repair a few frequent split artifacts from small-model decoding.
    t = re.sub(r"\bnon-\s+sm\s+art\b", "non-smart", t, flags=re.I)
    t = re.sub(r"\bsm\s+art\b", "smart", t, flags=re.I)
    t = re.sub(r"\bsurfactents\b", "surfactants", t, flags=re.I)
    t = re.sub(r"\bdisbursed\b", "discharged", t, flags=re.I)
    t = re.sub(r"\be\.\s*e\b", "e.g.", t, flags=re.I)
    return t.strip()


_AUTHOR_ONLY_RE = re.compile(
    r"^(?:according to\s+)?[A-Z][A-Za-z'`-]+(?:\s+[A-Z][A-Za-z'`-]+)?\s+et\.?(?:\s+al\.?)?\.?$",
    re.IGNORECASE,
)
_AUTHOR_MENTION_RE = re.compile(
    r"\b[A-Z][A-Za-z'`-]+(?:\s+[A-Z][A-Za-z'`-]+)?\s+et\.?(?:\s+al\.?)?\b",
    re.IGNORECASE,
)
_CITATION_QUESTION_RE = re.compile(r"\baccording to\s+[A-Z][A-Za-z'`-]+(?:\s+[A-Z][A-Za-z'`-]+)?\s+et\.?", re.IGNORECASE)
_RELATION_CUES = (
    ("between", re.compile(r"\bbetween\b", re.IGNORECASE)),
    ("within", re.compile(r"\bwithin\b", re.IGNORECASE)),
    ("respectively", re.compile(r"\brespectively\b", re.IGNORECASE)),
    ("range", re.compile(r"\brang(?:e|ed|es|ing)\b", re.IGNORECASE)),
    ("from_to", re.compile(r"\bfrom\b.*\bto\b", re.IGNORECASE)),
)
_RANGE_PATTERN_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:-|–|to)\s*\d+(?:\.\d+)?(?:\s*(?:%|°c|h|hours?|months?|days?))?\b",
    re.IGNORECASE,
)


def _answer_is_author_only_fragment(answer: str) -> bool:
    a = (answer or "").strip()
    if not a:
        return False
    if _AUTHOR_ONLY_RE.fullmatch(a):
        return True
    toks = _simple_tokenize(a)
    if len(toks) <= 4 and re.fullmatch(r"(?:According to\s+)?[A-Za-z .'-]+et\.?(?:\s+al\.?)?\.?", a, re.IGNORECASE):
        return True
    return False


def _answer_has_substantive_content_for_citation_question(question: str, answer: str) -> bool:
    if not _CITATION_QUESTION_RE.search(question or ""):
        return True
    if not answer or answer in _NOT_STATED_ANSWERS:
        return True
    if _extract_num_tokens(answer):
        return True
    stripped = _AUTHOR_MENTION_RE.sub(" ", answer or "")
    stripped_toks = [t for t in _content_tokens(stripped) if t not in {"according", "body", "study", "research", "cited"}]
    return len(stripped_toks) >= 2


def _answer_relation_is_supported(answer: str, support_text: str) -> bool:
    if not answer or answer in _NOT_STATED_ANSWERS:
        return True
    a = answer.strip()
    s = support_text.strip()
    if not s:
        return False
    for cue_name, cue_re in _RELATION_CUES:
        if not cue_re.search(a):
            continue
        if cue_name == "from_to":
            if cue_re.search(s) or _RANGE_PATTERN_RE.search(s):
                continue
            return False
        if cue_name == "between":
            if cue_re.search(s) or _RANGE_PATTERN_RE.search(s):
                continue
            return False
        if cue_name == "range":
            if cue_re.search(s) or _RANGE_PATTERN_RE.search(s):
                continue
            return False
        if not cue_re.search(s):
            return False
    return True


def _qa_is_near_duplicate(question: str, answer: str, seen_qas: List[Dict[str, Any]]) -> bool:
    q_toks = set(_content_tokens(question))
    a_toks = set(_content_tokens(answer))
    for prev in seen_qas:
        pq_toks = set(_content_tokens(prev.get("question", "")))
        pa_toks = set(_content_tokens(prev.get("answer", "")))
        q_inter = len(q_toks & pq_toks)
        q_union = max(1, len(q_toks | pq_toks))
        q_jaccard = q_inter / q_union
        a_inter = len(a_toks & pa_toks)
        a_union = max(1, len(a_toks | pa_toks))
        a_jaccard = a_inter / a_union if (a_toks or pa_toks) else 0.0
        return q_jaccard >= 0.8 and a_jaccard >= 0.45
    return False


def _pick_best_answer(candidates: List[str], body: str) -> str:
    def _is_bad(a: str) -> bool:
        if not a or not a.strip():
            return True
        t = a.strip()
        # too short
        if len(t) < 3:
            return True
        # repeated single char / symbol like "？？？？" or "aaaaa"
        if len(set(t)) <= 2 and len(t) >= 6:
            return True
        # excessive repeated punctuation
        if re.fullmatch(r"[\\?？!！,，.。;；:：\\-\\s]{6,}", t):
            return True
        return False

    cleaned = [a.strip() for a in candidates if not _is_bad(a)]
    if not cleaned:
        return (candidates[0].strip() if candidates else "")

    best = ""
    best_score = -1
    for a in cleaned:
        s = _answer_overlap_score(a, body)
        if s > best_score:
            best_score = s
            best = a
    return best or cleaned[0]


def build_chunk_context(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Collect minimal fields for prompt building.
    The prompt module can accept empty strings; we keep this flexible.
    """
    doc_kws = _to_list_keywords(
        _get_field(
            row,
            ["doc_keywords", "doc_kw", "Doc Keywords"],
            _get_meta_field(row, "keywords", ""),
        )
    )
    chunk_kws = _to_list_keywords(
        _get_field(
            row,
            ["chunk_keywords", "chunk_kw", "Chunk Keywords", "paragraph_keywords"],
            "",
        )
    )
    return {
        "article_summary": _normalize_chunk_text(
            _get_field(
            row,
            ["article_summary", "doc_summary", "Article Summary", "articleSummary"],
            _get_meta_field(row, "abstract", ""),
            )
        ),
        "doc_keywords": doc_kws,
        "chunk_summary": _normalize_chunk_text(_get_field(row, ["chunk_summary", "Chunk Summary", "summary"], "")),
        "chunk_keywords": chunk_kws,
        "body": _normalize_chunk_text(
            _get_field(
                row,
                ["body", "chunk_text", "text", "Body", "content", "chunk_anchor_text"],
                "",
            )
        ),
        "title": _normalize_chunk_text(_get_field(row, ["title", "doc_title"], "")),
        "doc_id": str(
            row.get(
                "doc_id",
                row.get("docid", row.get("document_id", row.get("articleId", ""))),
            )
            or ""
        ),
        "chunk_id": str(row.get("chunk_id", row.get("chunkid", row.get("cid", row.get("id", ""))))) or "",
    }


# -------------------------
# Probe routing
# -------------------------
@torch.no_grad()
def compute_anchor_energy(
    model: nn.Module,
    hook: PooledLayerHook,
    tokenizer,
    prompt: str,
    view_name: str,
    view_to_id: Dict[str, int],
    apt_token_id: int,
    a_doc: torch.Tensor,
    a_chunk: torch.Tensor,
    view_prototypes: Optional[torch.Tensor],
    layer_ids: List[int],
    lambda_doc: List[float],
    lambda_chunk: List[float],
    device: torch.device,
) -> Tuple[float, float, float, Optional[float], Optional[float]]:
    """
    Run a single forward pass on `prompt` and compute anchor energy at APT position.
    Energy definition matches training: sum_l (lambda_doc*(1-cos_doc) + lambda_chunk*(1-cos_chunk)).
    """
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)

    # locate APT position (last occurrence)
    ids = input_ids[0].tolist()
    try:
        apt_pos = len(ids) - 1 - ids[::-1].index(apt_token_id)
    except ValueError:
        raise ValueError("APT token id not found in prompt tokenization. Check --apt_token.")

    hook.pool_idx = torch.tensor([apt_pos], device=device, dtype=torch.long)
    hook.clear()
    _ = model(input_ids=input_ids, attention_mask=attn, use_cache=False)

    zs = [hook.cache[lid] for lid in layer_ids]  # [1,H]
    proj_doc, proj_chunk = _get_anchor_projs(model)

    # normalize anchors
    a_doc_n = F.normalize(a_doc.to(device=device, dtype=torch.float32).unsqueeze(0), p=2, dim=-1)
    a_chunk_n = F.normalize(a_chunk.to(device=device, dtype=torch.float32).unsqueeze(0), p=2, dim=-1)

    energy_doc = 0.0
    energy_chunk = 0.0
    for i, z in enumerate(zs):
        zf = z.to(dtype=proj_doc.weight.dtype)
        z_doc = F.normalize(proj_doc(zf), p=2, dim=-1)
        z_chunk = F.normalize(proj_chunk(zf), p=2, dim=-1)
        cos_doc = (z_doc * a_doc_n).sum(dim=-1).mean()
        cos_chunk = (z_chunk * a_chunk_n).sum(dim=-1).mean()
        e_doc = 1.0 - float(cos_doc.detach().cpu())
        e_chunk = 1.0 - float(cos_chunk.detach().cpu())
        energy_doc += lambda_doc[i] * e_doc
        energy_chunk += lambda_chunk[i] * e_chunk
    energy = energy_doc + energy_chunk
    hook.pool_idx = None
    route_logit_cur: Optional[float] = None
    view_energy_cur: Optional[float] = None
    z_route = zs[-1].to(dtype=proj_doc.weight.dtype)

    proj_route = _get_optional_proj(model, "anchor_proj_route")
    if proj_route is not None:
        logits = proj_route(z_route).float()  # [1,V]
        vid = int(view_to_id.get(view_name, -1))
        if 0 <= vid < logits.size(-1):
            route_logit_cur = float(logits[0, vid].detach().cpu())

    proj_view = _get_optional_proj(model, "anchor_proj_view")
    vid = int(view_to_id.get(view_name, -1))
    if proj_view is not None and view_prototypes is not None and 0 <= vid < int(view_prototypes.size(0)):
        z_view = F.normalize(proj_view(z_route).float(), p=2, dim=-1)  # [1,D]
        proto = F.normalize(view_prototypes[vid].to(device=device, dtype=torch.float32), p=2, dim=-1).unsqueeze(0)
        cos = (z_view * proto).sum(dim=-1)
        view_energy_cur = float((1.0 - cos).mean().detach().cpu())

    return float(energy), float(energy_doc), float(energy_chunk), route_logit_cur, view_energy_cur


@torch.no_grad()
def compute_anchor_energy_batch(
    model: nn.Module,
    hook: PooledLayerHook,
    tokenizer,
    prompts: List[str],
    view_names: List[str],
    view_to_id: Dict[str, int],
    apt_token_id: int,
    a_doc: torch.Tensor,
    a_chunk: torch.Tensor,
    view_prototypes: Optional[torch.Tensor],
    layer_ids: List[int],
    lambda_doc: List[float],
    lambda_chunk: List[float],
    device: torch.device,
) -> Tuple[List[float], List[float], List[float], List[Optional[float]], List[Optional[float]]]:
    """
    Batched forward pass for multiple prompts (views).
    Returns lists aligned with `prompts`:
      total/doc/chunk energy + route logit for current candidate view + view energy for current candidate view.
    """
    enc = tokenizer(prompts, return_tensors="pt", add_special_tokens=False, padding=True)
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)

    # locate APT position per row (last occurrence)
    ids = input_ids.tolist()
    apt_pos = []
    for row in ids:
        try:
            apt_pos.append(len(row) - 1 - row[::-1].index(apt_token_id))
        except ValueError:
            raise ValueError("APT token id not found in prompt tokenization. Check --apt_token.")

    hook.pool_idx = torch.tensor(apt_pos, device=device, dtype=torch.long)
    hook.clear()
    _ = model(input_ids=input_ids, attention_mask=attn, use_cache=False)

    zs = [hook.cache[lid] for lid in layer_ids]  # each [B,H]
    proj_doc, proj_chunk = _get_anchor_projs(model)

    # normalize anchors (broadcast to [B,D])
    a_doc_n = F.normalize(a_doc.to(device=device, dtype=proj_doc.weight.dtype), p=2, dim=-1).unsqueeze(0)
    a_chunk_n = F.normalize(a_chunk.to(device=device, dtype=proj_chunk.weight.dtype), p=2, dim=-1).unsqueeze(0)
    a_doc_n = a_doc_n.expand(input_ids.size(0), -1)
    a_chunk_n = a_chunk_n.expand(input_ids.size(0), -1)

    energy_doc = torch.zeros(input_ids.size(0), device=device, dtype=torch.float32)
    energy_chunk = torch.zeros(input_ids.size(0), device=device, dtype=torch.float32)

    for i, z in enumerate(zs):
        zf = z.to(dtype=proj_doc.weight.dtype)
        z_doc = F.normalize(proj_doc(zf), p=2, dim=-1)
        z_chunk = F.normalize(proj_chunk(zf), p=2, dim=-1)
        cos_doc = (z_doc * a_doc_n).sum(dim=-1)
        cos_chunk = (z_chunk * a_chunk_n).sum(dim=-1)
        e_doc = 1.0 - cos_doc.float()
        e_chunk = 1.0 - cos_chunk.float()
        energy_doc += float(lambda_doc[i]) * e_doc
        energy_chunk += float(lambda_chunk[i]) * e_chunk

    energy = energy_doc + energy_chunk
    hook.pool_idx = None
    route_list: List[Optional[float]] = [None] * input_ids.size(0)
    view_e_list: List[Optional[float]] = [None] * input_ids.size(0)

    z_route = zs[-1].to(dtype=proj_doc.weight.dtype)  # [B,H]
    proj_route = _get_optional_proj(model, "anchor_proj_route")
    if proj_route is not None:
        logits = proj_route(z_route).float()  # [B,V]
        for i, vname in enumerate(view_names):
            vid = int(view_to_id.get(vname, -1))
            if 0 <= vid < logits.size(-1):
                route_list[i] = float(logits[i, vid].detach().cpu())

    proj_view = _get_optional_proj(model, "anchor_proj_view")
    if proj_view is not None and view_prototypes is not None:
        z_view = F.normalize(proj_view(z_route).float(), p=2, dim=-1)  # [B,D]
        proto = F.normalize(view_prototypes.to(device=device, dtype=torch.float32), p=2, dim=-1)  # [V,D]
        for i, vname in enumerate(view_names):
            vid = int(view_to_id.get(vname, -1))
            if 0 <= vid < proto.size(0):
                cos = (z_view[i] * proto[vid]).sum()
                view_e_list[i] = float((1.0 - cos).detach().cpu())

    return (
        energy.detach().cpu().tolist(),
        energy_doc.detach().cpu().tolist(),
        energy_chunk.detach().cpu().tolist(),
        route_list,
        view_e_list,
    )


def coverage_bias(view: str, count_global: Dict[str, int], count_doc: Dict[str, int]) -> float:
    cg = count_global.get(view, 0)
    cd = count_doc.get(view, 0)
    return float(-math.log(1.0 + float(cg) + float(cd)))


def select_views_greedy(
    energies: Dict[str, float],
    count_global: Dict[str, int],
    count_doc: Dict[str, int],
    K: int,
) -> List[str]:
    """
    Greedy selection:
      score(view) = -energy + coverage_bias
    """
    selected: List[str] = []
    remaining = list(energies.keys())

    for _ in range(min(K, len(remaining))):
        best_v = None
        best_score = -1e18
        for v in remaining:
            base = -energies[v] + coverage_bias(v, count_global, count_doc)
            if base > best_score:
                best_score = base
                best_v = v
        assert best_v is not None
        selected.append(best_v)
        remaining.remove(best_v)
        # update doc counts immediately to encourage diversity within same chunk
        count_doc[best_v] = count_doc.get(best_v, 0) + 1

    return selected


def build_routing_energy(
    energies_docchunk: Dict[str, float],
    route_logits: Dict[str, Optional[float]],
    view_energies: Dict[str, Optional[float]],
) -> Dict[str, float]:
    """
    Convert heterogeneous signals to one comparable routing energy:
      E_route = E_docchunk + E_view - route_logit
    Lower is better.
    """
    out: Dict[str, float] = {}
    for v, e_dc in energies_docchunk.items():
        val = float(e_dc)
        ev = view_energies.get(v, None)
        if ev is not None:
            val += float(ev)
        rv = route_logits.get(v, None)
        if rv is not None:
            val -= float(rv)
        out[v] = float(val)
    return out


# -------------------------
# Generation helpers
# -------------------------
def _strip_prompt_from_decode(tokenizer, input_ids: torch.Tensor, gen_ids: torch.Tensor) -> str:
    """
    Decode only the newly generated part after prompt length.
    """
    plen = input_ids.shape[1]
    out_ids = gen_ids[0, plen:]
    return tokenizer.decode(out_ids, skip_special_tokens=True).strip()


_SENT_SPLIT_RE = re.compile(r"(?<=[。！？!?])\s+|\n+")


def _split_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    text = re.sub(r"\s*;\s*", " ; ", text)
    text = re.sub(r"\s*[；]\s*", " ； ", text)
    text = re.sub(r"(?<=\))\s*(?=\(\d+\))", "\n", text)
    text = re.sub(r"(?<=\.)\s*(?=\(\d+\))", "\n", text)
    parts = _SENT_SPLIT_RE.split(text)
    sents: List[str] = []
    for p in parts:
        p = re.sub(r"\s+", " ", (p or "").strip())
        if not p:
            continue
        clauses = re.split(r"\s[;；]\s|(?<=\))\s+(?=\(\d+\))", p)
        for c in clauses:
            c = re.sub(r"\s+", " ", c.strip())
            if c:
                sents.append(c)
    return sents


def _tokenize_cheap(s: str) -> List[str]:
    s = (s or "").lower()
    return re.findall(r"[\u4e00-\u9fff]|[a-z0-9]+", s)


def _simple_tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())


_BAD_TAIL_RE = re.compile(
    r"\b(?:For more details|According to th|This is supported by evi|forecasted to be rou)\b.*$",
    re.IGNORECASE,
)
_DANGLING_END_RE = re.compile(
    r"\b(?:via|by|with|for|to|from|of|in|on|at|than|because|due|according)\.$",
    re.IGNORECASE,
)
_STUDY_FRAGMENT_RE = re.compile(
    r"^(?:The study|This study|Research|The research)\b[^.]{0,80}\.$",
    re.IGNORECASE,
)
_HAS_VERB_RE = re.compile(
    r"\b(?:is|are|was|were|be|been|being|has|have|had|show|shows|showed|find|finds|found|indicate|indicates|indicated|"
    r"suggest|suggests|suggested|mean|means|meant|refer|refers|referred|depend|depends|depended|use|uses|used|"
    r"reduce|reduces|reduced|increase|increases|increased|include|includes|included|account|accounts|accounted|"
    r"require|requires|required|provide|provides|provided|drive|drives|driven|benefit|benefits|benefited|"
    r"rely|relies|relied|exist|exists|existed|cause|causes|caused|help|helps|helped|state|states|stated|"
    r"report|reports|reported|identify|identifies|identified|estimate|estimates|estimated)\b",
    re.IGNORECASE,
)


def _looks_fragmentary_answer(answer: str) -> bool:
    a = answer.strip()
    if not a:
        return True
    if _answer_is_author_only_fragment(a):
        return True
    if _DANGLING_END_RE.search(a):
        return True
    toks = _simple_tokenize(a)
    if _STUDY_FRAGMENT_RE.match(a) and not _HAS_VERB_RE.search(a):
        return True
    if len(toks) >= 6 and not _HAS_VERB_RE.search(a):
        return True
    return False


def _answer_ok(
    answer: str,
    question: str,
    body: str,
    *,
    min_chars: int,
    max_chars: int,
    min_unique_ratio: float,
    max_q_overlap: float,
) -> Tuple[bool, str]:
    if not answer:
        return False, "empty"
    a = answer.strip()
    if len(a) < min_chars:
        return False, "min_chars"
    if len(a) > max_chars:
        return False, "max_chars"
    if not re.search(r"(?<!\d)\.(?!\d)|[。！？!?]$", a):
        return False, "no_terminal_punct"
    if _BAD_TAIL_RE.search(a):
        return False, "bad_tail"
    if _looks_fragmentary_answer(a):
        return False, "incomplete"
    if _answer_is_author_only_fragment(a):
        return False, "author_only"
    if _answer_has_numeric_hallucination(a, body):
        return False, "numeric_hallucination"
    toks = _simple_tokenize(a)
    if _answer_overlap_score(a, body) < 1 and a not in _NOT_STATED_ANSWERS:
        return False, "low_body_overlap"
    if _QUESTION_NUMERIC_CUE_RE.search(question) and not _extract_num_tokens(a) and a not in _NOT_STATED_ANSWERS:
        return False, "missing_numeric_answer"
    if not _answer_has_substantive_content_for_citation_question(question, a):
        return False, "no_substantive_content"
    if not _answer_relation_is_supported(a, body):
        return False, "relation_unsupported"
    if len(toks) >= 8:
        uniq_ratio = len(set(toks)) / max(1, len(toks))
        if uniq_ratio < min_unique_ratio:
            return False, "low_unique_ratio"
    q_toks = _simple_tokenize(question)
    if q_toks:
        overlap = len(set(toks) & set(q_toks)) / max(1, len(set(q_toks)))
        if overlap > max_q_overlap:
            return False, "high_q_overlap"
    return True, ""


# -------------------------
# Postprocess helpers# -------------------------
# Postprocess helpers (QA v2)
# -------------------------
def _postprocess_single_line(txt: str) -> str:
    txt = (txt or "").strip()
    if not txt:
        return ""
    return txt.splitlines()[0].strip()


_ANS_PREFIX_RE = re.compile(r"^(?:\s*(?:Answer\s*:\s*|###\s*Response\s*:\s*|答\s*[:：]\s*))", re.IGNORECASE)
_NOT_STATED_ANSWERS = {"Not stated in the provided text.", "Not stated in the Body."}


def _postprocess_answer(txt: str, max_sentences: int = 2, max_chars_fallback: int = 220) -> str:
    """Postprocess answer text for small models.

    - Do NOT truncate by line (answers may span multiple lines).
    - Strip common prefixes (Answer:, ### Response:, etc.).
    - Keep at most `max_sentences` sentences by punctuation.
    """
    txt = (txt or "").strip()
    if not txt:
        return ""
    txt = _ANS_PREFIX_RE.sub("", txt).strip()

    # Collapse excessive whitespace/newlines.
    txt = re.sub(r"\s+", " ", txt).strip()
    txt = _BAD_TAIL_RE.sub("", txt).strip(" ,;:-")

    # Keep the first complete sentence(s). Treat '.' as a sentence end only
    # when it is not part of a decimal number.
    end_re = re.compile(r"(?<!\d)\.(?!\d)|[。！？!?]")
    ends = [m.end() for m in end_re.finditer(txt)]
    if ends:
        cut = ends[min(max_sentences, len(ends)) - 1]
        return txt[:cut].strip()

    # Fallback: trim to a natural boundary first, then to a word boundary.
    txt = txt[:max_chars_fallback].strip()
    for sep in ["; ", ", ", " and ", " but "]:
        pos = txt.rfind(sep)
        if pos >= max(24, max_chars_fallback // 3):
            candidate = txt[:pos].strip() + "."
            if not _looks_fragmentary_answer(candidate):
                return candidate
    if " " in txt:
        txt = txt.rsplit(" ", 1)[0].strip()
    candidate = txt.rstrip(" ,;:-") + "."
    if _looks_fragmentary_answer(candidate):
        return ""
    return candidate


def is_bad_question(q: str, min_len: int = 8) -> bool:
    if not q or len(q.strip()) < min_len:
        return True
    # must contain at least one '?', or end with '？'/'?'
    if "?" not in q and "？" not in q:
        # allow "What/How/Why..." as question-like
        if not re.match(r"^(what|how|why|when|where|which|who)\b", q.strip(), flags=re.I):
            return True
    return False


@torch.no_grad()
def generate_question_text(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    do_sample: bool,
) -> str:
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)

    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attn,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        repetition_penalty=repetition_penalty,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    gen_ids = model.generate(**gen_kwargs)
    txt = _strip_prompt_from_decode(tokenizer, input_ids, gen_ids).strip()
    return _postprocess_single_line(txt)


@torch.no_grad()
def generate_answer_text(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    do_sample: bool,
    no_repeat_ngram_size: int = 3,
    max_sentences: int = 2,
) -> str:
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)

    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attn,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    gen_ids = model.generate(**gen_kwargs)

    txt = _strip_prompt_from_decode(tokenizer, input_ids, gen_ids).strip()
    return _postprocess_answer(txt, max_sentences=max_sentences)


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()

    # Model
    ap.add_argument("--base_model", type=str, required=True, help="Base model name/path (same as training model_name_or_path)")
    ap.add_argument("--lora_dir", type=str, required=True, help="LoRA adapter dir (output_dir from training)")
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--device", type=str, default="cuda")

    # Anchors
    ap.add_argument("--doc_anchor_pt", type=str, required=True, help="doc_anchor_emb.pt (same alignment as input jsonl)")
    ap.add_argument("--chunk_anchor_pt", type=str, required=True, help="chunk_anchor_emb.pt")
    ap.add_argument("--anchor_row_key", type=str, default="anchor_row_idx", help="field name in input jsonl for anchor row index")

    # Data
    ap.add_argument("--input_jsonl", type=str, required=True, help="Input chunks jsonl")
    ap.add_argument("--max_chunks", type=int, default=0, help="Optional cap on input chunks; 0 keeps all.")
    ap.add_argument("--output_jsonl", type=str, required=True, help="Output QA jsonl")
    ap.add_argument("--log_jsonl", type=str, default="", help="Optional routing logs jsonl (energies, selected views)")
    ap.add_argument("--trace_jsonl", type=str, default="", help="Optional trace jsonl (per-QA prompts/outputs)")
    ap.add_argument("--trace_include_prompts", action="store_true", help="Include prompts in trace_jsonl (large)")

    # Routing
    ap.add_argument("--K", type=int, default=3, help="number of views per chunk")
    ap.add_argument("--apt_token", type=str, default="<|APT|>", help="Dedicated APT token inserted before ### Response.")

    # Generation
    ap.add_argument("--m_retry", type=int, default=1, help="question retry times if bad/duplicate")
    ap.add_argument("--q_max_new_tokens", type=int, default=256)
    ap.add_argument("--a_max_new_tokens", type=int, default=512)
    ap.add_argument("--temperature_q", type=float, default=0.7)
    # For answers, default deterministic decoding to avoid verbose / drifting outputs.
    ap.add_argument("--temperature_a", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--repetition_penalty", type=float, default=1.05)
    ap.add_argument("--a_repetition_penalty", type=float, default=1.12, help="repetition penalty for answer generation")
    ap.add_argument("--a_no_repeat_ngram_size", type=int, default=3, help="no-repeat ngram size for answer generation")
    ap.add_argument("--a_max_sentences", type=int, default=1, help="postprocess: keep at most N sentences for answers")
    ap.add_argument("--answer_min_chars", type=int, default=12, help="min answer length (characters)")
    ap.add_argument("--answer_max_chars", type=int, default=160, help="max answer length (characters)")
    ap.add_argument("--answer_min_unique_ratio", type=float, default=0.4, help="min unique token ratio for answer")
    ap.add_argument("--answer_max_q_overlap", type=float, default=0.55, help="max token overlap ratio with question")
    ap.add_argument("--view_prototypes_pt", type=str, default="", help="optional path to view_prototypes.pt; default uses <lora_dir>/view_prototypes.pt")
    ap.add_argument("--seed", type=int, default=42)

    # Probe routing
    ap.add_argument(
        "--probe_batch",
        action="store_true",
        default=True,
        help="Use batched probe routing (recommended). If OOM or failure, will fallback to per-view probing.",
    )

    # Coverage counts persistence
    ap.add_argument("--counts_json", type=str, default="", help="Optional json file to load/save global view counts")
    ap.add_argument("--summary_json", type=str, default="", help="Optional summary json path for view histograms/log stats")
    ap.add_argument("--log_file", type=str, default="", help="Optional log file path (tee stdout/stderr)")

    args = ap.parse_args()


    if args.log_file:
        os.makedirs(os.path.dirname(args.log_file) or ".", exist_ok=True)

        class _Tee:
            def __init__(self, *files):
                self.files = files

            def write(self, data):
                for f in self.files:
                    f.write(data)
                    f.flush()

            def flush(self):
                for f in self.files:
                    f.flush()

        log_f = open(args.log_file, "w", encoding="utf-8")
        sys.stdout = _Tee(sys.stdout, log_f)
        sys.stderr = _Tee(sys.stderr, log_f)
    set_seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    dtype = torch.bfloat16 if args.dtype == "bf16" else (torch.float16 if args.dtype == "fp16" else torch.float32)

    # tokenizer
    tok = load_tokenizer_for_adapter(args.lora_dir, args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # model (base + LoRA)
    try:
        from peft import PeftModel
    except Exception as e:
        raise RuntimeError("peft is required for loading LoRA adapters. Please `pip install peft`.") from e

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=dtype,
        trust_remote_code=True,
        device_map=None,
    ).to(device)

    apt_token_id = ensure_apt_token_and_resize(tok, base, args.apt_token)
    print(f"[INFO] Using APT token: {args.apt_token} (id={apt_token_id})")

    model = PeftModel.from_pretrained(base, args.lora_dir).to(device)
    model.eval()

    # quick compatibility check: base model path in adapter config should match --base_model
    adapter_cfg = os.path.join(args.lora_dir, "adapter_config.json")
    if os.path.exists(adapter_cfg):
        try:
            with open(adapter_cfg, "r", encoding="utf-8") as f:
                cfg_obj = json.load(f)
            trained_base = str(cfg_obj.get("base_model_name_or_path", "") or "")
            if trained_base and os.path.normpath(trained_base) != os.path.normpath(args.base_model):
                print(f"[WARN] adapter trained on base_model={trained_base}, but current --base_model={args.base_model}")
        except Exception:
            pass

    # Ensure the trained dual-anchor projection heads are available.
    _ensure_anchor_projs_from_adapter(model, args.lora_dir, device=device, dtype=dtype)
    model_base = _unwrap_base_model(model)
    route_proj = _get_optional_proj(model, "anchor_proj_route")
    view_proj = _get_optional_proj(model, "anchor_proj_view")
    if route_proj is None:
        print("[WARN] anchor_proj_route not found; routing will fallback to energy-only selection.")
    else:
        print("[INFO] Found anchor_proj_route for explicit routing logits.")
    if view_proj is None:
        print("[WARN] anchor_proj_view not found; view-energy term will be disabled.")

    # layers & hook for probing
    layers = find_transformer_layers(model)
    layer_ids = pick_layers(len(layers))
    lambda_doc, lambda_chunk = make_layer_weights(len(layer_ids))
    hook = PooledLayerHook(layers, layer_ids)
    hook.register()

    # anchors
    doc_anchor = torch.load(args.doc_anchor_pt, map_location="cpu")  # [N, D]
    chunk_anchor = torch.load(args.chunk_anchor_pt, map_location="cpu")
    assert doc_anchor.shape == chunk_anchor.shape, "doc/chunk anchor embeddings must have same shape"
    # Keep anchors on the same device as the model to avoid per-chunk CPU->GPU transfers.
    doc_anchor = doc_anchor.to(device=device, dtype=torch.float32)
    chunk_anchor = chunk_anchor.to(device=device, dtype=torch.float32)
    N = doc_anchor.shape[0]
    anchor_dim = int(doc_anchor.shape[1])
    proj_doc, _proj_chunk = _get_anchor_projs(model)
    proj_dim = int(getattr(proj_doc, "out_features"))
    if proj_dim != anchor_dim:
        raise ValueError(
            f"Anchor dim mismatch: model anchor_proj_doc out_features={proj_dim}, "
            f"but doc/chunk anchors dim={anchor_dim}."
        )

    view_names = list(VIEWS.keys())
    view_to_id = {v: i for i, v in enumerate(view_names)}
    view_prototypes = _load_view_prototypes_for_views(
        lora_dir=args.lora_dir,
        view_names=view_names,
        override_path=args.view_prototypes_pt,
    )
    if view_prototypes is None:
        print("[WARN] view prototypes not available; view-energy term will be disabled.")
    else:
        print(f"[INFO] Loaded aligned view prototypes: shape={tuple(view_prototypes.shape)}")

    # global counts
    count_global: Dict[str, int] = {k: 0 for k in VIEWS}
    if args.counts_json and os.path.exists(args.counts_json):
        with open(args.counts_json, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in count_global:
                    count_global[k] = int(v)

    # load data
    rows = read_jsonl(args.input_jsonl)
    total_input_rows = len(rows)
    if args.max_chunks < 0:
        raise ValueError(f"--max_chunks must be >= 0, got {args.max_chunks}")
    if args.max_chunks > 0:
        rows = rows[: args.max_chunks]
        print(f"[INFO] max_chunks={args.max_chunks}: using first {len(rows)}/{total_input_rows} chunks")
    chunk_id_to_index: Dict[str, int] = {}
    for idx, row in enumerate(rows):
        cid = str(row.get("chunk_id", row.get("chunkid", row.get("cid", row.get("id", ""))))) or str(idx)
        if cid not in chunk_id_to_index:
            chunk_id_to_index[cid] = idx

    # resume support: read existing outputs if present
    seen_chunk_ids = set()
    hist_view_selected: Dict[str, int] = {k: 0 for k in VIEWS}
    hist_view_in_final_questions: Dict[str, int] = {k: 0 for k in VIEWS}
    num_output_rows = 0
    num_qa_pairs = 0
    num_chunks_skipped = 0
    num_q_attempts = 0
    num_q_skipped = 0
    num_a_retry = 0
    num_a_filtered = 0
    last_doc_id = None
    count_doc: Dict[str, int] = {k: 0 for k in VIEWS}
    resume_start_idx = 0

    if args.output_jsonl and os.path.exists(args.output_jsonl):
        try:
            last_row = read_last_jsonl_row(args.output_jsonl)
            if last_row is not None:
                last_chunk_id = str(last_row.get("chunk_id", "")) or ""
                if last_chunk_id and last_chunk_id in chunk_id_to_index:
                    resume_start_idx = int(chunk_id_to_index[last_chunk_id]) + 1
                    print(
                        f"[INFO] resume from output tail: last chunk_id={last_chunk_id}, "
                        f"next input row={resume_start_idx + 1}/{len(rows)}"
                    )
            for r in iter_jsonl(args.output_jsonl):
                cid = str(r.get("chunk_id", "")) or ""
                if cid:
                    seen_chunk_ids.add(cid)
                num_output_rows += 1
                num_qa_pairs += 1
                v = r.get("view", "")
                if v in hist_view_in_final_questions:
                    hist_view_in_final_questions[v] = hist_view_in_final_questions.get(v, 0) + 1
                    if not args.counts_json:
                        count_global[v] = count_global.get(v, 0) + 1
                sv = r.get("selected_views", []) or []
                for vv in sv:
                    if vv in hist_view_selected:
                        hist_view_selected[vv] = hist_view_selected.get(vv, 0) + 1
                # track last doc_id + per-doc counts for resume
                doc_id = str(r.get("doc_id", "")) or ""
                if doc_id:
                    if doc_id != last_doc_id:
                        count_doc = {k: 0 for k in VIEWS}
                        last_doc_id = doc_id
                    if v in count_doc:
                        count_doc[v] = count_doc.get(v, 0) + 1
        except Exception as e:
            print(f"[WARN] Failed to read existing output for resume: {e}")

    # output writers (append for resume)
    out_f = open(args.output_jsonl, "a", encoding="utf-8") if args.output_jsonl else None
    log_f = open(args.log_jsonl, "a", encoding="utf-8") if args.log_jsonl else None
    trace_f = open(args.trace_jsonl, "a", encoding="utf-8") if args.trace_jsonl else None

    for i, row in enumerate(rows[resume_start_idx:], start=resume_start_idx):
        ctx = build_chunk_context(row)
        doc_id = ctx["doc_id"] or f"doc_{row.get('doc', '')}" or ""
        if doc_id != last_doc_id:
            count_doc = {k: 0 for k in VIEWS}
            last_doc_id = doc_id

        chunk_id = ctx["chunk_id"] or str(row.get("chunk_id", "")) or str(i)
        if chunk_id in seen_chunk_ids:
            print(f"[INFO] skip chunk {i+1}/{len(rows)} doc_id={doc_id} chunk_id={chunk_id} (already processed)")
            num_chunks_skipped += 1
            continue

        print(f"[INFO] chunk {i+1}/{len(rows)} doc_id={doc_id} chunk_id={chunk_id}")

        # anchor index
        anchor_row = row.get(args.anchor_row_key, row.get("anchor_row", row.get("anchor_idx", None)))
        if anchor_row is None:
            anchor_row = i
        anchor_row = int(anchor_row)
        if anchor_row < 0 or anchor_row >= N:
            raise IndexError(f"anchor_row_idx={anchor_row} out of range [0,{N}). Check input_jsonl alignment.")

        a_doc = doc_anchor[anchor_row]
        a_chunk = chunk_anchor[anchor_row]

        # --------- probe energies for all views (prefer batched) ---------
        probe_prompts = [
            _build_probe_prompt(
                ctx=ctx,
                view_name=vname,
                apt_token=args.apt_token,
            )
            for vname in view_names
        ]
        energies: Dict[str, float] = {}
        energies_doc: Dict[str, float] = {}
        energies_chunk: Dict[str, float] = {}
        route_logits_cur: Dict[str, Optional[float]] = {v: None for v in view_names}
        energies_view: Dict[str, Optional[float]] = {v: None for v in view_names}

        if args.probe_batch:
            try:
                e_total_list, e_doc_list, e_chunk_list, r_cur_list, e_view_list = compute_anchor_energy_batch(
                    model=model,
                    hook=hook,
                    tokenizer=tok,
                    prompts=probe_prompts,
                    view_names=view_names,
                    view_to_id=view_to_id,
                    apt_token_id=apt_token_id,
                    a_doc=a_doc,
                    a_chunk=a_chunk,
                    view_prototypes=view_prototypes,
                    layer_ids=layer_ids,
                    lambda_doc=lambda_doc,
                    lambda_chunk=lambda_chunk,
                    device=device,
                )
                energies = {v: e_total_list[i] for i, v in enumerate(view_names)}
                energies_doc = {v: e_doc_list[i] for i, v in enumerate(view_names)}
                energies_chunk = {v: e_chunk_list[i] for i, v in enumerate(view_names)}
                route_logits_cur = {v: r_cur_list[i] for i, v in enumerate(view_names)}
                energies_view = {v: e_view_list[i] for i, v in enumerate(view_names)}
            except RuntimeError as e:
                # Typical failure: CUDA OOM under long prompts.
                print(f"[WARN] Batched probe routing failed; fallback to per-view probing. err={e}")
                torch.cuda.empty_cache()

        if not energies:
            # per-view fallback (slower but robust)
            for vname, pp in zip(view_names, probe_prompts):
                e, e_doc, e_chunk, r_cur, e_view = compute_anchor_energy(
                    model=model,
                    hook=hook,
                    tokenizer=tok,
                    prompt=pp,
                    view_name=vname,
                    view_to_id=view_to_id,
                    apt_token_id=apt_token_id,
                    a_doc=a_doc,
                    a_chunk=a_chunk,
                    view_prototypes=view_prototypes,
                    layer_ids=layer_ids,
                    lambda_doc=lambda_doc,
                    lambda_chunk=lambda_chunk,
                    device=device,
                )
                energies[vname] = e
                energies_doc[vname] = e_doc
                energies_chunk[vname] = e_chunk
                route_logits_cur[vname] = r_cur
                energies_view[vname] = e_view

        routing_energy = build_routing_energy(
            energies_docchunk=energies,
            route_logits=route_logits_cur,
            view_energies=energies_view,
        )

        # --------- select K views (greedy with coverage bias) ---------
        # NOTE: select_views_greedy updates count_doc internally (for within-chunk diversity)
        selected_views = select_views_greedy(
            energies=routing_energy,
            count_global=count_global,
            count_doc=count_doc,
            K=args.K,
        )

        print(f"[INFO] selected_views={selected_views}")

        if log_f is not None:
            log_row = {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "anchor_row_idx": anchor_row,
                "routing_probe_energy_total": energies,
                "routing_probe_energy_doc": energies_doc,
                "routing_probe_energy_chunk": energies_chunk,
                "routing_probe_route_logit": route_logits_cur,
                "routing_probe_energy_view": energies_view,
                "routing_energy_fused": routing_energy,
                "selected_views": selected_views,
                "count_doc_before": {k: int(v) for k, v in count_doc.items()},
                "count_global_before": {k: int(v) for k, v in count_global.items()},
            }
            log_f.write(json.dumps(log_row, ensure_ascii=False) + "\n")
            log_f.flush()

        for v in selected_views:
            hist_view_selected[v] = hist_view_selected.get(v, 0) + 1
            count_global[v] = count_global.get(v, 0) + 1

        # --------- generate Q -> A sequentially (one QA per line) ---------
        seen_questions: List[str] = []
        seen_final_qas: List[Dict[str, Any]] = []
        view_to_questions: Dict[str, List[str]] = {}
        q_idx_global = 0  # question_idx within chunk across all views
        for vname in selected_views:
            questions: List[str] = []
            for _ in range(1):
                question = ""
                last_raw = ""
                for attempt in range(args.m_retry + 1):
                    q_prompt = _build_question_prompt(
                        ctx=ctx,
                        view_name=vname,
                        n_questions=1,
                        existing_questions=seen_questions + questions,
                    )
                    raw_q = generate_question_text(
                        model=model,
                        tokenizer=tok,
                        prompt=q_prompt,
                        device=device,
                        max_new_tokens=args.q_max_new_tokens,
                        temperature=args.temperature_q,
                        top_p=args.top_p,
                        repetition_penalty=args.repetition_penalty,
                        do_sample=True,
                    )
                    last_raw = raw_q
                    num_q_attempts += 1
                    # v2 prompt: raw_q should already be ONE question line
                    q = _normalize_question(_first_question(raw_q))
                    q_bad = (not q) or is_bad_question(q)
                    dup = _question_is_near_duplicate(q, seen_questions + questions)
                    if trace_f is not None:
                        trace_row = {
                            "type": "q_attempt",
                            "doc_id": doc_id,
                            "chunk_id": chunk_id,
                            "view": vname,
                            "attempt": int(attempt),
                            "q_raw": raw_q,
                            "q_norm": q,
                            "q_bad": q_bad,
                            "q_dup": dup,
                        }
                        if args.trace_include_prompts:
                            trace_row["q_prompt"] = q_prompt
                        trace_f.write(json.dumps(trace_row, ensure_ascii=False) + "\n")
                        trace_f.flush()
                    if q_bad or dup:
                        continue
                    question = q
                    break

                # If still invalid after retries, skip this routed view.
                if not question:
                    num_q_skipped += 1
                    continue

                questions.append(question)
                seen_questions.append(question)

                chunk_body = (ctx.get("body") or "").strip()
                a_prompt = _build_answer_prompt(ctx, question)
                raw_a = generate_answer_text(
                    model=model,
                    tokenizer=tok,
                    prompt=a_prompt,
                    device=device,
                    max_new_tokens=args.a_max_new_tokens,
                    temperature=args.temperature_a,
                    top_p=1.0,
                    repetition_penalty=args.a_repetition_penalty,
                    do_sample=(args.temperature_a > 1e-6),
                    no_repeat_ngram_size=args.a_no_repeat_ngram_size,
                    max_sentences=args.a_max_sentences,
                )
                answer = _clean_answer_surface(raw_a)
                ok_first, ok_first_reason = _answer_ok(
                    answer,
                    question,
                    chunk_body,
                    min_chars=args.answer_min_chars,
                    max_chars=args.answer_max_chars,
                    min_unique_ratio=args.answer_min_unique_ratio,
                    max_q_overlap=args.answer_max_q_overlap,
                )
                if trace_f is not None:
                    trace_row = {
                        "type": "a_attempt",
                        "doc_id": doc_id,
                        "chunk_id": chunk_id,
                        "view": vname,
                        "question": question,
                        "a_raw": raw_a,
                        "a_post": answer,
                        "a_ok": ok_first,
                        "a_ok_reason": ok_first_reason,
                    }
                    if args.trace_include_prompts:
                        trace_row["a_prompt"] = a_prompt
                    trace_f.write(json.dumps(trace_row, ensure_ascii=False) + "\n")
                    trace_f.flush()

                if not ok_first:
                    num_a_retry += 1
                    raw_a = generate_answer_text(
                        model=model,
                        tokenizer=tok,
                        prompt=a_prompt,
                        device=device,
                        max_new_tokens=args.a_max_new_tokens,
                        temperature=0.0,
                        top_p=1.0,
                        repetition_penalty=args.a_repetition_penalty,
                        do_sample=False,
                        no_repeat_ngram_size=args.a_no_repeat_ngram_size,
                        max_sentences=args.a_max_sentences,
                    )
                    answer = _clean_answer_surface(raw_a)
                    ok_second, ok_second_reason = _answer_ok(
                        answer,
                        question,
                        chunk_body,
                        min_chars=args.answer_min_chars,
                        max_chars=args.answer_max_chars,
                        min_unique_ratio=args.answer_min_unique_ratio,
                        max_q_overlap=args.answer_max_q_overlap,
                    )
                    if trace_f is not None:
                        trace_row = {
                            "type": "a_retry",
                            "doc_id": doc_id,
                            "chunk_id": chunk_id,
                            "view": vname,
                            "question": question,
                            "a_raw": raw_a,
                            "a_post": answer,
                            "a_ok": ok_second,
                            "a_ok_reason": ok_second_reason,
                        }
                        if args.trace_include_prompts:
                            trace_row["a_prompt"] = a_prompt
                        trace_f.write(json.dumps(trace_row, ensure_ascii=False) + "\n")
                        trace_f.flush()
                    if not ok_second:
                        num_a_filtered += 1
                        continue

                if _qa_is_near_duplicate(question, answer, seen_final_qas):
                    num_a_filtered += 1
                    continue

                out_row = {
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "anchor_row_idx": anchor_row,
                    "selected_views": selected_views,
                    "view": vname,
                    "question_idx": q_idx_global,
                    "question": question,
                    "answer": answer,
                }
                if out_f is not None:
                    out_f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                    out_f.flush()
                seen_final_qas.append(
                    {
                        "question": question,
                        "answer": answer,
                    }
                )
                num_output_rows += 1
                num_qa_pairs += 1
                if vname in hist_view_in_final_questions:
                    hist_view_in_final_questions[vname] = hist_view_in_final_questions.get(vname, 0) + 1
                q_idx_global += 1

            view_to_questions[vname] = questions
            print(f"[INFO] view={vname} n_questions={len(questions)} questions={questions}")

        seen_chunk_ids.add(chunk_id)

        if log_f is not None:
            log_row = {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "anchor_row_idx": anchor_row,
                "questions_by_view": view_to_questions,
                "count_doc_after": {k: int(v) for k, v in count_doc.items()},
                "count_global_after": {k: int(v) for k, v in count_global.items()},
            }
            log_f.write(json.dumps(log_row, ensure_ascii=False) + "\n")
            log_f.flush()

        if (i + 1) % 20 == 0:
            print(f"[INFO] processed {i+1}/{len(rows)} chunks")

    # close writers
    if out_f is not None:
        out_f.close()
    if log_f is not None:
        log_f.close()
    if trace_f is not None:
        trace_f.close()

    if args.counts_json:
        os.makedirs(os.path.dirname(args.counts_json) or ".", exist_ok=True)
        with open(args.counts_json, "w", encoding="utf-8") as f:
            json.dump(count_global, f, ensure_ascii=False, indent=2)

    summary_json = args.summary_json
    if not summary_json and args.output_jsonl:
        stem, _ = os.path.splitext(args.output_jsonl)
        summary_json = stem + ".summary.json"
    if summary_json:
        os.makedirs(os.path.dirname(summary_json) or ".", exist_ok=True)
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "num_chunks": len(rows),
                    "num_chunks_total_input": int(total_input_rows),
                    "max_chunks": int(args.max_chunks),
                    "num_chunks_skipped": int(num_chunks_skipped),
                    "num_output_rows": int(num_output_rows),
                    "num_qa_pairs": int(num_qa_pairs),
                    "num_q_attempts": int(num_q_attempts),
                    "num_q_skipped": int(num_q_skipped),
                    "num_a_retry": int(num_a_retry),
                    "num_a_filtered": int(num_a_filtered),
                    "K": int(args.K),
                    "hist_view_selected": {k: int(v) for k, v in hist_view_selected.items()},
                    "hist_view_in_final_questions": {k: int(v) for k, v in hist_view_in_final_questions.items()},
                    "layer_ids": layer_ids,
                    "lambda_doc": lambda_doc,
                    "lambda_chunk": lambda_chunk,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    # cleanup hook
    try:
        hook.remove()
    except Exception:
        pass

    print("[DONE] wrote:", args.output_jsonl)
    if args.log_jsonl:
        print("[DONE] wrote logs:", args.log_jsonl)
    if args.counts_json:
        print("[DONE] saved counts:", args.counts_json)
    if summary_json:
        print("[DONE] wrote summary:", summary_json)
    if args.trace_jsonl:
        print("[DONE] wrote trace:", args.trace_jsonl)


if __name__ == "__main__":
    main()
