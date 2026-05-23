#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phi-4-mini-instruct LoRA dual-anchor training script.
Adapted from train_lora_dual_anchor.py for Phi-4 architecture.

Key differences from Qwen version:
  - Default LoRA target modules: qkv_proj, o_proj, gate_up_proj, down_proj
    (Phi-4 uses combined QKV and gate_up projections)
"""

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import random
import re
from transformers import PreTrainedTokenizerBase
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from peft import LoraConfig, get_peft_model

# -------------------------
# View taxonomy (Bloom-inspired information-function taxonomy, fixed 8)
# -------------------------
VIEW_NAMES: List[str] = [
    "concept",
    "mechanism",
    "condition",
    "evidence",
    "causation",
    "comparison",
    "impact",
    "quantitative",
]
VIEW_TO_ID: Dict[str, int] = {v: i for i, v in enumerate(VIEW_NAMES)}


def normalize_view_name(view: str) -> str:
    return str(view or "").strip().lower()


# -------------------------
# Utilities
# -------------------------
import torch.nn as nn

def _unwrap_base_model(model: nn.Module) -> nn.Module:
    """
    Unwrap PEFT model to the underlying HF model if needed.
    """
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


def _token_is_single_id(tokenizer: PreTrainedTokenizerBase, s: str) -> bool:
    ids = tokenizer(s, add_special_tokens=False).get("input_ids", [])
    return isinstance(ids, list) and len(ids) == 1


def ensure_phi_cache_compat() -> None:
    """
    Some local Phi-4/Phi-3 modeling files import SlidingWindowCache, while a
    few transformers builds expose Cache/DynamicCache/StaticCache but not that
    symbol. The same local file imports LossKwargs from transformers.utils,
    which is only a typing helper in newer builds. Training runs with
    use_cache=False, so these shims only need to satisfy import-time/type
    checks in the dynamic module.
    """
    try:
        import transformers.cache_utils as cache_utils
    except Exception:
        cache_utils = None
    if cache_utils is not None and not hasattr(cache_utils, "SlidingWindowCache") and hasattr(cache_utils, "StaticCache"):
        cache_utils.SlidingWindowCache = cache_utils.StaticCache  # type: ignore[attr-defined]
        print("[WARN] transformers.cache_utils.SlidingWindowCache missing; aliased to StaticCache for Phi compatibility.")

    try:
        import transformers.utils as transformers_utils
        from typing import TypedDict
    except Exception:
        return
    if not hasattr(transformers_utils, "LossKwargs"):
        class LossKwargs(TypedDict, total=False):
            pass

        transformers_utils.LossKwargs = LossKwargs  # type: ignore[attr-defined]
        print("[WARN] transformers.utils.LossKwargs missing; installed empty TypedDict for Phi compatibility.")

    try:
        from transformers.modeling_utils import PreTrainedModel
    except Exception:
        return
    if getattr(PreTrainedModel, "_phi_list_tied_weights_compat", False):
        return
    orig_get_expanded = PreTrainedModel.get_expanded_tied_weights_keys

    def _get_expanded_tied_weights_keys_compat(self, all_submodels: bool = False):
        tied_mapping = getattr(self, "_tied_weights_keys", None)
        if not all_submodels and isinstance(tied_mapping, (list, tuple, set)):
            if not getattr(self.config, "tie_word_embeddings", False):
                return {}
            mapping = {}
            for key in tied_mapping:
                if key == "lm_head.weight" and hasattr(self, "model") and hasattr(self.model, "embed_tokens"):
                    mapping[key] = "model.embed_tokens.weight"
            return mapping
        return orig_get_expanded(self, all_submodels=all_submodels)

    PreTrainedModel.get_expanded_tied_weights_keys = _get_expanded_tied_weights_keys_compat  # type: ignore[method-assign]
    PreTrainedModel._phi_list_tied_weights_compat = True  # type: ignore[attr-defined]
    print("[WARN] Patched PreTrainedModel list-style _tied_weights_keys compatibility for local Phi modeling code.")


def repair_phi_longrope_meta_buffers(model: nn.Module) -> int:
    """
    The local Phi-4/Phi-3 dynamic modeling code keeps a second RoPE tensor in
    `original_inv_freq` as a plain attribute. Some transformers builds
    materialize the registered `inv_freq` buffer during from_pretrained(), but
    leave that plain attribute pointing at the meta tensor created at init time.
    LongRoPE then fails on the first forward when it calls
    `self.original_inv_freq.to(device)`.

    We always recompute inv_freq from the config to avoid carrying over any
    corrupted values from the checkpoint.
    """
    repaired = 0
    for module in model.modules():
        if not (hasattr(module, "inv_freq") and hasattr(module, "original_inv_freq")):
            continue
        if not (hasattr(module, "rope_init_fn") and hasattr(module, "config")):
            continue

        # Always recompute from config to ensure clean values
        inv_freq, attention_scaling = module.rope_init_fn(module.config, torch.device("cpu"))
        module.register_buffer("inv_freq", inv_freq, persistent=False)
        module.original_inv_freq = inv_freq.detach().clone()
        module.attention_scaling = attention_scaling
        long_inv_freq = getattr(module, "long_inv_freq", None)
        if isinstance(long_inv_freq, torch.Tensor) and long_inv_freq.is_meta:
            delattr(module, "long_inv_freq")
        repaired += 1

    if repaired:
        print(f"[WARN] Repaired Phi LongRoPE meta original_inv_freq buffers: {repaired}")
    return repaired


def _assert_finite_tensor(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        finite = torch.isfinite(value)
        finite_count = int(finite.sum().detach().cpu().item())
        total = int(value.numel())
        raise FloatingPointError(f"Non-finite tensor detected: {name}, finite={finite_count}/{total}")


def assert_trainable_parameters_finite(model: nn.Module, where: str) -> None:
    bad: List[str] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        data = param.detach()
        if not torch.isfinite(data).all():
            finite = torch.isfinite(data)
            finite_count = int(finite.sum().cpu().item())
            bad.append(f"{name} finite={finite_count}/{data.numel()}")
            if len(bad) >= 20:
                break
    if bad:
        raise FloatingPointError(f"Non-finite trainable parameters at {where}: " + "; ".join(bad))


def ensure_mid_token_and_resize(
    tokenizer: PreTrainedTokenizerBase,
    model: nn.Module,
    mid_token: str,
) -> int:
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    def _resize_only_if_needed() -> None:
        curr_vocab = int(model.get_input_embeddings().weight.shape[0])
        target_vocab = int(len(tokenizer))
        if target_vocab > curr_vocab:
            print(f"[INFO] Resizing token embeddings from {curr_vocab} to {target_vocab} to match tokenizer.")
            model.resize_token_embeddings(target_vocab)
        elif target_vocab < curr_vocab:
            print(
                f"[INFO] Keep model token embeddings at {curr_vocab}; tokenizer length is {target_vocab}. "
                "Phi models may reserve extra embedding rows, so training should not shrink them."
            )

    tid = tokenizer.convert_tokens_to_ids(mid_token)
    ok_single = _token_is_single_id(tokenizer, mid_token)
    if tid is None or tid == tokenizer.unk_token_id or not ok_single:
        tokenizer.add_special_tokens({"additional_special_tokens": [mid_token]})
        _resize_only_if_needed()
        tid = tokenizer.convert_tokens_to_ids(mid_token)
        if tid is None or tid == tokenizer.unk_token_id or not _token_is_single_id(tokenizer, mid_token):
            raise ValueError(f"Failed to install MID token '{mid_token}' as a single token id.")
    else:
        _resize_only_if_needed()

    if tokenizer.eos_token_id is not None and tid == tokenizer.eos_token_id:
        raise ValueError(f"MID token '{mid_token}' resolves to eos_token_id={tid}.")
    if tokenizer.pad_token_id is not None and tid == tokenizer.pad_token_id:
        raise ValueError(f"MID token '{mid_token}' resolves to pad_token_id={tid}.")
    return int(tid)


def find_transformer_layers(model: nn.Module):
    """
    Robustly locate decoder block list (ModuleList) by scanning the model.
    Works for Phi/Qwen/LLaMA/GPT-style and PEFT-wrapped models.
    """
    base = _unwrap_base_model(model)
    cfg = getattr(base, "config", None)

    num_layers = None
    for k in ["num_hidden_layers", "n_layer", "num_layers", "n_layers"]:
        if cfg is not None and hasattr(cfg, k):
            num_layers = int(getattr(cfg, k))
            break
    if num_layers is None:
        raise ValueError("Cannot infer num_layers from config. Please print base.config and adapt.")

    candidates = []
    for name, mod in base.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) == num_layers:
            candidates.append((name, mod))

    if not candidates:
        raise ValueError(
            f"Cannot locate transformer layers: no nn.ModuleList with length={num_layers} found."
        )

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

    print(f"[INFO] Found transformer layers at: {best_name} (len={len(best_list)}) "
          f"first_layer={best_list[0].__class__.__name__}")

    return list(best_list)



def pick_layers(num_layers: int, mode: str = "quartile") -> List[int]:
    if num_layers <= 0:
        raise ValueError("num_layers must be > 0")

    if mode == "quartile":
        s = sorted(set([num_layers // 4, num_layers // 2, (3 * num_layers) // 4, num_layers - 1]))
        return s
    elif mode == "middle_last":
        s = sorted(set([num_layers // 2, (3 * num_layers) // 4, num_layers - 1]))
        return s
    elif mode == "mid_last2":
        s = sorted(set([num_layers // 2, num_layers - 1]))
        return s
    else:
        raise ValueError(f"Unknown layer picking mode: {mode}")


def make_layer_weights(k: int) -> Tuple[List[float], List[float]]:
    if k == 2:
        return [1.0, 0.0], [0.0, 1.0]
    if k == 4:
        return [1.0, 0.7, 0.4, 0.2], [0.2, 0.4, 0.7, 1.0]
    doc = []
    chunk = []
    for i in range(k):
        t = i / max(k - 1, 1)
        doc.append(1.0 * (1 - t) + 0.2 * t)
        chunk.append(0.2 * (1 - t) + 1.0 * t)
    return doc, chunk


def make_view_layer_weights(lambda_chunk: List[float], view_lambda_scale: float) -> List[float]:
    s = float(view_lambda_scale)
    return [s * w for w in lambda_chunk]


def _safe_chunk_id(row: Dict[str, Any], fallback_idx: int) -> str:
    for k in ["chunk_id", "chunkid", "cid", "id"]:
        v = row.get(k, None)
        if v is not None and str(v).strip():
            return str(v).strip()
    return f"__row_{fallback_idx}"


def _safe_view_id(row: Dict[str, Any]) -> int:
    v = normalize_view_name(str(row.get("view", "") or "").strip())
    return int(VIEW_TO_ID.get(v, -1))


def build_view_prototypes_from_jsonl(
    jsonl_paths: List[str],
    view_anchor: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    if view_anchor is None:
        return None
    if len(view_anchor.shape) != 2:
        raise ValueError("view_anchor must be rank-2 [N, D].")

    num_views = len(VIEW_NAMES)
    d_anchor = int(view_anchor.shape[1])
    sums = torch.zeros(num_views, d_anchor, dtype=torch.float32)
    cnts = torch.zeros(num_views, dtype=torch.float32)

    for p in jsonl_paths:
        if not p:
            continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                vid = _safe_view_id(row)
                if vid < 0:
                    continue
                ridx = row.get("anchor_row_idx", row.get("anchor_row", row.get("anchor_idx", None)))
                if ridx is None:
                    continue
                ridx = int(ridx)
                if ridx < 0 or ridx >= int(view_anchor.shape[0]):
                    continue
                sums[vid] += view_anchor[ridx].float()
                cnts[vid] += 1.0

    global_mean = view_anchor.float().mean(dim=0)
    protos = torch.zeros_like(sums)
    for i in range(num_views):
        if cnts[i] > 0:
            protos[i] = sums[i] / cnts[i]
        else:
            protos[i] = global_mean
    return F.normalize(protos, p=2, dim=-1)


def last_label_index(labels: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    B, T = labels.shape
    idx = torch.full((B,), T - 1, device=labels.device, dtype=torch.long)

    for b in range(B):
        pos = torch.where(labels[b] != -100)[0]
        if len(pos) > 0:
            idx[b] = pos[-1]
        else:
            pos2 = torch.where(attention_mask[b] == 1)[0]
            idx[b] = pos2[-1] if len(pos2) > 0 else T - 1
    return idx


def prompt_end_index(labels: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    B, T = labels.shape
    idx = torch.full((B,), T - 1, device=labels.device, dtype=torch.long)

    for b in range(B):
        pos = torch.where(labels[b] != -100)[0]
        if len(pos) > 0:
            idx[b] = max(int(pos[0]) - 1, 0)
        else:
            pos2 = torch.where(attention_mask[b] == 1)[0]
            idx[b] = pos2[-1] if len(pos2) > 0 else T - 1
    return idx


ANCHOR_HEADERS = [
    "[Article Summary]",
    "[Doc Keywords]",
    "[Document Anchor]",
    "[Document Keywords]",
    "[Chunk Summary]",
    "[Chunk Keywords]",
    "[Chunk Anchor]",
]
DOC_ANCHOR_HEADERS = ["[Article Summary]", "[Doc Keywords]", "[Document Anchor]", "[Document Keywords]"]
CHUNK_ANCHOR_HEADERS = ["[Chunk Summary]", "[Chunk Keywords]", "[Chunk Anchor]"]

def _bucket_from_row(row: Dict[str, Any], prompt_text: str) -> int:
    mode = str(row.get("anchor_mode", "") or "").strip().lower()
    mode_map = {
        "dual": 0,
        "doc_only": 1,
        "chunk_only": 2,
        "none": 3,
    }
    if mode in mode_map:
        return mode_map[mode]

    has_doc = any(h in prompt_text for h in DOC_ANCHOR_HEADERS)
    has_chunk = any(h in prompt_text for h in CHUNK_ANCHOR_HEADERS)
    if has_doc and has_chunk:
        return 0
    if has_doc:
        return 1
    if has_chunk:
        return 2
    return 3

def _remove_anchor_block(text: str, header: str) -> str:
    heads = "|".join(re.escape(h) for h in ANCHOR_HEADERS)
    pat = rf"{re.escape(header)}\s*[\s\S]*?(?=(?:\n(?:{heads}))|\Z)"
    return re.sub(pat, "", text, count=1)

def _truncate_after_header(text: str, header: str, keep_words: int) -> str:
    if keep_words <= 0:
        return text
    heads = "|".join(re.escape(h) for h in ANCHOR_HEADERS)
    pat = rf"({re.escape(header)}\s*)([\s\S]*?)(?=(?:\n(?:{heads}))|\Z)"
    m = re.search(pat, text)
    if not m:
        return text
    prefix = m.group(1)
    body = m.group(2).strip()
    words = body.split()
    body2 = " ".join(words[:keep_words])
    repl = prefix + (body2 + "\n" if body2 else "\n")
    return text[:m.start()] + repl + text[m.end():]

# -------------------------
# Dataset & Collator
# -------------------------
class JSONLSFTDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        tokenizer,
        max_seq_len: int,
        add_task_prefix: bool = True,
        pooling_mode: str = "mid",
        mid_token: str = "<|MID|>",
        drop_doc_anchor_prob=0.0,
        drop_chunk_anchor_prob=0.0,
        anchor_trunc_words=0,
        anchor_dropout_seed=1234,
        forbid_both_drop=True,
        allow_both_drop: bool = False,
        drop_warmup_epochs=0.5,
        drop_ramp_epochs=0.5,
    ):
        self.jsonl_path = jsonl_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.add_task_prefix = add_task_prefix
        self.pooling_mode = str(pooling_mode)
        self.mid_token = str(mid_token)
        self.drop_doc_anchor_prob = drop_doc_anchor_prob
        self.drop_chunk_anchor_prob = drop_chunk_anchor_prob
        self.anchor_trunc_words = anchor_trunc_words
        self.rng = random.Random(anchor_dropout_seed)

        self.forbid_both_drop = bool(forbid_both_drop) and (not allow_both_drop)
        self.drop_warmup_epochs = float(drop_warmup_epochs)
        self.drop_ramp_epochs = float(drop_ramp_epochs)

        self._epoch_progress = 0.0

        self.rows = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.rows.append(json.loads(line))

        self._row_chunk_id: List[str] = []
        self._row_view_id: List[int] = []
        chunk_to_views: Dict[str, set] = {}
        for i, row in enumerate(self.rows):
            cid = _safe_chunk_id(row, i)
            vid = _safe_view_id(row)
            self._row_chunk_id.append(cid)
            self._row_view_id.append(vid)
            if cid not in chunk_to_views:
                chunk_to_views[cid] = set()
            if vid >= 0:
                chunk_to_views[cid].add(vid)
        self._row_observed_view_mask: List[List[int]] = []
        for cid in self._row_chunk_id:
            mask = [0] * len(VIEW_NAMES)
            for vid in chunk_to_views.get(cid, set()):
                if 0 <= vid < len(VIEW_NAMES):
                    mask[vid] = 1
            self._row_observed_view_mask.append(mask)

        self._raw_bucket: List[int] = []
        self._raw_has_doc: List[int] = []
        self._raw_has_chunk: List[int] = []

        for i, row in enumerate(self.rows):
            prompt_raw = (
                row.get("prompt")
                or row.get("instruction")
                or row.get("input")
                or row.get("text")
                or ""
            )
            b = _bucket_from_row(row, prompt_raw)
            has_doc = 1 if b in (0, 1) else 0
            has_chunk = 1 if b in (0, 2) else 0
            self._raw_bucket.append(b)
            self._raw_has_doc.append(has_doc)
            self._raw_has_chunk.append(has_chunk)

        self.tokenizer.padding_side = "right"

    def __len__(self):
        return len(self.rows)

    def set_epoch_progress(self, epoch_progress: float):
        self._epoch_progress = float(epoch_progress)

    def _curriculum_scale(self) -> float:
        e = self._epoch_progress
        if e < self.drop_warmup_epochs:
            return 0.0
        if self.drop_ramp_epochs <= 0:
            return 1.0
        t = (e - self.drop_warmup_epochs) / self.drop_ramp_epochs
        return max(0.0, min(1.0, t))

    def _format_pair(self, row: Dict[str, Any]) -> Tuple[str, str, int, int, int]:
        task = row.get("task", "")
        prompt = row.get("prompt", row.get("instruction", row.get("input", row.get("text", ""))))
        response = row.get("response", row.get("output", ""))

        task_line = ""
        if self.add_task_prefix and task:
            task_line = f"Task: {task}\n"

        p0 = prompt
        p = p0

        scale = self._curriculum_scale()
        p_doc = self.drop_doc_anchor_prob * scale
        p_chunk = self.drop_chunk_anchor_prob * scale

        drop_doc = 1 if (p_doc > 0 and self.rng.random() < p_doc) else 0
        drop_chunk = 1 if (p_chunk > 0 and self.rng.random() < p_chunk) else 0
        both_drop = 1 if (drop_doc == 1 and drop_chunk == 1) else 0

        if both_drop == 1 and self.forbid_both_drop:
            if self.rng.random() < 0.5:
                drop_doc = 0
            else:
                drop_chunk = 0
            both_drop = 0

        if drop_doc == 1:
            for header in DOC_ANCHOR_HEADERS:
                p = _remove_anchor_block(p, header)
        else:
            if self.anchor_trunc_words > 0:
                for header in DOC_ANCHOR_HEADERS:
                    p = _truncate_after_header(p, header, self.anchor_trunc_words)

        if drop_chunk == 1:
            for header in CHUNK_ANCHOR_HEADERS:
                p = _remove_anchor_block(p, header)
        else:
            if self.anchor_trunc_words > 0:
                for header in CHUNK_ANCHOR_HEADERS:
                    p = _truncate_after_header(p, header, self.anchor_trunc_words)

        if self.pooling_mode == "mid":
            prompt_text = f"{task_line}{p.strip()}\n{self.mid_token}\n\n### Response:\n"
        else:
            prompt_text = f"{task_line}{p.strip()}\n\n### Response:\n"
        resp_text = response.strip()
        return prompt_text, resp_text, drop_doc, drop_chunk, both_drop


    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row_idx = idx
        r = self.rows[row_idx]
        prompt_text, resp_text, drop_doc, drop_chunk, both_drop = self._format_pair(r)
        anchor_row_idx = r.get("anchor_row_idx", r.get("anchor_row", r.get("anchor_idx", None)))
        if anchor_row_idx is None:
            anchor_row_idx = row_idx

        raw_bucket = int(self._raw_bucket[row_idx])
        raw_has_doc = int(self._raw_has_doc[row_idx])
        raw_has_chunk = int(self._raw_has_chunk[row_idx])

        eff_has_doc = 1 if (raw_has_doc == 1 and drop_doc == 0) else 0
        eff_has_chunk = 1 if (raw_has_chunk == 1 and drop_chunk == 0) else 0
        eff_bucket = 0 if (eff_has_doc and eff_has_chunk) else (1 if eff_has_doc else (2 if eff_has_chunk else 3))

        tok_prompt = self.tokenizer(prompt_text, add_special_tokens=False, return_tensors=None)
        tok_resp = self.tokenizer(resp_text, add_special_tokens=False, return_tensors=None)
        prompt_ids = tok_prompt["input_ids"]
        resp_ids = tok_resp["input_ids"]

        total_len = len(prompt_ids) + len(resp_ids)
        overflow = max(0, total_len - self.max_seq_len)

        if overflow > 0:
            if overflow < len(prompt_ids):
                prompt_ids = prompt_ids[overflow:]
            else:
                overflow_on_resp = overflow - len(prompt_ids)
                prompt_ids = []
                resp_ids = resp_ids[overflow_on_resp:]

        input_ids = prompt_ids + resp_ids
        attention_mask = [1] * len(input_ids)
        labels = ([-100] * len(prompt_ids)) + resp_ids.copy()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "anchor_row_idx": int(anchor_row_idx),
            "view_id": int(self._row_view_id[row_idx]),
            "observed_view_mask": self._row_observed_view_mask[row_idx],
            "drop_doc": int(drop_doc),
            "drop_chunk": int(drop_chunk),
            "both_drop": int(both_drop),
            "raw_bucket": int(raw_bucket),
            "eff_bucket": int(eff_bucket),
            "raw_has_doc": int(raw_has_doc),
            "raw_has_chunk": int(raw_has_chunk),
        }

@dataclass
class DataCollatorWithPadding:
    tokenizer: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )
        if "anchor_row_idx" in batch and not isinstance(batch["anchor_row_idx"], torch.Tensor):
            batch["anchor_row_idx"] = torch.tensor(batch["anchor_row_idx"], dtype=torch.long)
        for k in [
            "anchor_row_idx",
            "view_id",
            "drop_doc",
            "drop_chunk",
            "both_drop",
            "raw_bucket",
            "eff_bucket",
            "raw_has_doc",
            "raw_has_chunk",
            "curr_phase",
        ]:
            if k in batch and not isinstance(batch[k], torch.Tensor):
                batch[k] = torch.tensor(batch[k], dtype=torch.long)
        if "observed_view_mask" in batch and not isinstance(batch["observed_view_mask"], torch.Tensor):
            batch["observed_view_mask"] = torch.tensor(batch["observed_view_mask"], dtype=torch.float32)
        return batch


# -------------------------
# Hook: store pooled z^l only (B x H)
# -------------------------
class PooledLayerHook:
    SKIP_DURING_EXTERNAL_FORWARD = False

    def __init__(self, layers: List[nn.Module], layer_ids: List[int], drop_doc_anchor_prob=0.0,
             drop_chunk_anchor_prob=0.0,
             anchor_trunc_words=0,
             anchor_dropout_seed=1234,
             forbid_both_drop=True,
             drop_warmup_epochs=0.5,
             drop_ramp_epochs=0.5):
        self.layers = layers
        self.layer_ids = layer_ids
        self.handles = []
        self.pool_idx: Optional[torch.Tensor] = None
        self.cache: Dict[int, torch.Tensor] = {}

    def clear(self):
        self.cache = {}

    def _make_hook(self, layer_id: int):
        def hook_fn(module, inp, out):
            if PooledLayerHook.SKIP_DURING_EXTERNAL_FORWARD:
                return
            hs = out[0] if isinstance(out, (tuple, list)) else out
            assert self.pool_idx is not None, "pool_idx must be set before forward."
            B = hs.size(0)
            idx = self.pool_idx
            z = hs[torch.arange(B, device=hs.device), idx]
            self.cache[layer_id] = z
        return hook_fn

    def register(self):
        for lid in self.layer_ids:
            h = self.layers[lid].register_forward_hook(self._make_hook(lid))
            self.handles.append(h)

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


# -------------------------
# Custom Trainer
# -------------------------
class DualAnchorTrainer(Trainer):
    def __init__(
        self,
        *args,
        doc_anchor_emb: torch.Tensor,
        chunk_anchor_emb: torch.Tensor,
        view_anchor_emb: Optional[torch.Tensor],
        view_prototypes: Optional[torch.Tensor],
        alpha: float,
        beta: float,
        gamma: float,
        rho_multi_pos: float,
        gamma_align: float,
        route_tau: float,
        energy_tau: float,
        layer_ids: List[int],
        lambda_doc: List[float],
        lambda_chunk: List[float],
        lambda_view: Optional[List[float]] = None,
        perturb_eps: float = 0.0,
        pooling_mode: str = "prompt_end",
        mid_token_id: Optional[int] = None,
        **kwargs,
    ):
        self._tok = kwargs.pop("tokenizer", None)
        super().__init__(*args, **kwargs)
        self.doc_anchor_emb = doc_anchor_emb
        self.chunk_anchor_emb = chunk_anchor_emb
        self.view_anchor_emb = view_anchor_emb
        self.view_prototypes = view_prototypes
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.rho_multi_pos = rho_multi_pos
        self.gamma_align = gamma_align
        self.route_tau = route_tau
        self.energy_tau = energy_tau
        self.layer_ids = layer_ids
        self.lambda_doc = lambda_doc
        self.lambda_chunk = lambda_chunk
        self.lambda_view = lambda_view
        self.use_view_anchor = self.view_anchor_emb is not None and self.lambda_view is not None
        self.perturb_eps = perturb_eps

        self.pooling_mode = str(pooling_mode)
        self.mid_token_id = int(mid_token_id) if mid_token_id is not None else None

        self._last_custom_metrics: Dict[str, float] = {}

        self._layers = find_transformer_layers(self.model)
        self._hook = PooledLayerHook(self._layers, self.layer_ids)
        self._hook.register()

    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        if self._last_custom_metrics:
            logs = dict(logs)
            for k, v in self._last_custom_metrics.items():
                if k not in logs:
                    logs[k] = v
        return super().log(logs, *args, **kwargs)


    def _get_anchors(
        self,
        anchor_row_idx: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        a_doc = self.doc_anchor_emb.index_select(0, anchor_row_idx.cpu()).to(device=device, dtype=torch.float32)
        a_chunk = self.chunk_anchor_emb.index_select(0, anchor_row_idx.cpu()).to(device=device, dtype=torch.float32)
        a_view = None
        if self.use_view_anchor and self.view_anchor_emb is not None:
            a_view = self.view_anchor_emb.index_select(0, anchor_row_idx.cpu()).to(device=device, dtype=torch.float32)
        return a_doc, a_chunk, a_view

    def _anchor_and_smooth_loss(
        self,
        zs: List[torch.Tensor],
        a_doc: torch.Tensor,
        a_chunk: torch.Tensor,
        a_view: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        device = zs[0].device
        B = zs[0].size(0)

        base = _unwrap_base_model(self.model)
        proj_doc: nn.Module = getattr(base, "anchor_proj_doc")
        proj_chunk: nn.Module = getattr(base, "anchor_proj_chunk")
        proj_view: Optional[nn.Module] = getattr(base, "anchor_proj_view", None) if self.use_view_anchor else None

        loss_anchor = 0.0
        loss_smooth = 0.0

        metrics: Dict[str, float] = {}

        for i in range(len(zs) - 1):
            z_next = torch.nan_to_num(zs[i + 1].float(), nan=0.0, posinf=1e4, neginf=-1e4)
            z_cur = torch.nan_to_num(zs[i].float(), nan=0.0, posinf=1e4, neginf=-1e4)
            loss_smooth = loss_smooth + (z_next - z_cur).pow(2).mean()

        for i, z in enumerate(zs):
            zf = torch.nan_to_num(z.float(), nan=0.0, posinf=1e4, neginf=-1e4)
            z_doc = proj_doc(zf)
            z_chunk = proj_chunk(zf)
            z_view = proj_view(zf) if (proj_view is not None and a_view is not None) else None

            z_doc = F.normalize(torch.nan_to_num(z_doc.float(), nan=0.0, posinf=1e4, neginf=-1e4), p=2, dim=-1, eps=1e-6)
            z_chunk = F.normalize(torch.nan_to_num(z_chunk.float(), nan=0.0, posinf=1e4, neginf=-1e4), p=2, dim=-1, eps=1e-6)
            if z_view is not None:
                z_view = F.normalize(torch.nan_to_num(z_view.float(), nan=0.0, posinf=1e4, neginf=-1e4), p=2, dim=-1, eps=1e-6)

            a_doc_n = F.normalize(torch.nan_to_num(a_doc.float(), nan=0.0, posinf=1e4, neginf=-1e4), p=2, dim=-1, eps=1e-6)
            a_chunk_n = F.normalize(torch.nan_to_num(a_chunk.float(), nan=0.0, posinf=1e4, neginf=-1e4), p=2, dim=-1, eps=1e-6)
            a_view_n = F.normalize(torch.nan_to_num(a_view.float(), nan=0.0, posinf=1e4, neginf=-1e4), p=2, dim=-1, eps=1e-6) if a_view is not None else None

            cos_doc = (z_doc * a_doc_n).sum(dim=-1)
            cos_chunk = (z_chunk * a_chunk_n).sum(dim=-1)
            cos_view = (z_view * a_view_n).sum(dim=-1) if (z_view is not None and a_view_n is not None) else None

            e_doc = (1.0 - cos_doc).mean()
            e_chunk = (1.0 - cos_chunk).mean()
            e_view = (1.0 - cos_view).mean() if cos_view is not None else None

            term = self.lambda_doc[i] * e_doc + self.lambda_chunk[i] * e_chunk
            if e_view is not None and self.lambda_view is not None:
                term = term + self.lambda_view[i] * e_view
            loss_anchor = loss_anchor + term

            lid = self.layer_ids[i]
            metrics[f"cos_doc_L{lid}"] = float(cos_doc.mean().detach().cpu())
            metrics[f"cos_chunk_L{lid}"] = float(cos_chunk.mean().detach().cpu())
            if cos_view is not None:
                metrics[f"cos_view_L{lid}"] = float(cos_view.mean().detach().cpu())

        metrics["loss_anchor"] = float(loss_anchor.detach().cpu())
        metrics["loss_smooth"] = float(loss_smooth.detach().cpu())
        return loss_anchor, loss_smooth, metrics

    def _compute_router_losses(
        self,
        zs: List[torch.Tensor],
        observed_view_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float], Optional[torch.Tensor]]:
        device = zs[-1].device
        z_route = zs[-1].float()
        base = _unwrap_base_model(self.model)
        proj_route: Optional[nn.Module] = getattr(base, "anchor_proj_route", None)
        proj_view: Optional[nn.Module] = getattr(base, "anchor_proj_view", None)

        zero = torch.tensor(0.0, device=device)
        metrics: Dict[str, float] = {}
        if proj_route is None:
            metrics["loss_multi_pos"] = 0.0
            metrics["loss_energy_align"] = 0.0
            return zero, zero, metrics, None

        route_logits = proj_route(z_route)
        vdim = int(route_logits.size(-1))

        loss_multi_pos = zero
        if observed_view_mask is not None and observed_view_mask.numel() > 0:
            mask = observed_view_mask.to(device=device, dtype=torch.float32)
            if mask.size(-1) != vdim:
                raise ValueError(f"observed_view_mask dim {mask.size(-1)} != route dim {vdim}")
            pos_cnt = mask.sum(dim=-1, keepdim=True)
            valid = (pos_cnt.squeeze(-1) > 0)
            if valid.any():
                tgt = mask[valid] / pos_cnt[valid].clamp_min(1.0)
                log_p = F.log_softmax(route_logits[valid] / max(self.route_tau, 1e-6), dim=-1)
                loss_multi_pos = -(tgt * log_p).sum(dim=-1).mean()

        loss_energy_align = zero
        if self.gamma_align > 0 and proj_view is not None and self.view_prototypes is not None:
            proto = torch.nan_to_num(self.view_prototypes.to(device=device, dtype=torch.float32), nan=0.0, posinf=1e4, neginf=-1e4)
            if proto.size(0) != vdim:
                raise ValueError(f"view_prototypes V={proto.size(0)} != route dim {vdim}")
            z_view = F.normalize(torch.nan_to_num(proj_view(z_route).float(), nan=0.0, posinf=1e4, neginf=-1e4), p=2, dim=-1, eps=1e-6)
            proto_n = F.normalize(proto, p=2, dim=-1, eps=1e-6)
            cos = torch.matmul(z_view, proto_n.t())
            e_view = 1.0 - cos
            tgt = F.softmax(-e_view / max(self.energy_tau, 1e-6), dim=-1).detach()
            log_p = F.log_softmax(route_logits / max(self.route_tau, 1e-6), dim=-1)
            loss_energy_align = F.kl_div(log_p, tgt, reduction="batchmean")
            metrics["e_view_mean"] = float(e_view.mean().detach().cpu())

        metrics["loss_multi_pos"] = float(loss_multi_pos.detach().cpu())
        metrics["loss_energy_align"] = float(loss_energy_align.detach().cpu())
        metrics["route_logit_mean"] = float(route_logits.mean().detach().cpu())
        metrics["route_logit_std"] = float(route_logits.std().detach().cpu())
        return loss_multi_pos, loss_energy_align, metrics, route_logits

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        anchor_row_idx = inputs.pop("anchor_row_idx")
        view_id = inputs.pop("view_id", None)
        observed_view_mask = inputs.pop("observed_view_mask", None)
        drop_doc = inputs.pop("drop_doc", None)
        drop_chunk = inputs.pop("drop_chunk", None)
        both_drop = inputs.pop("both_drop", None)
        raw_bucket = inputs.pop("raw_bucket", None)
        eff_bucket = inputs.pop("eff_bucket", None)
        raw_has_doc = inputs.pop("raw_has_doc", None)
        raw_has_chunk = inputs.pop("raw_has_chunk", None)
        curr_phase = inputs.pop("curr_phase", None)
        labels = inputs.get("labels")
        attention_mask = inputs.get("attention_mask")

        device = labels.device
        a_doc, a_chunk, a_view = self._get_anchors(anchor_row_idx, device=device)

        if self.pooling_mode == "mid":
            if self.mid_token_id is None:
                raise ValueError("pooling_mode=mid requires mid_token_id to be set in DualAnchorTrainer.")
            input_ids = inputs["input_ids"]
            B, _T = input_ids.shape
            pool_idx = torch.empty((B,), device=input_ids.device, dtype=torch.long)
            for b in range(B):
                row = input_ids[b].tolist()
                try:
                    pool_idx[b] = len(row) - 1 - row[::-1].index(self.mid_token_id)
                except ValueError:
                    pool_idx[b] = prompt_end_index(labels, attention_mask)[b]
        else:
            pool_idx = prompt_end_index(labels, attention_mask)
        self._hook.pool_idx = pool_idx
        self._hook.clear()

        outputs = model(**inputs, use_cache=False)
        lm_loss = outputs.loss

        zs = []
        for lid in self.layer_ids:
            if lid not in self._hook.cache:
                raise RuntimeError(f"Missing hook cache for layer {lid}.")
            zs.append(self._hook.cache[lid])

        loss_anchor, loss_smooth, metric = self._anchor_and_smooth_loss(zs, a_doc, a_chunk, a_view)
        loss_multi_pos, loss_energy_align, route_metric, route_logits = self._compute_router_losses(
            zs=zs,
            observed_view_mask=observed_view_mask,
        )
        metric.update(route_metric)

        total_loss = (
            lm_loss
            + self.alpha * loss_anchor
            + self.rho_multi_pos * loss_multi_pos
            + self.gamma_align * loss_energy_align
            + self.beta * loss_smooth
        )
        for name, value in {
            "loss_lm": lm_loss,
            "loss_anchor": loss_anchor,
            "loss_multi_pos": loss_multi_pos,
            "loss_energy_align": loss_energy_align,
            "loss_smooth": loss_smooth,
            "total_loss_pre_perturb": total_loss,
        }.items():
            _assert_finite_tensor(name, value)

        loss_pert = None
        if self.gamma > 0 and self.perturb_eps > 0:
            self._hook.clear()
            input_ids = inputs["input_ids"]
            embed = model.get_input_embeddings()(input_ids)
            noise = torch.randn_like(embed) * self.perturb_eps
            embed_pert = embed + noise

            _ = model(
                inputs_embeds=embed_pert,
                attention_mask=inputs["attention_mask"],
                use_cache=False,
            )

            zs_pert = []
            for lid in self.layer_ids:
                zs_pert.append(self._hook.cache[lid])

            loss_anchor_p, loss_smooth_p, metric_p = self._anchor_and_smooth_loss(zs_pert, a_doc, a_chunk, a_view)
            loss_pert = loss_anchor_p + loss_smooth_p
            total_loss = total_loss + self.gamma * loss_pert
            _assert_finite_tensor("loss_perturbed", loss_pert)
            _assert_finite_tensor("total_loss_post_perturb", total_loss)
            metric["loss_perturbed"] = float(loss_pert.detach().cpu())

        metric.update(
            {
                "loss_lm": float(lm_loss.detach().cpu()),
                "alpha": float(self.alpha),
                "beta": float(self.beta),
                "gamma": float(self.gamma),
                "rho_multi_pos": float(self.rho_multi_pos),
                "gamma_align": float(self.gamma_align),
                "use_view_anchor": 1.0 if self.use_view_anchor else 0.0,
            }
        )
        if route_logits is not None and view_id is not None:
            vid = view_id.to(device=route_logits.device, dtype=torch.long)
            valid = (vid >= 0) & (vid < route_logits.size(-1))
            if valid.any():
                pred = route_logits.argmax(dim=-1)
                acc = (pred[valid] == vid[valid]).float().mean()
                metric["route_current_view_acc"] = float(acc.detach().cpu())

        if drop_doc is not None:
            metric["drop_doc_rate"] = float(drop_doc.float().mean().detach().cpu())

        if drop_chunk is not None:
            metric["drop_chunk_rate"] = float(drop_chunk.float().mean().detach().cpu())

        if both_drop is not None:
            metric["both_drop_rate"] = float(both_drop.float().mean().detach().cpu())
        if raw_bucket is not None:
            metric["raw_bucket"] = float(raw_bucket.float().mean().detach().cpu())
        if eff_bucket is not None:
            metric["eff_bucket"] = float(eff_bucket.float().mean().detach().cpu())
        if raw_has_doc is not None:
            metric["raw_has_doc"] = float(raw_has_doc.float().mean().detach().cpu())
        if raw_has_chunk is not None:
            metric["raw_has_chunk"] = float(raw_has_chunk.float().mean().detach().cpu())
        if curr_phase is not None:
            metric["curr_phase"] = float(curr_phase.float().mean().detach().cpu())

        self._last_custom_metrics = metric

        return (total_loss, outputs) if return_outputs else total_loss

    def __del__(self):
        try:
            self._hook.remove()
        except Exception:
            pass


class AnchorDropoutCurriculumCallback(TrainerCallback):
    def __init__(self, train_dataset):
        self.ds = train_dataset

    def on_step_end(self, args, state, control, **kwargs):
        if hasattr(self.ds, "set_epoch_progress") and state.epoch is not None:
            self.ds.set_epoch_progress(float(state.epoch))
        return control


class JsonlLoggingCallback(TrainerCallback):
    def __init__(self, jsonl_path: str):
        self.jsonl_path = jsonl_path
        self._fp = None

    def on_train_begin(self, args, state, control, **kwargs):
        if self.jsonl_path:
            os.makedirs(os.path.dirname(self.jsonl_path), exist_ok=True)
            self._fp = open(self.jsonl_path, "a", encoding="utf-8")
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not self._fp or not logs:
            return control
        payload = dict(logs)
        payload["step"] = int(state.global_step)
        payload["epoch"] = float(state.epoch) if state.epoch is not None else None
        self._fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._fp.flush()
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if self._fp:
            self._fp.close()
            self._fp = None
        return control


class FiniteParameterCallback(TrainerCallback):
    def __init__(self, check_steps: int = 25):
        self.check_steps = max(1, int(check_steps))

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return control
        if int(state.global_step) > 0 and int(state.global_step) % self.check_steps == 0:
            assert_trainable_parameters_finite(model, where=f"step={int(state.global_step)}")
        return control

    def on_save(self, args, state, control, model=None, **kwargs):
        if model is not None:
            assert_trainable_parameters_finite(model, where=f"pre_save_step={int(state.global_step)}")
        return control


class BucketEvalCallback(TrainerCallback):
    def __init__(
        self,
        eval_dataset: Dataset,
        collator,
        eval_steps: int,
        log_jsonl_path: str,
        batch_size: int = 1,
        max_batches: int = 0,
    ):
        self.eval_dataset = eval_dataset
        self.collator = collator
        self.eval_steps = int(eval_steps)
        self.log_jsonl_path = log_jsonl_path
        self.batch_size = int(batch_size)
        self.max_batches = int(max_batches)

    @torch.no_grad()
    def _run_bucket_eval(self, model) -> Dict[str, float]:
        model_was_training = model.training
        model.eval()
        device = next(model.parameters()).device
        PooledLayerHook.SKIP_DURING_EXTERNAL_FORWARD = True

        try:
            dl = DataLoader(
                self.eval_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=self.collator,
            )

            loss_sums = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
            count_sums = {0: 0, 1: 0, 2: 0, 3: 0}

            for bi, batch in enumerate(dl):
                if self.max_batches > 0 and bi >= self.max_batches:
                    break

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                raw_bucket = batch["raw_bucket"].to(device)

                out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                logits = out.logits

                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()

                vocab_size = shift_logits.size(-1)
                token_loss = F.cross_entropy(
                    shift_logits.view(-1, vocab_size),
                    shift_labels.view(-1),
                    reduction="none",
                    ignore_index=-100,
                ).view(shift_labels.size(0), shift_labels.size(1))

                token_mask = (shift_labels != -100).float()
                per_sample_den = token_mask.sum(dim=1).clamp_min(1.0)
                per_sample_loss = (token_loss * token_mask).sum(dim=1) / per_sample_den

                for b in range(per_sample_loss.size(0)):
                    k = int(raw_bucket[b].item())
                    if k not in loss_sums:
                        continue
                    loss_sums[k] += float(per_sample_loss[b].item())
                    count_sums[k] += 1

            metrics: Dict[str, float] = {}
            total_n = 0
            total_loss = 0.0
            for k in [0, 1, 2, 3]:
                n = count_sums[k]
                mean_lm = loss_sums[k] / n if n > 0 else float("nan")
                metrics[f"eval_bucket{k}_lm_loss"] = mean_lm
                metrics[f"eval_bucket{k}_n"] = float(n)
                if n > 0:
                    total_n += n
                    total_loss += loss_sums[k]

            metrics["eval_bucket_all_lm_loss"] = (total_loss / total_n) if total_n > 0 else float("nan")
            metrics["eval_bucket3_no_anchor_lm_loss"] = metrics["eval_bucket3_lm_loss"]
            return metrics
        finally:
            PooledLayerHook.SKIP_DURING_EXTERNAL_FORWARD = False
            if model_was_training:
                model.train()

    def on_step_end(self, args, state, control, **kwargs):
        if self.eval_steps <= 0:
            return control
        if state.global_step <= 0 or (state.global_step % self.eval_steps != 0):
            return control

        model = kwargs.get("model", None)
        if model is None:
            return control

        metrics = self._run_bucket_eval(model)
        payload = dict(metrics)
        payload["bucket_eval"] = 1
        payload["step"] = int(state.global_step)
        payload["epoch"] = float(state.epoch) if state.epoch is not None else None

        if self.log_jsonl_path:
            os.makedirs(os.path.dirname(self.log_jsonl_path), exist_ok=True)
            with open(self.log_jsonl_path, "a", encoding="utf-8") as fp:
                fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return control


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()

    # IO
    ap.add_argument("--model_name_or_path", type=str, required=True, help="base model path or model id")
    ap.add_argument("--train_jsonl", type=str, required=True, help="*.with_anchor_index.jsonl")
    ap.add_argument("--doc_anchor_pt", type=str, required=True, help="doc_anchor_emb.pt")
    ap.add_argument("--chunk_anchor_pt", type=str, required=True, help="chunk_anchor_emb.pt")
    ap.add_argument("--view_anchor_pt", type=str, default="", help="optional: view_anchor_emb.pt")
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--eval_jsonl", type=str, default="", help="optional eval jsonl for bucketed lm-loss evaluation")

    # training
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=16)
    ap.add_argument("--num_train_epochs", type=int, default=2)
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--eval_steps", type=int, default=0, help="run bucket eval every N optimizer steps; 0 disables")
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--finite_check_steps", type=int, default=25, help="check trainable parameters for NaN/Inf every N optimizer steps")
    ap.add_argument("--max_steps", type=int, default=-1, help="override num_train_epochs if > 0")
    ap.add_argument(
        "--log_jsonl",
        type=str,
        default="auto",
        help="write training logs to this jsonl file; use 'auto' to place under output_dir",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"], help="model/training dtype")
    ap.add_argument(
        "--tokenizer_use_fast",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="load tokenizer with use_fast=True; pass --no-tokenizer_use_fast to force slow/backend fallback",
    )

    # lora - Phi-4 uses combined projections
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument(
        "--lora_target_modules",
        type=str,
        default="auto",
        help="auto selects Phi qkv_proj/o_proj/gate_up_proj/down_proj; otherwise pass comma-separated module suffixes",
    )

    # dual-anchor losses
    ap.add_argument("--alpha", type=float, default=0.10, help="weight for anchor loss")
    ap.add_argument("--beta", type=float, default=0.01, help="weight for smooth loss")
    ap.add_argument("--view_lambda_scale", type=float, default=0.5, help="per-layer view weight scale relative to chunk weights")
    ap.add_argument("--rho_multi_pos", type=float, default=0.05, help="weight for multi-positive routing loss")
    ap.add_argument("--gamma_align", type=float, default=0.05, help="weight for route-energy alignment loss")
    ap.add_argument("--route_tau", type=float, default=1.0, help="temperature for router logits")
    ap.add_argument("--energy_tau", type=float, default=1.0, help="temperature for view-energy target distribution")
    ap.add_argument("--gamma", type=float, default=0.0, help="weight for optional perturbed loss (default off)")
    ap.add_argument("--perturb_eps", type=float, default=0.0, help="std of gaussian noise on embeddings for perturbed branch")

    # pooling alignment (Scheme B)
    ap.add_argument(
        "--pooling_mode",
        type=str,
        default="mid",
        choices=["prompt_end", "mid"],
        help="Where to pool hidden states for anchor guidance. 'mid' pools at a dedicated MID token inserted before ### Response.",
    )
    ap.add_argument("--mid_token", type=str, default="<|MID|>", help="Dedicated MID token used when pooling_mode=mid")
    ap.add_argument(
        "--ensure_mid_token",
        action="store_true",
        default=True,
        help="Auto-add MID token to tokenizer and resize embeddings if missing.",
    )

    # layers
    ap.add_argument("--layer_pick_mode", type=str, default="quartile", choices=["quartile", "middle_last", "mid_last2"])
    ap.add_argument("--drop_doc_anchor_prob", type=float, default=0.0)
    ap.add_argument("--drop_chunk_anchor_prob", type=float, default=0.0)
    ap.add_argument("--anchor_trunc_words", type=int, default=0)
    ap.add_argument("--anchor_dropout_seed", type=int, default=1234)
    ap.add_argument("--drop_warmup_epochs", type=float, default=0.5)
    ap.add_argument("--drop_ramp_epochs", type=float, default=0.5)
    ap.add_argument("--allow_both_drop", action="store_true", help="allow dropping both doc and chunk anchors in one sample")
    ap.add_argument("--eval_max_batches", type=int, default=0, help="cap eval dataloader batches per eval run; 0 means full eval set")

    args = ap.parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    ensure_phi_cache_compat()

    model_cfg = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=False)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=bool(args.tokenizer_use_fast),
        trust_remote_code=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]

    # model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=False,
        device_map=None,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    mid_token_id = None
    if args.pooling_mode == "mid":
        if args.ensure_mid_token:
            mid_token_id = ensure_mid_token_and_resize(tokenizer, model, args.mid_token)
        else:
            mid_token_id = tokenizer.convert_tokens_to_ids(args.mid_token)
            if mid_token_id is None or mid_token_id == tokenizer.unk_token_id or not _token_is_single_id(tokenizer, args.mid_token):
                raise ValueError("MID token missing or not a single token id. Enable --ensure_mid_token.")

    # load anchor embeddings (CPU)
    doc_anchor = torch.load(args.doc_anchor_pt, map_location="cpu")
    chunk_anchor = torch.load(args.chunk_anchor_pt, map_location="cpu")
    view_anchor = torch.load(args.view_anchor_pt, map_location="cpu") if args.view_anchor_pt else None
    assert doc_anchor.shape == chunk_anchor.shape, "doc/chunk anchor embeddings must be line-aligned with same shape"
    if view_anchor is not None:
        assert doc_anchor.shape == view_anchor.shape, "view anchor embeddings must be line-aligned with doc/chunk shape"
    anchor_dim = doc_anchor.shape[1]

    # add projection heads: z(hidden_size) -> anchor_dim
    hidden_size = model.config.hidden_size
    model.anchor_proj_doc = nn.Linear(hidden_size, anchor_dim, bias=False)
    model.anchor_proj_chunk = nn.Linear(hidden_size, anchor_dim, bias=False)
    if view_anchor is not None:
        model.anchor_proj_view = nn.Linear(hidden_size, anchor_dim, bias=False)
    model.anchor_proj_route = nn.Linear(hidden_size, len(VIEW_NAMES), bias=False)

    # LoRA target modules. Phi-4-mini uses the Phi3 implementation with
    # combined qkv/gate_up projections; Qwen/LLaMA-style models use split
    # projections. Keep explicit override support for experiments.
    if args.lora_target_modules.strip().lower() == "auto":
        mt = str(getattr(model_cfg, "model_type", "")).lower()
        if mt in {"phi3", "phi"}:
            target_modules = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]
        else:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    else:
        target_modules = [s.strip() for s in args.lora_target_modules.split(",") if s.strip()]
    print(f"[INFO] model_type={getattr(model_cfg, 'model_type', '')} dtype={args.dtype}")
    print(f"[INFO] LoRA target_modules={target_modules}")
    modules_to_save = ["anchor_proj_doc", "anchor_proj_chunk", "anchor_proj_route"]
    if view_anchor is not None:
        modules_to_save.append("anchor_proj_view")
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
        modules_to_save=modules_to_save,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # layers & weights
    layers = find_transformer_layers(model)
    num_layers = len(layers)
    layer_ids = pick_layers(num_layers, mode=args.layer_pick_mode)
    lambda_doc, lambda_chunk = make_layer_weights(len(layer_ids))
    lambda_view = make_view_layer_weights(lambda_chunk, args.view_lambda_scale) if view_anchor is not None else None

    print(f"[INFO] num_layers={num_layers}, selected_layers={layer_ids}")
    print(f"[INFO] lambda_doc={lambda_doc}, lambda_chunk={lambda_chunk}")
    if lambda_view is not None:
        print(f"[INFO] lambda_view={lambda_view}")
    print(f"[INFO] anchors: N={doc_anchor.shape[0]}, D={anchor_dim}")

    # Build fixed view prototypes for energy-alignment target.
    view_prototypes = build_view_prototypes_from_jsonl(
        jsonl_paths=[args.train_jsonl],
        view_anchor=view_anchor,
    )
    if view_prototypes is not None:
        vp_path = os.path.join(args.output_dir, "view_prototypes.pt")
        torch.save(
            {
                "view_names": VIEW_NAMES,
                "prototypes": view_prototypes.cpu(),
            },
            vp_path,
        )
        print(f"[INFO] Saved view prototypes to: {vp_path}")

    train_ds = JSONLSFTDataset(
        jsonl_path=args.train_jsonl,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        add_task_prefix=True,
        pooling_mode=args.pooling_mode,
        mid_token=args.mid_token,
        drop_doc_anchor_prob=args.drop_doc_anchor_prob,
        drop_chunk_anchor_prob=args.drop_chunk_anchor_prob,
        anchor_trunc_words=args.anchor_trunc_words,
        anchor_dropout_seed=args.anchor_dropout_seed,
        allow_both_drop=args.allow_both_drop,
        drop_warmup_epochs=args.drop_warmup_epochs,
        drop_ramp_epochs=args.drop_ramp_epochs,
    )
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # training args
    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        bf16=(args.dtype == "bf16"),
        fp16=(args.dtype == "fp16"),
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        max_grad_norm=args.max_grad_norm,
        max_steps=args.max_steps,
        save_total_limit=2,
        report_to=["tensorboard"],
        logging_dir=os.path.join(args.output_dir, "runs"),
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
    )

    trainer = DualAnchorTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        data_collator=collator,
        doc_anchor_emb=doc_anchor,
        chunk_anchor_emb=chunk_anchor,
        view_anchor_emb=view_anchor,
        view_prototypes=view_prototypes,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        rho_multi_pos=args.rho_multi_pos,
        gamma_align=args.gamma_align,
        route_tau=args.route_tau,
        energy_tau=args.energy_tau,
        layer_ids=layer_ids,
        lambda_doc=lambda_doc,
        lambda_chunk=lambda_chunk,
        lambda_view=lambda_view,
        perturb_eps=args.perturb_eps,
        pooling_mode=args.pooling_mode,
        mid_token_id=mid_token_id,
        tokenizer=tokenizer,
    )

    log_path = ""
    if args.log_jsonl:
        if args.log_jsonl == "auto":
            log_path = os.path.join(args.output_dir, "train_log.jsonl")
        else:
            log_path = args.log_jsonl

    trainer.add_callback(AnchorDropoutCurriculumCallback(train_ds))
    trainer.add_callback(FiniteParameterCallback(check_steps=args.finite_check_steps))
    if args.eval_jsonl and args.eval_steps > 0 and log_path:
        eval_ds = JSONLSFTDataset(
            jsonl_path=args.eval_jsonl,
            tokenizer=tokenizer,
            max_seq_len=args.max_seq_len,
            add_task_prefix=True,
            pooling_mode=args.pooling_mode,
            mid_token=args.mid_token,
            drop_doc_anchor_prob=0.0,
            drop_chunk_anchor_prob=0.0,
            anchor_trunc_words=0,
            anchor_dropout_seed=args.anchor_dropout_seed,
            allow_both_drop=args.allow_both_drop,
            drop_warmup_epochs=0.0,
            drop_ramp_epochs=0.0,
        )
        trainer.add_callback(
            BucketEvalCallback(
                eval_dataset=eval_ds,
                collator=collator,
                eval_steps=args.eval_steps,
                log_jsonl_path=log_path,
                batch_size=args.per_device_train_batch_size,
                max_batches=args.eval_max_batches,
            )
        )
    if log_path:
        trainer.add_callback(JsonlLoggingCallback(log_path))


    assert_trainable_parameters_finite(model, where="before_train")
    trainer.train()
    assert_trainable_parameters_finite(model, where="after_train_before_save")
    trainer.save_model(args.output_dir)
    assert_trainable_parameters_finite(model, where="after_save")
    tokenizer.save_pretrained(args.output_dir)

    print("[DONE] Saved to:", args.output_dir)
    print("TensorBoard logs:", os.path.join(args.output_dir, "runs"))


if __name__ == "__main__":
    main()
