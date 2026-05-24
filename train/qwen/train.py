#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import random
from transformers import PreTrainedTokenizerBase
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers import (
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
    # PEFT >= 0.10 usually has get_base_model()
    if hasattr(model, "get_base_model"):
        try:
            return model.get_base_model()
        except Exception:
            pass
    # common PEFT wrapper fields
    if hasattr(model, "base_model"):
        bm = model.base_model
        # sometimes bm has .model
        if hasattr(bm, "model"):
            return bm.model
        return bm
    return model


def _token_is_single_id(tokenizer: PreTrainedTokenizerBase, s: str) -> bool:
    ids = tokenizer(s, add_special_tokens=False).get("input_ids", [])
    return isinstance(ids, list) and len(ids) == 1


def ensure_mid_token_and_resize(
    tokenizer: PreTrainedTokenizerBase,
    model: nn.Module,
    mid_token: str,
) -> int:
    """
    Universal B-scheme for training:
      - Ensure `mid_token` exists AND is a single token id
      - If not, add as additional_special_tokens, resize model embeddings
    Returns mid_token_id
    """
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    tid = tokenizer.convert_tokens_to_ids(mid_token)
    ok_single = _token_is_single_id(tokenizer, mid_token)
    if tid is None or tid == tokenizer.unk_token_id or not ok_single:
        tokenizer.add_special_tokens({"additional_special_tokens": [mid_token]})
        base = _unwrap_base_model(model)
        target = getattr(base, "model", base)
        if hasattr(target, "resize_token_embeddings"):
            target.resize_token_embeddings(len(tokenizer))
        else:
            model.resize_token_embeddings(len(tokenizer))
        tid = tokenizer.convert_tokens_to_ids(mid_token)
        if tid is None or tid == tokenizer.unk_token_id or not _token_is_single_id(tokenizer, mid_token):
            raise ValueError(f"Failed to install MID token '{mid_token}' as a single token id.")

    if tokenizer.eos_token_id is not None and tid == tokenizer.eos_token_id:
        raise ValueError(f"MID token '{mid_token}' resolves to eos_token_id={tid}.")
    if tokenizer.pad_token_id is not None and tid == tokenizer.pad_token_id:
        raise ValueError(f"MID token '{mid_token}' resolves to pad_token_id={tid}.")
    return int(tid)


def find_transformer_layers(model: nn.Module):
    """
    Robustly locate decoder block list (ModuleList) by scanning the model.
    Works for Qwen/LLaMA/GPT-style and PEFT-wrapped models.
    """
    base = _unwrap_base_model(model)
    cfg = getattr(base, "config", None)

    # number of layers in config (different naming across models)
    num_layers = None
    for k in ["num_hidden_layers", "n_layer", "num_layers", "n_layers"]:
        if cfg is not None and hasattr(cfg, k):
            num_layers = int(getattr(cfg, k))
            break
    if num_layers is None:
        raise ValueError("Cannot infer num_layers from config. Please print base.config and adapt.")

    candidates = []
    # scan all submodules, pick ModuleList with matching length
    for name, mod in base.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) == num_layers:
            candidates.append((name, mod))

    if not candidates:
        raise ValueError(
            f"Cannot locate transformer layers: no nn.ModuleList with length={num_layers} found."
        )

    # choose the most likely one: whose elements look like decoder layers
    def score(modlist: nn.ModuleList) -> int:
        m0 = modlist[0]
        s = 0
        for attr in ["self_attn", "attn", "mlp", "feed_forward", "input_layernorm", "post_attention_layernorm"]:
            if hasattr(m0, attr):
                s += 1
        # also favor typical class name hints
        cn = m0.__class__.__name__.lower()
        if "decoder" in cn or "layer" in cn or "block" in cn:
            s += 2
        return s

    candidates.sort(key=lambda x: score(x[1]), reverse=True)
    best_name, best_list = candidates[0]

    print(f"[INFO] Found transformer layers at: {best_name} (len={len(best_list)}) "
          f"first_layer={best_list[0].__class__.__name__}")

    return list(best_list)



def pick_layers(num_layers: int) -> List[int]:
    if num_layers <= 0:
        raise ValueError("num_layers must be > 0")
    return sorted(set([num_layers // 4, num_layers // 2, (3 * num_layers) // 4, num_layers - 1]))


def make_layer_weights(k: int) -> Tuple[List[float], List[float]]:
    """
    Make doc/chunk weights for k selected layers.
    Doc decreases, chunk increases.
    For k=4 -> doc: [1.0,0.7,0.4,0.2], chunk: [0.2,0.4,0.7,1.0]
    Otherwise do linear interpolation.
    """
    if k == 4:
        return [1.0, 0.7, 0.4, 0.2], [0.2, 0.4, 0.7, 1.0]
    # linear schedule
    doc = []
    chunk = []
    for i in range(k):
        t = i / max(k - 1, 1)
        # doc: 1 -> 0.2, chunk: 0.2 -> 1
        doc.append(1.0 * (1 - t) + 0.2 * t)
        chunk.append(0.2 * (1 - t) + 1.0 * t)
    return doc, chunk


def make_view_layer_weights(lambda_chunk: List[float], view_lambda_scale: float) -> List[float]:
    """
    Make view weights from chunk schedule.
    By default, view follows deeper-layer preference similar to chunk.
    """
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
    """
    Build one prototype per view by averaging row-level view anchors grouped by
    the row's `view` label. This keeps router-energy alignment in the same
    anchor space without requiring QA scores.
    """
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
    """
    For each sample, return the last index where labels != -100;
    fallback to last non-pad token if labels are all -100.
    labels: [B,T], attention_mask: [B,T]
    """
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
    """
    For each sample, return the last index of the prompt (i.e., the token
    immediately before the first label != -100). Fallback to last non-pad token.
    labels: [B,T], attention_mask: [B,T]
    """
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
    ):
        self.jsonl_path = jsonl_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.add_task_prefix = add_task_prefix
        self.pooling_mode = str(pooling_mode)
        self.mid_token = str(mid_token)

        self.rows = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.rows.append(json.loads(line))

        # Build chunk-level observed-view sets for weak multi-positive routing.
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

        # Tokenizer padding side is relevant for collator; we do truncation in __getitem__.
        self.tokenizer.padding_side = "right"

    def __len__(self):
        return len(self.rows)

    def _format_pair(self, row: Dict[str, Any]) -> Tuple[str, str]:
        task = row.get("task", "")
        prompt = row.get("prompt", row.get("instruction", row.get("input", row.get("text", ""))))
        response = row.get("response", row.get("output", ""))

        task_line = ""
        if self.add_task_prefix and task:
            task_line = f"Task: {task}\n"

        # Pooling position control:
        # - prompt_end: use the token just before the first supervised label (existing behavior)
        # - mid: insert a tokenizer-known special token as MID, and pool at its position
        if self.pooling_mode == "mid":
            prompt_text = f"{task_line}{prompt.strip()}\n{self.mid_token}\n\n### Response:\n"
        else:
            prompt_text = f"{task_line}{prompt.strip()}\n\n### Response:\n"
        resp_text = response.strip()
        return prompt_text, resp_text


    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row_idx = idx
        r = self.rows[row_idx]
        prompt_text, resp_text = self._format_pair(r)
        anchor_row_idx = r.get("anchor_row_idx", r.get("anchor_row", r.get("anchor_idx", None)))
        if anchor_row_idx is None:
            # If you didn't write indexed jsonl, fallback: use dataset line id
            anchor_row_idx = row_idx

        # Tokenize prompt/response separately to keep label masking correct under truncation
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
        }

@dataclass
class DataCollatorWithPadding:
    tokenizer: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        labels = [f["labels"] for f in features]
        model_features = [
            {
                "input_ids": f["input_ids"],
                "attention_mask": f["attention_mask"],
            }
            for f in features
        ]
        batch = self.tokenizer.pad(
            model_features,
            padding=True,
            return_tensors="pt",
        )

        max_len = int(batch["input_ids"].shape[1])
        padded_labels = []
        padding_side = getattr(self.tokenizer, "padding_side", "right")
        for label in labels:
            pad_len = max_len - len(label)
            if pad_len < 0:
                raise ValueError(f"label length {len(label)} exceeds padded input length {max_len}")
            if padding_side == "left":
                padded = ([-100] * pad_len) + list(label)
            else:
                padded = list(label) + ([-100] * pad_len)
            padded_labels.append(padded)
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)

        for k in [
            "anchor_row_idx",
            "view_id",
        ]:
            vals = [f.get(k, 0) for f in features]
            batch[k] = torch.tensor(vals, dtype=torch.long)

        if any("observed_view_mask" in f for f in features):
            batch["observed_view_mask"] = torch.tensor(
                [f.get("observed_view_mask", []) for f in features],
                dtype=torch.float32,
            )
        return batch


# -------------------------
# Hook: store pooled z^l only (B x H)
# -------------------------
class PooledLayerHook:
    SKIP_DURING_EXTERNAL_FORWARD = False

    def __init__(self, layers: List[nn.Module], layer_ids: List[int]):
        self.layers = layers
        self.layer_ids = layer_ids
        self.handles = []
        self.pool_idx: Optional[torch.Tensor] = None  # [B]
        self.cache: Dict[int, torch.Tensor] = {}

    def clear(self):
        self.cache = {}

    def _make_hook(self, layer_id: int):
        def hook_fn(module, inp, out):
            if PooledLayerHook.SKIP_DURING_EXTERNAL_FORWARD:
                return
            hs = out[0] if isinstance(out, (tuple, list)) else out  # [B,T,H]
            assert self.pool_idx is not None, "pool_idx must be set before forward."
            B = hs.size(0)
            idx = self.pool_idx  # [B]
            z = hs[torch.arange(B, device=hs.device), idx]  # [B,H]
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
        beta_route: float,
        delta_vgc: float,
        layer_ids: List[int],
        lambda_doc: List[float],
        lambda_chunk: List[float],
        lambda_view: Optional[List[float]] = None,
        pooling_mode: str = "prompt_end",
        mid_token_id: Optional[int] = None,
        **kwargs,
    ):
        self._tok = kwargs.pop("tokenizer", None) 
        super().__init__(*args, **kwargs)
        self.doc_anchor_emb = doc_anchor_emb  # [N, D_anchor] on CPU
        self.chunk_anchor_emb = chunk_anchor_emb
        self.view_anchor_emb = view_anchor_emb
        self.view_prototypes = view_prototypes
        self.alpha = alpha
        self.beta_route = beta_route
        self.delta_vgc = delta_vgc
        self.layer_ids = layer_ids
        self.lambda_doc = lambda_doc
        self.lambda_chunk = lambda_chunk
        self.lambda_view = lambda_view
        self.use_view_anchor = self.view_anchor_emb is not None and self.lambda_view is not None

        self.pooling_mode = str(pooling_mode)
        self.mid_token_id = int(mid_token_id) if mid_token_id is not None else None

        # cached metrics merged into logs
        self._last_custom_metrics: Dict[str, float] = {}

        # setup hooks
        self._layers = find_transformer_layers(self.model)
        self._hook = PooledLayerHook(self._layers, self.layer_ids)
        self._hook.register()

    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        # merge last custom metrics for tensorboard/console
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
        # anchor_row_idx: [B]
        a_doc = self.doc_anchor_emb.index_select(0, anchor_row_idx.cpu()).to(device=device, dtype=torch.float32)
        a_chunk = self.chunk_anchor_emb.index_select(0, anchor_row_idx.cpu()).to(device=device, dtype=torch.float32)
        a_view = None
        if self.use_view_anchor and self.view_anchor_emb is not None:
            a_view = self.view_anchor_emb.index_select(0, anchor_row_idx.cpu()).to(device=device, dtype=torch.float32)
        return a_doc, a_chunk, a_view

    def _anchor_loss(
        self,
        zs: List[torch.Tensor],   # list of [B,H] in same order as layer_ids
        a_doc: torch.Tensor,      # [B,D_anchor]
        a_chunk: torch.Tensor,    # [B,D_anchor]
        a_view: Optional[torch.Tensor] = None,  # [B,D_anchor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        zs in H-space; projection to anchor space inside.
        """
        device = zs[0].device
        B = zs[0].size(0)

        # project z -> anchor dim
        base = _unwrap_base_model(self.model)
        proj_doc: nn.Module = getattr(base, "anchor_proj_doc")
        proj_chunk: nn.Module = getattr(base, "anchor_proj_chunk")
        proj_view: Optional[nn.Module] = getattr(base, "anchor_proj_view", None) if self.use_view_anchor else None

        loss_anchor = torch.tensor(0.0, device=device)

        metrics: Dict[str, float] = {}

        # anchor in projected space
        for i, z in enumerate(zs):
            zf = z.float()  # [B,H]
            z_doc = proj_doc(zf)      # [B,D_anchor]
            z_chunk = proj_chunk(zf)  # [B,D_anchor]
            z_view = proj_view(zf) if (proj_view is not None and a_view is not None) else None

            z_doc = F.normalize(z_doc, p=2, dim=-1)
            z_chunk = F.normalize(z_chunk, p=2, dim=-1)
            if z_view is not None:
                z_view = F.normalize(z_view, p=2, dim=-1)

            # anchors already normalized offline; still safe to normalize
            a_doc_n = F.normalize(a_doc, p=2, dim=-1)
            a_chunk_n = F.normalize(a_chunk, p=2, dim=-1)
            a_view_n = F.normalize(a_view, p=2, dim=-1) if a_view is not None else None

            cos_doc = (z_doc * a_doc_n).sum(dim=-1)      # [B]
            cos_chunk = (z_chunk * a_chunk_n).sum(dim=-1)
            cos_view = (z_view * a_view_n).sum(dim=-1) if (z_view is not None and a_view_n is not None) else None

            # energy = 1 - cos (more intuitive)
            e_doc = (1.0 - cos_doc).mean()
            e_chunk = (1.0 - cos_chunk).mean()
            e_view = (1.0 - cos_view).mean() if cos_view is not None else None

            term = self.lambda_doc[i] * e_doc + self.lambda_chunk[i] * e_chunk
            if e_view is not None and self.lambda_view is not None:
                term = term + self.lambda_view[i] * e_view
            loss_anchor = loss_anchor + term

            # log per-layer cos (higher is better)
            lid = self.layer_ids[i]
            metrics[f"cos_doc_L{lid}"] = float(cos_doc.mean().detach().cpu())
            metrics[f"cos_chunk_L{lid}"] = float(cos_chunk.mean().detach().cpu())
            if cos_view is not None:
                metrics[f"cos_view_L{lid}"] = float(cos_view.mean().detach().cpu())

        metrics["loss_anchor"] = float(loss_anchor.detach().cpu())
        return loss_anchor, metrics

    def _compute_router_losses(
        self,
        zs: List[torch.Tensor],                # list of [B,H]
        observed_view_mask: Optional[torch.Tensor],  # [B,V]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float], Optional[torch.Tensor]]:
        """
        Router losses without QA scores:
          1) multi-label BCE over the observed view set per chunk
          2) alignment between router distribution and view-energy-induced distribution
        """
        device = zs[-1].device
        z_route = zs[-1].float()  # use deepest pooled state as routing state
        base = _unwrap_base_model(self.model)
        proj_route: Optional[nn.Module] = getattr(base, "anchor_proj_route", None)
        proj_view: Optional[nn.Module] = getattr(base, "anchor_proj_view", None)

        zero = torch.tensor(0.0, device=device)
        metrics: Dict[str, float] = {}
        if proj_route is None:
            metrics["loss_route"] = 0.0
            metrics["loss_vgc"] = 0.0
            return zero, zero, metrics, None

        route_logits = proj_route(z_route)  # [B,V]
        vdim = int(route_logits.size(-1))

        # Multi-label routing supervision from observed view set S(c).
        loss_route = zero
        if observed_view_mask is not None and observed_view_mask.numel() > 0:
            mask = observed_view_mask.to(device=device, dtype=torch.float32)
            if mask.size(-1) != vdim:
                raise ValueError(f"observed_view_mask dim {mask.size(-1)} != route dim {vdim}")
            pos_cnt = mask.sum(dim=-1, keepdim=True)
            valid = (pos_cnt.squeeze(-1) > 0)
            if valid.any():
                loss_route = F.binary_cross_entropy_with_logits(route_logits[valid], mask[valid], reduction="mean")

        # Align router distribution to geometry-induced view energy distribution.
        loss_vgc = zero
        if self.delta_vgc > 0 and proj_view is not None and self.view_prototypes is not None:
            proto = self.view_prototypes.to(device=device, dtype=torch.float32)  # [V,D]
            if proto.size(0) != vdim:
                raise ValueError(f"view_prototypes V={proto.size(0)} != route dim {vdim}")
            z_view = F.normalize(proj_view(z_route), p=2, dim=-1)               # [B,D]
            proto_n = F.normalize(proto, p=2, dim=-1)                           # [V,D]
            cos = torch.matmul(z_view, proto_n.t())                             # [B,V]
            tgt = F.softmax(cos, dim=-1).detach()
            log_p = F.log_softmax(route_logits, dim=-1)
            loss_vgc = F.kl_div(log_p, tgt, reduction="batchmean")
            metrics["view_cos_mean"] = float(cos.mean().detach().cpu())

        metrics["loss_route"] = float(loss_route.detach().cpu())
        metrics["loss_vgc"] = float(loss_vgc.detach().cpu())
        metrics["route_logit_mean"] = float(route_logits.mean().detach().cpu())
        metrics["route_logit_std"] = float(route_logits.std().detach().cpu())
        return loss_route, loss_vgc, metrics, route_logits

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Total loss:
          L = L_lm
            + alpha * L_anchor
            + beta_route * L_route
            + delta_vgc * L_vgc
        """
        anchor_row_idx = inputs.pop("anchor_row_idx")
        view_id = inputs.pop("view_id", None)
        observed_view_mask = inputs.pop("observed_view_mask", None)
        labels = inputs.get("labels")
        attention_mask = inputs.get("attention_mask")

        device = labels.device
        a_doc, a_chunk, a_view = self._get_anchors(anchor_row_idx, device=device)

        # pool index:
        # - prompt_end: token immediately before first supervised label
        # - mid: last occurrence of MID token in input_ids (should be in the prompt)
        if self.pooling_mode == "mid":
            if self.mid_token_id is None:
                raise ValueError("pooling_mode=mid requires mid_token_id to be set in DualAnchorTrainer.")
            input_ids = inputs["input_ids"]  # [B,T]
            B, _T = input_ids.shape
            pool_idx = torch.empty((B,), device=input_ids.device, dtype=torch.long)
            for b in range(B):
                row = input_ids[b].tolist()
                try:
                    # last occurrence
                    pool_idx[b] = len(row) - 1 - row[::-1].index(self.mid_token_id)
                except ValueError:
                    # If MID not found (bad data), fallback to prompt_end to avoid crashing
                    pool_idx[b] = prompt_end_index(labels, attention_mask)[b]
        else:
            pool_idx = prompt_end_index(labels, attention_mask)
        self._hook.pool_idx = pool_idx
        self._hook.clear()

        # clean forward
        outputs = model(**inputs, use_cache=False)
        lm_loss = outputs.loss

        # gather pooled z for selected layers in correct order
        zs = []
        for lid in self.layer_ids:
            if lid not in self._hook.cache:
                raise RuntimeError(f"Missing hook cache for layer {lid}.")
            zs.append(self._hook.cache[lid])  # [B,H]

        loss_anchor, metric = self._anchor_loss(zs, a_doc, a_chunk, a_view)
        loss_route, loss_vgc, route_metric, route_logits = self._compute_router_losses(
            zs=zs,
            observed_view_mask=observed_view_mask,
        )
        metric.update(route_metric)

        total_loss = (
            lm_loss
            + self.alpha * loss_anchor
            + self.beta_route * loss_route
            + self.delta_vgc * loss_vgc
        )

        # custom metrics to merge into Trainer logs
        metric.update(
            {
                "loss_lm": float(lm_loss.detach().cpu()),
                "alpha": float(self.alpha),
                "beta_route": float(self.beta_route),
                "delta_vgc": float(self.delta_vgc),
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

        self._last_custom_metrics = metric

        return (total_loss, outputs) if return_outputs else total_loss

    def __del__(self):
        # best-effort cleanup
        try:
            self._hook.remove()
        except Exception:
            pass


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

    # training
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=16)
    ap.add_argument("--num_train_epochs", type=int, default=2)
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--max_steps", type=int, default=-1, help="override num_train_epochs if > 0")
    ap.add_argument(
        "--log_jsonl",
        type=str,
        default="auto",
        help="write training logs to this jsonl file; use 'auto' to place under output_dir",
    )
    ap.add_argument("--seed", type=int, default=42)

    # lora
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )

    # formula losses
    ap.add_argument("--alpha", type=float, default=0.10, help="weight for anchor loss")
    ap.add_argument("--view_lambda_scale", type=float, default=0.5, help="per-layer view weight scale relative to chunk weights")
    ap.add_argument("--beta_route", type=float, default=0.05, help="weight for routing BCE loss")
    ap.add_argument("--delta_vgc", type=float, default=0.05, help="weight for view-geometry consistency loss")

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

    args = ap.parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
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
    doc_anchor = torch.load(args.doc_anchor_pt, map_location="cpu")   # [N, D_anchor]
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
    # Router head on MID hidden for the 8-view soft preference distribution.
    model.anchor_proj_route = nn.Linear(hidden_size, len(VIEW_NAMES), bias=False)

    # LoRA
    target_modules = [s.strip() for s in args.lora_target_modules.split(",") if s.strip()]
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
        # ensure projection heads are saved with adapters
        modules_to_save=modules_to_save,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # layers & weights
    layers = find_transformer_layers(model)
    num_layers = len(layers)
    layer_ids = pick_layers(num_layers)
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
        add_task_prefix=False,
        pooling_mode=args.pooling_mode,
        mid_token=args.mid_token,
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
        bf16=True,
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
        beta_route=args.beta_route,
        delta_vgc=args.delta_vgc,
        layer_ids=layer_ids,
        lambda_doc=lambda_doc,
        lambda_chunk=lambda_chunk,
        lambda_view=lambda_view,
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

    if log_path:
        trainer.add_callback(JsonlLoggingCallback(log_path))


    trainer.train()
    trainer.save_model(args.output_dir)  # saves LoRA adapters + modules_to_save

    print("[DONE] Saved to:", args.output_dir)
    print("TensorBoard logs:", os.path.join(args.output_dir, "runs"))


if __name__ == "__main__":
    main()
