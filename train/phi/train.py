#!/usr/bin/env python3

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
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from peft import LoraConfig, get_peft_model


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


import torch.nn as nn

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


def _token_is_single_id(tokenizer: PreTrainedTokenizerBase, s: str) -> bool:
    ids = tokenizer(s, add_special_tokens=False).get("input_ids", [])
    return isinstance(ids, list) and len(ids) == 1


def ensure_phi_cache_compat() -> None:

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

    repaired = 0
    for module in model.modules():
        if not (hasattr(module, "inv_freq") and hasattr(module, "original_inv_freq")):
            continue
        if not (hasattr(module, "rope_init_fn") and hasattr(module, "config")):
            continue


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


def ensure_apt_token_and_resize(
    tokenizer: PreTrainedTokenizerBase,
    model: nn.Module,
    apt_token: str,
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

    tid = tokenizer.convert_tokens_to_ids(apt_token)
    ok_single = _token_is_single_id(tokenizer, apt_token)
    if tid is None or tid == tokenizer.unk_token_id or not ok_single:
        tokenizer.add_special_tokens({"additional_special_tokens": [apt_token]})
        _resize_only_if_needed()
        tid = tokenizer.convert_tokens_to_ids(apt_token)
        if tid is None or tid == tokenizer.unk_token_id or not _token_is_single_id(tokenizer, apt_token):
            raise ValueError(f"Failed to install APT token '{apt_token}' as a single token id.")
    else:
        _resize_only_if_needed()

    if tokenizer.eos_token_id is not None and tid == tokenizer.eos_token_id:
        raise ValueError(f"APT token '{apt_token}' resolves to eos_token_id={tid}.")
    if tokenizer.pad_token_id is not None and tid == tokenizer.pad_token_id:
        raise ValueError(f"APT token '{apt_token}' resolves to pad_token_id={tid}.")
    return int(tid)


def find_transformer_layers(model: nn.Module):

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


def pick_layers(num_layers: int) -> List[int]:
    if num_layers <= 0:
        raise ValueError("num_layers must be > 0")
    return sorted(set([num_layers // 4, num_layers // 2, (3 * num_layers) // 4, num_layers - 1]))


def make_layer_weights(k: int) -> Tuple[List[float], List[float]]:
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


class JSONLSFTDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        tokenizer,
        max_seq_len: int,
        add_task_prefix: bool = True,
        apt_token: str = "<|APT|>",
    ):
        self.jsonl_path = jsonl_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.add_task_prefix = add_task_prefix
        self.apt_token = str(apt_token)

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

        prompt_text = f"{task_line}{prompt.strip()}\n{self.apt_token}\n\n### Response:\n"
        resp_text = response.strip()
        return prompt_text, resp_text


    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row_idx = idx
        r = self.rows[row_idx]
        prompt_text, resp_text = self._format_pair(r)
        anchor_row_idx = r.get("anchor_row_idx", r.get("anchor_row", r.get("anchor_idx", None)))
        if anchor_row_idx is None:
            anchor_row_idx = row_idx

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
        ]:
            if k in batch and not isinstance(batch[k], torch.Tensor):
                batch[k] = torch.tensor(batch[k], dtype=torch.long)
        if "observed_view_mask" in batch and not isinstance(batch["observed_view_mask"], torch.Tensor):
            batch["observed_view_mask"] = torch.tensor(batch["observed_view_mask"], dtype=torch.float32)
        return batch


class PooledLayerHook:
    SKIP_DURING_EXTERNAL_FORWARD = False

    def __init__(self, layers: List[nn.Module], layer_ids: List[int]):
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
        apt_token_id: int,
        **kwargs,
    ):
        self._tok = kwargs.pop("tokenizer", None)
        super().__init__(*args, **kwargs)
        self.doc_anchor_emb = doc_anchor_emb
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

        self.apt_token_id = int(apt_token_id)

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

    def _anchor_loss(
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

        loss_anchor = torch.tensor(0.0, device=device)

        metrics: Dict[str, float] = {}

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
        return loss_anchor, metrics

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
            metrics["loss_route"] = 0.0
            metrics["loss_vgc"] = 0.0
            return zero, zero, metrics, None

        route_logits = proj_route(z_route)
        vdim = int(route_logits.size(-1))

        loss_route = zero
        if observed_view_mask is not None and observed_view_mask.numel() > 0:
            mask = observed_view_mask.to(device=device, dtype=torch.float32)
            if mask.size(-1) != vdim:
                raise ValueError(f"observed_view_mask dim {mask.size(-1)} != route dim {vdim}")
            pos_cnt = mask.sum(dim=-1, keepdim=True)
            valid = (pos_cnt.squeeze(-1) > 0)
            if valid.any():
                loss_route = F.binary_cross_entropy_with_logits(route_logits[valid], mask[valid], reduction="mean")

        loss_vgc = zero
        if self.delta_vgc > 0 and proj_view is not None and self.view_prototypes is not None:
            proto = torch.nan_to_num(self.view_prototypes.to(device=device, dtype=torch.float32), nan=0.0, posinf=1e4, neginf=-1e4)
            if proto.size(0) != vdim:
                raise ValueError(f"view_prototypes V={proto.size(0)} != route dim {vdim}")
            z_view = F.normalize(torch.nan_to_num(proj_view(z_route).float(), nan=0.0, posinf=1e4, neginf=-1e4), p=2, dim=-1, eps=1e-6)
            proto_n = F.normalize(proto, p=2, dim=-1, eps=1e-6)
            cos = torch.matmul(z_view, proto_n.t())
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
        anchor_row_idx = inputs.pop("anchor_row_idx")
        view_id = inputs.pop("view_id", None)
        observed_view_mask = inputs.pop("observed_view_mask", None)
        labels = inputs.get("labels")

        device = labels.device
        a_doc, a_chunk, a_view = self._get_anchors(anchor_row_idx, device=device)

        input_ids = inputs["input_ids"]
        B, _T = input_ids.shape
        pool_idx = torch.empty((B,), device=input_ids.device, dtype=torch.long)
        for b in range(B):
            row = input_ids[b].tolist()
            try:
                pool_idx[b] = len(row) - 1 - row[::-1].index(self.apt_token_id)
            except ValueError as exc:
                raise ValueError("APT token id not found in training input.") from exc
        self._hook.pool_idx = pool_idx
        self._hook.clear()

        outputs = model(**inputs, use_cache=False)
        lm_loss = outputs.loss

        zs = []
        for lid in self.layer_ids:
            if lid not in self._hook.cache:
                raise RuntimeError(f"Missing hook cache for layer {lid}.")
            zs.append(self._hook.cache[lid])

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
        for name, value in {
            "loss_lm": lm_loss,
            "loss_anchor": loss_anchor,
            "loss_route": loss_route,
            "loss_vgc": loss_vgc,
            "total_loss": total_loss,
        }.items():
            _assert_finite_tensor(name, value)

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


def main():
    ap = argparse.ArgumentParser()


    ap.add_argument("--model_name_or_path", type=str, required=True, help="base model path or model id")
    ap.add_argument("--train_jsonl", type=str, required=True, help="*.with_anchor_index.jsonl")
    ap.add_argument("--doc_anchor_pt", type=str, required=True, help="doc_anchor_emb.pt")
    ap.add_argument("--chunk_anchor_pt", type=str, required=True, help="chunk_anchor_emb.pt")
    ap.add_argument("--view_anchor_pt", type=str, default="", help="optional: view_anchor_emb.pt")
    ap.add_argument("--output_dir", type=str, required=True)


    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--num_train_epochs", type=int, default=3)
    ap.add_argument("--learning_rate", type=float, default=5e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--logging_steps", type=int, default=10)
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


    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument(
        "--lora_target_modules",
        type=str,
        default="auto",
        help="auto selects Phi qkv_proj/o_proj/gate_up_proj/down_proj; otherwise pass comma-separated module suffixes",
    )


    ap.add_argument("--alpha", type=float, default=0.10, help="weight for anchor loss")
    ap.add_argument("--view_lambda_scale", type=float, default=0.5, help="per-layer view weight scale relative to chunk weights")
    ap.add_argument("--beta_route", type=float, default=0.05, help="weight for routing BCE loss")
    ap.add_argument("--delta_vgc", type=float, default=0.05, help="weight for view-geometry consistency loss")

    ap.add_argument("--apt_token", type=str, default="<|APT|>", help="Dedicated APT token inserted before ### Response.")

    args = ap.parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    ensure_phi_cache_compat()

    model_cfg = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=False)


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


    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=False,
        device_map=None,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    apt_token_id = ensure_apt_token_and_resize(tokenizer, model, args.apt_token)


    doc_anchor = torch.load(args.doc_anchor_pt, map_location="cpu")
    chunk_anchor = torch.load(args.chunk_anchor_pt, map_location="cpu")
    view_anchor = torch.load(args.view_anchor_pt, map_location="cpu") if args.view_anchor_pt else None
    assert doc_anchor.shape == chunk_anchor.shape, "doc/chunk anchor embeddings must be line-aligned with same shape"
    if view_anchor is not None:
        assert doc_anchor.shape == view_anchor.shape, "view anchor embeddings must be line-aligned with doc/chunk shape"
    anchor_dim = doc_anchor.shape[1]


    hidden_size = model.config.hidden_size
    model.anchor_proj_doc = nn.Linear(hidden_size, anchor_dim, bias=False)
    model.anchor_proj_chunk = nn.Linear(hidden_size, anchor_dim, bias=False)
    if view_anchor is not None:
        model.anchor_proj_view = nn.Linear(hidden_size, anchor_dim, bias=False)
    model.anchor_proj_route = nn.Linear(hidden_size, len(VIEW_NAMES), bias=False)


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
        apt_token=args.apt_token,
    )
    collator = DataCollatorWithPadding(tokenizer=tokenizer)


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
        beta_route=args.beta_route,
        delta_vgc=args.delta_vgc,
        layer_ids=layer_ids,
        lambda_doc=lambda_doc,
        lambda_chunk=lambda_chunk,
        lambda_view=lambda_view,
        apt_token_id=apt_token_id,
        tokenizer=tokenizer,
    )

    log_path = ""
    if args.log_jsonl:
        if args.log_jsonl == "auto":
            log_path = os.path.join(args.output_dir, "train_log.jsonl")
        else:
            log_path = args.log_jsonl

    trainer.add_callback(FiniteParameterCallback(check_steps=args.finite_check_steps))
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
