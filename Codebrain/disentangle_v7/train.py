"""Training entrypoints for Disentangle V7 (minimal fine-tune wrapper)."""

from __future__ import annotations

import argparse
import datetime
import math
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import numpy as np
import torch
from torch import amp

try:
    # Optional dependency.
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None  # type: ignore[assignment]

try:
    from sklearn.metrics import balanced_accuracy_score
except Exception:  # pragma: no cover
    balanced_accuracy_score = None

from Downstream.finetune_trainer import Trainer

try:
    from .standard_finetune import build_loader_and_model, build_model_only
    from .data import DisentangleDataConfig, build_dataloaders
    from .model import DisentangleV7Model, LossToggles, CodebookMode, ContentCodebookKind, ContentCodebookFuse, BranchImpl
    from .feature_dump import FeatureDumpConfig, dump_test_features
except ImportError:
    # Allow running as script (`python finetune_disentangle.py`) and module (`python -m ...`).
    from disentangle_v7.standard_finetune import build_loader_and_model, build_model_only
    from disentangle_v7.data import DisentangleDataConfig, build_dataloaders
    from disentangle_v7.model import (
        DisentangleV7Model,
        LossToggles,
        CodebookMode,
        ContentCodebookKind,
        ContentCodebookFuse,
        BranchImpl,
    )
    from disentangle_v7.feature_dump import FeatureDumpConfig, dump_test_features


def setup_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def _build_total_loss(losses: Dict[str, Any], lambdas: Dict[str, float]) -> torch.Tensor:
    """Safely aggregate weighted loss terms.

    Certain loss terms can become non-finite for edge batches (e.g., insufficient
    positive/negative pairs for contrastive computation). Skip non-finite terms to
    avoid poisoning the full objective.
    """
    total = torch.zeros((), dtype=torch.float32)
    for value in losses.values():
        if isinstance(value, torch.Tensor):
            total = torch.zeros((), device=value.device, dtype=value.dtype)
            break

    for name, weight in lambdas.items():
        w = float(weight)
        if w == 0.0:
            continue
        value = losses.get(name, None)
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            if not torch.isfinite(value).all():
                continue
            total = total + w * value
            continue
        try:
            scalar = float(value)
        except Exception:
            continue
        if math.isfinite(scalar):
            total = total + w * scalar
    return total


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Disentangle V7 (minimal): standard fine-tune wrapper")

    parser.add_argument("--data_backend", type=str, default="standard", choices=["standard", "disentangle"])

    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--log_every_epochs", type=int, default=10,
                        help="Disentangle backend: print a training log line every N epochs.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--use_amp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable CUDA AMP mixed precision. Disabled by default for cross-dataset stability.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-2)
    parser.add_argument("--optimizer", type=str, default="AdamW", choices=["AdamW", "SGD"])
    parser.add_argument("--clip_value", type=float, default=1.0)

    parser.add_argument(
        "--classifier",
        type=str,
        default="all_patch_reps",
        choices=[
            "all_patch_reps",
            "all_patch_reps_twolayer",
            "all_patch_reps_onelayer",
            "avgpooling_patch_reps",
        ],
    )
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--n_layer", type=int, default=8)

    parser.add_argument("--downstream_dataset", type=str, default="FACED")
    parser.add_argument("--datasets_dir", type=str, default="/path/to/eeg/Faced/processed")
    parser.add_argument("--num_of_classes", type=int, default=9)
    parser.add_argument(
        "--task_kind_override",
        choices=["multiclass", "binary", "regression"],
        default="",
        help="Override the task kind from the downstream dataset registry for audited reruns.",
    )
    parser.add_argument(
        "--num_classes_override",
        type=int,
        default=0,
        help="Override the class count from the downstream dataset registry for audited reruns.",
    )
    parser.add_argument("--model_dir", type=str, default="output_log/finetune_v7")
    parser.add_argument("--log_dir", type=str, default="")
    parser.add_argument("--dataset_name", type=str, default="",
                        help="Optional name hint used by DisentangleDataset backend (e.g., EMO_SEED_V).")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="val")
    parser.add_argument("--test_split", type=str, default="test")
    parser.add_argument("--subject_samples", type=int, default=0,
                        help="Disentangle backend only: samples per subject in each batch (0 disables).")
    parser.add_argument(
        "--train_shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disentangle backend only: enable/disable train DataLoader shuffle.",
    )
    parser.add_argument(
        "--dataloader_seed",
        type=int,
        default=-1,
        help="Disentangle backend only: if >=0, seed the train DataLoader generator.",
    )

    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--label_smoothing", type=float, default=0.1)

    parser.add_argument("--multi_lr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--frozen", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_pretrained_weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--foundation_dir", type=str, default="pretrained_weights/pretrained_weights.pth")
    parser.add_argument("--enable_progress_bar", action=argparse.BooleanOptionalAction, default=False)

    # TensorBoard logging (disentangle backend only).
    parser.add_argument(
        "--enable_tensorboard",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write TensorBoard event files (scalars) for losses/metrics.",
    )
    parser.add_argument(
        "--tensorboard_dir",
        type=str,
        default="",
        help="Optional override for TensorBoard log directory. If empty, defaults to a stable directory under ~/storage/logs/disentangle/tensorboard.",
    )
    parser.add_argument(
        "--save_best_checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save best-test checkpoint to <model_dir>/best_test_model.pt. "
        "Disable for large sweeps to avoid huge disk usage.",
    )

    parser.add_argument(
        "--select_best_by",
        type=str,
        default="test",
        choices=["test", "val", "val_kappa", "val_f1"],
        help="How to select the best epoch for reporting. 'test' matches historical behavior. "
        "'val' selects by val acc and reports test at that epoch. "
        "'val_kappa' matches original CodeBrain trainer selection. "
        "'val_f1' selects by validation weighted F1 and reports test at that epoch.",
    )
    parser.add_argument(
        "--selection_epoch_start",
        type=int,
        default=0,
        help="Optional inclusive lower bound (1-based) for epoch selection. "
        "Epochs outside the window are evaluated and logged but cannot be chosen as the reported best epoch.",
    )
    parser.add_argument(
        "--selection_epoch_end",
        type=int,
        default=0,
        help="Optional inclusive upper bound (1-based) for epoch selection. "
        "Epochs outside the window are evaluated and logged but cannot be chosen as the reported best epoch.",
    )
    parser.add_argument(
        "--fixed_selection_epoch",
        type=int,
        default=0,
        help="Policy-rerun helper: restrict best-epoch selection to exactly one epoch. "
        "Equivalent to setting --selection_epoch_start == --selection_epoch_end.",
    )
    parser.add_argument(
        "--scheduler_total_epochs",
        type=int,
        default=0,
        help="Optional cosine scheduler horizon in epochs. "
        "If >0, keep the LR schedule matched to a longer source run while training fewer actual epochs.",
    )

    # Curriculum / warmup: run task-only for first N epochs, then enable configured losses.
    parser.add_argument(
        "--warmup_task_only_epochs",
        type=int,
        default=0,
        help="If >0, run the first N epochs as task-only (all disentangle/codebook losses off).",
    )
    parser.add_argument(
        "--warmup_task_only_no_bypass_epochs",
        type=int,
        default=0,
        help="If >0, run an additional M epochs as task-only with the full path (no bypass) before enabling disentangle losses.",
    )
    parser.add_argument(
        "--warmup_bypass_modules",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="During warmup, force bypass_mid_mlp and bypass_content_proj to True to better match vanilla TaskOnly.",
    )
    parser.add_argument(
        "--warmup_exclude_from_selection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If enabled, do not select best epoch from warmup epochs (avoid picking best@warmup when using --select_best_by val).",
    )

    # Keep these for compatibility with eval_main / existing param passing.
    parser.add_argument(
        "--test_channel_permutation_mode",
        type=str,
        default="identity",
        choices=["identity", "reverse", "shuffle", "custom"],
    )
    parser.add_argument("--test_channel_permutation_custom", type=str, default="")
    parser.add_argument("--test_channel_permutation_seed", type=int, default=3407)
    parser.add_argument("--test_channel_axis", type=int, default=1)
    parser.add_argument("--test_channel_permutation_apply_to_val", action="store_true")

    # Disentangle backend: loss toggles for ablation.
    parser.add_argument("--lambda_task", type=float, default=1.0)
    parser.add_argument("--lambda_style_cl", type=float, default=1.0)
    parser.add_argument("--lambda_style_invar", type=float, default=0.1)
    parser.add_argument("--lambda_var_ratio", type=float, default=0.1)
    parser.add_argument("--lambda_rec", type=float, default=0.02)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--var_ratio_r", type=float, default=2.0)

    parser.add_argument("--use_style_contrastive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_style_invariance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_var_ratio", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_film_reconstruction", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_content_codebook", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--branch_impl",
        type=str,
        default="gaussian",
        choices=["mlp", "gaussian"],
        help="How to split features into dynamic/static branches after encoder.",
    )
    parser.add_argument("--bypass_mid_mlp", action=argparse.BooleanOptionalAction, default=True,
                        help="Bypass the extra mid MLP so task-only matches vanilla fine-tune more closely.")
    parser.add_argument("--bypass_content_proj", action=argparse.BooleanOptionalAction, default=True,
                        help="Bypass content projection so task head sees backbone features (vanilla-like).")
    parser.add_argument(
        "--gaussian_num_transition_components",
        type=int,
        default=16,
        help="RecXi-style dynamic transition components N for Gaussian branch disentanglement.",
    )
    parser.add_argument(
        "--gaussian_transition_hidden",
        type=int,
        default=0,
        help="Hidden size for Gaussian transition filter generator (<=0 uses 2*d_model).",
    )
    parser.add_argument(
        "--gaussian_remove_gate_init",
        type=float,
        default=-2.0,
        help="Initial logit for Gaussian counterpart-removal gate (sigmoid(logit)).",
    )
    parser.add_argument("--codebook_mode", type=str, default="window", choices=["window", "token"])
    parser.add_argument("--codebook_size", type=int, default=32)
    parser.add_argument("--vq_beta", type=float, default=0.25)
    parser.add_argument("--codebook_size_t", type=int, default=4096)
    parser.add_argument("--codebook_size_f", type=int, default=4096)
    parser.add_argument("--lambda_vq", type=float, default=0.1)
    parser.add_argument("--lambda_vq_usage", type=float, default=0.0)
    parser.add_argument("--vq_usage_temperature", type=float, default=1.0)
    parser.add_argument("--task_use_quantized_content", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--recon_use_quantized_content", action=argparse.BooleanOptionalAction, default=False)

    # Content codebook variants (experimental): VQ (hard, existing) vs retrieval (soft, multi-scale).
    # These only take effect when --use_content_codebook is enabled.
    parser.add_argument(
        "--content_codebook_kind",
        type=str,
        default="vq",
        choices=["vq", "retrieval_topk"],
        help="Which content codebook mechanism to use. 'vq' is VQ-VAE style hard assignment. "
        "'retrieval_topk' is soft top-k retrieval from a shared codebook.",
    )
    parser.add_argument(
        "--content_codebook_fuse",
        type=str,
        default="delta",
        choices=["delta", "concat_proj"],
        help="How to fuse retrieved codebook vectors into content tokens. "
        "'delta' adds (r - x) corrections. 'concat_proj' concatenates (x, r) and applies an identity-safe projection.",
    )
    parser.add_argument(
        "--content_codebook_use_global",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable trial-level (global pooled content) codebook retrieval/quantization.",
    )
    parser.add_argument(
        "--content_codebook_use_local",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable patch/window-level (local content) codebook retrieval/quantization.",
    )
    parser.add_argument(
        "--content_codebook_shared",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, global/local share the same codebook slots; each still has its own query projection.",
    )
    parser.add_argument("--content_codebook_topk", type=int, default=4)
    parser.add_argument("--content_codebook_tau", type=float, default=0.1)
    parser.add_argument("--content_codebook_alpha_global", type=float, default=0.1)
    parser.add_argument("--content_codebook_alpha_local", type=float, default=0.1)

    parser.add_argument("--run_suite", action=argparse.BooleanOptionalAction, default=False,
                        help="Run a small ablation suite (disentangle backend only).")
    parser.add_argument(
        "--suite_profile",
        type=str,
        default="taskfull",
        choices=["taskfull", "full_debug"],
        help="Which suite to run when --run_suite is enabled.",
    )
    parser.add_argument(
        "--analyze_val_test_correlation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disentangle backend only: evaluate test each epoch and print/save correlation between val metrics and test_acc across epochs (debugging selection metric).",
    )
    parser.add_argument(
        "--dump_feature_dir",
        type=str,
        default="",
        help="If set, dump content/style/mixed features from the selected checkpoint for probing.",
    )
    parser.add_argument(
        "--dump_feature_splits",
        type=str,
        default="train,val,test",
        help="Comma-separated splits to dump when --dump_feature_dir is set.",
    )
    parser.add_argument(
        "--dump_feature_include_windows",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also dump per-window content/style/mixed arrays for visualization.",
    )
    parser.add_argument(
        "--dump_feature_include_tokens",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also dump full token grids for probing all_patch_reps-like inputs.",
    )
    return parser


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size != y.size or x.size < 2:
        return float("nan")
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt((x * x).sum()) * np.sqrt((y * y).sum()))
    if denom <= 0:
        return float("nan")
    return float((x * y).sum() / denom)


def _rankdata_average_ties(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, x.size + 1, dtype=np.float64)

    sorted_x = x[order]
    i = 0
    while i < sorted_x.size:
        j = i + 1
        while j < sorted_x.size and sorted_x[j] == sorted_x[i]:
            j += 1
        if j - i > 1:
            avg = ranks[order[i:j]].mean()
            ranks[order[i:j]] = avg
        i = j
    return ranks


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size != y.size or x.size < 2:
        return float("nan")
    return _pearson_corr(_rankdata_average_ties(x), _rankdata_average_ties(y))


def run_finetune(params: argparse.Namespace) -> None:
    setup_seed(int(params.seed))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the original Trainer/model codepaths (they call .cuda()).")
    torch.cuda.set_device(int(params.cuda))

    dataset_dir = Path(params.datasets_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"datasets_dir not found: {dataset_dir}")

    if params.data_backend == "standard":
        # Keep standard backend behavior identical to original Downstream/finetune_main.py.
        params.model_dir = str(params.model_dir) + str(params.downstream_dataset) + "/"
        log_dir = str(getattr(params, "log_dir", "") or "")
        params.log_dir = log_dir + str(params.downstream_dataset) + "/"
        os.makedirs(params.log_dir, exist_ok=True)
        current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        params.file_name = str(params.log_dir) + str(current_time) + "_" + str(params.cuda) + ".txt"

        print(params)
        with open(params.file_name, "a", encoding="utf-8") as file:
            file.write(str(params) + "\n")
            file.write(f"The downstream dataset is {params.downstream_dataset}\n")

        loader, model, task = build_loader_and_model(params)
        trainer = Trainer(params, loader, model)

        if task == "multiclass":
            trainer.train_for_multiclass()
        elif task == "binary":
            trainer.train_for_binaryclass()
        elif task == "regression":
            trainer.train_for_regression()
        else:
            raise ValueError(f"Unsupported task kind: {task}")
        return

    if params.run_suite:
        run_ablation_suite(params)
        return
    run_disentangle_backend(params)


def _prepare_optimizer(params: argparse.Namespace, model: torch.nn.Module) -> torch.optim.Optimizer:
    backbone_params: List[torch.nn.Parameter] = []
    other_params: List[torch.nn.Parameter] = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(p)
        else:
            other_params.append(p)

    if params.frozen:
        for p in backbone_params:
            p.requires_grad = False
        backbone_params = []

    if params.optimizer == "AdamW":
        # Match original CodeBrain Trainer behavior: AdamW uses a single LR for all params.
        return torch.optim.AdamW(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)

    if params.multi_lr and backbone_params:
        return torch.optim.SGD(
            [{"params": backbone_params, "lr": params.lr}, {"params": other_params, "lr": params.lr * 5}],
            momentum=0.9,
            weight_decay=params.weight_decay,
        )
    return torch.optim.SGD([{"params": other_params, "lr": params.lr}], momentum=0.9, weight_decay=params.weight_decay)


def _balanced_acc(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    if balanced_accuracy_score is None:
        correct = (y_true == y_pred).float().mean().item() if y_true.numel() else 0.0
        return float(correct)
    return float(balanced_accuracy_score(y_true.cpu().numpy(), y_pred.cpu().numpy()))


def _json_float_dict(metrics: Optional[Dict[str, Any]]) -> Dict[str, float]:
    if not isinstance(metrics, dict):
        return {}
    out: Dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.floating)):
            out[str(key)] = float(value)
    return out


def _safe_metric(metrics: Optional[Dict[str, Any]], key: str) -> Optional[float]:
    if not isinstance(metrics, dict):
        return None
    value = metrics.get(key, None)
    if value is None:
        return None
    try:
        value_f = float(value)
    except Exception:
        return None
    if not math.isfinite(value_f):
        return None
    return value_f


def _epoch_snapshot(
    *,
    epoch: int,
    selection_allowed: bool,
    selection_metric_name: str,
    selection_metric_value: Optional[float],
    train_metrics: Optional[Dict[str, Any]],
    val_metrics: Optional[Dict[str, Any]],
    test_metrics: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    snapshot = {
        "epoch": int(epoch),
        "selection_allowed": bool(selection_allowed),
        "selection_metric_name": str(selection_metric_name),
        "selection_metric_value": (
            float(selection_metric_value)
            if selection_metric_value is not None and math.isfinite(float(selection_metric_value))
            else None
        ),
        "train": _json_float_dict(train_metrics),
        "val": _json_float_dict(val_metrics),
        "test": _json_float_dict(test_metrics),
    }
    return snapshot


def _eval_one_epoch(
    model: DisentangleV7Model,
    loader,
    device: torch.device,
    params: argparse.Namespace,
    task_kind: str,
    *,
    epoch: Optional[int] = None,
    split: str = "",
) -> Dict[str, float]:
    model.eval()
    totals: Dict[str, float] = {}
    total_samples = 0
    all_true: List[torch.Tensor] = []
    all_pred: List[torch.Tensor] = []
    all_probs: List[torch.Tensor] = []
    all_logits: List[torch.Tensor] = []
    vq_counts: Optional[torch.Tensor] = None
    vq_total = 0

    cb_global_counts: Optional[torch.Tensor] = None
    cb_global_total = 0
    cb_local_counts: Optional[torch.Tensor] = None
    cb_local_total = 0
    toggles = LossToggles(
        use_style_contrastive=params.use_style_contrastive,
        use_style_invariance=params.use_style_invariance,
        use_var_ratio=params.use_var_ratio,
        use_film_reconstruction=params.use_film_reconstruction,
        use_content_codebook=params.use_content_codebook,
    )
    with torch.no_grad():
        for batch in loader:
            x = batch["eeg"].to(device)
            y = batch["label"].to(device)
            s = batch["subject"].to(device)
            out = model(x)
            losses = model.compute_losses(
                x=x,
                labels=y,
                subject_ids=s,
                outputs=out,
                task_kind=task_kind,
                label_smoothing=params.label_smoothing,
                toggles=toggles,
                temperature=params.temperature,
                var_ratio_r=params.var_ratio_r,
            )
            total = _build_total_loss(
                losses,
                {
                    "task": float(params.lambda_task),
                    "style_cl": float(params.lambda_style_cl),
                    "style_invar": float(params.lambda_style_invar),
                    "var_ratio": float(params.lambda_var_ratio),
                    "recon": float(params.lambda_rec),
                    "vq": float(params.lambda_vq),
                    "vq_usage": float(params.lambda_vq_usage),
                },
            )
            batch_size = x.size(0)
            total_samples += batch_size
            for k, v in losses.items():
                totals[k] = totals.get(k, 0.0) + float(v.item()) * batch_size
            totals["total"] = totals.get("total", 0.0) + float(total.item()) * batch_size

            logits = out["task_logits"]
            if task_kind == "multiclass":
                if logits.dim() == y.dim() + 1 and y.dim() >= 2:
                    pred = logits.argmax(dim=-1)
                else:
                    pred = logits.argmax(dim=1)
                all_logits.append(logits.detach().cpu())
            else:
                probs = torch.sigmoid(logits.view(-1)).detach().cpu()
                pred = (probs > 0.5).long()
                y = y.view(-1).long()
                all_probs.append(probs)
            all_true.append(y.detach().cpu())
            all_pred.append(pred.detach().cpu())

            # VQ code usage stats (accumulate on CPU, optional).
            indices = out.get("vq_indices", None)
            if isinstance(indices, torch.Tensor) and indices.numel() > 0 and getattr(model, "vq", None) is not None:
                k = int(getattr(model.vq, "codebook_size", 0) or 0)
                if k > 0:
                    flat = indices.detach().view(-1).to(dtype=torch.long, device="cpu")
                    counts = torch.bincount(flat, minlength=k)
                    if vq_counts is None:
                        vq_counts = counts
                    else:
                        vq_counts += counts
                    vq_total += int(flat.numel())

            # Retrieval code usage stats (global/local top-1 indices).
            k_cb = int(getattr(model, "content_codebook_size", 0) or 0)
            if k_cb > 0:
                g_idx = out.get("cb_global_top1", None)
                if isinstance(g_idx, torch.Tensor) and g_idx.numel() > 0:
                    flat_g = g_idx.detach().view(-1).to(dtype=torch.long, device="cpu")
                    counts_g = torch.bincount(flat_g, minlength=k_cb)
                    cb_global_counts = counts_g if cb_global_counts is None else (cb_global_counts + counts_g)
                    cb_global_total += int(flat_g.numel())

                l_idx = out.get("cb_local_top1", None)
                if isinstance(l_idx, torch.Tensor) and l_idx.numel() > 0:
                    flat_l = l_idx.detach().view(-1).to(dtype=torch.long, device="cpu")
                    counts_l = torch.bincount(flat_l, minlength=k_cb)
                    cb_local_counts = counts_l if cb_local_counts is None else (cb_local_counts + counts_l)
                    cb_local_total += int(flat_l.numel())

    if total_samples == 0:
        return {k: 0.0 for k in ["total", "task", "acc"]}
    metrics = {k: v / total_samples for k, v in totals.items()}
    y_true = torch.cat(all_true, dim=0)
    y_pred = torch.cat(all_pred, dim=0)
    if task_kind == "multiclass" and y_true.dim() > 1:
        y_true = y_true.reshape(-1)
        y_pred = y_pred.reshape(-1)
    metrics["acc"] = _balanced_acc(y_true, y_pred)
    metrics["bacc"] = float(metrics["acc"])

    # Additional metrics (sklearn-backed when available).
    try:
        from sklearn.metrics import cohen_kappa_score, f1_score

        if y_true.numel():
            metrics["kappa"] = float(cohen_kappa_score(y_true.numpy(), y_pred.numpy()))
            metrics["f1"] = float(f1_score(y_true.numpy(), y_pred.numpy(), average="weighted"))
    except Exception:
        pass
    if task_kind != "multiclass":
        try:
            from sklearn.metrics import roc_auc_score, precision_recall_curve, auc as sk_auc

            if all_probs:
                probs = torch.cat(all_probs, dim=0).numpy()
                yt = y_true.numpy()
                uniq = np.unique(yt)
                if uniq.size > 1:
                    metrics["roc_auc"] = float(roc_auc_score(yt, probs))
                    precision, recall, _ = precision_recall_curve(yt, probs, pos_label=1)
                    metrics["pr_auc"] = float(sk_auc(recall, precision))
        except Exception:
            pass

    if vq_counts is not None and vq_total > 0:
        total = float(vq_counts.sum().item())
        if total > 0:
            probs = (vq_counts.to(dtype=torch.float64) / total).clamp_min(1e-12)
            entropy = float(-(probs * probs.log()).sum().item())
            metrics["vq_used_codes"] = float((vq_counts > 0).sum().item())
            metrics["vq_top1_frac"] = float((vq_counts.max().item()) / total)
            metrics["vq_perplexity"] = float(np.exp(entropy))

    def _add_codebook_metrics(prefix: str, counts: Optional[torch.Tensor], total_n: int) -> None:
        if counts is None or total_n <= 0:
            return
        total = float(counts.sum().item())
        if total <= 0:
            return
        probs = (counts.to(dtype=torch.float64) / total).clamp_min(1e-12)
        entropy = float(-(probs * probs.log()).sum().item())
        metrics[f"{prefix}_used_codes"] = float((counts > 0).sum().item())
        metrics[f"{prefix}_top1_frac"] = float((counts.max().item()) / total)
        metrics[f"{prefix}_perplexity"] = float(np.exp(entropy))
        metrics[f"{prefix}_entropy"] = float(entropy)

    _add_codebook_metrics("cb_global", cb_global_counts, cb_global_total)
    _add_codebook_metrics("cb_local", cb_local_counts, cb_local_total)

    # Persist raw code usage counts for later plotting.
    if epoch is not None and (cb_global_counts is not None or cb_local_counts is not None):
        try:
            import json
            from pathlib import Path

            out_path = Path(params.model_dir) / "codebook_stats.jsonl"
            record = {
                "epoch": int(epoch),
                "split": str(split),
                "dataset": str(getattr(params, "downstream_dataset", "")),
                "codebook_size": int(getattr(model, "content_codebook_size", 0) or 0),
                "counts_global": cb_global_counts.tolist() if cb_global_counts is not None else None,
                "counts_local": cb_local_counts.tolist() if cb_local_counts is not None else None,
                "metrics": {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
            }
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            # Never break evaluation for logging.
            pass
    return metrics


def run_disentangle_backend(params: argparse.Namespace) -> Dict[str, List[Dict[str, float]]]:
    if balanced_accuracy_score is None:
        print("[Warn] sklearn not available; using plain accuracy as a fallback for 'acc'.")

    setup_seed(int(params.seed))
    device = torch.device(f"cuda:{params.cuda}" if torch.cuda.is_available() else "cpu")

    data_cfg = DisentangleDataConfig(
        dataset_dir=params.datasets_dir,
        dataset_name=params.dataset_name,
        train_split=params.train_split,
        val_split=params.val_split,
        test_split=params.test_split,
        batch_size=params.batch_size,
        subject_samples=params.subject_samples,
        num_workers=params.num_workers,
        seed=params.seed,
        train_shuffle=bool(getattr(params, "train_shuffle", True)),
        dataloader_seed=int(getattr(params, "dataloader_seed", -1)),
    )
    train_loader, val_loader, test_loader, train_dataset = build_dataloaders(data_cfg)

    base_model, task_kind = build_model_only(params)
    if task_kind == "regression":
        raise ValueError("Disentangle backend currently disables regression tasks.")

    sample_shape = train_dataset[0]["eeg"].shape  # (C,S,P)
    patch_size = int(sample_shape[-1])
    d_model = patch_size  # original models use patch_size==d_model
    model = DisentangleV7Model(
        base_model=base_model,
        d_model=d_model,
        patch_size=patch_size,
        dropout=float(params.dropout),
        enable_mid_mlp=True,
        bypass_mid_mlp=bool(params.bypass_mid_mlp),
        bypass_content_proj=bool(params.bypass_content_proj),
        branch_impl=cast(BranchImpl, getattr(params, "branch_impl", "gaussian")),
        gaussian_num_transition_components=int(getattr(params, "gaussian_num_transition_components", 16)),
        gaussian_transition_hidden=(
            None
            if int(getattr(params, "gaussian_transition_hidden", 0)) <= 0
            else int(getattr(params, "gaussian_transition_hidden", 0))
        ),
        gaussian_remove_gate_init=float(getattr(params, "gaussian_remove_gate_init", -2.0)),
        use_content_codebook=bool(params.use_content_codebook),
        codebook_size=int(params.codebook_size),
        vq_beta=float(params.vq_beta),
        codebook_mode=cast(CodebookMode, params.codebook_mode),
        task_use_quantized_content=bool(params.task_use_quantized_content),
        recon_use_quantized_content=bool(params.recon_use_quantized_content),
        compute_vq_usage_loss=bool(params.use_content_codebook and float(params.lambda_vq_usage) > 0.0),
        vq_usage_temperature=float(params.vq_usage_temperature),
        content_codebook_kind=cast(ContentCodebookKind, getattr(params, "content_codebook_kind", "vq")),
        content_codebook_fuse=cast(ContentCodebookFuse, getattr(params, "content_codebook_fuse", "delta")),
        content_codebook_use_global=bool(getattr(params, "content_codebook_use_global", True)),
        content_codebook_use_local=bool(getattr(params, "content_codebook_use_local", True)),
        content_codebook_shared=bool(getattr(params, "content_codebook_shared", True)),
        content_codebook_topk=int(getattr(params, "content_codebook_topk", 4)),
        content_codebook_tau=float(getattr(params, "content_codebook_tau", 0.1)),
        content_codebook_alpha_global=float(getattr(params, "content_codebook_alpha_global", 0.1)),
        content_codebook_alpha_local=float(getattr(params, "content_codebook_alpha_local", 0.1)),
    ).to(device)

    optimiser = _prepare_optimizer(params, model)
    # Match original Trainer behavior: cosine schedule steps per iteration (not per epoch).
    data_length = len(train_loader)
    scheduler_total_epochs = int(getattr(params, "scheduler_total_epochs", 0) or 0)
    if scheduler_total_epochs > 0 and scheduler_total_epochs < int(params.epochs):
        raise ValueError("--scheduler_total_epochs cannot be smaller than actual --epochs.")
    scheduler_horizon_epochs = scheduler_total_epochs if scheduler_total_epochs > 0 else int(params.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=int(scheduler_horizon_epochs) * int(data_length), eta_min=1e-6
    )
    use_amp = bool(getattr(params, "use_amp", False)) and device.type == "cuda"
    scaler = amp.GradScaler(enabled=use_amp)

    def _default_tensorboard_dir(model_dir: str) -> str:
        # Keep event files on stable storage by default.
        # Mirror the model_dir relative path under ~/storage/logs/disentangle/tensorboard.
        home = Path.home()
        if (home / "storage").is_dir():
            tb_root = home / "storage" / "logs" / "disentangle" / "tensorboard"
            p = Path(model_dir)
            rel = None
            if "finetune_weights" in p.parts:
                idx = p.parts.index("finetune_weights")
                rel = Path(*p.parts[idx + 1 :])
            if rel is None or str(rel) == "":
                rel = Path(p.name)
            return str(tb_root / rel)
        return str(Path(model_dir) / "tb")

    writer = None
    if bool(getattr(params, "enable_tensorboard", False)):
        if SummaryWriter is None:
            raise RuntimeError(
                "--enable_tensorboard was set but TensorBoard is not available. "
                "Install it (pip install tensorboard) or disable tensorboard logging."
            )
        tb_dir = str(getattr(params, "tensorboard_dir", "") or "")
        if not tb_dir:
            tb_dir = _default_tensorboard_dir(str(params.model_dir))
        Path(tb_dir).mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_dir)
        print(f"[V7|{params.downstream_dataset}] TensorBoard logdir: {tb_dir}")

    base_toggles = LossToggles(
        use_style_contrastive=params.use_style_contrastive,
        use_style_invariance=params.use_style_invariance,
        use_var_ratio=params.use_var_ratio,
        use_film_reconstruction=params.use_film_reconstruction,
        use_content_codebook=params.use_content_codebook,
    )

    warmup_stage1_epochs = int(getattr(params, "warmup_task_only_epochs", 0) or 0)
    warmup_stage2_epochs = int(getattr(params, "warmup_task_only_no_bypass_epochs", 0) or 0)
    warmup_bypass = bool(getattr(params, "warmup_bypass_modules", True))
    warmup_exclude_from_selection = bool(getattr(params, "warmup_exclude_from_selection", True))

    # Cache the "steady-state" module bypass flags.
    base_bypass_mid_mlp = bool(getattr(model, "bypass_mid_mlp", False))
    base_bypass_content_proj = bool(getattr(model, "bypass_content_proj", False))

    # Cache the steady-state lambda weights.
    base_lambdas = {
        "task": float(getattr(params, "lambda_task", 0.0)),
        "style_cl": float(getattr(params, "lambda_style_cl", 0.0)),
        "style_invar": float(getattr(params, "lambda_style_invar", 0.0)),
        "var_ratio": float(getattr(params, "lambda_var_ratio", 0.0)),
        "recon": float(getattr(params, "lambda_rec", 0.0)),
        "vq": float(getattr(params, "lambda_vq", 0.0)),
        "vq_usage": float(getattr(params, "lambda_vq_usage", 0.0)),
    }

    # Only exclude warmup epochs from best-epoch selection when warmup actually changes
    # the objective/path compared to the steady-state config.
    warmup_changes_losses = bool(
        (base_toggles.use_style_contrastive and base_lambdas["style_cl"] > 0.0)
        or (base_toggles.use_style_invariance and base_lambdas["style_invar"] > 0.0)
        or (base_toggles.use_var_ratio and base_lambdas["var_ratio"] > 0.0)
        or (base_toggles.use_film_reconstruction and base_lambdas["recon"] > 0.0)
        or (base_toggles.use_content_codebook and (base_lambdas["vq"] > 0.0 or base_lambdas["vq_usage"] > 0.0))
    )
    warmup_changes_modules = bool(warmup_bypass and ((not base_bypass_mid_mlp) or (not base_bypass_content_proj)))
    warmup_total_epochs = int(warmup_stage1_epochs + warmup_stage2_epochs)
    warmup_is_curriculum = bool(warmup_total_epochs > 0 and (warmup_changes_losses or warmup_changes_modules))
    warmup_exclude_effective = bool(warmup_exclude_from_selection and warmup_is_curriculum)

    save_dir = Path(params.model_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    history: Dict[str, List[Dict[str, float]]] = {"train": [], "val": [], "test": []}

    select_best_by = str(getattr(params, "select_best_by", "test") or "test")
    if select_best_by not in {"test", "val", "val_kappa", "val_f1"}:
        raise ValueError(f"Unsupported --select_best_by: {select_best_by}")

    fixed_selection_epoch = int(getattr(params, "fixed_selection_epoch", 0) or 0)
    selection_epoch_start = int(getattr(params, "selection_epoch_start", 0) or 0)
    selection_epoch_end = int(getattr(params, "selection_epoch_end", 0) or 0)
    if fixed_selection_epoch > 0:
        selection_epoch_start = fixed_selection_epoch
        selection_epoch_end = fixed_selection_epoch
    if selection_epoch_start < 0 or selection_epoch_end < 0 or fixed_selection_epoch < 0:
        raise ValueError("Selection epoch arguments must be >= 0.")
    if selection_epoch_start > 0 and selection_epoch_start > int(params.epochs):
        raise ValueError("--selection_epoch_start exceeds total epochs.")
    if selection_epoch_end > 0 and selection_epoch_end > int(params.epochs):
        raise ValueError("--selection_epoch_end exceeds total epochs.")
    if selection_epoch_start > 0 and selection_epoch_end > 0 and selection_epoch_start > selection_epoch_end:
        raise ValueError("--selection_epoch_start cannot exceed --selection_epoch_end.")

    best_epoch = -1
    best_path = save_dir / "best_test_model.pt"  # historical filename; now means "best by selection metric"

    best_sel = -1.0
    best_test = -1.0  # test acc at best selection epoch
    best_val_acc: Optional[float] = None
    save_best_checkpoint = bool(getattr(params, "save_best_checkpoint", True))
    if not save_best_checkpoint and best_path.exists():
        try:
            best_path.unlink()
        except Exception:
            pass
    best_test_metrics: Optional[Dict[str, float]] = None
    best_val_metrics: Optional[Dict[str, float]] = None
    selected_epoch_info: Optional[Dict[str, Any]] = None
    best_epoch_by_metric_all: Dict[str, Dict[str, Any]] = {}
    best_epoch_by_metric_selectable: Dict[str, Dict[str, Any]] = {}
    epoch_metrics_path = save_dir / "epoch_metrics.jsonl"
    try:
        if epoch_metrics_path.exists():
            epoch_metrics_path.unlink()
    except Exception:
        pass

    def _update_best_epoch_metric(
        tracker: Dict[str, Dict[str, Any]],
        metric_name: str,
        metric_value: Optional[float],
        snapshot: Dict[str, Any],
    ) -> None:
        if metric_value is None or not math.isfinite(float(metric_value)):
            return
        current = tracker.get(metric_name)
        if current is None or float(metric_value) >= float(current.get("metric_value", float("-inf"))):
            tracker[metric_name] = {
                "metric_name": str(metric_name),
                "metric_value": float(metric_value),
                **snapshot,
            }

    def _selection_window_allows(epoch: int) -> bool:
        if selection_epoch_start > 0 and epoch < selection_epoch_start:
            return False
        if selection_epoch_end > 0 and epoch > selection_epoch_end:
            return False
        return True

    for epoch in range(1, int(params.epochs) + 1):
        in_stage1 = warmup_stage1_epochs > 0 and epoch <= warmup_stage1_epochs
        in_stage2 = (warmup_stage2_epochs > 0 and epoch > warmup_stage1_epochs and epoch <= warmup_total_epochs)

        # Curriculum stage config.
        if in_stage1:
            # Stage 1: task-only with optional bypass to mimic vanilla task-only.
            toggles = LossToggles(
                use_style_contrastive=False,
                use_style_invariance=False,
                use_var_ratio=False,
                use_film_reconstruction=False,
                use_content_codebook=False,
            )
            lambdas = {
                "task": base_lambdas["task"],
                "style_cl": 0.0,
                "style_invar": 0.0,
                "var_ratio": 0.0,
                "recon": 0.0,
                "vq": 0.0,
                "vq_usage": 0.0,
            }

            # Try to match vanilla TaskOnly behavior by bypassing extra modules.
            try:
                if warmup_bypass:
                    setattr(model, "bypass_mid_mlp", True)
                    setattr(model, "bypass_content_proj", True)
                else:
                    setattr(model, "bypass_mid_mlp", base_bypass_mid_mlp)
                    setattr(model, "bypass_content_proj", base_bypass_content_proj)
            except Exception:
                pass
        elif in_stage2:
            # Stage 2: task-only with full path (no bypass) to align representations.
            toggles = LossToggles(
                use_style_contrastive=False,
                use_style_invariance=False,
                use_var_ratio=False,
                use_film_reconstruction=False,
                use_content_codebook=False,
            )
            lambdas = {
                "task": base_lambdas["task"],
                "style_cl": 0.0,
                "style_invar": 0.0,
                "var_ratio": 0.0,
                "recon": 0.0,
                "vq": 0.0,
                "vq_usage": 0.0,
            }
            try:
                setattr(model, "bypass_mid_mlp", base_bypass_mid_mlp)
                setattr(model, "bypass_content_proj", base_bypass_content_proj)
            except Exception:
                pass
        else:
            toggles = base_toggles
            lambdas = dict(base_lambdas)
            # Restore original bypass behavior.
            try:
                setattr(model, "bypass_mid_mlp", base_bypass_mid_mlp)
                setattr(model, "bypass_content_proj", base_bypass_content_proj)
            except Exception:
                pass

        # Print stage boundary once.
        if warmup_is_curriculum:
            if warmup_stage1_epochs > 0 and epoch == warmup_stage1_epochs + 1 and warmup_stage2_epochs > 0:
                print(
                    f"[V7|{params.downstream_dataset}] Curriculum: stage1 finished at epoch={warmup_stage1_epochs}; "
                    "switching to task-only full path"
                )
            if epoch == warmup_total_epochs + 1:
                print(
                    f"[V7|{params.downstream_dataset}] Curriculum: stage2 finished at epoch={warmup_total_epochs}; "
                    "enabling disentangle losses"
                )

        model.train()
        totals: Dict[str, float] = {}
        total_samples = 0

        for batch in train_loader:
            x = batch["eeg"].to(device)
            y = batch["label"].to(device)
            s = batch["subject"].to(device)

            optimiser.zero_grad()
            with amp.autocast(device_type=device.type, enabled=use_amp):
                out = model(x)
                losses = model.compute_losses(
                    x=x,
                    labels=y,
                    subject_ids=s,
                    outputs=out,
                    task_kind=task_kind,
                    label_smoothing=params.label_smoothing,
                    toggles=toggles,
                    temperature=params.temperature,
                    var_ratio_r=params.var_ratio_r,
                )
                total = _build_total_loss(losses, lambdas)

            if use_amp:
                scaler.scale(total).backward()
                scaler.unscale_(optimiser)
            else:
                total.backward()

            if params.clip_value and float(params.clip_value) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(params.clip_value))

            stepped = True
            if use_amp:
                scale_before = float(scaler.get_scale())
                scaler.step(optimiser)
                scaler.update()
                scale_after = float(scaler.get_scale())
                # When gradients overflow, GradScaler skips the optimizer step and reduces the scale.
                stepped = scale_after >= scale_before
            else:
                optimiser.step()
            if stepped:
                scheduler.step()

            batch_size = x.size(0)
            total_samples += batch_size
            for k, v in losses.items():
                totals[k] = totals.get(k, 0.0) + float(v.item()) * batch_size
            totals["total"] = totals.get("total", 0.0) + float(total.item()) * batch_size

        train_metrics = {k: v / max(1, total_samples) for k, v in totals.items()}
        history["train"].append(train_metrics)

        if writer is not None:
            for k, v in train_metrics.items():
                if isinstance(v, (int, float)):
                    writer.add_scalar(f"train/{k}", float(v), int(epoch))

        val_metrics: Optional[Dict[str, float]] = None
        if val_loader is not None:
            val_metrics = _eval_one_epoch(model, val_loader, device, params, task_kind, epoch=epoch, split="val")
            history["val"].append(val_metrics)

            if writer is not None:
                for k, v in val_metrics.items():
                    if isinstance(v, (int, float)):
                        writer.add_scalar(f"val/{k}", float(v), int(epoch))

        test_metrics: Optional[Dict[str, float]] = None
        if test_loader is not None:
            test_metrics = _eval_one_epoch(model, test_loader, device, params, task_kind, epoch=epoch, split="test")
            history["test"].append(test_metrics)

            if writer is not None:
                for k, v in test_metrics.items():
                    if isinstance(v, (int, float)):
                        writer.add_scalar(f"test/{k}", float(v), int(epoch))
            # Selection logic:
            # - historical: select best epoch by test_acc
            # - best-val: select by val_acc but report test_acc at that epoch
            val_acc: Optional[float]
            if isinstance(val_metrics, dict):
                val_acc = float(val_metrics.get("bacc", val_metrics.get("acc", 0.0)))
                val_kappa = _safe_metric(val_metrics, "kappa")
                val_f1 = _safe_metric(val_metrics, "f1")
                val_roc_auc = _safe_metric(val_metrics, "roc_auc")
                val_pr_auc = _safe_metric(val_metrics, "pr_auc")
            else:
                val_acc = None
                val_kappa = None
                val_f1 = None
                val_roc_auc = None
                val_pr_auc = None
            test_acc = float(test_metrics.get("bacc", test_metrics.get("acc", 0.0))) if isinstance(test_metrics, dict) else 0.0

            # Ensure `sel` is always a float.
            if select_best_by == "val_kappa" and val_kappa is not None:
                sel = float(val_kappa)
                selection_metric_name = "val_kappa"
            elif select_best_by == "val_f1" and val_f1 is not None:
                sel = float(val_f1)
                selection_metric_name = "val_f1"
            elif select_best_by == "val" and val_acc is not None:
                sel = float(val_acc)
                selection_metric_name = "val_bacc"
            else:
                sel = float(test_acc)
                selection_metric_name = "test_bacc"

            selection_allowed = not ((in_stage1 or in_stage2) and warmup_exclude_effective and epoch != int(params.epochs))
            selection_allowed = bool(selection_allowed and _selection_window_allows(epoch))
            epoch_snapshot = _epoch_snapshot(
                epoch=epoch,
                selection_allowed=selection_allowed,
                selection_metric_name=selection_metric_name,
                selection_metric_value=sel,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                test_metrics=test_metrics,
            )
            try:
                with open(epoch_metrics_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(epoch_snapshot, ensure_ascii=False) + "\n")
            except Exception:
                pass

            _update_best_epoch_metric(best_epoch_by_metric_all, "test_bacc", test_acc, epoch_snapshot)
            if val_acc is not None:
                _update_best_epoch_metric(best_epoch_by_metric_all, "val_bacc", val_acc, epoch_snapshot)
            if val_kappa is not None:
                _update_best_epoch_metric(best_epoch_by_metric_all, "val_kappa", val_kappa, epoch_snapshot)
            if val_f1 is not None:
                _update_best_epoch_metric(best_epoch_by_metric_all, "val_f1", val_f1, epoch_snapshot)
            if val_roc_auc is not None:
                _update_best_epoch_metric(best_epoch_by_metric_all, "val_roc_auc", val_roc_auc, epoch_snapshot)
            if val_pr_auc is not None:
                _update_best_epoch_metric(best_epoch_by_metric_all, "val_pr_auc", val_pr_auc, epoch_snapshot)

            if selection_allowed:
                _update_best_epoch_metric(best_epoch_by_metric_selectable, "test_bacc", test_acc, epoch_snapshot)
                if val_acc is not None:
                    _update_best_epoch_metric(best_epoch_by_metric_selectable, "val_bacc", val_acc, epoch_snapshot)
                if val_kappa is not None:
                    _update_best_epoch_metric(best_epoch_by_metric_selectable, "val_kappa", val_kappa, epoch_snapshot)
                if val_f1 is not None:
                    _update_best_epoch_metric(best_epoch_by_metric_selectable, "val_f1", val_f1, epoch_snapshot)
                if val_roc_auc is not None:
                    _update_best_epoch_metric(best_epoch_by_metric_selectable, "val_roc_auc", val_roc_auc, epoch_snapshot)
                if val_pr_auc is not None:
                    _update_best_epoch_metric(best_epoch_by_metric_selectable, "val_pr_auc", val_pr_auc, epoch_snapshot)

            if not selection_allowed:
                # Still log metrics into history, but don't let warmup epochs win selection.
                # Edge case: if the whole run is warmup (e.g. smoke test or very short epochs),
                # allow selection so best_epoch/best_test_acc are meaningful.
                if epoch == int(params.epochs) and best_epoch < 0:
                    best_sel = float(sel)
                    best_epoch = epoch
                    best_test = test_acc
                    best_val_acc = val_acc
                    best_test_metrics = dict(test_metrics)
                    best_val_metrics = dict(val_metrics) if isinstance(val_metrics, dict) else None
                    selected_epoch_info = dict(epoch_snapshot)
                else:
                    pass
            elif sel >= best_sel:
                best_sel = float(sel)
                best_epoch = epoch
                best_test = test_acc
                best_val_acc = val_acc
                best_test_metrics = dict(test_metrics)
                best_val_metrics = dict(val_metrics) if isinstance(val_metrics, dict) else None
                selected_epoch_info = dict(epoch_snapshot)
                if save_best_checkpoint:
                    torch.save(model.state_dict(), best_path)

        log_every = int(getattr(params, "log_every_epochs", 1) or 1)
        if (epoch % log_every == 0) or (epoch == 1) or (epoch == int(params.epochs)):
            msg = f"[V7|{params.downstream_dataset}] epoch {epoch}/{params.epochs} "
            if val_loader is not None and history["val"]:
                msg += f"val_acc={history['val'][-1].get('acc', 0.0):.4f} "
            if test_loader is not None and history["test"]:
                msg += f"test_acc={history['test'][-1].get('acc', 0.0):.4f} "
            msg += f"train_total={train_metrics.get('total', 0.0):.4f}"
            print(msg)

        if writer is not None:
            writer.flush()

    # Final: re-eval test metrics for the best-selected checkpoint (suite-friendly).
    # When checkpoints are disabled, reuse metrics captured at the best epoch.
    if test_loader is not None and save_best_checkpoint and best_path.exists():
        state = torch.load(best_path, map_location=device)
        model.load_state_dict(state, strict=True)
        best_test_metrics = _eval_one_epoch(
            model,
            test_loader,
            device,
            params,
            task_kind,
            epoch=int(best_epoch) if best_epoch is not None else None,
            split="test_best",
        )
        best_test = float(best_test_metrics.get("acc", best_test))
        if selected_epoch_info is not None:
            selected_epoch_info["test"] = _json_float_dict(best_test_metrics)

    # Persist val metrics for the selected checkpoint as well.
    if val_loader is not None and save_best_checkpoint and best_path.exists():
        try:
            state = torch.load(best_path, map_location=device)
            model.load_state_dict(state, strict=True)
            best_val_metrics = _eval_one_epoch(
                model,
                val_loader,
                device,
                params,
                task_kind,
                epoch=int(best_epoch) if best_epoch is not None else None,
                split="val_best",
            )
            acc_v = best_val_metrics.get("bacc", best_val_metrics.get("acc", None))
            if isinstance(acc_v, (int, float)):
                best_val_acc = float(acc_v)
            if selected_epoch_info is not None:
                selected_epoch_info["val"] = _json_float_dict(best_val_metrics)
                if select_best_by == "val_kappa":
                    selected_epoch_info["selection_metric_value"] = _safe_metric(best_val_metrics, "kappa")
                elif select_best_by == "val_f1":
                    selected_epoch_info["selection_metric_value"] = _safe_metric(best_val_metrics, "f1")
                elif select_best_by == "val":
                    selected_epoch_info["selection_metric_value"] = _safe_metric(best_val_metrics, "bacc")
                else:
                    selected_epoch_info["selection_metric_value"] = _safe_metric(best_test_metrics, "bacc")
        except Exception:
            pass

    feature_dump_paths: Dict[str, str] = {}
    dump_feature_dir = str(getattr(params, "dump_feature_dir", "") or "").strip()
    if dump_feature_dir:
        if save_best_checkpoint and best_path.exists():
            state = torch.load(best_path, map_location=device)
            model.load_state_dict(state, strict=True)
        split_to_loader = {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader,
        }
        requested_splits = [
            item.strip()
            for item in str(getattr(params, "dump_feature_splits", "train,val,test") or "").split(",")
            if item.strip()
        ]
        dump_root = Path(dump_feature_dir)
        dump_root.mkdir(parents=True, exist_ok=True)
        for split_name in requested_splits:
            loader = split_to_loader.get(split_name)
            if loader is None:
                continue
            out_path = dump_root / f"{split_name}_content_style_features.npz"
            saved_path = dump_test_features(
                model=model,
                loader=loader,
                device=device,
                task_kind=task_kind,
                cfg=FeatureDumpConfig(
                    out_path=str(out_path),
                    include_windows=bool(getattr(params, "dump_feature_include_windows", False)),
                    include_tokens=bool(getattr(params, "dump_feature_include_tokens", False)),
                ),
            )
            feature_dump_paths[split_name] = str(saved_path)

    # Optional: analyze correlation between val metrics and test_acc across epochs.
    val_test_corr: Dict[str, float] = {}
    selection_debug: Dict[str, object] = {}
    if bool(getattr(params, "analyze_val_test_correlation", False)) and history["val"] and history["test"]:
        n = min(len(history["val"]), len(history["test"]))
        if n >= 3:
            val_keys: List[str]
            if task_kind == "multiclass":
                val_keys = ["acc", "f1", "kappa"]
            else:
                val_keys = ["acc", "roc_auc", "pr_auc"]

            val_accs = np.array([float(history["val"][i].get("acc", 0.0)) for i in range(n)], dtype=np.float64)
            test_accs = np.array([float(history["test"][i].get("acc", 0.0)) for i in range(n)], dtype=np.float64)

            best_metric = None
            best_abs_rho = -1.0
            print(f"\n[V7 Corr|{params.downstream_dataset}] n={n} target=test_acc")
            for k in val_keys:
                vx = np.array([float(history["val"][i].get(k, float("nan"))) for i in range(n)], dtype=np.float64)
                mask = np.isfinite(vx) & np.isfinite(test_accs)
                if int(mask.sum()) < 3:
                    continue
                pear = _pearson_corr(vx[mask], test_accs[mask])
                rho = _spearman_corr(vx[mask], test_accs[mask])
                val_test_corr[f"val_{k}_pearson"] = float(pear)
                val_test_corr[f"val_{k}_spearman"] = float(rho)
                print(f"- val_{k}: pearson={pear:.4f} spearman={rho:.4f}")
                if not math.isnan(rho) and abs(rho) > best_abs_rho:
                    best_abs_rho = abs(rho)
                    best_metric = f"val_{k}"

            if best_metric is not None:
                selection_debug["best_by_abs_spearman"] = {"metric": best_metric, "abs_rho": float(best_abs_rho)}
                print(f"- best_by_|spearman|: {best_metric} (|rho|={best_abs_rho:.4f})")

            # What would happen if we select by different val metrics?
            selection_debug["by_metric"] = {}
            for k in val_keys:
                vx = np.array([float(history['val'][i].get(k, float('nan'))) for i in range(n)], dtype=np.float64)
                if not np.isfinite(vx).any():
                    continue
                best_i = int(np.nanargmax(vx))
                selection_debug["by_metric"][f"val_{k}"] = {
                    "best_epoch": int(best_i + 1),
                    "val": float(vx[best_i]),
                    "test_acc_at_best": float(test_accs[best_i]),
                }
            # Also include the "val_acc" chosen checkpoint result for reference.
            if np.isfinite(val_accs).any():
                best_i = int(np.nanargmax(val_accs))
                selection_debug["val_acc_best_epoch"] = int(best_i + 1)
                selection_debug["val_acc_best_test_acc"] = float(test_accs[best_i])

            corr_path = save_dir / "val_test_correlation.json"
            corr_path.write_text(
                json.dumps(
                    {
                        "downstream_dataset": params.downstream_dataset,
                        "task_kind": task_kind,
                        "n_epochs": int(n),
                        "correlations": val_test_corr,
                        "selection_debug": selection_debug,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

    active_val_metric_name: Optional[str] = None
    if val_loader is not None:
        if select_best_by == "val_kappa":
            active_val_metric_name = "val_kappa"
        elif select_best_by == "val_f1":
            active_val_metric_name = "val_f1"
        else:
            active_val_metric_name = "val_bacc"
    best_val_epoch_info = (
        dict(best_epoch_by_metric_all.get(active_val_metric_name, {}))
        if active_val_metric_name is not None and active_val_metric_name in best_epoch_by_metric_all
        else (dict(selected_epoch_info) if selected_epoch_info is not None else {})
    )
    best_test_epoch_info = dict(best_epoch_by_metric_all.get("test_bacc", {}))
    selectable_best_val_epoch_info = (
        dict(best_epoch_by_metric_selectable.get(active_val_metric_name, {}))
        if active_val_metric_name is not None and active_val_metric_name in best_epoch_by_metric_selectable
        else (dict(selected_epoch_info) if selected_epoch_info is not None else {})
    )
    selectable_best_test_epoch_info = dict(best_epoch_by_metric_selectable.get("test_bacc", {}))

    if selected_epoch_info is None:
        selected_epoch_info = {}

    summary = {
        "downstream_dataset": params.downstream_dataset,
        "dataset_name": getattr(params, "dataset_name", ""),
        "datasets_dir": params.datasets_dir,
        "task_kind": task_kind,
        "selection": {
            "select_best_by": select_best_by,
            "best_selection_value": float(best_sel),
            "selection_epoch_start": int(selection_epoch_start),
            "selection_epoch_end": int(selection_epoch_end),
            "fixed_selection_epoch": int(fixed_selection_epoch),
            "scheduler_total_epochs": int(scheduler_total_epochs),
            "selection_window_active": bool(
                fixed_selection_epoch > 0 or selection_epoch_start > 0 or selection_epoch_end > 0
            ),
        },
        "curriculum": {
            "warmup_task_only_epochs": int(warmup_stage1_epochs),
            "warmup_task_only_no_bypass_epochs": int(warmup_stage2_epochs),
            "warmup_bypass_modules": bool(warmup_bypass),
            "warmup_exclude_from_selection": bool(warmup_exclude_from_selection),
        },
        "model_config": {
            "enable_mid_mlp": True,
            "branch_impl": str(getattr(params, "branch_impl", "gaussian")),
            "bypass_mid_mlp": bool(getattr(params, "bypass_mid_mlp", False)),
            "bypass_content_proj": bool(getattr(params, "bypass_content_proj", False)),
            "gaussian_num_transition_components": int(getattr(params, "gaussian_num_transition_components", 16)),
            "gaussian_transition_hidden": int(getattr(params, "gaussian_transition_hidden", 0)),
            "gaussian_remove_gate_init": float(getattr(params, "gaussian_remove_gate_init", -2.0)),
        },
        "codebook_config": {
            "use_content_codebook": bool(getattr(params, "use_content_codebook", False)),
            "content_codebook_kind": str(getattr(params, "content_codebook_kind", "vq")),
            "content_codebook_use_global": bool(getattr(params, "content_codebook_use_global", True)),
            "content_codebook_use_local": bool(getattr(params, "content_codebook_use_local", True)),
            "content_codebook_shared": bool(getattr(params, "content_codebook_shared", True)),
            "content_codebook_topk": int(getattr(params, "content_codebook_topk", 4)),
            "content_codebook_tau": float(getattr(params, "content_codebook_tau", 0.1)),
            "content_codebook_alpha_global": float(getattr(params, "content_codebook_alpha_global", 0.1)),
            "content_codebook_alpha_local": float(getattr(params, "content_codebook_alpha_local", 0.1)),
            "codebook_mode": str(getattr(params, "codebook_mode", "window")),
            "codebook_size": int(getattr(params, "codebook_size", 0) or 0),
            "task_use_quantized_content": bool(getattr(params, "task_use_quantized_content", False)),
            "recon_use_quantized_content": bool(getattr(params, "recon_use_quantized_content", False)),
            "vq_beta": float(getattr(params, "vq_beta", 0.0)),
            "vq_usage_temperature": float(getattr(params, "vq_usage_temperature", 1.0)),
        },
        "loss_config": {
            "lambda_task": float(getattr(params, "lambda_task", 0.0)),
            "lambda_style_cl": float(getattr(params, "lambda_style_cl", 0.0)),
            "lambda_style_invar": float(getattr(params, "lambda_style_invar", 0.0)),
            "lambda_var_ratio": float(getattr(params, "lambda_var_ratio", 0.0)),
            "lambda_rec": float(getattr(params, "lambda_rec", 0.0)),
            "lambda_vq": float(getattr(params, "lambda_vq", 0.0)),
            "lambda_vq_usage": float(getattr(params, "lambda_vq_usage", 0.0)),
            "temperature": float(getattr(params, "temperature", 0.0)),
            "var_ratio_r": float(getattr(params, "var_ratio_r", 0.0)),
        },
        "run_config": {
            "epochs": int(getattr(params, "epochs", 0)),
            "scheduler_total_epochs": int(scheduler_total_epochs),
            "batch_size": int(getattr(params, "batch_size", 0)),
            "num_workers": int(getattr(params, "num_workers", 0)),
            "subject_samples": int(getattr(params, "subject_samples", 0)),
            "train_shuffle": bool(getattr(params, "train_shuffle", True)),
            "dataloader_seed": int(getattr(params, "dataloader_seed", -1)),
            "lr": float(getattr(params, "lr", 0.0)),
            "weight_decay": float(getattr(params, "weight_decay", 0.0)),
            "optimizer": str(getattr(params, "optimizer", "")),
            "label_smoothing": float(getattr(params, "label_smoothing", 0.0)),
            "classifier": str(getattr(params, "classifier", "")),
            "dropout": float(getattr(params, "dropout", 0.0)),
            "foundation_dir": str(getattr(params, "foundation_dir", "")),
        },
        # Always report the test acc at the selected best epoch.
        "best_test_acc": float(best_test),
        "best_test_bacc": float(best_test),
        "best_val_acc": float(best_val_acc) if best_val_acc is not None else None,
        "best_val_bacc": float(best_val_acc) if best_val_acc is not None else None,
        "best_epoch": int(best_epoch),
        "best_checkpoint": str(best_path) if save_best_checkpoint else "",
        "test_at_best": best_test_metrics or {},
        "val_at_best": best_val_metrics or {},
        "selected_epoch_info": selected_epoch_info,
        "best_val_epoch": int(best_val_epoch_info.get("epoch", -1)) if best_val_epoch_info else -1,
        "best_val_metric_name": active_val_metric_name,
        "best_val_metric_value": best_val_epoch_info.get("metric_value", None) if best_val_epoch_info else None,
        "val_at_best_val": best_val_epoch_info.get("val", {}) if best_val_epoch_info else {},
        "test_at_best_val": best_val_epoch_info.get("test", {}) if best_val_epoch_info else {},
        "best_val_epoch_info": best_val_epoch_info,
        "best_val_epoch_selectable": int(selectable_best_val_epoch_info.get("epoch", -1))
        if selectable_best_val_epoch_info
        else -1,
        "best_val_metric_value_selectable": selectable_best_val_epoch_info.get("metric_value", None)
        if selectable_best_val_epoch_info
        else None,
        "val_at_best_val_selectable": selectable_best_val_epoch_info.get("val", {})
        if selectable_best_val_epoch_info
        else {},
        "test_at_best_val_selectable": selectable_best_val_epoch_info.get("test", {})
        if selectable_best_val_epoch_info
        else {},
        "best_val_epoch_info_selectable": selectable_best_val_epoch_info,
        "best_test_epoch": int(best_test_epoch_info.get("epoch", -1)) if best_test_epoch_info else -1,
        "best_test_metric_name": "test_bacc",
        "best_test_metric_value": best_test_epoch_info.get("metric_value", None) if best_test_epoch_info else None,
        "val_at_best_test": best_test_epoch_info.get("val", {}) if best_test_epoch_info else {},
        "test_at_best_test": best_test_epoch_info.get("test", {}) if best_test_epoch_info else {},
        "best_test_epoch_info": best_test_epoch_info,
        "best_test_epoch_selectable": int(selectable_best_test_epoch_info.get("epoch", -1))
        if selectable_best_test_epoch_info
        else -1,
        "best_test_metric_value_selectable": selectable_best_test_epoch_info.get("metric_value", None)
        if selectable_best_test_epoch_info
        else None,
        "val_at_best_test_selectable": selectable_best_test_epoch_info.get("val", {})
        if selectable_best_test_epoch_info
        else {},
        "test_at_best_test_selectable": selectable_best_test_epoch_info.get("test", {})
        if selectable_best_test_epoch_info
        else {},
        "best_test_epoch_info_selectable": selectable_best_test_epoch_info,
        "best_epoch_by_metric": best_epoch_by_metric_all,
        "best_epoch_by_metric_selectable": best_epoch_by_metric_selectable,
        "epoch_metrics_path": str(epoch_metrics_path),
        "feature_dump_paths": feature_dump_paths,
        "val_test_correlation": val_test_corr,
        "val_test_selection_debug": selection_debug,
        "last_val": history["val"][-1] if history["val"] else {},
        "last_test": history["test"][-1] if history["test"] else {},
    }
    with open(save_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if best_test_metrics is not None:
        saved_msg = str(best_path) if save_best_checkpoint else "<disabled>"
        val_msg = ""
        if best_val_acc is not None:
            val_msg = f" best_val_bacc={best_val_acc:.4f}"
        best_test_epoch_msg = ""
        if best_test_epoch_info:
            best_test_epoch_msg = (
                f" best_test_bacc_epoch={int(best_test_epoch_info.get('epoch', -1))}"
                f" best_test_bacc={float(best_test_epoch_info.get('metric_value', -1.0)):.4f}"
            )
        print(
            f"[V7 Summary] select_by={select_best_by}{val_msg} best_epoch={best_epoch} "
            f"best_test_acc={best_test:.4f}{best_test_epoch_msg} saved={saved_msg}"
        )
    else:
        saved_msg = str(best_path) if save_best_checkpoint else "<disabled>"
        print(f"[V7 Summary] best_epoch={best_epoch} best_test_acc={best_test:.4f} saved={saved_msg}")

    if writer is not None:
        try:
            writer.close()
        except Exception:
            pass

    return history


def run_ablation_suite(params: argparse.Namespace):
    if params.data_backend != "disentangle":
        raise ValueError("--run_suite is supported only with --data_backend disentangle.")

    root = Path(params.model_dir)
    root.mkdir(parents=True, exist_ok=True)

    def with_overrides(name: str, **overrides):
        cfg = argparse.Namespace(**vars(params))
        for k, v in overrides.items():
            setattr(cfg, k, v)
        cfg.model_dir = str(root / name)
        return cfg

    # Previous longer suites are kept for reference below.

    # Previous codebook suite (kept for reference):
    # base_overrides = {
    #     "use_style_contrastive": True,
    #     "use_style_invariance": True,
    #     "use_var_ratio": True,
    #     "use_film_reconstruction": True,
    #     "use_content_codebook": True,
    #     "codebook_mode": "window",
    #     "vq_beta": params.vq_beta,
    #     "codebook_size": 32,
    #     "lambda_style_cl": params.lambda_style_cl,
    #     "lambda_style_invar": params.lambda_style_invar,
    #     "lambda_var_ratio": params.lambda_var_ratio,
    #     "lambda_rec": params.lambda_rec if params.lambda_rec > 0 else 0.1,
    #     "lambda_vq": params.lambda_vq,
    #     "recon_use_quantized_content": True,
    # }
    # suites = [
    #     ("run6_codebook_cont", {**base_overrides, "task_use_quantized_content": False, "lambda_vq_usage": 0.0}),
    #     ("run7_codebook_quant", {**base_overrides, "task_use_quantized_content": True, "lambda_vq_usage": 0.0}),
    #     ("run8_codebook_quant_usage", {**base_overrides, "task_use_quantized_content": True, "lambda_vq_usage": params.lambda_vq_usage if params.lambda_vq_usage > 0 else 0.1}),
    # ]

    # Suite (short): task-only + full. Others are commented out.
    # - Selection is by best test_acc (see training loop).
    base_full = {
        "use_style_contrastive": True,
        "use_style_invariance": True,
        "use_var_ratio": True,
        "use_film_reconstruction": True,
        "use_content_codebook": False,
        # Bypass flags: task-only uses vanilla-like path; full enables extra modules.
        "bypass_mid_mlp": False,
        "bypass_content_proj": False,
        "lambda_style_cl": params.lambda_style_cl,
        "lambda_style_invar": params.lambda_style_invar,
        "lambda_var_ratio": params.lambda_var_ratio,
        "lambda_rec": 0.02,
        "lambda_vq": 0.0,
        "lambda_vq_usage": 0.0,
        "task_use_quantized_content": False,
        "recon_use_quantized_content": False,
    }

    suites_taskfull = [
        ("run_task_only", {**base_full, "use_style_contrastive": False, "use_style_invariance": False, "use_var_ratio": False,
                           "use_film_reconstruction": False, "lambda_style_cl": 0.0, "lambda_style_invar": 0.0, "lambda_var_ratio": 0.0, "lambda_rec": 0.0,
                           "bypass_mid_mlp": True, "bypass_content_proj": True}),
        ("run_full", {**base_full, "bypass_mid_mlp": False, "bypass_content_proj": False}),
    ]

    suites_full_debug = [
        # Baseline sanity check.
        suites_taskfull[0],
        # Is it the extra modules (mid/content_proj) causing collapse?
        ("run_modules_only", {**base_full,
                              "use_style_contrastive": False, "use_style_invariance": False, "use_var_ratio": False, "use_film_reconstruction": False,
                              "lambda_style_cl": 0.0, "lambda_style_invar": 0.0, "lambda_var_ratio": 0.0, "lambda_rec": 0.0,
                              "bypass_mid_mlp": False, "bypass_content_proj": False}),
        # Is it the disentangle losses causing collapse (without extra modules)?
        ("run_losses_only", {**base_full,
                             "bypass_mid_mlp": True, "bypass_content_proj": True}),
        # Full model (modules + losses).
        suites_taskfull[1],
        # More options (commented):
        # ("run_full_no_stylecl", {**base_full, "use_style_contrastive": False, "lambda_style_cl": 0.0}),
        # ("run_full_no_rec", {**base_full, "use_film_reconstruction": False, "lambda_rec": 0.0}),
        # ("run_full_no_var", {**base_full, "use_var_ratio": False, "lambda_var_ratio": 0.0}),
    ]

    suites = suites_taskfull if str(getattr(params, "suite_profile", "taskfull")) == "taskfull" else suites_full_debug

    runs = []
    suite_summary = []
    print("\n[V7 Suite] Starting ablation suite...\n")
    for name, overrides in suites:
        cfg = with_overrides(name, **overrides)
        print(f"[V7 Suite] === {name} ===")
        runs.append(run_disentangle_backend(cfg))
        summary_path = Path(cfg.model_dir) / "summary.json"
        if summary_path.exists():
            try:
                suite_summary.append(json.loads(summary_path.read_text(encoding="utf-8")))
            except Exception:
                pass
    print("\n[V7 Suite] Done.\n")
    if suite_summary:
        (root / "suite_summary.json").write_text(
            json.dumps(suite_summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        best = max(suite_summary, key=lambda x: float(x.get("best_val_acc", -1.0)))
        print(
            "[V7 Suite Summary] "
            f"best_run={Path(best.get('best_checkpoint','')).parent.name} "
            f"best_val_acc={float(best.get('best_val_acc', 0.0)):.4f} "
            f"test_acc={float(best.get('test_at_best_val', {}).get('acc', 0.0)):.4f}"
        )
    return runs


def run_from_dict(config: Dict[str, Any]) -> None:
    parser = build_arg_parser()
    defaults = parser.parse_args([])
    for k, v in config.items():
        setattr(defaults, k, v)
    run_finetune(defaults)


def run_cli(argv: Optional[list[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_finetune(args)


def main() -> None:
    run_cli()


if __name__ == "__main__":
    main()
