from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score, precision_recall_curve, roc_auc_score, auc as sk_auc
from torch.utils.data import DataLoader

from .dataset_specs import resolve_channel_names
from .gaussian_model import DisentangleV7LiteModel, LossToggles
from .lmdb_dataset import GenericLMDBDataset
from .sequence_folder_dataset import GenericSequenceFolderDataset
from .samplers import SubjectBalancedBatchSampler


def validate_device_arg(device_str: str) -> str:
    device_str = str(device_str).strip()
    if device_str == "cpu":
        return device_str
    if device_str.startswith("cuda:"):
        try:
            gpu_index = int(device_str.split(":", 1)[1])
        except ValueError as exc:
            raise SystemExit(f"invalid --device value: {device_str}") from exc
        if gpu_index >= 0:
            return device_str
    raise SystemExit(f"for current experiment scheduling, --device must be cpu or cuda:<nonnegative index>; got {device_str}")


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0, warmup_steps=-1):
    total_iters = max(1, int(epochs) * int(niter_per_ep))
    warmup_schedule = np.array([])
    warmup_iters = int(warmup_epochs) * int(niter_per_ep)
    if warmup_steps > 0:
        warmup_iters = int(warmup_steps)
    warmup_iters = max(0, min(total_iters, warmup_iters))
    if warmup_iters > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
    remain = total_iters - warmup_iters
    if remain > 0:
        iters = np.arange(remain)
        schedule = np.array([
            final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / max(1, len(iters))))
            for i in iters
        ])
        schedule = np.concatenate((warmup_schedule, schedule))
    else:
        schedule = warmup_schedule
    assert len(schedule) == total_iters
    return schedule


@dataclass
class RunSummary:
    dataset: str
    lmdb_dir: str
    finetune: str
    seed: int
    backend: str
    branch_impl: str
    classifier: str
    task_kind: str
    num_classes: int
    select_metric: str
    best_epoch: int
    best_val_score: float
    best_test_score_at_best: float
    best_val_metrics: Dict[str, float]
    best_test_metrics_at_best: Dict[str, float]
    last_epoch: int
    last_val_metrics: Dict[str, float]
    last_test_metrics: Dict[str, float]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Fine-tune LaBraM on generic LMDB datasets.")
    ap.add_argument("--lmdb_dir", type=str, required=True)
    ap.add_argument("--dataset_name", type=str, default="")
    ap.add_argument("--channel_names", type=str, default="", help="Comma-separated list or a JSON/text file.")
    ap.add_argument("--finetune", type=str, required=True)
    ap.add_argument("--model", type=str, default="labram_base_patch200_200")
    ap.add_argument("--backend", type=str, default="baseline", choices=["baseline", "disentangle"])
    ap.add_argument("--branch_impl", type=str, default="gaussian", choices=["gaussian", "mlp"])
    ap.add_argument("--classifier", type=str, default="all_patch_reps", choices=["all_patch_reps", "all_patch_reps_twolayer", "all_patch_reps_onelayer", "avgpooling_patch_reps"])
    ap.add_argument("--task_kind", type=str, default="auto", choices=["auto", "binary", "multiclass"])
    ap.add_argument("--num_classes", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--opt", type=str, default="adamw")
    ap.add_argument("--opt_eps", type=float, default=1e-8)
    ap.add_argument("--opt_betas", type=float, nargs="+", default=None)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--min_lr", type=float, default=1e-6)
    ap.add_argument("--warmup_epochs", type=int, default=3)
    ap.add_argument("--warmup_steps", type=int, default=-1)
    ap.add_argument("--layer_decay", type=float, default=0.65)
    ap.add_argument("--clip_grad", type=float, default=None)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--weight_decay_end", type=float, default=None)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--subject_samples", type=int, default=2)
    ap.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--dataloader_seed", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--log_dir", type=str, default="")
    ap.add_argument("--save_ckpt", action="store_true")
    ap.add_argument("--eval_metric", type=str, default="balanced_accuracy", choices=["accuracy", "balanced_accuracy"])
    ap.add_argument("--fixed_selection_epoch", type=int, default=0)
    ap.add_argument("--frozen", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--use_pretrained_weights", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--gaussian_num_transition_components", type=int, default=16)
    ap.add_argument("--gaussian_transition_hidden", type=int, default=0)
    ap.add_argument("--gaussian_remove_gate_init", type=float, default=-2.0)
    ap.add_argument("--lambda_task", type=float, default=1.0)
    ap.add_argument("--lambda_style_cl", type=float, default=0.2)
    ap.add_argument("--lambda_style_invar", type=float, default=0.02)
    ap.add_argument("--lambda_var_ratio", type=float, default=0.1)
    ap.add_argument("--lambda_rec", type=float, default=0.02)
    ap.add_argument("--use_style_contrastive", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--use_style_invariance", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--use_var_ratio", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--use_film_reconstruction", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--var_ratio_r", type=float, default=2.0)
    ap.add_argument("--bypass_mid_mlp", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--bypass_content_proj", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--time_embed_resize_mode", type=str, default="interp", choices=["interp", "crop"])
    return ap


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _invert(mapping: Dict[int, str]) -> Dict[str, int]:
    return {value: int(index) for index, value in mapping.items()}


def _open_dataset(args: argparse.Namespace, split: str, *, subject_mapping=None, label_mapping=None):
    root = Path(args.lmdb_dir)
    if (root / "data.mdb").exists():
        return GenericLMDBDataset(args.lmdb_dir, split=split, subject_mapping=subject_mapping, label_mapping=label_mapping)
    if (root / "seq").is_dir() and (root / "labels").is_dir():
        return GenericSequenceFolderDataset(args.lmdb_dir, split=split, subject_mapping=subject_mapping, label_mapping=label_mapping)
    raise FileNotFoundError(f"Unsupported dataset layout under: {args.lmdb_dir}")


def build_dataloaders(args: argparse.Namespace):
    train_dataset = _open_dataset(args, split="train")
    subject_mapping = _invert(train_dataset.subject_index_to_id())
    label_mapping = _invert(train_dataset.label_index_to_value())
    val_dataset = _open_dataset(args, split="val", subject_mapping=subject_mapping, label_mapping=label_mapping)
    test_dataset = _open_dataset(args, split="test", subject_mapping=subject_mapping, label_mapping=label_mapping)
    if int(args.subject_samples) > 0:
        sampler_seed = int(args.dataloader_seed) if int(args.dataloader_seed) >= 0 else int(args.seed)
        sampler = SubjectBalancedBatchSampler(
            subject_ids=train_dataset.subject_ids(),
            batch_size=args.batch_size,
            samples_per_subject=args.subject_samples,
            generator=np.random.default_rng(sampler_seed),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=train_dataset.collate_fn,
        )
    else:
        generator = None
        if int(args.dataloader_seed) >= 0:
            generator = torch.Generator()
            generator.manual_seed(int(args.dataloader_seed))
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=bool(args.shuffle),
            drop_last=True,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=train_dataset.collate_fn,
            generator=generator,
        )
    val_loader = DataLoader(val_dataset, batch_size=max(1, int(1.5 * args.batch_size)), shuffle=False, drop_last=False, num_workers=args.num_workers, pin_memory=True, collate_fn=val_dataset.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=max(1, int(1.5 * args.batch_size)), shuffle=False, drop_last=False, num_workers=args.num_workers, pin_memory=True, collate_fn=test_dataset.collate_fn)
    return train_loader, val_loader, test_loader, train_dataset


def infer_task_kind(args: argparse.Namespace, train_dataset: GenericLMDBDataset) -> tuple[str, int]:
    num_classes = int(args.num_classes) if int(args.num_classes) > 0 else int(train_dataset.num_labels)
    if args.task_kind != "auto":
        return str(args.task_kind), num_classes
    return ("binary" if num_classes == 2 else "multiclass"), num_classes


def build_model(args: argparse.Namespace, ch_names: list[str], train_dataset: GenericLMDBDataset, task_kind: str, num_classes: int) -> torch.nn.Module:
    from .labram_models import LaBraMBaselineModel, LaBraMTokenBackbone, PatchTokenClassifier
    n_chan, n_win, patch_size = train_dataset.feature_shape
    backbone = LaBraMTokenBackbone(
        model_name=args.model,
        ch_names=ch_names,
        finetune_ckpt=args.finetune,
        use_pretrained_weights=bool(args.use_pretrained_weights),
        freeze=bool(args.frozen),
        time_embed_resize_mode=str(args.time_embed_resize_mode),
    )
    classifier = PatchTokenClassifier(
        num_channels=n_chan,
        num_windows=n_win,
        d_model=patch_size,
        num_classes=num_classes,
        task_kind=task_kind,
        classifier=args.classifier,
        dropout=float(args.dropout),
    )
    base_model = LaBraMBaselineModel(backbone, classifier)
    if args.backend == "baseline":
        return base_model
    return DisentangleV7LiteModel(
        base_model,
        d_model=patch_size,
        patch_size=patch_size,
        dropout=float(args.dropout),
        branch_impl=args.branch_impl,
        gaussian_num_transition_components=int(args.gaussian_num_transition_components),
        gaussian_transition_hidden=(None if int(args.gaussian_transition_hidden) <= 0 else int(args.gaussian_transition_hidden)),
        gaussian_remove_gate_init=float(args.gaussian_remove_gate_init),
        bypass_mid_mlp=bool(args.bypass_mid_mlp),
        bypass_content_proj=bool(args.bypass_content_proj),
    )


def _layer_id_for_name(name: str, assigner, num_layers: int) -> int:
    prefix = "backbone.encoder."
    if name.startswith(prefix):
        return assigner.get_layer_id(name[len(prefix):])
    return num_layers + 1


def prepare_optimizer(args: argparse.Namespace, model: torch.nn.Module) -> torch.optim.Optimizer:
    decay_params = []
    no_decay_params = []
    skip = model.no_weight_decay() if hasattr(model, "no_weight_decay") else set()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias") or name in skip:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    param_groups = [
        {"params": decay_params, "weight_decay": float(args.weight_decay)},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    if str(args.opt).lower() == "sgd":
        return torch.optim.SGD(param_groups, lr=float(args.lr), momentum=float(args.momentum), nesterov=True)
    return torch.optim.AdamW(param_groups, lr=float(args.lr), betas=tuple(args.opt_betas) if args.opt_betas else (0.9, 0.999), eps=float(args.opt_eps))


def _compute_metrics(task_kind: str, logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    labels_np = labels.detach().cpu().numpy()
    if task_kind == "binary":
        probs = torch.sigmoid(logits.view(-1)).detach().cpu().numpy()
        preds = (probs > 0.5).astype(np.int64)
        metrics = {
            "accuracy": float((preds == labels_np).mean()),
            "balanced_accuracy": float(balanced_accuracy_score(labels_np, preds)),
            "kappa": float(cohen_kappa_score(labels_np, preds)),
            "macro_f1": float(f1_score(labels_np, preds, average="macro")),
        }
        if np.unique(labels_np).size > 1:
            metrics["roc_auc"] = float(roc_auc_score(labels_np, probs))
            precision, recall, _ = precision_recall_curve(labels_np, probs, pos_label=1)
            metrics["pr_auc"] = float(sk_auc(recall, precision))
        return metrics
    preds = logits.argmax(dim=1).detach().cpu().numpy()
    metrics = {
        "accuracy": float((preds == labels_np).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(labels_np, preds)),
        "kappa": float(cohen_kappa_score(labels_np, preds)),
        "macro_f1": float(f1_score(labels_np, preds, average="macro")),
        "f1": float(f1_score(labels_np, preds, average="weighted")),
    }
    return metrics


def _loss_bundle(args: argparse.Namespace, model: torch.nn.Module, batch: Dict[str, torch.Tensor], task_kind: str) -> tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
    eeg = batch["eeg"]
    labels = batch["label"]
    subjects = batch["subject"]
    if args.backend == "baseline":
        logits = model(eeg)
        if task_kind == "binary":
            task_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.view(-1), labels.float().view(-1))
        else:
            task_loss = torch.nn.functional.cross_entropy(logits, labels, label_smoothing=float(args.label_smoothing))
        return task_loss, {"task": task_loss}, logits
    outputs = model(eeg)
    toggles = LossToggles(
        use_style_contrastive=bool(args.use_style_contrastive),
        use_style_invariance=bool(args.use_style_invariance),
        use_var_ratio=bool(args.use_var_ratio),
        use_film_reconstruction=bool(args.use_film_reconstruction),
    )
    losses = model.compute_losses(
        eeg,
        labels,
        subjects,
        outputs,
        task_kind=task_kind,
        label_smoothing=float(args.label_smoothing),
        toggles=toggles,
        temperature=float(args.temperature),
        var_ratio_r=float(args.var_ratio_r),
    )
    total = (
        float(args.lambda_task) * losses.get("task", torch.zeros((), device=eeg.device))
        + float(args.lambda_style_cl) * losses.get("style_cl", torch.zeros((), device=eeg.device))
        + float(args.lambda_style_invar) * losses.get("style_invar", torch.zeros((), device=eeg.device))
        + float(args.lambda_var_ratio) * losses.get("var_ratio", torch.zeros((), device=eeg.device))
        + float(args.lambda_rec) * losses.get("recon", torch.zeros((), device=eeg.device))
    )
    return total, losses, outputs["task_logits"]


def run_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    task_kind: str,
    args: argparse.Namespace,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    lr_schedule_values: Optional[np.ndarray] = None,
    wd_schedule_values: Optional[np.ndarray] = None,
    epoch: int = 0,
    train: bool,
) -> Dict[str, float]:
    model.train(train)
    totals: Dict[str, float] = {}
    total_n = 0
    all_logits = []
    all_labels = []
    num_training_steps_per_epoch = len(loader)
    for step, batch in enumerate(loader):
        eeg = batch["eeg"].to(device, non_blocking=True).float()
        batch["eeg"] = eeg
        batch["label"] = batch["label"].to(device, non_blocking=True)
        batch["subject"] = batch["subject"].to(device, non_blocking=True)
        if train and optimizer is not None and lr_schedule_values is not None:
            it = epoch * num_training_steps_per_epoch + step
            for param_group in optimizer.param_groups:
                param_group["lr"] = float(lr_schedule_values[it]) * param_group.get("lr_scale", 1.0)
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = float(wd_schedule_values[it])
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            total_loss, losses, logits = _loss_bundle(args, model, batch, task_kind)
        if train and optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and device.type == "cuda":
                scaler.scale(total_loss).backward()
                if args.clip_grad is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.clip_grad))
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if args.clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.clip_grad))
                optimizer.step()
        batch_size = int(eeg.size(0))
        total_n += batch_size
        totals["total"] = totals.get("total", 0.0) + float(total_loss.item()) * batch_size
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.item()) * batch_size
        all_logits.append(logits.detach().cpu())
        all_labels.append(batch["label"].detach().cpu())
    if total_n == 0:
        return {"total": 0.0}
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    metrics = {key: value / total_n for key, value in totals.items()}
    metrics.update(_compute_metrics(task_kind, logits, labels))
    return metrics


def main(argv: Optional[list[str]] = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    args.device = validate_device_arg(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "args.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    dataset_name, ch_names = resolve_channel_names(args.dataset_name, args.lmdb_dir, args.channel_names)
    args.dataset_name = dataset_name
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    cudnn.benchmark = True
    _seed_everything(int(args.seed))
    train_loader, val_loader, test_loader, train_dataset = build_dataloaders(args)
    task_kind, num_classes = infer_task_kind(args, train_dataset)
    model = build_model(args, ch_names, train_dataset, task_kind, num_classes).to(device)
    optimizer = prepare_optimizer(args, model)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    num_training_steps_per_epoch = len(train_loader)
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    lr_schedule_values = cosine_scheduler(args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch, warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps)
    wd_schedule_values = cosine_scheduler(args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    best_epoch = -1
    best_val_score = -1.0
    best_test_score_at_best = -1.0
    best_val_metrics: Dict[str, float] = {}
    best_test_metrics_at_best: Dict[str, float] = {}
    fixed_selection_epoch = int(args.fixed_selection_epoch)
    if fixed_selection_epoch < 0:
        fixed_selection_epoch = 0
    for epoch in range(args.epochs):
        t0 = time.time()
        train_metrics = run_epoch(model=model, loader=train_loader, device=device, task_kind=task_kind, args=args, optimizer=optimizer, scaler=scaler, lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values, epoch=epoch, train=True)
        val_metrics = run_epoch(model=model, loader=val_loader, device=device, task_kind=task_kind, args=args, train=False)
        test_metrics = run_epoch(model=model, loader=test_loader, device=device, task_kind=task_kind, args=args, train=False)
        dt = time.time() - t0
        val_score = float(val_metrics.get(args.eval_metric, 0.0))
        test_score = float(test_metrics.get(args.eval_metric, 0.0))
        print(
            f"epoch {epoch + 1}/{args.epochs} train_total={train_metrics.get('total', 0.0):.4f} "
            f"val_{args.eval_metric}={val_score:.4f} test_{args.eval_metric}={test_score:.4f} sec={dt:.1f}"
        )
        should_select = False
        if fixed_selection_epoch > 0:
            should_select = (epoch + 1) == fixed_selection_epoch
        else:
            should_select = val_score > best_val_score
        if should_select:
            best_val_score = val_score
            best_test_score_at_best = test_score
            best_epoch = epoch + 1
            best_val_metrics = dict(val_metrics)
            best_test_metrics_at_best = dict(test_metrics)
            if args.save_ckpt:
                torch.save({"model": model.state_dict(), "epoch": best_epoch, "args": vars(args)}, output_dir / "best.pth")
        if args.save_ckpt:
            torch.save({"model": model.state_dict(), "epoch": epoch + 1, "args": vars(args)}, output_dir / "last.pth")
    last_val_metrics = run_epoch(model=model, loader=val_loader, device=device, task_kind=task_kind, args=args, train=False)
    last_test_metrics = run_epoch(model=model, loader=test_loader, device=device, task_kind=task_kind, args=args, train=False)
    if fixed_selection_epoch > int(args.epochs) and best_epoch < 0:
        best_epoch = int(args.epochs)
        best_val_score = float(last_val_metrics.get(args.eval_metric, 0.0))
        best_test_score_at_best = float(last_test_metrics.get(args.eval_metric, 0.0))
        best_val_metrics = dict(last_val_metrics)
        best_test_metrics_at_best = dict(last_test_metrics)
    summary = RunSummary(
        dataset=dataset_name,
        lmdb_dir=args.lmdb_dir,
        finetune=args.finetune,
        seed=int(args.seed),
        backend=str(args.backend),
        branch_impl=str(args.branch_impl if args.backend == "disentangle" else "baseline"),
        classifier=str(args.classifier),
        task_kind=task_kind,
        num_classes=int(num_classes),
        select_metric=str(args.eval_metric),
        best_epoch=int(best_epoch),
        best_val_score=float(best_val_score),
        best_test_score_at_best=float(best_test_score_at_best),
        best_val_metrics=best_val_metrics,
        best_test_metrics_at_best=best_test_metrics_at_best,
        last_epoch=int(args.epochs),
        last_val_metrics=last_val_metrics,
        last_test_metrics=last_test_metrics,
    )
    (output_dir / "summary.json").write_text(json.dumps(asdict(summary), indent=2) + "\n")
    print("wrote", output_dir / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
