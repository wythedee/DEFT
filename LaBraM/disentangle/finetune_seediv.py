"""Fine-tune LaBraM on SEED-IV (LMDB) without modifying upstream scripts.

Usage (single GPU):
  python -m disentangle.finetune_seediv \
    --lmdb_dir /path/to/processed_average \
    --finetune ./checkpoints/labram-base.pth \
    --output_dir ./checkpoints/finetune_seediv_seed0 \
    --log_dir ./log/finetune_seediv_seed0 \
    --eval_metric balanced_accuracy \
    --epochs 30 --batch_size 64 --lr 5e-4 --seed 0
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from timm.loss import LabelSmoothingCrossEntropy
from timm.models import create_model

import modeling_finetune  # registers timm models
from engine_for_finetuning import evaluate, train_one_epoch
from optim_factory import LayerDecayValueAssigner, create_optimizer
import utils

from .seediv_lmdb import SEED_IV_CH_NAMES, build_seediv_datasets


@dataclass
class Summary:
    dataset: str
    lmdb_dir: str
    finetune: str
    seed: int
    eval_metric: str
    best_epoch: int
    best_val_score: float
    best_test_score_at_best: float
    last_epoch: int
    last_val_score: float
    last_test_score: float


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _load_checkpoint(model: torch.nn.Module, ckpt_path: str) -> None:
    """Load LaBraM pretrain ckpt for finetuning."""

    from collections import OrderedDict

    # LaBraM checkpoints may contain non-tensor metadata; opt out of the
    # PyTorch>=2.6 weights-only default for compatibility with upstream files.
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    checkpoint_model = None
    if isinstance(checkpoint, dict):
        for model_key in ("model", "module"):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print(f"Load state_dict by model_key = {model_key}")
                break
    if checkpoint_model is None:
        checkpoint_model = checkpoint

    # Keep only student.* weights.
    if isinstance(checkpoint_model, dict):
        new_dict = OrderedDict()
        for k, v in checkpoint_model.items():
            if k.startswith("student."):
                new_dict[k[8:]] = v
        checkpoint_model = new_dict

    # Drop head if shape mismatched.
    state_dict = model.state_dict()
    for k in ("head.weight", "head.bias"):
        if k in checkpoint_model and k in state_dict and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # Remove relative position index (buffer).
    for k in list(checkpoint_model.keys()):
        if "relative_position_index" in k:
            checkpoint_model.pop(k)

    utils.load_state_dict(model, checkpoint_model, prefix="")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb_dir", type=str, required=True)
    ap.add_argument("--finetune", type=str, required=True)
    ap.add_argument("--model", type=str, default="labram_base_patch200_200")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--opt", type=str, default="adamw")
    ap.add_argument("--opt_eps", type=float, default=1e-8)
    ap.add_argument("--opt_betas", type=float, nargs="+", default=None)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--min_lr", type=float, default=1e-6)
    ap.add_argument("--warmup_epochs", type=int, default=3)
    ap.add_argument("--warmup_steps", type=int, default=-1)
    ap.add_argument("--layer_decay", type=float, default=0.65)
    ap.add_argument("--clip_grad", type=float, default=None)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--weight_decay_end", type=float, default=None)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--log_dir", type=str, default="")
    ap.add_argument(
        "--save_ckpt",
        action="store_true",
        help="If set, save best/last model checkpoints. Default: disabled.",
    )
    ap.add_argument(
        "--eval_metric",
        type=str,
        default="balanced_accuracy",
        choices=("balanced_accuracy", "accuracy"),
        help="Metric used for model selection and reporting.",
    )
    args = ap.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "args.json").write_text(json.dumps(vars(args), indent=2) + "\n")

    device = torch.device(args.device)
    cudnn.benchmark = True
    _seed_everything(args.seed)

    datasets = build_seediv_datasets(args.lmdb_dir)
    train_sampler = torch.utils.data.DistributedSampler(
        datasets["train"], num_replicas=1, rank=0, shuffle=True
    )
    train_loader = torch.utils.data.DataLoader(
        datasets["train"],
        sampler=train_sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        datasets["val"],
        batch_size=int(1.5 * args.batch_size),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets["test"],
        batch_size=int(1.5 * args.batch_size),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # SEED-IV has 4 classes.
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=4,
        drop_rate=0.0,
        drop_path_rate=0.1,
        attn_drop_rate=0.0,
        drop_block_rate=None,
        use_mean_pooling=True,
        init_scale=0.001,
        use_rel_pos_bias=False,
        use_abs_pos_emb=True,
        init_values=0.1,
        qkv_bias=False,
    )

    _load_checkpoint(model, args.finetune)
    model.to(device)

    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    num_layers = model.get_num_layers()
    assigner = LayerDecayValueAssigner(
        list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2))
    )
    skip_weight_decay_list = model.no_weight_decay()
    optimizer = create_optimizer(
        args,
        model,
        skip_list=skip_weight_decay_list,
        get_num_layer=assigner.get_layer_id,
        get_layer_scale=assigner.get_scale,
    )
    loss_scaler = utils.NativeScalerWithGradNormCount()

    total_batch_size = args.batch_size
    num_training_steps_per_epoch = len(datasets["train"]) // total_batch_size
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    lr_schedule_values = utils.cosine_scheduler(
        args.lr,
        args.min_lr,
        args.epochs,
        num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs,
        warmup_steps=args.warmup_steps,
    )
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs,
        num_training_steps_per_epoch,
    )

    best_epoch = -1
    best_val_score = -1.0
    best_test_score_at_best = -1.0

    for epoch in range(args.epochs):
        train_loader.sampler.set_epoch(epoch)
        t0 = time.time()
        train_one_epoch(
            model=model,
            criterion=criterion,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            loss_scaler=loss_scaler,
            max_norm=args.clip_grad,
            model_ema=None,
            log_writer=None,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch,
            update_freq=1,
            ch_names=SEED_IV_CH_NAMES,
            is_binary=False,
        )

        val_ret = evaluate(
            val_loader,
            model,
            device,
            header=f"Val@{epoch}:",
            ch_names=SEED_IV_CH_NAMES,
            metrics=[args.eval_metric],
            is_binary=False,
        )
        test_ret = evaluate(
            test_loader,
            model,
            device,
            header=f"Test@{epoch}:",
            ch_names=SEED_IV_CH_NAMES,
            metrics=[args.eval_metric],
            is_binary=False,
        )

        val_score = float(val_ret.get(args.eval_metric, 0.0))
        test_score = float(test_ret.get(args.eval_metric, 0.0))
        dt = time.time() - t0
        print(
            f"epoch {epoch+1}/{args.epochs} "
            f"val_{args.eval_metric}={val_score:.4f} "
            f"test_{args.eval_metric}={test_score:.4f} sec={dt:.1f}"
        )

        if val_score > best_val_score:
            best_val_score = val_score
            best_epoch = epoch + 1
            best_test_score_at_best = test_score

            if args.save_ckpt:
                ckpt = {
                    "model": model.state_dict(),
                    "epoch": best_epoch,
                    "eval_metric": args.eval_metric,
                    "val_score": best_val_score,
                    "test_score": best_test_score_at_best,
                    "args": vars(args),
                }
                torch.save(ckpt, output_dir / "best.pth")

        if args.save_ckpt:
            torch.save({"model": model.state_dict(), "epoch": epoch + 1, "args": vars(args)}, output_dir / "last.pth")

    last_val = evaluate(
        val_loader,
        model,
        device,
        header="Val@last:",
        ch_names=SEED_IV_CH_NAMES,
        metrics=[args.eval_metric],
        is_binary=False,
    )
    last_test = evaluate(
        test_loader,
        model,
        device,
        header="Test@last:",
        ch_names=SEED_IV_CH_NAMES,
        metrics=[args.eval_metric],
        is_binary=False,
    )

    summ = Summary(
        dataset="SEED-IV",
        lmdb_dir=args.lmdb_dir,
        finetune=args.finetune,
        seed=args.seed,
        eval_metric=args.eval_metric,
        best_epoch=best_epoch,
        best_val_score=best_val_score,
        best_test_score_at_best=best_test_score_at_best,
        last_epoch=args.epochs,
        last_val_score=float(last_val.get(args.eval_metric, 0.0)),
        last_test_score=float(last_test.get(args.eval_metric, 0.0)),
    )
    (output_dir / "summary.json").write_text(json.dumps(asdict(summ), indent=2) + "\n")
    print("wrote", output_dir / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
