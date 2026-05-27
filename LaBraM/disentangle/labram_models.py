from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from timm.models import create_model

import modeling_finetune  # noqa: F401

from .standard_1020 import STANDARD_1020


class LaBraMTokenBackbone(nn.Module):
    def __init__(
        self,
        *,
        model_name: str,
        ch_names: list[str],
        finetune_ckpt: str,
        use_pretrained_weights: bool = True,
        freeze: bool = False,
        time_embed_resize_mode: str = "interp",
    ) -> None:
        super().__init__()
        self.encoder = create_model(
            model_name,
            pretrained=False,
            num_classes=0,
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
            time_embed_resize_mode=time_embed_resize_mode,
        )
        self.default_input_chans = [0] + [STANDARD_1020.index(ch) + 1 for ch in ch_names]
        if use_pretrained_weights:
            self._load_checkpoint(finetune_ckpt)
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def _load_checkpoint(self, ckpt_path: str) -> None:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        checkpoint_model = None
        if isinstance(checkpoint, dict):
            for model_key in ("model", "module"):
                if model_key in checkpoint:
                    checkpoint_model = checkpoint[model_key]
                    break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        if isinstance(checkpoint_model, dict):
            new_dict = OrderedDict()
            for key, value in checkpoint_model.items():
                if key.startswith("student."):
                    new_dict[key[8:]] = value
                else:
                    new_dict[key] = value
            checkpoint_model = new_dict
        state_dict = self.encoder.state_dict()
        for key in ("head.weight", "head.bias"):
            if key in checkpoint_model and key in state_dict and checkpoint_model[key].shape != state_dict[key].shape:
                del checkpoint_model[key]
        for key in list(checkpoint_model.keys()):
            if "relative_position_index" in key:
                checkpoint_model.pop(key)
        incompatible = self.encoder.load_state_dict(checkpoint_model, strict=False)
        if getattr(incompatible, "missing_keys", None):
            print("Missing keys:", incompatible.missing_keys)
        if getattr(incompatible, "unexpected_keys", None):
            print("Unexpected keys:", incompatible.unexpected_keys)

    def forward(self, x: torch.Tensor, input_chans: Optional[list[int]] = None) -> torch.Tensor:
        batch, n_chan, n_win, _ = x.shape
        tokens = self.encoder.forward_features(
            x,
            input_chans=input_chans if input_chans is not None else self.default_input_chans,
            return_patch_tokens=True,
        )
        return tokens.reshape(batch, n_chan, n_win, -1)

    def get_num_layers(self) -> int:
        return int(self.encoder.get_num_layers())

    def no_weight_decay(self) -> set[str]:
        skip = set()
        if hasattr(self.encoder, "no_weight_decay"):
            skip = {f"backbone.encoder.{name}" for name in self.encoder.no_weight_decay()}
        return skip


class PatchTokenClassifier(nn.Module):
    def __init__(
        self,
        *,
        num_channels: int,
        num_windows: int,
        d_model: int,
        num_classes: int,
        task_kind: str,
        classifier: str,
        dropout: float,
    ) -> None:
        super().__init__()
        self.task_kind = str(task_kind)
        self.classifier = str(classifier)
        self.num_channels = int(num_channels)
        self.num_windows = int(num_windows)
        self.d_model = int(d_model)
        self.out_dim = 1 if self.task_kind == "binary" else int(num_classes)
        flat_dim = self.num_channels * self.num_windows * self.d_model
        hidden_mid = self.num_windows * self.d_model
        self.dropout = nn.Dropout(dropout)
        if self.classifier == "avgpooling_patch_reps":
            self.proj = nn.Linear(self.d_model, self.out_dim)
        elif self.classifier == "all_patch_reps_onelayer":
            self.proj = nn.Linear(flat_dim, self.out_dim)
        elif self.classifier == "all_patch_reps_twolayer":
            self.proj = nn.Sequential(
                nn.Linear(flat_dim, self.d_model),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(self.d_model, self.out_dim),
            )
        elif self.classifier == "all_patch_reps":
            self.proj = nn.Sequential(
                nn.Linear(flat_dim, hidden_mid),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_mid, self.d_model),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(self.d_model, self.out_dim),
            )
        else:
            raise ValueError(f"Unsupported classifier: {classifier}")

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        if self.classifier == "avgpooling_patch_reps":
            pooled = feats.mean(dim=(1, 2))
            logits = self.proj(pooled)
        else:
            logits = self.proj(feats.reshape(feats.size(0), -1))
        if self.task_kind == "binary":
            return logits.view(-1)
        return logits


class LaBraMBaselineModel(nn.Module):
    def __init__(self, backbone: LaBraMTokenBackbone, classifier: PatchTokenClassifier) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x: torch.Tensor, input_chans: Optional[list[int]] = None) -> torch.Tensor:
        feats = self.backbone(x, input_chans=input_chans)
        return self.classifier(feats)

    def get_num_layers(self) -> int:
        return self.backbone.get_num_layers()

    def no_weight_decay(self) -> set[str]:
        return self.backbone.no_weight_decay()
