from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch


@dataclass(frozen=True)
class FeatureDumpConfig:
    out_path: str
    include_windows: bool = False
    include_tokens: bool = False


@torch.no_grad()
def dump_test_features(
    *,
    model,
    loader,
    device: torch.device,
    task_kind: str,
    cfg: FeatureDumpConfig,
) -> Path:
    out_path = Path(cfg.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dyn_vecs: List[np.ndarray] = []
    sty_vecs: List[np.ndarray] = []
    mix_vecs: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    subjects: List[np.ndarray] = []

    dyn_wins: List[np.ndarray] = []
    sty_wins: List[np.ndarray] = []
    mix_wins: List[np.ndarray] = []

    dyn_tokens: List[np.ndarray] = []
    sty_tokens: List[np.ndarray] = []
    mix_tokens_full: List[np.ndarray] = []

    probs_list: List[np.ndarray] = []
    preds_list: List[np.ndarray] = []

    model.eval()
    for batch in loader:
        x = batch["eeg"].to(device)
        y = batch["label"].to(device)
        s = batch.get("subject")
        s = s.to(device) if isinstance(s, torch.Tensor) else torch.zeros((x.size(0),), device=device, dtype=torch.long)

        out: Dict[str, torch.Tensor] = model(x)

        content_tokens = out["content_tokens"]
        dynamic_windows = content_tokens.mean(dim=1)
        dynamic_vec = dynamic_windows.mean(dim=1)

        static_tokens = out["style_tokens"]
        static_vec = out["style_vec"]
        static_windows = out["style_windows"]

        mixed_tokens = out["mixed_tokens"]
        mixed_windows = mixed_tokens.mean(dim=1)
        mixed_vec = mixed_windows.mean(dim=1)

        dyn_vecs.append(dynamic_vec.detach().cpu().numpy().astype(np.float32))
        sty_vecs.append(static_vec.detach().cpu().numpy().astype(np.float32))
        mix_vecs.append(mixed_vec.detach().cpu().numpy().astype(np.float32))
        labels.append(y.detach().view(-1).cpu().numpy().astype(np.int64))
        subjects.append(s.detach().view(-1).cpu().numpy().astype(np.int64))

        if cfg.include_windows:
            dyn_wins.append(dynamic_windows.detach().cpu().numpy().astype(np.float32))
            sty_wins.append(static_windows.detach().cpu().numpy().astype(np.float32))
            mix_wins.append(mixed_windows.detach().cpu().numpy().astype(np.float32))

        if cfg.include_tokens:
            dyn_tokens.append(content_tokens.detach().cpu().numpy().astype(np.float16))
            sty_tokens.append(static_tokens.detach().cpu().numpy().astype(np.float16))
            mix_tokens_full.append(mixed_tokens.detach().cpu().numpy().astype(np.float16))

        logits = out["task_logits"].detach()
        if task_kind == "multiclass":
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy().astype(np.float32)
            pred = logits.argmax(dim=1).detach().cpu().numpy().astype(np.int64)
        else:
            probs = torch.sigmoid(logits.view(-1)).detach().cpu().numpy().astype(np.float32)
            pred = (probs > 0.5).astype(np.int64)
        probs_list.append(probs)
        preds_list.append(pred)

    payload: Dict[str, np.ndarray] = {
        "dynamic_vec": np.concatenate(dyn_vecs, axis=0) if dyn_vecs else np.empty((0, 0), dtype=np.float32),
        "static_vec": np.concatenate(sty_vecs, axis=0) if sty_vecs else np.empty((0, 0), dtype=np.float32),
        "mixed_vec": np.concatenate(mix_vecs, axis=0) if mix_vecs else np.empty((0, 0), dtype=np.float32),
        "labels": np.concatenate(labels, axis=0) if labels else np.empty((0,), dtype=np.int64),
        "subjects": np.concatenate(subjects, axis=0) if subjects else np.empty((0,), dtype=np.int64),
        "preds": np.concatenate(preds_list, axis=0) if preds_list else np.empty((0,), dtype=np.int64),
        "probs": np.concatenate(probs_list, axis=0) if probs_list else np.empty((0,), dtype=np.float32),
    }
    if cfg.include_windows:
        payload.update(
            {
                "dynamic_windows": np.concatenate(dyn_wins, axis=0) if dyn_wins else np.empty((0, 0, 0), dtype=np.float32),
                "static_windows": np.concatenate(sty_wins, axis=0) if sty_wins else np.empty((0, 0, 0), dtype=np.float32),
                "mixed_windows": np.concatenate(mix_wins, axis=0) if mix_wins else np.empty((0, 0, 0), dtype=np.float32),
            }
        )
    if cfg.include_tokens:
        payload.update(
            {
                "dynamic_tokens": np.concatenate(dyn_tokens, axis=0) if dyn_tokens else np.empty((0, 0, 0, 0), dtype=np.float16),
                "static_tokens": np.concatenate(sty_tokens, axis=0) if sty_tokens else np.empty((0, 0, 0, 0), dtype=np.float16),
                "mixed_tokens_full": (
                    np.concatenate(mix_tokens_full, axis=0) if mix_tokens_full else np.empty((0, 0, 0, 0), dtype=np.float16)
                ),
            }
        )

    np.savez_compressed(out_path, **payload)
    return out_path
