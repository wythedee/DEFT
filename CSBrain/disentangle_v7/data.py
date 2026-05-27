from __future__ import annotations

from dataclasses import dataclass
import os
import re
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import get_worker_info


@dataclass(frozen=True)
class DisentangleDataConfig:
    dataset_dir: str
    dataset_name: str = ""
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"
    batch_size: int = 64
    subject_samples: int = 0
    num_workers: int = 8
    seed: int = 3407


def _invert_mapping(index_to_value: dict[int, str]) -> dict[str, int]:
    return {value: int(index) for index, value in index_to_value.items()}


def _reset_lmdb_handles_in_worker(_: int) -> None:
    worker_info = get_worker_info()
    if worker_info is None:
        return
    dataset = worker_info.dataset
    reset_fn = getattr(dataset, "reset_db_state_for_worker", None)
    if callable(reset_fn):
        reset_fn()


def _subject_from_key(key: str) -> int:
    match = re.search(r"\d+", key)
    if match is None:
        return 0
    return int(match.group(0))


class SleepSequenceDisentangleDataset(Dataset):
    """Sequence dataset used by ISRUC-style sleep staging folders."""

    def __init__(self, dataset_dir: str, split: str) -> None:
        super().__init__()
        self.dataset_dir = dataset_dir
        self.seqs_dir = os.path.join(dataset_dir, "seq")
        self.labels_dir = os.path.join(dataset_dir, "labels")
        if not (os.path.isdir(self.seqs_dir) and os.path.isdir(self.labels_dir)):
            raise FileNotFoundError(f"Expected seq/labels dirs under {dataset_dir}")
        self.subject_pairs = self._load_subject_pairs()
        self.samples = self._split_subjects(split)
        self._subjects = np.asarray([sid for sid, _, _ in self.samples], dtype=np.int64)

    def _load_subject_pairs(self):
        seq_dirs = sorted(
            [name for name in os.listdir(self.seqs_dir) if os.path.isdir(os.path.join(self.seqs_dir, name))]
        )
        label_dirs = {
            name: os.path.join(self.labels_dir, name)
            for name in os.listdir(self.labels_dir)
            if os.path.isdir(os.path.join(self.labels_dir, name))
        }
        pairs = []
        for seq_dir in seq_dirs:
            if seq_dir not in label_dirs:
                continue
            subject_id = _subject_from_key(seq_dir)
            seq_path = os.path.join(self.seqs_dir, seq_dir)
            label_path = label_dirs[seq_dir]
            seq_files = sorted(os.listdir(seq_path))
            label_files = sorted(os.listdir(label_path))
            subject_samples = []
            for seq_fname, label_fname in zip(seq_files, label_files):
                subject_samples.append((os.path.join(seq_path, seq_fname), os.path.join(label_path, label_fname)))
            pairs.append((subject_id, subject_samples))
        return pairs

    def _split_subjects(self, split: str):
        num_subjects = len(self.subject_pairs)
        if num_subjects >= 100:
            train_end, val_end = 80, 90
        else:
            train_end, val_end = 6, 8
        items = []
        for index, (subject_id, samples) in enumerate(self.subject_pairs):
            target_split = "test"
            if index < train_end:
                target_split = "train"
            elif index < val_end:
                target_split = "val"
            if split == target_split:
                for seq_path, label_path in samples:
                    items.append((subject_id, seq_path, label_path))
        return items

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        subject_id, seq_path, label_path = self.samples[idx]
        seq = np.asarray(np.load(seq_path), dtype=np.float32) / 100.0
        label = np.asarray(np.load(label_path), dtype=np.int64)
        if seq.ndim != 3:
            raise ValueError(f"Expected ISRUC seq with ndim=3, got {seq.shape}")
        seq_len, ch_num, epoch_size = seq.shape
        if epoch_size != 6000:
            raise ValueError(f"Expected epoch_size=6000, got {epoch_size}")
        eeg = seq.reshape(seq_len, ch_num, 30, 200).reshape(seq_len, ch_num * 30, 200)
        return {
            "eeg": torch.from_numpy(eeg),
            "label": torch.from_numpy(label).long(),
            "subject": torch.tensor(int(subject_id), dtype=torch.long),
        }

    @staticmethod
    def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        eeg = torch.stack([item["eeg"] for item in batch], dim=0)
        label = torch.stack([item["label"] for item in batch], dim=0)
        subject = torch.stack([item["subject"] for item in batch], dim=0)
        return {"eeg": eeg, "label": label, "subject": subject}

    def subject_ids(self) -> np.ndarray:
        return self._subjects


def build_datasets(
    cfg: DisentangleDataConfig,
) -> Tuple[Dataset, Optional[Dataset], Optional[Dataset]]:
    if os.path.isdir(os.path.join(cfg.dataset_dir, "seq")) and os.path.isdir(os.path.join(cfg.dataset_dir, "labels")):
        train_dataset = SleepSequenceDisentangleDataset(cfg.dataset_dir, split=cfg.train_split)

        val_dataset: Optional[SleepSequenceDisentangleDataset] = None
        if cfg.val_split:
            val_dataset = SleepSequenceDisentangleDataset(cfg.dataset_dir, split=cfg.val_split)

        test_dataset: Optional[SleepSequenceDisentangleDataset] = None
        if cfg.test_split:
            test_dataset = SleepSequenceDisentangleDataset(cfg.dataset_dir, split=cfg.test_split)

        return train_dataset, val_dataset, test_dataset

    try:
        from datasets.disentangle_dataset import DisentangleDataset
    except ModuleNotFoundError as exc:
        if exc.name == "lmdb":
            raise ModuleNotFoundError(
                "Missing optional dependency 'lmdb'. Install project deps first (e.g. `pip install -r requirements.txt`)."
            ) from exc
        raise
    train_dataset = DisentangleDataset(
        cfg.dataset_dir,
        split=cfg.train_split,
        dataset_name=cfg.dataset_name or None,
    )

    subject_mapping = _invert_mapping(train_dataset.subject_index_to_id())
    label_mapping = _invert_mapping(train_dataset.label_index_to_value())

    val_dataset: Optional[DisentangleDataset] = None
    if cfg.val_split:
        val_dataset = DisentangleDataset(
            cfg.dataset_dir,
            split=cfg.val_split,
            dataset_name=cfg.dataset_name or None,
            subject_mapping=subject_mapping,
            label_mapping=label_mapping,
        )

    test_dataset: Optional[DisentangleDataset] = None
    if cfg.test_split:
        test_dataset = DisentangleDataset(
            cfg.dataset_dir,
            split=cfg.test_split,
            dataset_name=cfg.dataset_name or None,
            subject_mapping=subject_mapping,
            label_mapping=label_mapping,
        )

    return train_dataset, val_dataset, test_dataset


def build_dataloaders(
    cfg: DisentangleDataConfig,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader], "DisentangleDataset"]:
    try:
        from datasets.samplers import SubjectBalancedBatchSampler
    except ModuleNotFoundError:
        # Safe: samplers has no heavy deps, but keep a defensive import path.
        from datasets.samplers import SubjectBalancedBatchSampler
    train_dataset, val_dataset, test_dataset = build_datasets(cfg)

    if cfg.subject_samples > 0:
        sampler = SubjectBalancedBatchSampler(
            subject_ids=train_dataset.subject_ids(),
            batch_size=cfg.batch_size,
            samples_per_subject=cfg.subject_samples,
            generator=np.random.default_rng(cfg.seed),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=sampler,
            num_workers=cfg.num_workers,
            pin_memory=True,
            collate_fn=train_dataset.collate_fn,
            worker_init_fn=_reset_lmdb_handles_in_worker if cfg.num_workers > 0 else None,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=train_dataset.collate_fn,
            worker_init_fn=_reset_lmdb_handles_in_worker if cfg.num_workers > 0 else None,
        )

    val_loader: Optional[DataLoader] = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=val_dataset.collate_fn,
            worker_init_fn=_reset_lmdb_handles_in_worker if cfg.num_workers > 0 else None,
        )

    test_loader: Optional[DataLoader] = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=test_dataset.collate_fn,
            worker_init_fn=_reset_lmdb_handles_in_worker if cfg.num_workers > 0 else None,
        )

    return train_loader, val_loader, test_loader, train_dataset
