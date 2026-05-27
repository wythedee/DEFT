from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import get_worker_info

from utils.util import to_tensor


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
    train_shuffle: bool = True


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


def _torch_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


class _ISRUCSequenceDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        split: str,
        variant: str,
        subject_mapping: Optional[Dict[str, int]] = None,
        label_mapping: Optional[Dict[str, int]] = None,
    ) -> None:
        super().__init__()
        self.dataset_dir = dataset_dir
        self.split = split
        self.variant = variant
        self.seq_dir = os.path.join(dataset_dir, "seq")
        self.labels_dir = os.path.join(dataset_dir, "labels")
        if not os.path.isdir(self.seq_dir) or not os.path.isdir(self.labels_dir):
            raise FileNotFoundError(
                f"ISRUC disentangle backend expects 'seq' and 'labels' folders under {dataset_dir!r}."
            )

        self._subject_to_index: Dict[str, int] = {} if subject_mapping is None else dict(subject_mapping)
        self._label_to_index: Dict[str, int] = {} if label_mapping is None else dict(label_mapping)
        self._samples: List[tuple[str, str, str, str]] = []
        self._subject_cache: List[int] = []
        self._label_cache: List[int] = []
        self._build_index()

    def _split_subject_numbers(self) -> List[int]:
        split_name = str(self.split or "").lower()
        if self.variant == "ISRUC-S3":
            if split_name == "train":
                return list(range(1, 7))
            if split_name == "val":
                return list(range(7, 9))
            if split_name == "test":
                return list(range(9, 11))
            raise KeyError(f"Unsupported ISRUC-S3 split: {self.split}")
        if split_name == "train":
            return list(range(1, 81))
        if split_name == "val":
            return list(range(81, 91))
        if split_name == "test":
            return list(range(91, 101))
        raise KeyError(f"Unsupported ISRUC split: {self.split}")

    def _subject_prefix(self) -> str:
        return "ISRUC-group3" if self.variant == "ISRUC-S3" else "ISRUC-group1"

    def _normalise_scalar_label(self, label: Any) -> str:
        arr = np.asarray(label)
        if arr.size == 1:
            return str(arr.item())
        return str(label)

    def _label_values(self, label: Any) -> List[str]:
        arr = np.asarray(label)
        if arr.size == 1:
            return [str(arr.item())]
        return [self._normalise_scalar_label(item) for item in arr.reshape(-1)]

    def _build_index(self) -> None:
        prefix = self._subject_prefix()
        next_subject_index = len(self._subject_to_index)
        next_label_index = len(self._label_to_index)

        for subject_num in self._split_subject_numbers():
            subject_name = f"{prefix}-{subject_num}"
            subject_seq_dir = os.path.join(self.seq_dir, subject_name)
            subject_label_dir = os.path.join(self.labels_dir, subject_name)
            if not os.path.isdir(subject_seq_dir) or not os.path.isdir(subject_label_dir):
                raise FileNotFoundError(
                    f"Missing ISRUC subject folder for {subject_name!r} under {self.dataset_dir!r}."
                )

            seq_fnames = sorted(os.listdir(subject_seq_dir))
            label_fnames = sorted(os.listdir(subject_label_dir))
            if len(seq_fnames) != len(label_fnames):
                raise ValueError(
                    f"Mismatched seq/label counts for {subject_name!r}: "
                    f"{len(seq_fnames)} vs {len(label_fnames)}."
                )

            if subject_name not in self._subject_to_index:
                self._subject_to_index[subject_name] = next_subject_index
                next_subject_index += 1
            subject_idx = self._subject_to_index[subject_name]

            for seq_fname, label_fname in zip(seq_fnames, label_fnames):
                seq_path = os.path.join(subject_seq_dir, seq_fname)
                label_path = os.path.join(subject_label_dir, label_fname)
                raw_label = np.load(label_path, allow_pickle=True)
                label_values = self._label_values(raw_label)
                for label_str in label_values:
                    if label_str not in self._label_to_index:
                        self._label_to_index[label_str] = next_label_index
                        next_label_index += 1
                label_idx = self._label_to_index[label_values[0]]
                sample_key = f"{subject_name}/{seq_fname}"
                self._samples.append((seq_path, label_path, subject_name, sample_key))
                self._subject_cache.append(subject_idx)
                self._label_cache.append(label_idx)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        seq_path, label_path, subject_name, sample_key = self._samples[index]
        seq = np.load(seq_path, allow_pickle=True)
        label = np.load(label_path, allow_pickle=True)
        eeg = to_tensor(seq) / 100
        subject_idx = self._subject_to_index[subject_name]
        label_arr = np.asarray(label)
        if label_arr.size == 1:
            label_tensor = torch.tensor(self._label_to_index[self._normalise_scalar_label(label_arr.item())], dtype=torch.long)
        else:
            mapped = [self._label_to_index[self._normalise_scalar_label(item)] for item in label_arr.reshape(-1)]
            label_tensor = torch.tensor(mapped, dtype=torch.long).view(*label_arr.shape)
        return {
            "eeg": eeg,
            "subject": torch.tensor(subject_idx, dtype=torch.long),
            "label": label_tensor,
            "key": sample_key,
        }

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        eeg = torch.stack([item["eeg"] for item in batch], dim=0)
        subject = torch.stack([item["subject"] for item in batch], dim=0)
        label = torch.stack([item["label"] for item in batch], dim=0)
        keys = [item["key"] for item in batch]
        return {"eeg": eeg, "subject": subject, "label": label, "keys": keys}

    def subject_ids(self) -> List[int]:
        return list(self._subject_cache)

    def subject_index_to_id(self) -> Dict[int, str]:
        return {index: subject for subject, index in self._subject_to_index.items()}

    def label_index_to_value(self) -> Dict[int, str]:
        return {index: label for label, index in self._label_to_index.items()}


def _maybe_build_isruc_datasets(
    cfg: DisentangleDataConfig,
) -> Optional[Tuple["_ISRUCSequenceDataset", Optional["_ISRUCSequenceDataset"], Optional["_ISRUCSequenceDataset"]]]:
    dataset_name = str(cfg.dataset_name or "")
    lowered_name = dataset_name.lower()
    lowered_path = cfg.dataset_dir.lower()
    if "isruc" not in lowered_name and "sleep_05_isruc" not in lowered_path:
        return None

    variant = "ISRUC-S3" if ("s3" in lowered_name or "processed_average_s3" in lowered_path) else "ISRUC"
    train_dataset = _ISRUCSequenceDataset(cfg.dataset_dir, cfg.train_split, variant)

    val_dataset: Optional[_ISRUCSequenceDataset] = None
    if cfg.val_split:
        val_dataset = _ISRUCSequenceDataset(
            cfg.dataset_dir,
            cfg.val_split,
            variant,
            subject_mapping=train_dataset._subject_to_index,
            label_mapping=train_dataset._label_to_index,
        )

    test_dataset: Optional[_ISRUCSequenceDataset] = None
    if cfg.test_split:
        test_dataset = _ISRUCSequenceDataset(
            cfg.dataset_dir,
            cfg.test_split,
            variant,
            subject_mapping=train_dataset._subject_to_index,
            label_mapping=train_dataset._label_to_index,
        )

    return train_dataset, val_dataset, test_dataset


def build_datasets(
    cfg: DisentangleDataConfig,
) -> Tuple["DisentangleDataset", Optional["DisentangleDataset"], Optional["DisentangleDataset"]]:
    isruc_datasets = _maybe_build_isruc_datasets(cfg)
    if isruc_datasets is not None:
        return isruc_datasets

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
            shuffle=cfg.train_shuffle,
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
            shuffle=cfg.train_shuffle,
            generator=_torch_generator(cfg.seed) if cfg.train_shuffle else None,
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
