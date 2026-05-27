from __future__ import annotations

import io
import os
import pickle
import re
from dataclasses import dataclass
from typing import Optional, Tuple

import lmdb
import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset


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
    dataloader_seed: int = -1


class _CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def _safe_pickle_loads(payload: bytes):
    try:
        return pickle.loads(payload)
    except ModuleNotFoundError as exc:
        # Numpy module path changed across versions (`numpy._core` -> `numpy.core`).
        if not (exc.name and exc.name.startswith("numpy._core")):
            raise
        return _CompatUnpickler(io.BytesIO(payload)).load()


def _subject_from_key(key: str) -> int:
    # Supports key styles such as:
    # - "1_..." (SEED-V/SEED-IV)
    # - "sub000.pkl-..." (FACED)
    # - "A01E-..." (BCIC)
    m = re.search(r"\d+", key)
    if m is None:
        return 0
    return int(m.group(0))


class LmdbDisentangleDataset(Dataset):
    def __init__(self, dataset_dir: str, split: str):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.db = lmdb.open(dataset_dir, readonly=True, lock=False, readahead=True, meminit=False)
        with self.db.begin(write=False) as txn:
            payload = txn.get(b"__keys__")
        if payload is None:
            raise KeyError("Missing __keys__ in LMDB dataset")
        split_keys = _safe_pickle_loads(payload)
        if not isinstance(split_keys, dict):
            raise TypeError("Invalid __keys__ metadata type")
        if split not in split_keys:
            raise KeyError(f"Split {split!r} not found in LMDB keys; available={list(split_keys.keys())}")
        self.keys = list(split_keys[split])
        self._subjects = np.asarray([_subject_from_key(k) for k in self.keys], dtype=np.int64)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        with self.db.begin(write=False) as txn:
            payload = txn.get(key.encode())
        if payload is None:
            raise KeyError(f"Sample key not found in LMDB: {key}")
        pair = _safe_pickle_loads(payload)
        # Match original CodeBrain dataset loaders: inputs are scaled by 1/100.
        eeg = np.asarray(pair["sample"], dtype=np.float32) / 100.0
        label = int(pair["label"])
        subject = _subject_from_key(key)
        return {
            "eeg": torch.from_numpy(eeg),
            "label": torch.tensor(label, dtype=torch.long),
            "subject": torch.tensor(subject, dtype=torch.long),
        }

    @staticmethod
    def collate_fn(batch):
        eeg = torch.stack([item["eeg"] for item in batch], dim=0)
        label = torch.stack([item["label"] for item in batch], dim=0)
        subject = torch.stack([item["subject"] for item in batch], dim=0)
        return {"eeg": eeg, "label": label, "subject": subject}

    def subject_ids(self) -> np.ndarray:
        return self._subjects


class SleepSequenceDisentangleDataset(Dataset):
    def __init__(self, dataset_dir: str, split: str):
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
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
    def collate_fn(batch):
        eeg = torch.stack([item["eeg"] for item in batch], dim=0)
        label = torch.stack([item["label"] for item in batch], dim=0)
        subject = torch.stack([item["subject"] for item in batch], dim=0)
        return {"eeg": eeg, "label": label, "subject": subject}

    def subject_ids(self) -> np.ndarray:
        return self._subjects


class SubjectBalancedBatchSampler(BatchSampler):
    def __init__(
        self,
        subject_ids: np.ndarray,
        batch_size: int,
        samples_per_subject: int,
        generator: np.random.Generator,
    ):
        if samples_per_subject <= 0:
            raise ValueError("samples_per_subject must be > 0")
        if batch_size < samples_per_subject:
            raise ValueError("batch_size must be >= samples_per_subject")
        self.subject_ids = np.asarray(subject_ids, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.samples_per_subject = int(samples_per_subject)
        self.generator = generator

        self.indices_by_subject = {}
        for idx, sid in enumerate(self.subject_ids.tolist()):
            self.indices_by_subject.setdefault(int(sid), []).append(idx)
        self.unique_subjects = np.asarray(sorted(self.indices_by_subject.keys()), dtype=np.int64)
        self.batches_per_epoch = max(1, len(self.subject_ids) // self.batch_size)

    def __len__(self):
        return self.batches_per_epoch

    def __iter__(self):
        subjects_per_batch = max(1, self.batch_size // self.samples_per_subject)
        for _ in range(self.batches_per_epoch):
            chosen_subjects = self.generator.choice(
                self.unique_subjects,
                size=subjects_per_batch,
                replace=(subjects_per_batch > len(self.unique_subjects)),
            )
            batch_indices = []
            for sid in chosen_subjects.tolist():
                pool = self.indices_by_subject[int(sid)]
                take = self.samples_per_subject
                replace = len(pool) < take
                picks = self.generator.choice(pool, size=take, replace=replace)
                batch_indices.extend(int(x) for x in picks.tolist())

            while len(batch_indices) < self.batch_size:
                sid = int(self.generator.choice(self.unique_subjects))
                pool = self.indices_by_subject[sid]
                batch_indices.append(int(self.generator.choice(pool)))

            yield batch_indices[: self.batch_size]


def build_datasets(
    cfg: DisentangleDataConfig,
) -> Tuple[LmdbDisentangleDataset, Optional[LmdbDisentangleDataset], Optional[LmdbDisentangleDataset]]:
    if os.path.isdir(os.path.join(cfg.dataset_dir, "seq")) and os.path.isdir(os.path.join(cfg.dataset_dir, "labels")):
        train_dataset = SleepSequenceDisentangleDataset(cfg.dataset_dir, split=cfg.train_split)

        val_dataset: Optional[SleepSequenceDisentangleDataset] = None
        if cfg.val_split:
            val_dataset = SleepSequenceDisentangleDataset(cfg.dataset_dir, split=cfg.val_split)

        test_dataset: Optional[SleepSequenceDisentangleDataset] = None
        if cfg.test_split:
            test_dataset = SleepSequenceDisentangleDataset(cfg.dataset_dir, split=cfg.test_split)

        return train_dataset, val_dataset, test_dataset

    train_dataset = LmdbDisentangleDataset(cfg.dataset_dir, split=cfg.train_split)

    val_dataset: Optional[LmdbDisentangleDataset] = None
    if cfg.val_split:
        val_dataset = LmdbDisentangleDataset(cfg.dataset_dir, split=cfg.val_split)

    test_dataset: Optional[LmdbDisentangleDataset] = None
    if cfg.test_split:
        test_dataset = LmdbDisentangleDataset(cfg.dataset_dir, split=cfg.test_split)

    return train_dataset, val_dataset, test_dataset


def build_dataloaders(
    cfg: DisentangleDataConfig,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader], LmdbDisentangleDataset]:
    train_dataset, val_dataset, test_dataset = build_datasets(cfg)

    if cfg.subject_samples > 0:
        sampler_seed = int(cfg.dataloader_seed) if cfg.dataloader_seed >= 0 else int(cfg.seed)
        sampler = SubjectBalancedBatchSampler(
            subject_ids=train_dataset.subject_ids(),
            batch_size=cfg.batch_size,
            samples_per_subject=cfg.subject_samples,
            generator=np.random.default_rng(sampler_seed),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=sampler,
            num_workers=cfg.num_workers,
            pin_memory=True,
            collate_fn=train_dataset.collate_fn,
        )
    else:
        generator = None
        if cfg.dataloader_seed >= 0:
            generator = torch.Generator()
            generator.manual_seed(int(cfg.dataloader_seed))
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=bool(cfg.train_shuffle),
            num_workers=cfg.num_workers,
            pin_memory=True,
            # Match original CodeBrain loaders (no drop_last).
            drop_last=False,
            collate_fn=train_dataset.collate_fn,
            generator=generator,
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
        )

    return train_loader, val_loader, test_loader, train_dataset
