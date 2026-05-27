from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class GenericSequenceFolderDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        *,
        scale: float = 100.0,
        subject_mapping: Optional[Dict[str, int]] = None,
        label_mapping: Optional[Dict[str, int]] = None,
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = str(split)
        self.scale = float(scale)
        self.seq_dir = self.root_dir / "seq"
        self.labels_dir = self.root_dir / "labels"
        if not self.seq_dir.is_dir() or not self.labels_dir.is_dir():
            raise FileNotFoundError(f"Sequence-folder dataset not found under: {self.root_dir}")
        self._subject_to_index: Dict[str, int] = {} if subject_mapping is None else dict(subject_mapping)
        self._label_to_index: Dict[str, int] = {} if label_mapping is None else dict(label_mapping)
        self._subject_cache: list[int] = []
        self._label_cache: list[int] = []
        self._feature_shape: Optional[tuple[int, ...]] = None
        self.entries: list[tuple[str, int, str, str]] = []
        self._build_entries()

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        seq_path, sample_index, subject, label = self.entries[index]
        seq = np.load(seq_path, mmap_mode="r")
        sample = np.asarray(seq[int(sample_index)], dtype=np.float32)
        if sample.ndim != 2 or sample.shape[-1] % 200 != 0:
            raise ValueError(f"Unexpected sequence sample shape: {sample.shape} from {seq_path}")
        sample = sample.reshape(sample.shape[0], sample.shape[1] // 200, 200)
        eeg = torch.as_tensor(sample, dtype=torch.float32)
        if self.scale:
            eeg = eeg / self.scale
        subject_idx = self._subject_to_index[subject]
        label_idx = self._label_to_index[label]
        return {
            "eeg": eeg,
            "subject": torch.tensor(subject_idx, dtype=torch.long),
            "label": torch.tensor(label_idx, dtype=torch.long),
            "key": f"{Path(seq_path).stem}:{sample_index}",
        }

    @property
    def feature_shape(self) -> tuple[int, ...]:
        if self._feature_shape is None:
            raise RuntimeError("feature_shape unavailable")
        return self._feature_shape

    @property
    def num_labels(self) -> int:
        return len(self._label_to_index)

    def subject_ids(self) -> list[int]:
        return list(self._subject_cache)

    def subject_index_to_id(self) -> Dict[int, str]:
        return {index: subject for subject, index in self._subject_to_index.items()}

    def label_index_to_value(self) -> Dict[int, str]:
        return {index: label for label, index in self._label_to_index.items()}

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "eeg": torch.stack([item["eeg"] for item in batch], dim=0),
            "subject": torch.stack([item["subject"] for item in batch], dim=0),
            "label": torch.stack([item["label"] for item in batch], dim=0),
            "key": [item["key"] for item in batch],
        }

    def _build_entries(self) -> None:
        subject_dirs = sorted(path for path in self.seq_dir.iterdir() if path.is_dir())
        if not subject_dirs:
            raise FileNotFoundError(f"No subject dirs found under {self.seq_dir}")
        selected_subjects = self._select_split_subjects([path.name for path in subject_dirs])
        next_subject = len(self._subject_to_index)
        next_label = len(self._label_to_index)
        for subject_name in selected_subjects:
            seq_subject_dir = self.seq_dir / subject_name
            label_subject_dir = self.labels_dir / subject_name
            seq_files = sorted(seq_subject_dir.glob("*.npy"))
            label_files = sorted(label_subject_dir.glob("*.npy"))
            if len(seq_files) != len(label_files):
                raise ValueError(f"Mismatched seq/label file count for {subject_name}")
            if subject_name not in self._subject_to_index:
                self._subject_to_index[subject_name] = next_subject
                next_subject += 1
            for seq_path, label_path in zip(seq_files, label_files):
                labels = np.load(label_path)
                labels = np.asarray(labels).reshape(-1)
                if self._feature_shape is None:
                    seq = np.load(seq_path, mmap_mode="r")
                    sample = np.asarray(seq[0])
                    if sample.ndim != 2 or sample.shape[-1] % 200 != 0:
                        raise ValueError(f"Unexpected sequence sample shape: {sample.shape} from {seq_path}")
                    self._feature_shape = (sample.shape[0], sample.shape[1] // 200, 200)
                for sample_index, label in enumerate(labels.tolist()):
                    label_name = str(int(label))
                    if label_name not in self._label_to_index:
                        self._label_to_index[label_name] = next_label
                        next_label += 1
                    self.entries.append((str(seq_path), int(sample_index), subject_name, label_name))
                    self._subject_cache.append(self._subject_to_index[subject_name])
                    self._label_cache.append(self._label_to_index[label_name])
        if self._feature_shape is None:
            raise RuntimeError("No sequence samples found")

    def _select_split_subjects(self, subject_names: list[str]) -> list[str]:
        if all(name.startswith("ISRUC-group1-") for name in subject_names):
            train_cut, val_cut = 80, 90
        elif all(name.startswith("ISRUC-group3-") for name in subject_names):
            train_cut, val_cut = 6, 8
        else:
            total = len(subject_names)
            train_cut = max(1, int(round(total * 0.8)))
            val_cut = max(train_cut + 1, int(round(total * 0.9)))
        if self.split == "train":
            return subject_names[:train_cut]
        if self.split == "val":
            return subject_names[train_cut:val_cut]
        if self.split == "test":
            return subject_names[val_cut:]
        raise KeyError(f"Unsupported split: {self.split}")
