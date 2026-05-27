from __future__ import annotations

import os
import pickle
import warnings
from typing import Any, Dict, List, Optional

import lmdb
import numpy as np
import sys
import torch

if "numpy._core" not in sys.modules:
    import numpy.core as _np_core
    sys.modules["numpy._core"] = _np_core
    sys.modules["numpy._core.multiarray"] = np.core.multiarray
    sys.modules["numpy._core.numeric"] = np.core.numeric
from torch.utils.data import Dataset


class GenericLMDBDataset(Dataset):
    SUBJECT_FIELD_CANDIDATES = (
        "subject",
        "subject_id",
        "sid",
        "subjectId",
        "subjectID",
        "SubjectID",
        "participant",
        "participant_id",
        "subject_idx",
    )
    LABEL_FIELD_CANDIDATES = ("label", "target", "y", "Label", "class")

    def __init__(
        self,
        lmdb_dir: str,
        split: str = "train",
        *,
        scale: float = 100.0,
        subject_mapping: Optional[Dict[str, int]] = None,
        label_mapping: Optional[Dict[str, int]] = None,
    ) -> None:
        super().__init__()
        if not os.path.isdir(lmdb_dir):
            raise FileNotFoundError(f"LMDB directory does not exist: {lmdb_dir}")
        self.lmdb_dir = lmdb_dir
        self.split = split
        self.scale = float(scale)
        self._db_pid: Optional[int] = None
        self.db: Optional[lmdb.Environment] = None
        self._warned_subject_inference = False
        self._subject_to_index: Dict[str, int] = {} if subject_mapping is None else dict(subject_mapping)
        self._label_to_index: Dict[str, int] = {} if label_mapping is None else dict(label_mapping)
        self._subject_cache: list[int] = []
        self._label_cache: list[int] = []
        self._feature_shape: Optional[tuple[int, ...]] = None
        self._ensure_db_open()
        with self.db.begin(write=False) as txn:
            keys_blob = txn.get(b"__keys__")
            if keys_blob is None:
                raise ValueError("Missing '__keys__' entry in LMDB.")
            metadata = pickle.loads(keys_blob)
        if isinstance(metadata, dict):
            if split not in metadata:
                raise KeyError(f"Split '{split}' not found in __keys__. Available: {list(metadata.keys())}")
            self.keys = list(metadata[split])
        else:
            self.keys = list(metadata)
        self._build_metadata_caches()

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        self._ensure_db_open()
        key = self.keys[index]
        with self.db.begin(write=False) as txn:
            blob = txn.get(str(key).encode())
            if blob is None:
                raise KeyError(f"Missing LMDB key: {key}")
            record = pickle.loads(blob)
        sample, subject, label = self._extract_record(record, str(key))
        eeg = torch.as_tensor(np.asarray(sample, dtype=np.float32))
        if self.scale:
            eeg = eeg / self.scale
        subject_idx = self._subject_to_index[self._normalise(subject)]
        label_idx = self._label_to_index[self._normalise(label)]
        return {
            "eeg": eeg,
            "subject": torch.tensor(subject_idx, dtype=torch.long),
            "label": torch.tensor(label_idx, dtype=torch.long),
            "key": str(key),
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

    def _ensure_db_open(self) -> None:
        pid = os.getpid()
        if self.db is not None and self._db_pid == pid:
            return
        if self.db is not None:
            try:
                self.db.close()
            except Exception:
                pass
        self.db = lmdb.open(self.lmdb_dir, readonly=True, lock=False, readahead=True, meminit=False)
        self._db_pid = pid

    def _normalise(self, value: Any) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)

    def _infer_subject_from_key(self, key: str) -> Optional[str]:
        base = key.split("/")[-1].split(".")[0]
        for sep in ("-", "_"):
            if sep in base:
                candidate = base.split(sep, 1)[0].strip()
                if candidate:
                    if not self._warned_subject_inference:
                        warnings.warn(
                            "Subject field missing in LMDB records; falling back to parsing the key prefix.",
                            RuntimeWarning,
                        )
                        self._warned_subject_inference = True
                    return candidate
        if base and not self._warned_subject_inference:
            warnings.warn(
                "Subject field missing in LMDB records; falling back to the full LMDB key.",
                RuntimeWarning,
            )
            self._warned_subject_inference = True
        return base or None

    def _extract_record(self, record: Any, key: str) -> tuple[Any, Any, Any]:
        if isinstance(record, dict):
            sample = record.get("sample")
            if sample is None:
                sample = record.get("data")
            subject = None
            for field in self.SUBJECT_FIELD_CANDIDATES:
                if field in record:
                    subject = record[field]
                    break
            label = None
            for field in self.LABEL_FIELD_CANDIDATES:
                if field in record:
                    label = record[field]
                    break
        elif isinstance(record, (list, tuple)) and len(record) >= 2:
            sample, subject = record[0], record[1]
            label = record[2] if len(record) >= 3 else None
        else:
            raise ValueError("Unsupported LMDB record format.")

        if sample is None:
            raise ValueError(f"Missing sample for key '{key}'")
        if subject is None:
            subject = self._infer_subject_from_key(key)
        if subject is None:
            raise ValueError(f"Missing subject for key '{key}'")
        if label is None:
            raise ValueError(f"Missing label for key '{key}'")
        return sample, subject, label

    def _build_metadata_caches(self) -> None:
        self._ensure_db_open()
        next_subject = len(self._subject_to_index)
        next_label = len(self._label_to_index)
        with self.db.begin(write=False) as txn:
            for key in self.keys:
                blob = txn.get(str(key).encode())
                if blob is None:
                    raise KeyError(f"Missing LMDB key: {key}")
                record = pickle.loads(blob)
                sample, subject, label = self._extract_record(record, str(key))
                subject_s = self._normalise(subject)
                label_s = self._normalise(label)
                if subject_s not in self._subject_to_index:
                    self._subject_to_index[subject_s] = next_subject
                    next_subject += 1
                if label_s not in self._label_to_index:
                    self._label_to_index[label_s] = next_label
                    next_label += 1
                self._subject_cache.append(self._subject_to_index[subject_s])
                self._label_cache.append(self._label_to_index[label_s])
                if self._feature_shape is None:
                    self._feature_shape = tuple(np.asarray(sample).shape)
