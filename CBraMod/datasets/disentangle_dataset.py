"""Dataset helper for the disentanglement fine-tuning task.

The dataset expects an LMDB store where each value is a pickled dictionary
containing the fields:

- ``sample`` or ``data``: the EEG tensor with shape (C, P, S).
- ``subject`` / ``subject_id`` / ``sid``: the subject identifier used for the
  static contrastive loss.

The ``__keys__`` entry in the LMDB must either be a list of keys or a mapping
from split names (``train``, ``val``, ``test``) to lists of keys.
"""

from __future__ import annotations

import os
import pickle
import warnings
from typing import Any, Dict, List, Optional, Sequence

import lmdb
import torch
from torch.utils.data import Dataset

from utils.util import to_tensor


class DisentangleDataset(Dataset):
    """LMDB-backed dataset that serves EEG samples and subject identifiers."""

    _ENV_CACHE: Dict[tuple[int, str], lmdb.Environment] = {}
    _ENV_REFCOUNT: Dict[tuple[int, str], int] = {}

    SUPPORTED_DATASETS: Dict[str, Sequence[str]] = {
        "ATTEN_GoNogo": ("ATTEN_GoNogo",),
        "BCIC_IV_2a": ("MI_BCI_IV_2a", "BCIC_IV_2a"),
        "BCIC_Track4": ("CS_04_BCIC_Track3", "CS-BCIC-Track4"),
        "EMO_SEED": ("EMO_SEED", "SEED"),
        "EMO_SEED_IV": ("EMO_SEED_IV", "SEED_IV"),
        "EMO_SEED_V": ("EMO_SEED_V", "SEED_V"),
        "EMO_SEED_VIG": ("EMO_SEED_VIG", "SEED_VIG", "SEED-VIG"),
        "EMO_FACED": ("EMO_FACED", "FACED", "EMO-FACED"),
        "HeBin2021_LR": ("MI_HeBin2021_LR",),
        "HeBin2021_UD": ("MI_HeBin2021_UD",),
        "MI_08_Track1_Few_shot": ("MI_08_Track1_Few_shot",),
        "MI_Cho2017": ("MI_Cho2017", "MI-Cho2017", "Cho2017"),
        "MI_KoreaU": ("MI_KoreaU",),
        "MI_Mumtaz": ("MDD_Mumtaz", "Mumtaz"),
        "MI_PhysioNet": ("MI_PhysioNet", "PhysioNet"),
        "MI_Schirrmeister2017": ("MI_Schirrmeister2017", "Schirrmeister2017"),
        "MI_SHU": ("MI_SHU", "SHU"),
        "ISRUC": ("ISRUC", "ISRUC-S3", "SLEEP_05_isruc"),
        "SEEDV_Cross": ("EMO_SEED_V", "SEEDV_cross"),
        "Speech": ("Imagined speech", "Speech"),
        "STR_MentalArithmetic": ("STR_MentalArithmetic", "MentalArithmetic"),
        "TUH_TUEV_Events": ("TUH_TUEV_Events", "TUH_TUEV", "TUEV_Events"),
    }

    SUBJECT_FIELD_CANDIDATES: Sequence[str] = (
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
    LABEL_FIELD_CANDIDATES: Sequence[str] = (
        "label",
        "target",
        "y",
        "Label",
        "class",
    )
    SUBJECT_FALLBACK_DATASETS = {"MI_KoreaU", "HeBin2021_LR", "HeBin2021_UD"}

    def __init__(
        self,
        lmdb_path: str,
        split: str = "train",
        scale: float = 100.0,
        dataset_name: Optional[str] = None,
        keys: Optional[List[str]] = None,
        subject_mapping: Optional[Dict[str, int]] = None,
        label_mapping: Optional[Dict[str, int]] = None,
    ) -> None:
        super().__init__()
        if not os.path.isdir(lmdb_path):
            raise FileNotFoundError(f"Dataset directory does not exist: {lmdb_path}")
        self._lmdb_path = lmdb_path
        self._db_pid: Optional[int] = None
        self._db_cache_key: Optional[tuple[int, str]] = None
        self.db: Optional[lmdb.Environment] = None
        self.dataset_name = self._resolve_dataset_name(dataset_name, lmdb_path)
        self._ensure_db_open()
        if keys is not None:
            self.keys = list(keys)
        else:
            with self.db.begin(write=False) as txn:
                keys_blob = txn.get("__keys__".encode())
                if keys_blob is None:
                    raise ValueError("Missing '__keys__' entry in LMDB.")
                metadata = pickle.loads(keys_blob)
            if isinstance(metadata, dict):
                if split not in metadata:
                    raise KeyError(f"Split '{split}' not found in '__keys__'. Available: {list(metadata.keys())}")
                self.keys = metadata[split]
            else:
                self.keys = metadata
        self.scale = scale
        self.split = split
        self._subject_cache: Optional[List[int]] = None
        self._subject_strings: Optional[List[str]] = None
        self._subject_to_index: Dict[str, int] = {} if subject_mapping is None else dict(subject_mapping)
        self._label_cache: Optional[List[int]] = None
        self._label_strings: Optional[List[str]] = None
        self._label_to_index: Dict[str, int] = {} if label_mapping is None else dict(label_mapping)
        self._warned_subject_inference = False
        self._build_metadata_caches()
        # Drop the parent-process handle after metadata bootstrap. Workers and
        # later accesses reopen lazily in their own PID.
        self._release_db_handle()

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        self._ensure_db_open()
        key = self.keys[index]
        with self.db.begin(write=False) as txn:
            record_blob = txn.get(key.encode())
            if record_blob is None:
                raise KeyError(f"Key '{key}' is missing from the LMDB store.")
            record = pickle.loads(record_blob)

        data, subject, label = self._extract_sample_subject_label(record, key)

        if data is None:
            raise ValueError(f"EEG sample is missing for key '{key}'.")
        if subject is None:
            raise ValueError(
                "Subject identifier is required for the contrastive objective. "
                f"Please ensure the record for key '{key}' includes one of ['subject', 'subject_id', 'sid']."
            )

        eeg = to_tensor(data)
        if self.scale:
            eeg = eeg / self.scale
        subject_idx = self._subject_to_index[self._normalise_subject(subject)]
        label_idx = self._label_to_index[self._normalise_label(label)]
        subject_tensor = torch.tensor(subject_idx, dtype=torch.long)
        label_tensor = torch.tensor(label_idx, dtype=torch.long)
        return {"eeg": eeg, "subject": subject_tensor, "label": label_tensor, "key": key}

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        eeg = torch.stack([item["eeg"] for item in batch], dim=0)
        subject = torch.stack([item["subject"] for item in batch], dim=0)
        label = torch.stack([item["label"] for item in batch], dim=0)
        keys = [item["key"] for item in batch]
        return {"eeg": eeg, "subject": subject, "label": label, "keys": keys}

    def subject_ids(self) -> List[int]:
        """Return cached subject identifiers aligned with dataset indexing."""

        if self._subject_cache is None:
            raise RuntimeError("Subject cache was not initialised correctly.")
        return self._subject_cache

    def subject_index_to_id(self) -> Dict[int, str]:
        """Mapping between numeric subject indices and their raw identifiers."""

        return {index: subject for subject, index in self._subject_to_index.items()}

    def label_ids(self) -> List[int]:
        """Return cached class-label identifiers aligned with dataset indexing."""

        if self._label_cache is None:
            raise RuntimeError("Label cache was not initialised correctly.")
        return self._label_cache

    def label_index_to_value(self) -> Dict[int, str]:
        """Mapping between numeric class indices and raw stored labels."""

        return {index: label for label, index in self._label_to_index.items()}

    def key_subject_strings(self) -> List[str]:
        """Return the raw subject identifier for each key in order."""

        if self._subject_strings is None:
            raise RuntimeError("Subject cache was not initialised correctly.")
        return list(self._subject_strings)

    def close(self) -> None:
        self._release_db_handle()
        if hasattr(self, "_subject_cache"):
            self._subject_cache = None
        if hasattr(self, "_subject_strings"):
            self._subject_strings = None
        subject_map = getattr(self, "_subject_to_index", None)
        if subject_map is not None:
            subject_map.clear()
        if hasattr(self, "_label_cache"):
            self._label_cache = None
        if hasattr(self, "_label_strings"):
            self._label_strings = None
        label_map = getattr(self, "_label_to_index", None)
        if label_map is not None:
            label_map.clear()

    def __del__(self) -> None:  # pragma: no cover - defensive clean-up
        self.close()

    @classmethod
    def reset_all_db_handles_for_worker(cls) -> None:
        for env in list(cls._ENV_CACHE.values()):
            try:
                env.close()
            except Exception:
                pass
        cls._ENV_CACHE.clear()
        cls._ENV_REFCOUNT.clear()

    def reset_db_state_for_worker(self) -> None:
        type(self).reset_all_db_handles_for_worker()
        self.db = None
        self._db_pid = None
        self._db_cache_key = None

    def _ensure_db_open(self) -> None:
        """Ensure the LMDB env is opened in the current process.

        Torch DataLoader may fork worker processes. Reusing an LMDB environment
        opened in the parent process inside forked workers can lead to crashes
        (e.g. segfaults). We therefore reopen the environment lazily per PID.
        """

        pid = os.getpid()
        if self.db is not None and self._db_pid == pid:
            return
        cache_key = (pid, self._lmdb_path)
        if self._db_cache_key is not None and self._db_cache_key != cache_key:
            self._release_db_handle()
        env = self._ENV_CACHE.get(cache_key)
        if env is None:
            env = lmdb.open(self._lmdb_path, readonly=True, lock=False, readahead=True, meminit=False)
            self._ENV_CACHE[cache_key] = env
            self._ENV_REFCOUNT[cache_key] = 0
        self._ENV_REFCOUNT[cache_key] = self._ENV_REFCOUNT.get(cache_key, 0) + 1
        self.db = env
        self._db_pid = pid
        self._db_cache_key = cache_key

    def _release_db_handle(self) -> None:
        cache_key = getattr(self, "_db_cache_key", None)
        if cache_key is not None:
            refcount = self._ENV_REFCOUNT.get(cache_key, 0)
            if refcount <= 1:
                env = self._ENV_CACHE.pop(cache_key, None)
                self._ENV_REFCOUNT.pop(cache_key, None)
                if env is not None:
                    try:
                        env.close()
                    except Exception:
                        pass
            else:
                self._ENV_REFCOUNT[cache_key] = refcount - 1
        self.db = None
        self._db_pid = None
        self._db_cache_key = None

    def _resolve_dataset_name(self, dataset_name: Optional[str], lmdb_path: str) -> str:
        if dataset_name:
            for canonical, aliases in self.SUPPORTED_DATASETS.items():
                if dataset_name == canonical or dataset_name in aliases:
                    return canonical
            raise ValueError(
                f"Unsupported dataset '{dataset_name}'. Supported options: {sorted(self.SUPPORTED_DATASETS)}"
            )

        lowered_path = lmdb_path.lower()
        for canonical, aliases in self.SUPPORTED_DATASETS.items():
            if any(alias.lower() in lowered_path for alias in aliases):
                return canonical
        raise ValueError(
            "Could not infer dataset name from path. Please provide 'dataset_name' explicitly. "
            f"Supported datasets: {sorted(self.SUPPORTED_DATASETS)}"
        )

    def _extract_sample_subject_label(self, record: Any, key: str) -> tuple[Any, Any, Any]:
        if isinstance(record, dict):
            data = record.get("sample")
            if data is None:
                data = record.get("data")
            subject = None
            for candidate in self.SUBJECT_FIELD_CANDIDATES:
                if candidate in record:
                    subject = record[candidate]
                    break
            label = None
            for candidate in self.LABEL_FIELD_CANDIDATES:
                if candidate in record:
                    label = record[candidate]
                    break
        elif isinstance(record, (list, tuple)) and len(record) >= 2:
            data, subject = record[0], record[1]
            label = record[2] if len(record) >= 3 else None
        else:
            raise ValueError("Unsupported record format encountered in LMDB entry.")

        if subject is None:
            subject = self._infer_subject_from_key(key)
        if subject is None:
            raise ValueError(
                "Subject identifier is required for the contrastive objective. "
                f"Please ensure the record for key '{key}' includes one of ['subject', 'subject_id', 'sid']."
            )
        if label is None:
            raise ValueError(
                "Class label is required for linear probing. "
                f"Please ensure the record for key '{key}' includes one of ['label', 'target', 'y']."
            )
        return data, subject, label

    def _normalise_subject(self, subject: Any) -> str:
        if isinstance(subject, bytes):
            return subject.decode("utf-8")
        return str(subject)

    def _normalise_label(self, label: Any) -> str:
        if isinstance(label, bytes):
            return label.decode("utf-8")
        return str(label)

    def _infer_subject_from_key(self, key: str) -> Optional[str]:
        if self.dataset_name not in self.SUBJECT_FALLBACK_DATASETS:
            return None
        base = key.split("/")[-1]
        base = base.split(".")[0]
        separators = ("-", "_")
        for sep in separators:
            if sep in base:
                candidate = base.split(sep, 1)[0].strip()
                if candidate:
                    if not self._warned_subject_inference:
                        warnings.warn(
                            f"Subject field missing in records for dataset '{self.dataset_name}'. "
                            "Falling back to parsing the LMDB key prefix.",
                            RuntimeWarning,
                        )
                        self._warned_subject_inference = True
                    return candidate
        candidate = base.strip() or None
        if candidate and not self._warned_subject_inference:
            warnings.warn(
                f"Subject field missing in records for dataset '{self.dataset_name}'. "
                "Using the LMDB key as subject identifier.",
                RuntimeWarning,
            )
            self._warned_subject_inference = True
        return candidate

    def _build_metadata_caches(self) -> None:
        self._ensure_db_open()
        subjects_numeric: List[int] = []
        subject_strings: List[str] = []
        labels_numeric: List[int] = []
        label_strings: List[str] = []
        next_subject_index = len(self._subject_to_index)
        next_label_index = len(self._label_to_index)
        with self.db.begin(write=False) as txn:
            for key in self.keys:
                record_blob = txn.get(key.encode())
                if record_blob is None:
                    raise KeyError(f"Key '{key}' is missing from the LMDB store.")
                record = pickle.loads(record_blob)
                _, subject, label = self._extract_sample_subject_label(record, key)
                subject_str = self._normalise_subject(subject)
                label_str = self._normalise_label(label)

                if subject_str not in self._subject_to_index:
                    self._subject_to_index[subject_str] = next_subject_index
                    next_subject_index += 1
                if label_str not in self._label_to_index:
                    self._label_to_index[label_str] = next_label_index
                    next_label_index += 1

                subjects_numeric.append(self._subject_to_index[subject_str])
                subject_strings.append(subject_str)
                labels_numeric.append(self._label_to_index[label_str])
                label_strings.append(label_str)

        self._subject_cache = subjects_numeric
        self._subject_strings = subject_strings
        self._label_cache = labels_numeric
        self._label_strings = label_strings
