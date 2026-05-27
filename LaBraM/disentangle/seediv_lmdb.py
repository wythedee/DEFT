from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import lmdb
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset


# SEED-IV uses 60 EEG channels in this preprocessed LMDB.
# Names are uppercase to match LaBraM's `utils.standard_1020`.
SEED_IV_CH_NAMES: List[str] = [
    "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4",
    "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "T7",
    "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3", "CP1",
    "CPZ", "CP2", "CP4", "CP6", "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4",
    "P6", "P8", "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "O1", "OZ", "O2",
]


@dataclass(frozen=True)
class SEEDIVLMDBSpec:
    lmdb_dir: str
    split: str  # train|val|test


class SEEDIVLMDBDataset(Dataset):
    """SEED-IV dataset backed by the CodeBrain/disentangle LMDB format."""

    def __init__(self, spec: SEEDIVLMDBSpec):
        self.spec = spec
        self._env = lmdb.open(
            spec.lmdb_dir,
            readonly=True,
            lock=False,
            readahead=True,
            meminit=False,
        )
        with self._env.begin(write=False) as txn:
            keys_obj = pickle.loads(txn.get(b"__keys__"))
        self._keys = list(keys_obj[spec.split])

    def __len__(self) -> int:
        return len(self._keys)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        k = self._keys[idx]
        with self._env.begin(write=False) as txn:
            pair = pickle.loads(txn.get(k.encode()))
        x = np.asarray(pair["sample"], dtype=np.float32)  # (60, A, 200)
        y = int(pair["label"])
        x = x.reshape(x.shape[0], -1)  # (60, A*200)
        return torch.from_numpy(x), y


def build_seediv_datasets(lmdb_dir: str) -> Dict[str, SEEDIVLMDBDataset]:
    return {
        split: SEEDIVLMDBDataset(SEEDIVLMDBSpec(lmdb_dir=lmdb_dir, split=split))
        for split in ("train", "val", "test")
    }
