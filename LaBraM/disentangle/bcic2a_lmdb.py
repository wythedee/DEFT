from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import lmdb
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset


# BCI Competition IV-2a (22 EEG channels). Must match the dataset's channel order.
# Names are uppercase to match LaBraM's `utils.standard_1020`.
BCIC2A_CH_NAMES: List[str] = [
    "FZ",
    "FC3",
    "FC1",
    "FCZ",
    "FC2",
    "FC4",
    "C5",
    "C3",
    "C1",
    "CZ",
    "C2",
    "C4",
    "C6",
    "CP3",
    "CP1",
    "CPZ",
    "CP2",
    "CP4",
    "P1",
    "PZ",
    "P2",
    "POZ",
]


@dataclass(frozen=True)
class BCIC2aLMDBSpec:
    lmdb_dir: str
    split: str  # train|val|test


class BCIC2aLMDBDataset(Dataset):
    """BCIC-IV-2a dataset backed by the shared LMDB format.

    LMDB format assumptions:
    - key '__keys__' stores a dict: {split: [key1, key2, ...]}
    - each sample is a dict with:
      - 'sample': np.ndarray shaped (22, 4, 200) (uV)
      - 'label': int in [0..3]

    We return X shaped (22, 800) so LaBraM's engine can reshape it back to
    (22, 4, 200) via einops.
    """

    def __init__(self, spec: BCIC2aLMDBSpec):
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
        x = np.asarray(pair["sample"], dtype=np.float32)  # (22, 4, 200)
        y = int(pair["label"])
        x = x.reshape(x.shape[0], -1)  # (22, 800)
        return torch.from_numpy(x), y


def build_bcic2a_datasets(lmdb_dir: str) -> Dict[str, BCIC2aLMDBDataset]:
    return {
        split: BCIC2aLMDBDataset(BCIC2aLMDBSpec(lmdb_dir=lmdb_dir, split=split))
        for split in ("train", "val", "test")
    }
