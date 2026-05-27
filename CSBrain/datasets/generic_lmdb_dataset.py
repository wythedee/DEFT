import pickle

import lmdb
import numpy as np
from torch.utils.data import DataLoader, Dataset

from datasets.lmdb_utils import get_lmdb_env
from utils.util import to_tensor


class CustomDataset(Dataset):
    def __init__(self, data_dir, mode="train"):
        super().__init__()
        self.data_dir = data_dir
        self.db = None
        self.db_pid = None
        tmp_db = lmdb.open(data_dir, readonly=True, lock=False, readahead=True, meminit=False)
        with tmp_db.begin(write=False) as txn:
            self.keys = pickle.loads(txn.get("__keys__".encode()))[mode]
        tmp_db.close()

    def __len__(self):
        return len(self.keys)

    def _ensure_db_open(self):
        current_db = get_lmdb_env(self.data_dir)
        current_pid = id(current_db)
        if self.db is None or self.db_pid != current_pid:
            self.db = current_db
            self.db_pid = current_pid

    def __getitem__(self, idx):
        self._ensure_db_open()
        key = self.keys[idx]
        with self.db.begin(write=False) as txn:
            pair = pickle.loads(txn.get(key.encode()))
        data = pair["sample"]
        label = pair["label"]
        return data / 100.0, label

    @staticmethod
    def collate(batch):
        x_data = np.array([x[0] for x in batch])
        y_label = np.array([x[1] for x in batch])
        return to_tensor(x_data), to_tensor(y_label)


class LoadDataset(object):
    def __init__(self, params):
        self.params = params
        self.datasets_dir = params.datasets_dir

    def get_data_loader(self):
        train_set = CustomDataset(self.datasets_dir, mode="train")
        val_set = CustomDataset(self.datasets_dir, mode="val")
        test_set = CustomDataset(self.datasets_dir, mode="test")
        print(len(train_set), len(val_set), len(test_set))
        print(len(train_set) + len(val_set) + len(test_set))
        common_kwargs = {}
        if getattr(self.params, "num_workers", 0) > 0:
            common_kwargs["num_workers"] = self.params.num_workers
            common_kwargs["persistent_workers"] = True
        data_loader = {
            "train": DataLoader(
                train_set,
                batch_size=self.params.batch_size,
                collate_fn=train_set.collate,
                shuffle=True,
                **common_kwargs,
            ),
            "val": DataLoader(
                val_set,
                batch_size=self.params.batch_size,
                collate_fn=val_set.collate,
                shuffle=False,
                **common_kwargs,
            ),
            "test": DataLoader(
                test_set,
                batch_size=self.params.batch_size,
                collate_fn=test_set.collate,
                shuffle=False,
                **common_kwargs,
            ),
        }
        return data_loader
