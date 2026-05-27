import lmdb
import pickle
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os

from utils.util import to_tensor


class CustomDataset(Dataset):
    def __init__(self, data_dir, mode='train'):
        super().__init__()
        self._data_dir = data_dir
        self._db_pid = None
        self.db = None
        tmp_db = lmdb.open(data_dir, readonly=True, lock=False, readahead=True, meminit=False)
        with tmp_db.begin(write=False) as txn:
            self.keys = pickle.loads(txn.get('__keys__'.encode()))[mode]
        tmp_db.close()
        if mode == 'train':
            random.shuffle(self.keys)

    def __len__(self):
        return len(self.keys)

    def _ensure_db_open(self):
        pid = os.getpid()
        if self.db is not None and self._db_pid == pid:
            return
        if self.db is not None:
            try:
                self.db.close()
            except Exception:
                pass
            self.db = None
        self.db = lmdb.open(self._data_dir, readonly=True, lock=False, readahead=True, meminit=False)
        self._db_pid = pid

    def __getitem__(self, idx):
        self._ensure_db_open()
        key = self.keys[idx]
        with self.db.begin(write=False) as txn:
            pair = pickle.loads(txn.get(key.encode()))
        sample = pair['sample']
        label = pair['label']
        return sample / 100.0, label

    @staticmethod
    def collate(batch):
        x = np.array([b[0] for b in batch])
        y = np.array([b[1] for b in batch])
        return to_tensor(x), to_tensor(y).long()


class LoadDataset:
    def __init__(self, params):
        self.params = params
        self.datasets_dir = params.datasets_dir

    def get_data_loader(self):
        train_set = CustomDataset(self.datasets_dir, mode='train')
        val_set = CustomDataset(self.datasets_dir, mode='val')
        test_set = CustomDataset(self.datasets_dir, mode='test')
        print(len(train_set), len(val_set), len(test_set))
        loaders = {
            'train': DataLoader(train_set, batch_size=self.params.batch_size, collate_fn=train_set.collate,
                                 shuffle=True, num_workers=self.params.num_workers),
            'val': DataLoader(val_set, batch_size=self.params.batch_size, collate_fn=val_set.collate,
                               shuffle=False, num_workers=self.params.num_workers),
            'test': DataLoader(test_set, batch_size=self.params.batch_size, collate_fn=test_set.collate,
                                shuffle=False, num_workers=self.params.num_workers),
        }
        return loaders
