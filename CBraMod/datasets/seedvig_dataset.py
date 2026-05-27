import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils.util import to_tensor
import os
import random
import lmdb
import pickle
from datasets.lmdb_utils import LazyLMDBDatasetMixin


class CustomDataset(LazyLMDBDatasetMixin, Dataset):
    def __init__(
            self,
            data_dir,
            mode='train',
    ):
        super(CustomDataset, self).__init__()
        self._init_lmdb_dataset(data_dir, mode)

    def __len__(self):
        return len((self.keys))

    def __getitem__(self, idx):
        self._ensure_db_open()
        key = self.keys[idx]
        with self.db.begin(write=False) as txn:
            pair = pickle.loads(txn.get(key.encode()))
        data = pair['sample']
        label = pair['label']
        # print(key)
        # print(data.shape)
        return data/100, label

    def collate(self, batch):
        x_data = np.array([x[0] for x in batch])
        y_label = np.array([x[1] for x in batch])
        return to_tensor(x_data), to_tensor(y_label)


class LoadDataset(object):
    def __init__(self, params):
        self.params = params
        self.datasets_dir = params.datasets_dir

    def get_data_loader(self):
        train_set = CustomDataset(self.datasets_dir, mode='train')
        val_set = CustomDataset(self.datasets_dir, mode='val')
        test_set = CustomDataset(self.datasets_dir, mode='test')
        print(len(train_set), len(val_set), len(test_set))
        data_loader = {
            'train': DataLoader(
                train_set,
                batch_size=self.params.batch_size,
                collate_fn=train_set.collate,
                shuffle=True,
            ),
            'val': DataLoader(
                val_set,
                batch_size=self.params.batch_size,
                collate_fn=val_set.collate,
                shuffle=False,
            ),
            'test': DataLoader(
                test_set,
                batch_size=self.params.batch_size,
                collate_fn=test_set.collate,
                shuffle=False,
            ),
        }
        return data_loader
