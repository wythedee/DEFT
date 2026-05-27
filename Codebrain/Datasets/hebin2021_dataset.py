import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils.util import to_tensor
from Datasets.lmdb_pickle import loads_lmdb_pickle


class CustomDataset(Dataset):
    def __init__(self, data_dir, mode='train'):
        super().__init__()
        self.db = lmdb.open(data_dir, readonly=True, lock=False, readahead=True, meminit=False)
        with self.db.begin(write=False) as txn:
            self.keys = loads_lmdb_pickle(txn.get('__keys__'.encode()))[mode]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        with self.db.begin(write=False) as txn:
            pair = loads_lmdb_pickle(txn.get(key.encode()))
        data, label = pair['sample'], pair['label']
        return data / 100, label

    @staticmethod
    def collate(batch):
        x = np.array([b[0] for b in batch])
        y = np.array([b[1] for b in batch])
        return to_tensor(x), to_tensor(y).long()


class LoadDataset:
    """
    params.datasets_dir should point to processed_average directory of either LR or UD dataset.
    """
    def __init__(self, params):
        self.params = params
        self.datasets_dir = params.datasets_dir

    def get_data_loader(self):
        train_set = CustomDataset(self.datasets_dir, 'train')
        val_set = CustomDataset(self.datasets_dir, 'val')
        test_set = CustomDataset(self.datasets_dir, 'test')
        print(len(train_set), len(val_set), len(test_set))
        loaders = {
            'train': DataLoader(train_set, batch_size=self.params.batch_size, collate_fn=train_set.collate, shuffle=True),
            'val': DataLoader(val_set, batch_size=self.params.batch_size, collate_fn=val_set.collate, shuffle=False),
            'test': DataLoader(test_set, batch_size=self.params.batch_size, collate_fn=test_set.collate, shuffle=False),
        }
        return loaders
