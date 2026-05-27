import lmdb
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils.util import to_tensor
from datasets.lmdb_utils import LazyLMDBDatasetMixin


class CustomDataset(LazyLMDBDatasetMixin, Dataset):
    def __init__(self, data_dir, mode='train'):
        super().__init__()
        self._init_lmdb_dataset(data_dir, mode)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        self._ensure_db_open()
        key = self.keys[idx]
        with self.db.begin(write=False) as txn:
            pair = pickle.loads(txn.get(key.encode()))
        data, label = pair['sample'], pair['label']
        return data / 100, label

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
        train_set = CustomDataset(self.datasets_dir, 'train')
        val_set = CustomDataset(self.datasets_dir, 'val')
        test_set = CustomDataset(self.datasets_dir, 'test')
        print(len(train_set), len(val_set), len(test_set))
        return {
            'train': DataLoader(train_set, batch_size=self.params.batch_size, collate_fn=train_set.collate, shuffle=True),
            'val': DataLoader(val_set, batch_size=self.params.batch_size, collate_fn=val_set.collate, shuffle=False),
            'test': DataLoader(test_set, batch_size=self.params.batch_size, collate_fn=test_set.collate, shuffle=False),
        }
