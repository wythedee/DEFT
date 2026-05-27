import pickle
import lmdb
from torch.utils.data import Dataset
from Utils.util import to_tensor
import numpy as np

class PretrainingDataset(Dataset):
    def __init__(
            self,
            dataset_dir
    ):
        super(PretrainingDataset, self).__init__()
        self.db = lmdb.open(dataset_dir, readonly=True, lock=False, readahead=True, meminit=False)
        with self.db.begin(write=False) as txn:
            all_keys = pickle.loads(txn.get('__keys__'.encode()))
            self.keys = [key for key in all_keys if key != '__keys__']

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
            key = self.keys[idx]
            with self.db.begin(write=False) as txn:
                value = txn.get(key.encode())
                if value is None:
                    raise KeyError(f"Key {key} not found in the database.")
                patch = pickle.loads(value)
            patch = np.array(patch)
            patch = to_tensor(patch)
            return patch