import h5py
import scipy
from scipy import signal
import os
import lmdb
import pickle
import numpy as np
import pandas as pd

try:
    from preprocessing.path_utils import eeg_dataset_root, first_existing, processed_root
except ModuleNotFoundError:  # allow running as a script from this directory
    from path_utils import eeg_dataset_root, first_existing, processed_root


DATASET_ROOT = eeg_dataset_root()

# Allow overriding with env vars because SEED-VIG is often stored separately.
data_dir = first_existing(
    os.environ.get("SEEDVIG_RAW_DIR", ""),
    os.path.join(DATASET_ROOT, "EMO", "EMO_SEED_VIG", "mat", "Raw_Data"),
    os.path.join(DATASET_ROOT, "SEED-VIG", "mat", "Raw_Data"),
)
labels_dir = first_existing(
    os.environ.get("SEEDVIG_LABEL_DIR", ""),
    os.path.join(DATASET_ROOT, "EMO", "EMO_SEED_VIG", "mat", "perclos_labels"),
    os.path.join(DATASET_ROOT, "SEED-VIG", "mat", "perclos_labels"),
)

files = [file for file in os.listdir(data_dir)]
files = sorted(files)

files_dict = {
    'train': files[:15],
    'val': files[15:19],
    'test': files[19:23],
}

print(files_dict)

dataset = {
    'train': list(),
    'val': list(),
    'test': list(),
}

db_path = os.path.join(processed_root(), "EMO_SEED_VIG", "processed_average")
os.makedirs(os.path.dirname(db_path), exist_ok=True)
db = lmdb.open(db_path, map_size=6000000000)

for files_key in files_dict.keys():
    for file in files_dict[files_key]:
        eeg = scipy.io.loadmat(os.path.join(data_dir, file))['EEG'][0][0][0]
        labels = scipy.io.loadmat(os.path.join(labels_dir, file))['perclos']
        print(eeg.shape, labels.shape)
        eeg = eeg.reshape(885, 8, 200, 17)
        eeg = eeg.transpose(0, 3, 1, 2)
        labels = labels[:, 0]
        print(eeg.shape, labels.shape)
        for i, (sample, label) in enumerate(zip(eeg, labels)):
            sample_key = f'{file[:-4]}-{i}'
            print(sample_key)
            data_dict = {
                'sample': sample,
                'label': label,
                'subject': file[:-4],
            }
            txn = db.begin(write=True)
            txn.put(key=sample_key.encode(), value=pickle.dumps(data_dict))
            txn.commit()
            dataset[files_key].append(sample_key)

txn = db.begin(write=True)
txn.put(key='__keys__'.encode(), value=pickle.dumps(dataset))
txn.commit()
db.close()
