import scipy
from scipy import signal
import os
import lmdb
import pickle

try:
    from preprocessing.path_utils import eeg_dataset_root, first_existing, processed_root
except ModuleNotFoundError:  # allow running as a script from this directory
    from path_utils import eeg_dataset_root, first_existing, processed_root

DATASET_ROOT = eeg_dataset_root()
PROCESSED_ROOT = processed_root()

root_dir = first_existing(
    os.path.join(DATASET_ROOT, "MI", "MI_SHU", "files"),
    os.path.join(DATASET_ROOT, "MI_SHU", "files"),
)
files = [file for file in os.listdir(root_dir)]
files = sorted(files)
# print(files)

files_dict = {
    'train':files[:75],
    'val':files[75:100],
    'test':files[100:],
}

dataset = {
    'train': list(),
    'val': list(),
    'test': list(),
}
db_path = os.path.join(PROCESSED_ROOT, "MI_SHU", "processed_average")
os.makedirs(os.path.dirname(db_path), exist_ok=True)
db = lmdb.open(db_path, map_size=5<<30)
for files_key in files_dict.keys():
    for file in files_dict[files_key]:
        data = scipy.io.loadmat(os.path.join(root_dir, file))
        eeg = data['data']
        labels = data['labels'][0]
        bz, ch_num, points = eeg.shape
        print(eeg.shape)
        eeg_resample = signal.resample(eeg, 800, axis=2)
        eeg_ = eeg_resample.reshape(bz, ch_num, 4, 200)
        print(eeg_.shape, labels.shape)
        for i, (sample, label) in enumerate(zip(eeg_, labels)):
            sample_key = f'{file[:-4]}-{i}'
            # print(sample_key)
            data_dict = {
                'sample': sample,
                'label': label - 1,
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
