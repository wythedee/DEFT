import scipy
from scipy import signal
import os
import lmdb
import pickle
import mne

try:
    from preprocessing.path_utils import eeg_dataset_root, first_existing, processed_root
except ModuleNotFoundError:  # allow running as a script from this directory
    from path_utils import eeg_dataset_root, first_existing, processed_root

DATASET_ROOT = eeg_dataset_root()
PROCESSED_ROOT = processed_root()

root_dir = first_existing(
    os.path.join(DATASET_ROOT, "STR", "STR_01_MentalArithmetic", "eeg-during-mental-arithmetic-tasks-1.0.0"),
    os.path.join(DATASET_ROOT, "STR_01_MentalArithmetic", "eeg-during-mental-arithmetic-tasks-1.0.0"),
)
files = [file for file in os.listdir(root_dir) if file.endswith('.edf')]
files = sorted(files)
print(files)

files_dict = {
    'train':files[:52],
    'val':files[52:62],
    'test':files[62:],
}
print(files_dict)
dataset = {
    'train': list(),
    'val': list(),
    'test': list(),
}


selected_channels = ['EEG Fp1', 'EEG Fp2', 'EEG F3', 'EEG F4', 'EEG F7', 'EEG F8', 'EEG T3', 'EEG T4',
                     'EEG C3', 'EEG C4', 'EEG T5', 'EEG T6', 'EEG P3', 'EEG P4', 'EEG O1', 'EEG O2',
                     'EEG Fz', 'EEG Cz', 'EEG Pz', 'EEG A2-A1']



db_path = os.path.join(PROCESSED_ROOT, "STR_MentalArithmetic", "processed_average")
os.makedirs(os.path.dirname(db_path), exist_ok=True)
db = lmdb.open(db_path, map_size=1000000000)
for files_key in files_dict.keys():
    for file in files_dict[files_key]:
        raw = mne.io.read_raw_edf(os.path.join(root_dir, file), preload=True)
        raw.pick(selected_channels)
        raw.reorder_channels(selected_channels)
        raw.resample(200)

        eeg = raw.get_data(units='uV')
        chs, points = eeg.shape
        a = points % (5 * 200)
        if a != 0:
            eeg = eeg[:, :-a]
        eeg = eeg.reshape(20, -1, 5, 200).transpose(1, 0, 2, 3)
        label = int(file[-5])

        for i, sample in enumerate(eeg):
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
