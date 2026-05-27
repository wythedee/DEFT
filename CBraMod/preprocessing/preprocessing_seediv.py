import os
import glob
import pickle
import lmdb
import numpy as np
import mne
mne.set_log_level('ERROR')
import scipy.io
import torch
import einops
from multiprocessing import Pool, cpu_count

# Paths
try:
    from preprocessing.path_utils import eeg_dataset_root, first_existing, processed_root
except ModuleNotFoundError:  # allow running as a script from this directory
    from path_utils import eeg_dataset_root, first_existing, processed_root

DATASET_ROOT = eeg_dataset_root()
PROCESSED_ROOT = processed_root()

SRC_FOLDER = first_existing(
    os.path.join(DATASET_ROOT, "EMO", "EMO_02_SEED_IV"),
    os.path.join(DATASET_ROOT, "EMO_02_SEED_IV"),
)
LMDB_PATH = os.path.join(PROCESSED_ROOT, "EMO_SEED_IV", "processed_average")
os.makedirs(os.path.dirname(LMDB_PATH), exist_ok=True)

# Channels and processing params
ORIGINAL_CH_NAMES = [
    'Fp1','Fpz','Fp2','AF3','AF4','F7','F5','F3','F1','Fz','F2','F4','F6','F8',
    'FT7','FC5','FC3','FC1','FCz','FC2','FC4','FC6','FT8','T7','C5','C3','C1',
    'Cz','C2','C4','C6','T8','TP7','CP5','CP3','CP1','CPz','CP2','CP4','CP6',
    'TP8','P7','P5','P3','P1','Pz','P2','P4','P6','P8','PO7','PO5','PO3','POz',
    'PO4','PO6','PO8','CB1','O1','Oz','O2','CB2']
CH_NAMES = [c for c in ORIGINAL_CH_NAMES if c not in ['CB1', 'CB2']]
L_FREQ = 0.3
H_FREQ = 50
RESAMPLE_RATE = 200
TIME_LENGTH = 4  # seconds
PATCH_SIZE = 200
SEQ_LEN = 4

SUBJECTS = ['1_', '2_', '3_', '4_', '5_', '6_', '7_', '8_', '9_', '10_', '11_',
            '12_', '13_', '14_', '15_']

# Subject split: 10/2/3
SPLIT = {
    'train': SUBJECTS[:10],
    'val': SUBJECTS[10:12],
    'test': SUBJECTS[12:]
}

session_labels = {
    1: [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],
    2: [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1],
    3: [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0],
}


def process_subject(sub_prefix: str):
    keys = []
    samples = []
    labels = []
    for session in [1, 2, 3]:
        fns = glob.glob(os.path.join(SRC_FOLDER, 'eeg_raw_data', str(session), f'{sub_prefix}*.mat'))
        if len(fns) == 0:
            continue
        fn = fns[0]
        try:
            data = scipy.io.loadmat(fn, squeeze_me=True)
        except Exception as e:
            print(f'Error loading {fn}: {e}')
            continue
        prefix = list(data.keys())[-1].split('_')[0]
        sess_label = session_labels[session]
        for i in range(1, 25):
            x = data[f'{prefix}_eeg{i}']
            raw = mne.io.RawArray(x, mne.create_info(ch_names=ORIGINAL_CH_NAMES, sfreq=200, ch_types='eeg'), verbose=False)
            raw.drop_channels(['CB1', 'CB2'])
            raw.filter(l_freq=L_FREQ, h_freq=H_FREQ, verbose=False)
            if abs(raw.info['sfreq'] - RESAMPLE_RATE) > 1e-6:
                raw.resample(RESAMPLE_RATE)
            arr = raw.get_data().astype(np.float32)
            # segment into non-overlapping 4s windows (800 samples)
            total_len = arr.shape[1]
            seg_len = TIME_LENGTH * RESAMPLE_RATE
            usable = (total_len // seg_len) * seg_len
            if usable == 0:
                continue
            arr = arr[:, :usable]
            arr = arr.reshape(arr.shape[0], -1, seg_len)  # C, N, 800
            arr = np.transpose(arr, (1, 0, 2))  # N, C, 800
            # patches 4x200
            arr = arr.reshape(arr.shape[0], arr.shape[1], SEQ_LEN, PATCH_SIZE)
            y = np.full((arr.shape[0],), sess_label[i - 1], dtype=np.uint8)
            for idx in range(arr.shape[0]):
                key = f'{sub_prefix}s{session}t{i}w{idx}'
                keys.append(key)
                samples.append(arr[idx])
                labels.append(int(y[idx]))
    return sub_prefix, keys, samples, labels


if __name__ == '__main__':
    db = lmdb.open(LMDB_PATH, map_size=50 * 1024 * 1024 * 1024)  # 50GB
    split_keys = {'train': [], 'val': [], 'test': []}

    with Pool(min(8, cpu_count())) as pool:
        results = pool.map(process_subject, SUBJECTS)

    for sub, keys, samples, labels in results:
        if sub in SPLIT['train']:
            split = 'train'
        elif sub in SPLIT['val']:
            split = 'val'
        else:
            split = 'test'
        for key, sample, label in zip(keys, samples, labels):
            with db.begin(write=True) as txn:
                txn.put(
                    key.encode(),
                    pickle.dumps({'sample': sample, 'label': label, 'subject': sub.rstrip('_')}),
                )
            split_keys[split].append(key)
        print(sub, len(keys))

    with db.begin(write=True) as txn:
        txn.put('__keys__'.encode(), pickle.dumps(split_keys))
    db.close()
    print('SEED-IV preprocessing finished.')
