import os
import pickle
import lmdb
import numpy as np
import mne
import multiprocessing as mp

try:
    from preprocessing.path_utils import eeg_dataset_root, first_existing, processed_root
except ModuleNotFoundError:  # allow running as a script from this directory
    from path_utils import eeg_dataset_root, first_existing, processed_root

DATASET_ROOT = eeg_dataset_root()
PROCESSED_ROOT = processed_root()

ROOT_DIR = first_existing(
    os.path.join(DATASET_ROOT, "MI", "MI_Weibo2014"),
    os.path.join(DATASET_ROOT, "MI_Weibo2014"),
)
OUTPUT_LMDB = os.path.join(PROCESSED_ROOT, "MI_05_Weibo2014", "processed_average")
TARGET_SFREQ = 200
NUM_PATCHES = 4
PATCH_LEN = 200

SUBJECTS = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10']

CH_NAMES = [
    'Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6',
    'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5',
    'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2',
    'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7',
    'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'PO8', 'O1', 'Oz', 'O2'
]

SPLITS = {
    'train': SUBJECTS[:6],
    'val': SUBJECTS[6:8],
    'test': SUBJECTS[8:],
}


def _segment_trials(x: np.ndarray, labels: np.ndarray):
    seg_len = NUM_PATCHES * PATCH_LEN
    usable = (x.shape[-1] // seg_len) * seg_len
    if usable == 0:
        return None, None
    x = x[:, :, :usable]
    num_windows = x.shape[-1] // seg_len
    x = x.reshape(x.shape[0], x.shape[1], num_windows, seg_len)
    x = x.transpose(0, 2, 1, 3).reshape(-1, x.shape[1], seg_len)
    x = x.reshape(x.shape[0], x.shape[1], NUM_PATCHES, PATCH_LEN)
    labels = np.repeat(labels, num_windows)
    return x.astype(np.float32), labels.astype(np.uint8)


def _process_subject(args):
    split, subject = args
    path = os.path.join(ROOT_DIR, f'{subject}.npz')
    if not os.path.exists(path):
        print(f'[Weibo2014] Missing file for {subject}')
        return split, subject, []
    data = np.load(path, allow_pickle=True)
    x = data['x_data'].astype(np.float32)
    y = data['y_data'].astype(np.int64)
    mask = (y == 1) | (y == 2)
    x = x[mask]
    y = y[mask]
    if x.shape[0] == 0:
        print(f'[Weibo2014] No LR trials for {subject}')
        return split, subject, []
    y = y - y.min()

    info = mne.create_info(ch_names=CH_NAMES, sfreq=float(data['fs']), ch_types='eeg')
    epochs = mne.EpochsArray(x, info, tmin=0, verbose=False)
    epochs.filter(l_freq=1, h_freq=40, verbose=False)
    epochs.resample(TARGET_SFREQ, npad='auto')
    arr = epochs.get_data(copy=False)
    segmented, labels = _segment_trials(arr, y)
    if segmented is None:
        print(f'[Weibo2014] No usable segments for {subject}')
        return split, subject, []

    kv = []
    for idx, (sample, label) in enumerate(zip(segmented, labels)):
        key = f'{subject}-{idx}'
        value = pickle.dumps({'sample': sample, 'label': int(label), 'subject': subject})
        kv.append((key, value))
    return split, subject, kv


def main():
    os.makedirs(os.path.dirname(OUTPUT_LMDB), exist_ok=True)
    tasks = []
    for split, subs in SPLITS.items():
        for sub in subs:
            tasks.append((split, sub))

    print(f'[Weibo2014] Total subjects: {len(SUBJECTS)}')
    num_workers = min(len(tasks), mp.cpu_count())
    results = []
    with mp.Pool(processes=num_workers) as pool:
        for res in pool.imap_unordered(_process_subject, tasks):
            results.append(res)
            split, subject, kv = res
            print(f'[Weibo2014] {subject} ({split}) -> {len(kv)} segments')

    db = lmdb.open(OUTPUT_LMDB, map_size=5 << 30)
    keys = {split: [] for split in SPLITS}
    for split, _, kv in results:
        for key, value in kv:
            with db.begin(write=True) as txn:
                txn.put(key.encode(), value)
            keys[split].append(key)
    with db.begin(write=True) as txn:
        txn.put('__keys__'.encode(), pickle.dumps(keys))
    db.close()
    print(f'[Weibo2014] LMDB saved to {OUTPUT_LMDB}')


if __name__ == '__main__':
    main()
