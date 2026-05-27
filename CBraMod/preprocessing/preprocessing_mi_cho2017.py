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
    os.path.join(DATASET_ROOT, "MI", "MI_Cho2017"),
    os.path.join(DATASET_ROOT, "MI_Cho2017"),
)
OUTPUT_LMDB = os.path.join(PROCESSED_ROOT, "MI_Cho2017", "processed_average")
TARGET_SFREQ = 200
NUM_PATCHES = 4
PATCH_LEN = 200

SUBJECTS = ['s1', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18',
            's19', 's2', 's20', 's21', 's22', 's23', 's24', 's25', 's26', 's27',
            's28', 's29', 's3', 's30', 's31', 's33', 's34', 's35', 's36', 's37',
            's38', 's39', 's4', 's40', 's41', 's42', 's43', 's44', 's45', 's47',
            's48', 's5', 's50', 's51', 's52', 's6', 's7', 's8', 's9']

CH_NAMES = [
    'Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1',
    'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7',
    'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'Fp2',
    'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4',
    'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2',
    'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2'
]

SPLITS = {
    'train': SUBJECTS[:35],
    'val': SUBJECTS[35:42],
    'test': SUBJECTS[42:],
}


def _segment_trials(x: np.ndarray, labels: np.ndarray):
    seg_len = NUM_PATCHES * PATCH_LEN
    orig_len = x.shape[-1]
    num_windows = int(np.ceil(orig_len / seg_len))
    if num_windows == 0:
        return None, None
    pad_len = num_windows * seg_len - orig_len
    if pad_len > 0:
        x = np.pad(x, ((0, 0), (0, 0), (0, pad_len)), mode='edge')
    x = x.reshape(x.shape[0], x.shape[1], num_windows, seg_len)
    x = x.transpose(0, 2, 1, 3).reshape(-1, x.shape[1], seg_len)
    x = x.reshape(x.shape[0], x.shape[1], NUM_PATCHES, PATCH_LEN)
    labels = np.repeat(labels, num_windows)
    return x.astype(np.float32), labels.astype(np.uint8)


def _process_subject(args):
    split, subject = args
    path = os.path.join(ROOT_DIR, f'{subject}.npz')
    if not os.path.exists(path):
        print(f'[Cho2017] Missing file for {subject}')
        return split, subject, []
    data = np.load(path, allow_pickle=True)
    x = data['x_data'].astype(np.float32)
    y_raw = data['y_data'].astype(np.int64)
    _, y = np.unique(y_raw, return_inverse=True)
    fs = float(data['fs'])

    info = mne.create_info(ch_names=CH_NAMES, sfreq=fs, ch_types='eeg')
    epochs = mne.EpochsArray(x, info, tmin=0, verbose=False)
    epochs.filter(l_freq=1, h_freq=40, verbose=False)
    if abs(fs - TARGET_SFREQ) > 1e-3:
        epochs.resample(TARGET_SFREQ, npad='auto')
    data_arr = epochs.get_data(copy=False)
    segmented, labels = _segment_trials(data_arr, y)
    if segmented is None:
        print(f'[Cho2017] No usable segments for {subject}')
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

    print(f'[Cho2017] Total subjects: {len(SUBJECTS)}')
    num_workers = min(mp.cpu_count(), len(tasks))
    results = []
    with mp.Pool(processes=num_workers) as pool:
        for res in pool.imap_unordered(_process_subject, tasks):
            results.append(res)
            split, sub, kv = res
            print(f'[Cho2017] {sub} ({split}) -> {len(kv)} segments')

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
    print(f'[Cho2017] LMDB saved to {OUTPUT_LMDB}')


if __name__ == '__main__':
    main()
