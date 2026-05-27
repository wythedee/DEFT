import os
import sys
import lmdb
import pickle
import numpy as np
import mne
import scipy.io
import multiprocessing as mp

try:
    from preprocessing.path_utils import processed_root
except ModuleNotFoundError:  # allow running as a script from this directory
    from path_utils import processed_root


# Try to import FAST2 share to reuse its SRC paths
_FAST2_SHARE_IMPORTED = False
try:
    fast2_share_dir = os.path.expanduser(
        os.environ.get("FAST2_EEG_DATASET_DIR", "/path/to/raw_eeg")
    )
    if os.path.isdir(fast2_share_dir) and fast2_share_dir not in sys.path:
        sys.path.insert(0, fast2_share_dir)
    from share import SRC_FOLDER as FAST2_SRC_FOLDER  # type: ignore
    _FAST2_SHARE_IMPORTED = True
except Exception:
    FAST2_SRC_FOLDER = ''


# Dataset config (align with FAST2 MI_08_Track1_Few_shot)
SRC_NAME = 'MI_Track1_Few_shot'
CH_NAMES = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2',
    'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2',
    'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz',
    'O2', 'PO10', 'FC3', 'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CPz',
    'CP4', 'P1', 'P2', 'POz', 'FT9', 'FTT9h', 'TTP7h', 'TP7', 'TPP9h',
    'FT10', 'FTT10h', 'TPP8h', 'TP8', 'TPP10h', 'F9', 'F10', 'AF7',
    'AF3', 'AF4', 'AF8', 'PO3', 'PO4'
]
SUBJECTS = [
    '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11',
    '12', '13', '14', '15', '16', '17', '18', '19', '20'
]


def get_src_root() -> str:
    # Prefer FAST2 SRC folder if available; else use env var override; else raise
    if FAST2_SRC_FOLDER:
        return os.path.join(FAST2_SRC_FOLDER, 'MI', SRC_NAME)
    env_root = os.environ.get('MI08_TRACK1_FEWSHOT_SRC', '')
    if env_root:
        return env_root
    raise RuntimeError('Set MI08_TRACK1_FEWSHOT_SRC to the MI_Track1_Few_shot directory or install FAST2 and share.py.')


def split_subjects(subjects):
    # Deterministic subject-wise split

    return {
        'train': subjects[:10],
        'val': subjects[10:15],
        'test': subjects[15:20]
    }


def load_one_subject(src_root: str, sub: str):
    # Paths from the competition data layout
    fname_train = os.path.join(src_root, 'Training set', f'Data_Sample{sub}.mat')
    fname_valid = os.path.join(src_root, 'Validation set', f'Data_Sample{sub}.mat')
    if not (os.path.exists(fname_train) and os.path.exists(fname_valid)):
        print('Missing:', fname_train, fname_valid)
        return None, None

    data_1 = scipy.io.loadmat(fname_train, squeeze_me=True, struct_as_record=False)['Training']
    data_2 = scipy.io.loadmat(fname_valid, squeeze_me=True, struct_as_record=False)['Validation']

    sfreq = float(data_1.fs)
    x_1, y_1 = data_1.x, data_1.y_dec - 1
    x_2, y_2 = data_2.x, data_2.y_dec - 1

    # Concat along trials, transpose to (n, ch, t)
    x = np.concatenate([x_1, x_2], axis=1).transpose((1, 2, 0))
    y = np.concatenate([y_1, y_2], axis=0)

    # Swap labels: description says right=1 left=2; map to 0/1 as (left=0, right=1)
    y = 1 - y

    info = mne.create_info(ch_names=CH_NAMES, sfreq=sfreq, ch_types='eeg')
    epochs = mne.EpochsArray(x, info, tmin=0, verbose=False)
    epochs.filter(l_freq=1, h_freq=40, verbose=False)
    # Resample to 200 Hz to form 4x200 patches
    epochs.resample(200, npad='auto')
    X = epochs.get_data(copy=False).astype(np.float32)
    Y = y.astype(np.uint8)
    return X, Y


def to_patched(x: np.ndarray) -> np.ndarray:
    # Expect x: (n, ch, t). Keep last 800 samples (4s@200Hz), reshape to (n, ch, 4, 200)
    x = x[:, :, -800:]
    n, c, t = x.shape
    return x.reshape(n, c, 4, 200)


def _process_one(args):
    split, sub, src_root = args
    x, y = load_one_subject(src_root, sub)
    if x is None:
        return split, sub, []
    x = to_patched(x)
    kv = []
    for i, (sample, label) in enumerate(zip(x, y)):
        key = f'{sub}-{i}'
        value = pickle.dumps({'sample': sample, 'label': int(label), 'subject': sub})
        kv.append((key, value))
    return split, sub, kv


def main():
    src_root = get_src_root()
    print('Source root:', src_root)

    # Output LMDB path
    lmdb_path = os.path.join(processed_root(), 'MI_08_Track1_Few_shot', 'processed_average')
    os.makedirs(os.path.dirname(lmdb_path), exist_ok=True)

    files_dict = split_subjects(SUBJECTS)
    dataset_keys = {'train': [], 'val': [], 'test': []}

    tasks = []
    for split in ['train', 'val', 'test']:
        for sub in files_dict[split]:
            tasks.append((split, sub, src_root))

    num_workers = int(os.environ.get('MI08_TRACK1_FEWSHOT_NUM_WORKERS', 8))
    print('Num workers:', num_workers)

    results = []
    with mp.Pool(processes=num_workers) as pool:
        for res in pool.imap_unordered(_process_one, tasks):
            results.append(res)
            split, sub, kv = res
            print(sub, split, 'samples:', len(kv))

    db = lmdb.open(lmdb_path, map_size=5 << 30)
    for split, sub, kv in results:
        for key, value in kv:
            with db.begin(write=True) as txn:
                txn.put(key.encode(), value)
        dataset_keys[split].extend([k for k, _ in kv])
    with db.begin(write=True) as txn:
        txn.put('__keys__'.encode(), pickle.dumps(dataset_keys))
    db.close()
    print('Preprocessing finished. LMDB saved to:', lmdb_path)


if __name__ == '__main__':
    main()
