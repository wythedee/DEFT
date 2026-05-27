import os
import sys
import lmdb
import pickle
import numpy as np
import mne
import scipy.io
from scipy import signal
import multiprocessing as mp

try:
    from preprocessing.path_utils import processed_root
except ModuleNotFoundError:  # allow running as a script from this directory
    from path_utils import processed_root


# Try to import FAST2 share to reuse its SRC paths and channel pipeline
_FAST2_SHARE_IMPORTED = False
try:
    fast2_share_dir = os.path.expanduser(
        os.environ.get("FAST2_EEG_DATASET_DIR", "/path/to/raw_eeg")
    )
    if os.path.isdir(fast2_share_dir) and fast2_share_dir not in sys.path:
        sys.path.insert(0, fast2_share_dir)
    from share import SRC_FOLDER as FAST2_SRC_FOLDER, pipeline as fast2_pipeline  # type: ignore
    _FAST2_SHARE_IMPORTED = True
except Exception:
    fast2_pipeline = None
    FAST2_SRC_FOLDER = ''


# Dataset config (align with FAST2 MI_01_SSVEP_KoreaU)
SRC_NAME = 'MI_KoreaU'
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
    's10', 's22', 's28', 's5', 's31', 's46', 's41', 's36', 's8',
    's2', 's52', 's25', 's17', 's19', 's13', 's21', 's6', 's45',
    's32', 's38', 's35', 's42', 's48', 's1', 's26', 's51', 's14',
    's24', 's53', 's16', 's37', 's40', 's3', 's9', 's4', 's47',
    's30', 's11', 's29', 's54', 's23', 's50', 's27', 's15', 's49',
    's43', 's34', 's7', 's39', 's33', 's44', 's12', 's18', 's20'
]


def get_src_root() -> str:
    # Prefer FAST2 SRC folder if available; else use env var override; else raise
    if FAST2_SRC_FOLDER:
        return os.path.join(FAST2_SRC_FOLDER, 'MI', SRC_NAME, 'BCI_dataset', 'DB_mat')
    env_root = os.environ.get('MI_KOREAU_SRC', '')
    if env_root:
        return env_root
    raise RuntimeError('Set MI_KOREAU_SRC to the ".../MI_KoreaU/BCI_dataset/DB_mat" directory or install FAST2 and share.py.')


def split_subjects(subjects):
    # Deterministic subject-wise split similar to earlier convention
    n = len(subjects)
    n_train = min(45, max(1, int(0.79 * n)))
    n_val = min(6, max(1, int(0.10 * n)))
    return {
        'train': subjects[:n_train],
        'val': subjects[n_train:n_train + n_val],
        'test': subjects[n_train + n_val:]
    }


def load_one_subject(src_db_root: str, sub: str):
    # Load four partitions (session1/2 x train/test) for MI
    paths = [
        os.path.join(src_db_root, 'session1', sub, 'EEG_MI.mat'),
        os.path.join(src_db_root, 'session2', sub, 'EEG_MI.mat'),
    ]
    mats = []
    for p in paths:
        if not os.path.exists(p):
            print('Missing:', p)
            return None, None
        mats.append(scipy.io.loadmat(p))

    # Concatenate train+test within each session, then across sessions
    data_list = []
    label_list = []
    for m in mats:
        # shapes: smt: (trials, channels, time) after transpose below; labels vector
        d_tr = np.transpose(m['EEG_MI_train']['smt'][0, 0], (1, 2, 0))
        l_tr = m['EEG_MI_train']['y_dec'][0, 0][0]
        d_te = np.transpose(m['EEG_MI_test']['smt'][0, 0], (1, 2, 0))
        l_te = m['EEG_MI_test']['y_dec'][0, 0][0]
        data_list.extend([d_tr, d_te])
        label_list.extend([l_tr, l_te])

    epoch = np.concatenate(data_list, axis=0)
    label = np.concatenate(label_list, axis=0)

    # MNE preprocessing: filter 1-40 Hz, resample to 200 Hz for 4x200 patches
    info = mne.create_info(ch_names=CH_NAMES, sfreq=1000, ch_types='eeg')
    epochs = mne.EpochsArray(epoch, info, verbose=False)
    epochs.filter(l_freq=1, h_freq=40, verbose=False)
    epochs.resample(200, npad='auto')
    x = epochs.get_data(copy=False).astype(np.float32)  # (n, ch, t)
    y = label - 1

    # Special label notice: 0 -> right, 1 -> left in KoreaU; flip to (0:left,1:right)
    y = 1 - y

    # Optional FAST2 channel pipeline if available

    return x, y


def to_patched(x: np.ndarray) -> np.ndarray:
    # Expect x: (n, ch, t). Keep last 800 samples (4s@200Hz), reshape to (n, ch, 4, 200)
    x = x[:, :, -800:]
    n, c, t = x.shape
    return x.reshape(n, c, 4, 200)


def _process_one(args):
    split, sub, src_db_root = args
    x, y = load_one_subject(src_db_root, sub)
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
    src_db_root = get_src_root()
    print('Source root:', src_db_root)

    # Output LMDB path
    lmdb_path = os.path.join(processed_root(), 'MI_KoreaU', 'processed_average')
    os.makedirs(os.path.dirname(lmdb_path), exist_ok=True)

    files_dict = split_subjects(SUBJECTS)
    dataset_keys = {'train': [], 'val': [], 'test': []}

    # Build tasks
    tasks = []
    for split in ['train', 'val', 'test']:
        for sub in files_dict[split]:
            tasks.append((split, sub, src_db_root))

    num_workers = int(os.environ.get('MI_KOREAU_NUM_WORKERS', 8))
    print('Num workers:', num_workers)

    # Parallel preprocess; serialize LMDB writes to avoid contention
    results = []
    with mp.Pool(processes=num_workers) as pool:
        for res in pool.imap_unordered(_process_one, tasks):
            results.append(res)
            split, sub, kv = res
            print(sub, split, 'samples:', len(kv))

    # Write LMDB
    db = lmdb.open(lmdb_path, map_size=5 << 30)
    for split, sub, kv in results:
        for key, value in kv:
            with db.begin(write=True) as txn:
                txn.put(key.encode(), value)
            dataset_keys[split].append(key)
    with db.begin(write=True) as txn:
        txn.put('__keys__'.encode(), pickle.dumps(dataset_keys))
    db.close()
    print('Preprocessing finished. LMDB saved to:', lmdb_path)


if __name__ == '__main__':
    main()
