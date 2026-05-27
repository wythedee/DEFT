import os
import os.path as osp
import lmdb
import pickle
import numpy as np
import mne
mne.set_log_level('ERROR')
import scipy.io as sio
import multiprocessing as mp

# Paths
try:
    from preprocessing.path_utils import eeg_dataset_root, first_existing, processed_root
except ModuleNotFoundError:  # allow running as a script from this directory
    from path_utils import eeg_dataset_root, first_existing, processed_root

DATASET_ROOT = eeg_dataset_root()
PROCESSED_ROOT = processed_root()

ROOT_DIR = first_existing(
    os.path.join(DATASET_ROOT, "EMO", "EMO_04_SEED", "SEED", "Preprocessed_EEG"),
    os.path.join(DATASET_ROOT, "EMO_04_SEED", "SEED", "Preprocessed_EEG"),
)
LMDB_PATH = os.path.join(PROCESSED_ROOT, "EMO_SEED", "processed_average")
os.makedirs(os.path.dirname(LMDB_PATH), exist_ok=True)

# Channels
CH_NAMES = ['FP1','FPZ','FP2','AF3','AF4','F7','F5','F3','F1','FZ','F2','F4','F6','F8','FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8','T7','C5','C3','C1','CZ','C2','C4','C6','T8','TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8','P7','P5','P3','P1','PZ','P2','P4','P6','P8','PO7','PO5','PO3','POZ','PO4','PO6','PO8','O1','OZ','O2']
ORIGINAL_CH_NAMES = CH_NAMES + ['CB1','CB2']

# Processing params
RESAMPLE_RATE = 200
TIME_LENGTH = 4  # seconds
PATCH_SIZE = 200
SEQ_LEN = 4

SUBJECTS = list(range(1, 16))  # 1..15
SPLIT = {
    'train': SUBJECTS[:10],
    'val': SUBJECTS[10:12],
    'test': SUBJECTS[12:],
}


def load_one_session(file_to_load: str, feature_key: str = 'eeg'):
    data = sio.loadmat(file_to_load, verify_compressed_data_integrity=False, squeeze_me=True)
    keys = [k for k in data.keys() if feature_key in k]
    trials = [data[k] for k in keys]
    if len(trials) == 0:
        return None
    min_len = min(t.shape[1] for t in trials)
    trials = [t[:, :min_len] for t in trials]
    return np.asarray(trials)  # (num_trials, C, T)


def process_subject(sub_idx: int):
    sub_code_prefix = f"{sub_idx}_"
    data_files = []
    for root, _, files in os.walk(ROOT_DIR, topdown=False):
        for name in files:
            if name.startswith(sub_code_prefix):
                data_files.append(osp.join(root, name))
    data_files.sort()
    if len(data_files) == 0:
        print(f'sub{sub_idx}: no files found')
        return sub_idx, [], [], []

    label_mat = sio.loadmat(osp.join(ROOT_DIR, 'label.mat'))['label']
    label_mat = np.squeeze(label_mat) + 1  # shift labels

    samples = []
    labels = []
    keys = []
    for fpath in data_files:
        sess_trials = load_one_session(fpath, 'eeg')
        if sess_trials is None:
            continue
        # two-class: drop class==1 (neutral), map 2->1 (pos), others->0 (neg)
        idx_keep = np.delete(np.arange(label_mat.shape[-1]), np.where(label_mat == 1)[0])
        sess_trials = sess_trials[idx_keep]
        label_sel = label_mat[idx_keep]
        label_sel = np.where(label_sel == 2, 1, 0).astype(np.uint8)

        info = mne.create_info(ch_names=ORIGINAL_CH_NAMES, sfreq=1000, ch_types='eeg')
        epochs = mne.EpochsArray(sess_trials, info, tmin=0, verbose=False)
        epochs.drop_channels(['CB1', 'CB2'])
        if abs(epochs.info['sfreq'] - RESAMPLE_RATE) > 1e-6:
            epochs.resample(RESAMPLE_RATE, npad='auto')
        data = epochs.get_data(copy=False).astype(np.float32)  # (n, 60, T)

        seg_len = TIME_LENGTH * RESAMPLE_RATE  # 800
        usable = (data.shape[-1] // seg_len) * seg_len
        if usable == 0:
            continue
        data = data[:, :, :usable]
        data = data.reshape(data.shape[0], data.shape[1], -1, seg_len)  # n, C, N, 800
        n_win = data.shape[2]
        data = data.transpose(0, 2, 1, 3).reshape(-1, data.shape[1], seg_len)  # (n*N, C, 800)
        data = data.reshape(data.shape[0], data.shape[1], SEQ_LEN, PATCH_SIZE)
        labels_rep = np.repeat(label_sel, n_win)

        base_key = osp.basename(fpath).split('.')[0]
        for i in range(data.shape[0]):
            key = f'{sub_idx}-{base_key}-{i}'
            keys.append(key)
            samples.append(data[i])
            labels.append(int(labels_rep[i]))

    return sub_idx, keys, samples, labels


if __name__ == '__main__':
    with mp.Pool(8) as pool:
        results = pool.map(process_subject, SUBJECTS)

    db = lmdb.open(LMDB_PATH, map_size=50 * 1024 * 1024 * 1024)
    split_keys = {'train': [], 'val': [], 'test': []}

    for sub, keys, samples, labels in results:
        split = 'train' if sub in SPLIT['train'] else ('val' if sub in SPLIT['val'] else 'test')
        for k, s, y in zip(keys, samples, labels):
            with db.begin(write=True) as txn:
                txn.put(
                    k.encode(),
                    pickle.dumps({'sample': s, 'label': y, 'subject': sub}),
                )
            split_keys[split].append(k)
        print(f'sub{sub}: {len(keys)}')

    with db.begin(write=True) as txn:
        txn.put('__keys__'.encode(), pickle.dumps(split_keys))
    db.close()
    print('SEED preprocessing finished.')
