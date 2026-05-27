import os
import pickle
import lmdb
import numpy as np
import mne
import scipy.io
import multiprocessing as mp

try:
    from preprocessing.path_utils import eeg_dataset_root, first_existing, processed_root
except ModuleNotFoundError:  # allow running as a script from this directory
    from path_utils import eeg_dataset_root, first_existing, processed_root

# Dataset paths
DATASET_ROOT = eeg_dataset_root()
PROCESSED_ROOT = processed_root()

ROOT_DIR = first_existing(
    os.path.join(DATASET_ROOT, "MI", "MI_HeBin2021"),
    os.path.join(DATASET_ROOT, "MI_HeBin2021"),
)
CACHE_DIR = os.path.expanduser(
    os.environ.get(
        "HEBIN2021_CACHE_DIR",
        os.path.join(os.path.expanduser("~"), "blob_mount", "data", "_cache", "hebin2021"),
    )
)
os.makedirs(CACHE_DIR, exist_ok=True)

SUBJECTS = [
    'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14',
    'S15', 'S16', 'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27',
    'S28', 'S29', 'S30', 'S31', 'S32', 'S33', 'S35', 'S36', 'S37', 'S38', 'S39', 'S40',
    'S41', 'S42', 'S43', 'S44', 'S45', 'S46', 'S47', 'S48', 'S49', 'S50', 'S51', 'S52', 'S53',
    'S54', 'S55', 'S56', 'S57', 'S58', 'S59'
]

ORIGINAL_CH_NAMES = ['Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2',
                     'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6',
                     'FT8', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5',
                     'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1',
                     'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POz', 'PO4', 'PO6',
                     'PO8', 'CB1', 'O1', 'Oz', 'O2', 'CB2']

CH_NAMES = [c for c in ORIGINAL_CH_NAMES if c not in ['CB1', 'CB2']]

# Output LMDB paths
# DB_LR_PATH = '/path/to/standardized_eeg/cbramod/MI_HeBin2021_LR/processed_average'
# DB_UD_PATH = '/path/to/standardized_eeg/cbramod/MI_HeBin2021_UD/processed_average'

# for p in [DB_LR_PATH, DB_UD_PATH]:
#     os.makedirs(os.path.dirname(p), exist_ok=True)

# Split subjects (40/9/9)
train_subs = SUBJECTS[:40]
val_subs = SUBJECTS[40:49]
test_subs = SUBJECTS[49:]
SUB_SPLIT = {s: ('train' if s in train_subs else 'val' if s in val_subs else 'test') for s in SUBJECTS}

start_sec, end_sec = 1, 6  # cut 1-6 s (5 s window)


def _proc_one_mat(mat_path):
    """Return trials (n, chans, time) and labels (n,) from one session MAT."""
    try:
        bci = scipy.io.loadmat(mat_path, struct_as_record=False, squeeze_me=True)['BCI']
        trials, labels = [], []
        for i in range(450):
            trial = bci.data[i]
            target = int(bci.TrialData[i].targetnumber)
            if trial.shape[1] < end_sec * 1000:
                trial = np.pad(trial, ((0, 0), (0, end_sec * 1000 - trial.shape[1])), 'edge')
            trials.append(trial[:, start_sec * 1000:end_sec * 1000])
            labels.append(target)
        trials = np.asarray(trials, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.uint8) - 1  # 0-3 where 0=Right,1=Left,2=Up,3=Down
        return trials, labels
    except Exception as e:
        print(f"Error processing {mat_path}: {e}")
        return None, None


def process_subject(sub):
    """Process all sessions for one subject, return dictionary."""
    x_acc_lr, y_acc_lr, x_acc_ud, y_acc_ud = [], [], [], []
    for sess in range(1, 12):
        mat_path = f'{ROOT_DIR}/{sub}_Session_{sess}.mat'
        if not os.path.exists(mat_path):
            continue
        trials, labels = _proc_one_mat(mat_path)
        if trials is None:
            continue
        try:
            info = mne.create_info(ORIGINAL_CH_NAMES, 1000, 'eeg')
            epochs = mne.EpochsArray(trials, info, tmin=0, verbose=False)
            epochs.drop_channels(['CB1', 'CB2'])
            epochs.filter(l_freq=1, h_freq=40, verbose=False)
            epochs.resample(200, npad='auto')
            data = epochs.get_data(copy=False)  # (n, 60, T≈1000)
            # keep last 800 samples and reshape to 4 patches
            data = data[:, :, -800:]
            b, c, _ = data.shape
            data = data.reshape(b, c, 4, 200)

            # Split LR / UD
            mask_lr = (labels == 0) | (labels == 1)
            mask_ud = (labels == 2) | (labels == 3)
            x_acc_lr.append(data[mask_lr])
            y_acc_lr.append(1 - labels[mask_lr])  # swap: 0=Left,1=Right
            x_acc_ud.append(data[mask_ud])
            y_acc_ud.append(labels[mask_ud] - 2)
        except Exception as e:
            print(f"Error processing {sub}_Session_{sess}: {e}")
            continue
    if x_acc_lr:
        x_lr = np.concatenate(x_acc_lr)
        y_lr = np.concatenate(y_acc_lr)
    else:
        x_lr = np.empty((0, len(CH_NAMES), 4, 200), dtype=np.float32)
        y_lr = np.empty((0,), dtype=np.uint8)
    if x_acc_ud:
        x_ud = np.concatenate(x_acc_ud)
        y_ud = np.concatenate(y_acc_ud)
    else:
        x_ud = np.empty((0, len(CH_NAMES), 4, 200), dtype=np.float32)
        y_ud = np.empty((0,), dtype=np.uint8)
    return sub, x_lr, y_lr, x_ud, y_ud


if __name__ == '__main__':
    with mp.Pool(min(8, mp.cpu_count())) as pool:
        results = pool.map(process_subject, SUBJECTS)

    # LMDB setup
    lmdb_path_lr = os.path.join(PROCESSED_ROOT, "MI_HeBin2021_LR", "processed_average")
    os.makedirs(os.path.dirname(lmdb_path_lr), exist_ok=True)
    db_lr = lmdb.open(lmdb_path_lr, map_size=50 * 1024 * 1024 * 1024)  # 50GB

    lmdb_path_ud = os.path.join(PROCESSED_ROOT, "MI_HeBin2021_UD", "processed_average")
    os.makedirs(os.path.dirname(lmdb_path_ud), exist_ok=True)
    db_ud = lmdb.open(lmdb_path_ud, map_size=50 * 1024 * 1024 * 1024)  # 50GB

    keys_lr = {'train': [], 'val': [], 'test': []}
    keys_ud = {'train': [], 'val': [], 'test': []}

    for sub, x_lr, y_lr, x_ud, y_ud in results:
        split = SUB_SPLIT[sub]
        # LR samples
        for idx, (sample, label) in enumerate(zip(x_lr, y_lr)):
            key = f'{sub}-LR-{idx}'
            with db_lr.begin(write=True) as txn:
                txn.put(
                    key.encode(),
                    pickle.dumps({'sample': sample, 'label': int(label), 'subject': sub}),
                )
            keys_lr[split].append(key)
        # UD samples
        for idx, (sample, label) in enumerate(zip(x_ud, y_ud)):
            key = f'{sub}-UD-{idx}'
            with db_ud.begin(write=True) as txn:
                txn.put(
                    key.encode(),
                    pickle.dumps({'sample': sample, 'label': int(label), 'subject': sub}),
                )
            keys_ud[split].append(key)
        print(sub, 'LR', x_lr.shape, y_lr.shape, 'UD', x_ud.shape, y_ud.shape)

    # Save keys
    with db_lr.begin(write=True) as txn:
        txn.put('__keys__'.encode(), pickle.dumps(keys_lr))
    with db_ud.begin(write=True) as txn:
        txn.put('__keys__'.encode(), pickle.dumps(keys_ud))

    db_lr.close()
    db_ud.close()
    print('HeBin2021 preprocessing finished.')
