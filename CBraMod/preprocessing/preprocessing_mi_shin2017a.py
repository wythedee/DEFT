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

DATASET_ROOT = eeg_dataset_root()
PROCESSED_ROOT = processed_root()

ROOT_DIR = first_existing(
    os.path.join(DATASET_ROOT, "MI", "MI_Shin2017A"),
    os.path.join(DATASET_ROOT, "MI_Shin2017A"),
)
OUTPUT_LMDB = os.path.join(PROCESSED_ROOT, "MI_Shin2017A", "processed_average")
TARGET_SFREQ = 200
WINDOW_SEC = 4
PATCH_LEN = 200
NUM_PATCHES = 4
RUN_INDICES = [0, 2, 4]

SUBJECTS = [
    '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11',
    '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22',
    '23', '24', '25', '26', '27', '28', '29'
]

CH_NAMES = [
    'F7', 'AFF5h', 'F3', 'AFp1', 'AFp2', 'AFF6h', 'F4', 'F8', 'AFF1h',
    'AFF2h', 'Cz', 'Pz', 'FCC5h', 'FCC3h', 'CCP5h', 'CCP3h', 'T7', 'P7',
    'P3', 'PPO1h', 'POO1', 'POO2', 'PPO2h', 'P4', 'FCC4h', 'FCC6h', 'CCP4h',
    'CCP6h', 'P8', 'T8'
]

SPLITS = {
    'train': SUBJECTS[:20],
    'val': SUBJECTS[20:25],
    'test': SUBJECTS[25:],
}


def _ensure_dirs():
    os.makedirs(os.path.dirname(OUTPUT_LMDB), exist_ok=True)


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
    return x, labels


def _reorder_channels(x: np.ndarray, current_names):
    idx = [current_names.index(ch) for ch in CH_NAMES]
    return x[:, idx, :]


def _create_epochs(run_data, sfreq, run_mrk):
    ch_names = list(run_data.clab)
    ch_types = ['eeg'] * 30 + ['eog'] * 2
    eeg = run_data.x.T
    info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
    raw = mne.io.RawArray(eeg, info, verbose=False)
    drop_list = [c for c in ['VEOG', 'HEOG'] if c in raw.ch_names]
    if drop_list:
        raw.drop_channels(drop_list)
    raw.filter(l_freq=1, h_freq=40, verbose=False)

    trig_offset = 0
    mkr_time = ((run_mrk.time - 1) // 5) / sfreq
    mkr = run_mrk.event.desc // 16 + trig_offset
    if len(mkr) == 0:
        return None, None
    events = np.column_stack(((mkr_time * sfreq).astype(int), np.zeros_like(mkr), mkr)).astype(int)
    epochs = mne.Epochs(raw, events=events, event_id=None, tmin=0, tmax=WINDOW_SEC,
                        preload=True, baseline=None, verbose=False)
    if abs(epochs.info['sfreq'] - TARGET_SFREQ) > 1e-3:
        epochs.resample(TARGET_SFREQ, npad='auto')
    data = epochs.get_data(copy=False).astype(np.float32)
    labels = epochs.events[:, 2] - 1
    mask = (labels == 0) | (labels == 1)
    data = data[mask]
    labels = labels[mask]
    if data.size == 0:
        return None, None
    data = _reorder_channels(data, epochs.ch_names)
    return data, labels.astype(np.uint8)


def _process_subject(args):
    split, subject = args
    cnt_path = os.path.join(ROOT_DIR, f'subject {subject}', 'with occular artifact', 'cnt.mat')
    mrk_path = os.path.join(ROOT_DIR, f'subject {subject}', 'with occular artifact', 'mrk.mat')
    if not (os.path.exists(cnt_path) and os.path.exists(mrk_path)):
        print(f'[Shin2017A] Missing files for subject {subject}')
        return split, subject, []

    mat_cnt = scipy.io.loadmat(cnt_path, squeeze_me=True, struct_as_record=False).get('cnt')
    mat_mrk = scipy.io.loadmat(mrk_path, squeeze_me=True, struct_as_record=False).get('mrk')
    if mat_cnt is None or mat_mrk is None:
        print(f'[Shin2017A] Invalid structure for subject {subject}')
        return split, subject, []

    data_list, label_list = [], []
    for idx in RUN_INDICES:
        try:
            data, labels = _create_epochs(mat_cnt[idx], 200, mat_mrk[idx])
        except Exception as err:
            print(f'[Shin2017A] Error subject {subject} run {idx}: {err}')
            continue
        if data is None:
            continue
        segmented, seg_labels = _segment_trials(data, labels)
        if segmented is None:
            continue
        data_list.append(segmented)
        label_list.append(seg_labels)

    if not data_list:
        print(f'[Shin2017A] No valid trials for subject {subject}')
        return split, subject, []

    data = np.concatenate(data_list, axis=0)
    labels = np.concatenate(label_list, axis=0)

    kv = []
    for i, (sample, label) in enumerate(zip(data, labels)):
        key = f'{subject}-{i}'
        value = pickle.dumps({'sample': sample, 'label': int(label), 'subject': subject})
        kv.append((key, value))
    return split, subject, kv


def main():
    _ensure_dirs()
    tasks = []
    for split, subs in SPLITS.items():
        for sub in subs:
            tasks.append((split, sub))

    print(f'[Shin2017A] Total tasks: {len(tasks)}')
    num_workers = min(len(tasks), mp.cpu_count())
    results = []
    with mp.Pool(processes=num_workers) as pool:
        for res in pool.imap_unordered(_process_subject, tasks):
            results.append(res)
            split, subject, kv = res
            print(f'[Shin2017A] {subject} ({split}) -> {len(kv)} samples')

    db = lmdb.open(OUTPUT_LMDB, map_size=5 << 30)
    keys = {split: [] for split in SPLITS}
    for split, subject, kv in results:
        for key, value in kv:
            with db.begin(write=True) as txn:
                txn.put(key.encode(), value)
            keys[split].append(key)
    with db.begin(write=True) as txn:
        txn.put('__keys__'.encode(), pickle.dumps(keys))
    db.close()
    print(f'[Shin2017A] LMDB saved to {OUTPUT_LMDB}')


if __name__ == '__main__':
    main()
