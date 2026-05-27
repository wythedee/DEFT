import os
import lmdb
import pickle
import numpy as np
import mne

try:
    from preprocessing.path_utils import eeg_dataset_root, first_existing, processed_root
except ModuleNotFoundError:  # allow running as a script from this directory
    from path_utils import eeg_dataset_root, first_existing, processed_root

"""Cross-subject preprocessing for SEED-V dataset.

This script keeps **the same output format** as `preprocessing_SEEDV.py`:
1. LMDB database where each sample is a 1-second patch with shape (62, 1, 200)
2. A special key `__keys__` recording train/val/test lists.

Difference: splitting is done **by subject**, not by trial**.  
Edit `TRAIN_SUBJECTS / VAL_SUBJECTS / TEST_SUBJECTS` below as needed.
"""

# -------- configurable lists (subject IDs as strings, leading zeros removed) --------
TRAIN_SUBJECTS = ["1", "2", "3", "4", "5",]
VAL_SUBJECTS   = ["6", "8", "9", "10", "11"]
TEST_SUBJECTS  = ["12", "13", "14", "15", "16"]
# -------------------------------------------------------------------------------

DATASET_ROOT = eeg_dataset_root()
PROCESSED_ROOT = processed_root()

root_dir = first_existing(
    os.path.join(DATASET_ROOT, "EMO", "EMO_03_SEED_V", "EEG_raw"),
    os.path.join(DATASET_ROOT, "EMO_03_SEED_V", "EEG_raw"),
)
output_lmdb = os.path.join(PROCESSED_ROOT, "EMO_SEED_V", "processed_average_cross")
os.makedirs(os.path.dirname(output_lmdb), exist_ok=True)

a2_remove = ["M1", "M2", "VEO", "HEO"]

# hard-coded trial boundaries & labels (same as original script)
trials_of_sessions = {
    "1": {"start": [30, 132, 287, 555, 773, 982, 1271, 1628, 1730, 2025, 2227, 2435, 2667, 2932, 3204],
           "end":   [102, 228, 524, 742, 920, 1240, 1568, 1697, 1994, 2166, 2401, 2607, 2901, 3172, 3359]},
    "2": {"start": [30, 299, 548, 646, 836, 1000, 1091, 1392, 1657, 1809, 1966, 2186, 2333, 2490, 2741],
           "end":   [267, 488, 614, 773, 967, 1059, 1331, 1622, 1777, 1908, 2153, 2302, 2428, 2709, 2817]},
    "3": {"start": [30, 353, 478, 674, 825, 908, 1200, 1346, 1451, 1711, 2055, 2307, 2457, 2726, 2888],
           "end":   [321, 418, 643, 764, 877, 1147, 1284, 1418, 1679, 1996, 2275, 2425, 2664, 2857, 3066]},
}
labels_of_sessions = {
    "1": [4, 1, 3, 2, 0, 4, 1, 3, 2, 0, 4, 1, 3, 2, 0],
    "2": [2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0],
    "3": [2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0],
}

# mapping subject-id -> split
SUB_SPLIT = {sub: "train" for sub in TRAIN_SUBJECTS} | {sub: "val" for sub in VAL_SUBJECTS} | {sub: "test" for sub in TEST_SUBJECTS}

dataset_keys = {"train": [], "val": [], "test": []}

db = lmdb.open(output_lmdb, map_size=20_000_000_000)  # 20 GB

# --- helper to read both Neuroscan & ANT CNT ---
from mne.io import read_raw_cnt, read_raw_ant

def safe_read_cnt(path):
    try:
        return read_raw_cnt(path, preload=True, verbose=False)
    except Exception as e:
        # fallback to ANT reader
        return read_raw_ant(path, preload=True, verbose=False)

files = sorted([f for f in os.listdir(root_dir) if f.endswith('.cnt')])
print(f"Found {len(files)} files")

for file in files:
    subject_id = file.split('_')[0].lstrip('0')  # remove leading zeros like '01' -> '1'
    split = SUB_SPLIT.get(subject_id)
    if split is None:
        # subject not in any split -> skip
        continue

    try:
        raw = safe_read_cnt(os.path.join(root_dir, file))
    except Exception as e:
        print(f"[Skip] {file}: {e}")
        continue

    raw.drop_channels(a2_remove)
    raw.resample(200)
    raw.filter(l_freq=0.3, h_freq=75, verbose=False)
    data_matrix = raw.get_data(units='uV')

    session_index = file.split('_')[1]
    starts = trials_of_sessions[session_index]['start']
    ends   = trials_of_sessions[session_index]['end']
    labels = labels_of_sessions[session_index]

    for t_idx in range(15):
        trial = data_matrix[:, starts[t_idx]*200: ends[t_idx]*200]  # (62, T)
        # reshape to patches
        patches = trial.reshape(62, -1, 1, 200).transpose(1, 0, 2, 3)  # (seg, 62,1,200)
        label = labels[t_idx]
        for p_idx, sample in enumerate(patches):
            key = f"{file}-{t_idx}-{p_idx}"
            subject_id = file.split('_')[0].lstrip('0') or file.split('_')[0]
            data_dict = {
                "sample": sample.astype(np.float32),
                "label": label,
                "subject": subject_id,
            }
            txn = db.begin(write=True)
            txn.put(key.encode(), pickle.dumps(data_dict))
            txn.commit()
            dataset_keys[split].append(key)
    print(f"Processed {file} -> {split}")

# write key lists
with db.begin(write=True) as txn:
    txn.put(b'__keys__', pickle.dumps(dataset_keys))

db.close()
print("Done. keys count:", {k: len(v) for k,v in dataset_keys.items()})
