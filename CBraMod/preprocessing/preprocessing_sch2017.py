import os
import pickle
import lmdb
import mne
import numpy as np

try:
    from preprocessing.path_utils import eeg_dataset_root, first_existing, processed_root
except ModuleNotFoundError:  # allow running as a script from this directory
    from path_utils import eeg_dataset_root, first_existing, processed_root

# Dataset parameters
DATASET_ROOT = eeg_dataset_root()
PROCESSED_ROOT = processed_root()

root_dir = first_existing(
    os.path.join(DATASET_ROOT, "MI", "MI_Schirrmeister2017"),
    os.path.join(DATASET_ROOT, "MI_Schirrmeister2017"),
)
subjects = sorted([f.split('.')[0] for f in os.listdir(root_dir) if f.endswith('.npz')])

# Split subjects into train/val/test (8/3/3)
files_dict = {
    'train': subjects[:8],
    'val': subjects[8:11],
    'test': subjects[11:],
}
print(files_dict)

# All 128 channels provided by the dataset
CH_NAMES = [
    'Fp1','Fp2','Fpz','F7','F3','Fz','F4','F8','FC5','FC1','FC2','FC6',
    'M1','T7','C3','Cz','C4','T8','M2','CP5','CP1','CP2','CP6','P7','P3',
    'Pz','P4','P8','POz','O1','Oz','O2','AF7','AF3','AF4','AF8','F5','F1',
    'F2','F6','FC3','FCz','FC4','C5','C1','C2','C6','CP3','CPz','CP4','P5',
    'P1','P2','P6','PO5','PO3','PO4','PO6','FT7','FT8','TP7','TP8','PO7','PO8',
    'FT9','FT10','TPP9h','TPP10h','PO9','PO10','P9','P10','AFF1','AFz','AFF2',
    'FFC5h','FFC3h','FFC4h','FFC6h','FCC5h','FCC3h','FCC4h','FCC6h','CCP5h','CCP3h',
    'CCP4h','CCP6h','CPP5h','CPP3h','CPP4h','CPP6h','PPO1','PPO2','I1','Iz','I2','AFp3h',
    'AFp4h','AFF5h','AFF6h','FFT7h','FFC1h','FFC2h','FFT8h','FTT9h','FTT7h','FCC1h',
    'FCC2h','FTT8h','FTT10h','TTP7h','CCP1h','CCP2h','TTP8h','TPP7h','CPP1h','CPP2h',
    'TPP8h','PPO9h','PPO5h','PPO6h','PPO10h','POO9h','POO3h','POO4h','POO10h','OI1h','OI2h'
]

# LMDB setup
lmdb_path = os.path.join(PROCESSED_ROOT, "MI_Schirrmeister2017", "processed_average")
os.makedirs(os.path.dirname(lmdb_path), exist_ok=True)
db = lmdb.open(lmdb_path, map_size=4614542346)

dataset_keys = {'train': [], 'val': [], 'test': []}

for split in files_dict:
    for subj in files_dict[split]:
        npz_path = os.path.join(root_dir, f'{subj}.npz')
        data = np.load(npz_path, allow_pickle=True)
        x_data, y_data, fs = data['x_data'], data['y_data'].astype(np.uint8), int(data['fs'])

        # Keep only left (2) and right (3) motor-imagery trials
        keep = (y_data == 2) | (y_data == 3)
        x_data, y_data = x_data[keep], y_data[keep] - 2  # labels 0 and 1

        # MNE preprocessing per subject
        info = mne.create_info(ch_names=CH_NAMES, sfreq=fs, ch_types='eeg')
        epochs = mne.EpochsArray(x_data, info, tmin=0, verbose=False)
        epochs.filter(l_freq=1, h_freq=40, verbose=False)
        epochs.resample(200, npad='auto')
        x_data = epochs.get_data(copy=False).astype(np.float32)  # (n, 128, T)

        # Take the last 4 seconds (800 samples) and split into 4 patches
        x_data = x_data[:, :, -800:]
        bsz, chn, _ = x_data.shape
        x_data = x_data.reshape(bsz, chn, 4, 200)

        print(subj, x_data.shape, y_data.shape, np.unique(y_data, return_counts=True))

        for idx, (sample, label) in enumerate(zip(x_data, y_data)):
            key = f'{subj}-{idx}'
            value = pickle.dumps({'sample': sample, 'label': int(label), 'subject': subj})
            with db.begin(write=True) as txn:
                txn.put(key.encode(), value)
            dataset_keys[split].append(key)

# Save split information
with db.begin(write=True) as txn:
    txn.put('__keys__'.encode(), pickle.dumps(dataset_keys))

db.close()
print('Preprocessing finished.')
