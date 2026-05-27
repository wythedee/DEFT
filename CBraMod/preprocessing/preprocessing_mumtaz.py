import os
import mne
import numpy as np
import lmdb
import pickle

from preprocessing.path_utils import eeg_dataset_root, first_existing, processed_root

#遍历文件夹
def iter_files(rootDir):
    #遍历根目录
    files_H, files_MDD = [], []
    for file in os.listdir(rootDir):
        if 'TASK' not in file and file.endswith('.edf'):
            if 'MDD' in file:
                files_MDD.append(file)
            else:
                files_H.append(file)
    return files_H, files_MDD


selected_channels = ['EEG Fp1-LE', 'EEG Fp2-LE', 'EEG F3-LE', 'EEG F4-LE', 'EEG C3-LE', 'EEG C4-LE', 'EEG P3-LE',
                     'EEG P4-LE', 'EEG O1-LE', 'EEG O2-LE', 'EEG F7-LE', 'EEG F8-LE', 'EEG T3-LE', 'EEG T4-LE',
                     'EEG T5-LE', 'EEG T6-LE', 'EEG Fz-LE', 'EEG Cz-LE', 'EEG Pz-LE']
DATASET_ROOT = eeg_dataset_root()
PROCESSED_ROOT = processed_root()

rootDir = first_existing(
    os.path.join(DATASET_ROOT, "MDD", "MDD_01_Mumtaz"),
    os.path.join(DATASET_ROOT, "MDD_01_Mumtaz"),
)
files_H, files_MDD = iter_files(rootDir)
files_H = sorted(files_H)
files_MDD = sorted(files_MDD)
print(files_H)
print(files_MDD)
print(len(files_H), len(files_MDD))


files_dict = {
    'train':[],
    'val':[],
    'test':[],
}

dataset = {
    'train': list(),
    'val': list(),
    'test': list(),
}

files_dict['train'].extend(files_H[:40])
files_dict['train'].extend(files_MDD[:42])
files_dict['val'].extend(files_H[40:48])
files_dict['val'].extend(files_MDD[42:52])
files_dict['test'].extend(files_H[48:])
files_dict['test'].extend(files_MDD[52:])

print(files_dict['train'])
print(files_dict['val'])
print(files_dict['test'])


db_path = os.path.join(PROCESSED_ROOT, "MDD_Mumtaz", "processed_average")
os.makedirs(os.path.dirname(db_path), exist_ok=True)
db = lmdb.open(db_path, map_size=1273741824)

for files_key in files_dict.keys():
    for file in files_dict[files_key]:
        raw = mne.io.read_raw_edf(os.path.join(rootDir, file), preload=True)
        print(raw.info['ch_names'])
        raw.pick_channels(selected_channels, ordered=True)
        print(raw.info['ch_names'])
        raw.resample(200)
        raw.filter(l_freq=0.3, h_freq=75)
        raw.notch_filter((50))
        # raw.plot_psd(average=True)
        eeg_array = raw.to_data_frame().values
        # print(raw.info)
        eeg_array = eeg_array[:, 1:]
        points, chs = eeg_array.shape
        print(eeg_array.shape)
        a = points % (5 * 200)
        print(a)
        if a != 0:
            eeg_array = eeg_array[:-a, :]
        eeg_array = eeg_array.reshape(-1, 5, 200, chs)
        eeg_array = eeg_array.transpose(0, 3, 1, 2)
        print(eeg_array.shape)
        label = 1 if 'MDD' in file else 0
        for i, sample in enumerate(eeg_array):
            sample_key = f'{file[:-4]}_{i}'
            subject_id = file.split('_')[0]
            data_dict = {
                'sample': sample,
                'label': label,
                'subject': subject_id,
            }
            txn = db.begin(write=True)
            txn.put(key=sample_key.encode(), value=pickle.dumps(data_dict))
            txn.commit()
            dataset[files_key].append(sample_key)


txn = db.begin(write=True)
txn.put(key='__keys__'.encode(), value=pickle.dumps(dataset))
txn.commit()
db.close()
