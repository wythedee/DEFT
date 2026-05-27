import numpy as np
import scipy
from scipy import signal
import os
import lmdb
import pickle
import mne
from scipy.signal import butter, lfilter, resample, filtfilt

try:
    from preprocessing.path_utils import eeg_dataset_root, first_existing, processed_root
except ModuleNotFoundError:  # allow running as a script from this directory
    from path_utils import eeg_dataset_root, first_existing, processed_root

def butter_bandpass(low_cut, high_cut, fs, order=5):
    nyq = 0.5 * fs
    low = low_cut / nyq
    high = high_cut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

# === 新增：读取标签与gdf数据的辅助函数 ===
def load_BCI_IV_IIa_label(label_path):
    labelmat = scipy.io.loadmat(label_path)
    label = np.squeeze(np.array(labelmat['classlabel'])) - 1
    return label


def load_BCI_IV_IIa_data(data_path, tmin=0, tmax=4, baseline=None):
    """使用 mne 读取 gdf 文件并返回 shape=(trials, 22, time) 的数据"""
    raw_data = mne.io.read_raw_gdf(data_path, preload=True, verbose=False)
    events, event_ids = mne.events_from_annotations(raw_data)
    stimcodes = ('769', '770', '771', '772', '783')
    stims = [value for key, value in event_ids.items() if key in stimcodes]
    epochs = mne.Epochs(
        raw_data,
        events,
        event_id=stims,
        tmin=tmin,
        tmax=7,  # 保证后续可以截到 6 s
        event_repeated='drop',
        baseline=baseline,
        preload=True,
        proj=False,
        reject_by_annotation=False,
        verbose=False,
    )
    # 基本带通滤波，与原脚本保持一致频段
    epochs.filter(l_freq=1, h_freq=45, verbose=False)
    # 去除 EOG 通道，仅保留 22 个 EEG 通道
    channels_to_remove = [ch for ch in epochs.ch_names if 'EOG' in ch]
    epochs = epochs.drop_channels(channels_to_remove)
    # 重采样到 250 Hz
    epochs.resample(250, npad='auto')
    x_data = (epochs.get_data(copy=False) * 1e6)  # 转换为微伏
    return x_data


DATASET_ROOT = eeg_dataset_root()
PROCESSED_ROOT = processed_root()

root_dir = first_existing(
    os.path.join(DATASET_ROOT, "MI", "MI_BCI_IV_2a"),
    os.path.join(DATASET_ROOT, "MI_BCI_IV_2a"),
)
files = [file for file in os.listdir(root_dir) if file.endswith('.gdf')]
files = sorted(files)

# files.remove('A04E.mat')
# files.remove('A04T.mat')
# files.remove('A06E.mat')
# files.remove('A06T.mat')
print(files)

files_dict = {
    'train': ['A01E.gdf', 'A01T.gdf', 'A02E.gdf', 'A02T.gdf', 'A03E.gdf', 'A03T.gdf',
              'A04E.gdf', 'A04T.gdf',
              'A05E.gdf', 'A05T.gdf'],
    'val': ['A06E.gdf', 'A06T.gdf', 'A07E.gdf', 'A07T.gdf'],
    'test': ['A08E.gdf', 'A08T.gdf', 'A09E.gdf', 'A09T.gdf'],
}



dataset = {
    'train': list(),
    'val': list(),
    'test': list(),
}

# for file in files:
#     if 'E' in file:
#         files_dict['train'].append(file)
#     else:
#         files_dict['test'].append(file)
#
# print(files_dict)


db_path = os.path.join(PROCESSED_ROOT, "MI_BCI_IV_2a", "processed_average")
os.makedirs(os.path.dirname(db_path), exist_ok=True)
db = lmdb.open(db_path, map_size=1610612736)
for files_key in files_dict.keys():
    for file in files_dict[files_key]:
        print(file)
        # 读取数据与标签
        x_data = load_BCI_IV_IIa_data(os.path.join(root_dir, file))
        label_path = os.path.join(root_dir, 'true_labels', f'{file[:-4]}.mat')
        y_data = load_BCI_IV_IIa_label(label_path)

        # 处理长度不一致问题，截取较小长度
        min_len = min(x_data.shape[0], len(y_data))
        x_data = x_data[:min_len]
        y_data = y_data[:min_len]

        # 保留四类(0:左手,1:右手,2:脚,3:舌)
        # 若存在其他标签可在此过滤，此处不做额外筛选

        # 遍历 trial
        for i, (sample, label) in enumerate(zip(x_data, y_data)):
            # 去直流分量（按通道均值）
            sample = sample - np.mean(sample, axis=0, keepdims=True)
            # 与旧脚本相同的带通滤波
            b, a = butter_bandpass(0.3, 50, 250)
            sample = lfilter(b, a, sample, -1)
            # 截取 2 s – 6 s 区段
            sample = sample[:, 2 * 250:6 * 250]
            # 重采样到 800 (4 秒 × 200 Hz)
            sample = resample(sample, 800, axis=-1)
            # (22, 800) -> (22, 4, 200)
            sample = sample.reshape(22, 4, 200)

            sample_key = f'{file[:-4]}-{i}'
            data_dict = {
                'sample': sample,
                'label': int(label),
                'subject': file[:3],
            }
            txn = db.begin(write=True)
            txn.put(key=sample_key.encode(), value=pickle.dumps(data_dict))
            txn.commit()
            dataset[files_key].append(sample_key)


txn = db.begin(write=True)
txn.put(key='__keys__'.encode(), value=pickle.dumps(dataset))
txn.commit()
db.close()
