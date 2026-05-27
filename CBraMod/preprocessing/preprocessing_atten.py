#!/usr/bin/env python3
"""
Preprocessing script for ATTEN GoNogo dataset for CBraMod
This script processes the raw ATTEN GoNogo dataset and converts it to LMDB format
for use with the CBraMod model.
"""

import os
import re
import mne
import numpy as np
import lmdb
import pickle
from scipy.signal import resample
from tqdm import tqdm
import traceback

try:
    from preprocessing.path_utils import eeg_dataset_root, processed_root
except ModuleNotFoundError:  # allow running as a script from this directory
    from path_utils import eeg_dataset_root, processed_root


# Configuration
SRC_FOLDER = os.environ.get("EEG_DATASET_ROOT") or os.environ.get("EEG_RAW_ROOT") or eeg_dataset_root()
DATA_FOLDER = os.path.dirname(processed_root())
OUTPUT_DIR = os.path.join(processed_root(), "ATTEN_GoNogo", "processed_average")

# Dataset split configuration (provided by user)
ATTEN_01_GoNogo_split = {
    'train': range(0, 16),   # 26 subjects: 16/5/5 (approx 60/20/20)
    'val': range(16, 21),
    'test': range(21, 26),
}

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def find_dataset_root():
    """Find the root directory for the ATTEN dataset"""
    candidates = [
        os.path.join(SRC_FOLDER, 'ATTEN', 'ATTEN_01_GoNogo'),
        os.path.join(SRC_FOLDER, 'ATT', 'ATTEN_01_GoNogo'),
        os.path.join(SRC_FOLDER, 'ATTEN', 'ThreeCognitiveTasks'),
    ]
    for folder in candidates:
        if os.path.exists(folder):
            return folder
    raise FileNotFoundError(f'None of the candidate paths exist: {candidates}')

def list_subjects(root):
    """List all subjects in the dataset"""
    subs = []
    for d in os.listdir(root):
        # Only consider directories, accept VPxxx or VPxxx-EEG
        if not os.path.isdir(os.path.join(root, d)):
            continue
        if re.match(r'^VP0?\d+(-EEG)?$', d):
            subs.append(d)
    subs.sort(key=lambda s: int(re.findall(r'\d+', s)[0]))
    print(f"[ATTEN][list_subjects] scanned '{root}', found {len(subs)} subjects")
    return subs

def safe_read_brainvision(vhdr_path):
    """Safely read a BrainVision file"""
    print(f"[ATTEN][read] reading BrainVision: {vhdr_path}")
    raw = mne.io.read_raw_brainvision(vhdr_path, preload=True, verbose='ERROR')
    print(f"[ATTEN][read] loaded: sfreq={raw.info.get('sfreq')}, n_ch={len(raw.ch_names)}")
    # Normalize channel types for EOG if present
    eog_map = {}
    for k in ['HEOG', 'VEOG']:
        if k in raw.ch_names:
            eog_map[k] = 'eog'
    if eog_map:
        raw.set_channel_types(eog_map)
    # Usual capitalization for Fp1/Fp2 that appears as FP1/FP2 in some recordings
    rename = {}
    if 'FP1' in raw.ch_names:
        rename['FP1'] = 'Fp1'
    if 'FP2' in raw.ch_names:
        rename['FP2'] = 'Fp2'
    if rename:
        raw.rename_channels(rename)
    try:
        raw.set_montage(mne.channels.make_standard_montage('standard_1005'))
    except Exception:
        pass
    print(f"[ATTEN][read] post-setup: n_ch={len(raw.ch_names)} (EOG may be present)")
    return raw

def extract_epochs_att_rest(raw):
    """Extract attention and rest epochs from raw data"""
    # We expect blocks starting with code 48 (dataset-specific). We'll robustly locate it.
    events, event_id = mne.events_from_annotations(raw, verbose='ERROR')
    print(f"[ATTEN][events] got {len(events)} events, ids={list(event_id.keys())}")
    codes = events[:, 2]

    # Try to map a key that ends with ' 48' or equals '48'
    target_codes = []
    for k, v in event_id.items():
        ks = str(k)
        if ks.endswith(' 48') or ks == '48' or ks.endswith('/S 48'):
            target_codes.append(v)
    if not target_codes:
        # Fallback: some BrainVision exporters write raw code 48 directly in events
        target_codes = [48]
    print(f"[ATTEN][events] target_codes={target_codes}")

    # Build attention and rest epochs around each start code (20s each)
    # Attention: [0, 20]s after the cue; Rest: [-20, 0]s before the next cue
    # We emulate the original script using the same 20s windows.
    # First, construct an events array that contains only the target code
    att_events = events[np.isin(codes, target_codes)]
    if len(att_events) == 0:
        print(f"[ATTEN][epochs] no attention events found")
        return np.zeros((0, 0, 0), dtype=np.float32), np.zeros((0, 0, 0), dtype=np.float32), raw.info['sfreq']

    # For rest, use the same onsets but take the preceding 20s
    rest_events = att_events.copy()
    # Disable the very first trial's rest (no preceding window)
    rest_events[0, 2] = 0
    # Add a synthetic end mark to safely bound the final window
    last = events[-1, 0]
    rest_events = np.concatenate([rest_events, np.array([[last + int(20 * raw.info['sfreq']), 0, target_codes[0]]])])

    epochs_att = mne.Epochs(raw, att_events, event_id=dict(att=target_codes[0]), tmin=0.0, tmax=20.0,
                            baseline=None, preload=True, reject_by_annotation=False, verbose='ERROR')
    epochs_rest = mne.Epochs(raw, rest_events, event_id=dict(rest=target_codes[0]), tmin=-20.0, tmax=0.0,
                             baseline=None, preload=True, reject_by_annotation=False, verbose='ERROR')
    print(f"[ATTEN][epochs] att={len(epochs_att)} rest={len(epochs_rest)}")

    # Resample to project standard (250 Hz)
    RESAMPLE_RATE = 250
    epochs_att.resample(RESAMPLE_RATE, npad='auto')
    epochs_rest.resample(RESAMPLE_RATE, npad='auto')

    X_att = epochs_att.get_data(copy=False).astype(np.float32)  # (N, C, T)
    X_rest = epochs_rest.get_data(copy=False).astype(np.float32)
    print(f"[ATTEN][epochs] X_att={X_att.shape} X_rest={X_rest.shape}")
    return X_att, X_rest, RESAMPLE_RATE

def segment_and_pipeline(X, Y):
    """与ATTEN_01_GoNogo.py一致的分段与标准化流程"""
    # X: (N, C, T) at 250Hz
    if X.size == 0:
        return np.zeros((0, 75, 4 * 200), dtype=np.float32), np.zeros((0,), dtype=np.uint8)
    print(f"[ATTEN][segment] input X={X.shape}, Y={Y.shape}")
    # 分段参数
    segment_length = 4  # 秒
    resample_rate = 200
    window_length = segment_length * 250  # 4s*250Hz=1000
    step = window_length  # 无重叠
    seg_X, seg_Y = [], []
    for i in range(X.shape[0]):
        for start in range(0, X.shape[2] - window_length + 1, step):
            seg = X[i, :, start:start+window_length]  # (C, 1000)
            # 只做重采样，不做robust标准化和通道映射
            seg = resample(seg, 800, axis=-1)
            seg_X.append(seg)
            seg_Y.append(Y[i])
    if len(seg_X) == 0:
        return np.zeros((0, X.shape[1], 4 * 200), dtype=np.float32), np.zeros((0,), dtype=np.uint8)
    X_arr = np.stack(seg_X, axis=0)  # (N_seg, C, 800)
    Y_arr = np.array(seg_Y, dtype=np.uint8)
    # 切分成4个patch
    X_final = X_arr.reshape(X_arr.shape[0], X_arr.shape[1], 4, 200)
    print(f"[ATTEN][segment] final shape: X={X_final.shape}, Y={Y_arr.shape}")
    return X_final, Y_arr

def find_session_vhdr(sub_dir_path, ses):
    """Find the BrainVision header file for a given session"""
    # Search any BrainVision header file that corresponds to the given session index
    # Common patterns: gonogo1.vhdr, gonogo_1.vhdr, GoNogo1.vhdr
    pat = re.compile(r'(?i)gonogo\D*0?%d\.vhdr$' % ses)
    # search in the subject folder recursively (some exports nest files)
    print(f"[ATTEN][vhdr] searching session {ses} under {sub_dir_path}")
    for root, _, files in os.walk(sub_dir_path):
        for fn in files:
            if fn.lower().endswith('.vhdr') and pat.search(fn):
                path = os.path.join(root, fn)
                print(f"[ATTEN][vhdr] found session {ses}: {path}")
                return path
    # Fallback: if there is exactly one .vhdr and ses==1, take it
    vhdrs = []
    for root, _, files in os.walk(sub_dir_path):
        for fn in files:
            if fn.lower().endswith('.vhdr'):
                vhdrs.append(os.path.join(root, fn))
    if len(vhdrs) == 1 and ses == 1:
        print(f"[ATTEN][vhdr] fallback single vhdr for ses=1: {vhdrs[0]}")
        return vhdrs[0]
    print(f"[ATTEN][vhdr] no vhdr for session {ses}")
    return ''

def process_one_subject(sub_dir, sub_idx):
    """Process a single subject"""
    X_all = []
    Y_all = []

    root = find_dataset_root()
    sub_path = os.path.join(root, sub_dir)
    print(f"{sub_dir}: sub_path {sub_path}")
    
    # Process each session
    for ses in (1, 2, 3):
        try:
            print(f"[ATTEN][subject {sub_dir}] processing session {ses}")
            vhdr = find_session_vhdr(sub_path, ses)
            if not vhdr:
                print(f"[ATTEN][subject {sub_dir}] session {ses} skipped: vhdr not found")
                continue
            raw = safe_read_brainvision(vhdr)

            X_att, X_rest, _ = extract_epochs_att_rest(raw)
            if X_att.size == 0:
                print(f"[ATTEN][subject {sub_dir}] session {ses}: no epochs extracted")
                continue
            X = np.concatenate([X_att, X_rest], axis=0)
            Y = np.array([1] * len(X_att) + [0] * len(X_rest), dtype=np.uint8)
            print(f"[ATTEN][subject {sub_dir}] session {ses}: concatenated X={X.shape}, Y={Y.shape}")

            # Use all channels (75 channels as per the unified template)
            # No need to pick specific channels as data is already mapped to template
            X_seg, Y_seg = segment_and_pipeline(X, Y)
            if X_seg.size:
                print(f"{sub_dir}: segments {X_seg.shape}, labels {np.unique(Y_seg, return_counts=True)}")
                X_all.append(X_seg)
                Y_all.append(Y_seg)
            else:
                print(f"[ATTEN][subject {sub_dir}] session {ses}: no segments after pipeline")
        except Exception as e:
            print(f"[ATTEN][subject {sub_dir}] session {ses} ERROR: {e}\n{traceback.format_exc()}")

    if not X_all:
        return sub_dir, np.zeros((0, 75, 4, 200), dtype=np.float32), np.zeros((0,), dtype=np.uint8)
    X_cat = np.concatenate(X_all, axis=0)
    Y_cat = np.concatenate(Y_all, axis=0)
    return sub_dir, X_cat, Y_cat

def process_all():
    """Process all subjects in the dataset"""
    root = find_dataset_root()
    print(f"[ATTEN] root: {root}")
    subjects_src = list_subjects(root)
    print(f"[ATTEN] subjects: {len(subjects_src)} -> {subjects_src[:5]}{'...' if len(subjects_src)>5 else ''}")
    
    # Process subjects in parallel or sequentially
    results = []
    for i, sub_dir in enumerate(tqdm(subjects_src, desc="Processing subjects")):
        try:
            result = process_one_subject(sub_dir, i)
            results.append(result)
        except Exception as e:
            print(f"[ATTEN] Error processing subject {sub_dir}: {e}\n{traceback.format_exc()}")
            # Add empty result for this subject
            results.append((sub_dir, np.zeros((0, 75, 4, 200), dtype=np.float32), np.zeros((0,), dtype=np.uint8)))

    # Open LMDB database
    db = lmdb.open(OUTPUT_DIR, map_size=6612500172)  # ~6GB
    
    try:
        # Create dataset dictionary
        dataset = {
            'train': [],
            'val': [],
            'test': []
        }
        
        # Process results and write to LMDB
        for subject_idx, (sub_dir, X, Y) in enumerate(tqdm(results, desc="Writing to LMDB")):
            if X is None or X.size == 0:
                print(f"[ATTEN] skip {sub_dir}: no usable segments")
                continue
                
            # Determine which set this subject belongs to
            if subject_idx in ATTEN_01_GoNogo_split['train']:
                set_name = 'train'
            elif subject_idx in ATTEN_01_GoNogo_split['val']:
                set_name = 'val'
            elif subject_idx in ATTEN_01_GoNogo_split['test']:
                set_name = 'test'
            else:
                print(f"Warning: Subject {sub_dir} (index {subject_idx}) not assigned to any set, skipping")
                continue
            
            print(f"Processing subject {sub_dir} (index {subject_idx}) -> {set_name}: X shape {X.shape}, Y shape {Y.shape}")
            
            # Process each sample
            for i in range(X.shape[0]):
                sample = X[i]  # Shape: (75, 4, 200)
                label = Y[i]
                
                # Create sample key
                sample_key = f'{sub_dir}-{i}'
                
                # Create data dictionary
                data_dict = {
                    'sample': sample,  # Shape: (75, 4, 200)
                    'label': label,
                    'subject': sub_dir,
                }
                
                # Write to LMDB
                txn = db.begin(write=True)
                txn.put(key=sample_key.encode(), value=pickle.dumps(data_dict))
                txn.commit()
                
                # Add to dataset
                dataset[set_name].append(sample_key)
        
        # Write dataset keys to LMDB
        txn = db.begin(write=True)
        txn.put(key='__keys__'.encode(), value=pickle.dumps(dataset))
        txn.commit()
        
        print("Dataset split summary:")
        print(f"Train samples: {len(dataset['train'])}")
        print(f"Validation samples: {len(dataset['val'])}")
        print(f"Test samples: {len(dataset['test'])}")
        print(f"Total samples: {sum([len(dataset[k]) for k in dataset.keys()])}")
        
    finally:
        db.close()
        
    print(f"Successfully processed ATTEN GoNogo dataset and saved to {OUTPUT_DIR}")

if __name__ == '__main__':
    process_all()
