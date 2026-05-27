import os
import lmdb
import pickle
import re
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import scipy.io
from scipy import signal

try:
    from preprocessing.path_utils import eeg_dataset_root, first_existing, processed_root
except ModuleNotFoundError:
    from path_utils import eeg_dataset_root, first_existing, processed_root

DATASET_ROOT = eeg_dataset_root()
PROCESSED_ROOT = processed_root()

BASE_DIR = first_existing(
    os.environ.get("CS_BCIC_TRACK4_ROOT", ""),
    os.path.join(DATASET_ROOT, "CS", "CS_04_BCIC_Track3"),
    os.path.join(DATASET_ROOT, "CS_04_BCIC_Track3"),
)
TRAIN_DIR = os.path.join(BASE_DIR, "Training set")
VAL_DIR = os.path.join(BASE_DIR, "Validation set")
TEST_DIR = first_existing(
    os.path.join(BASE_DIR, "Test set"),
    os.path.join(BASE_DIR, "Test set (new)"),
)
PROCESSED_DIR = os.path.join(PROCESSED_ROOT, "CS_04_BCIC_Track3", "processed")

XML_NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _safe_listdir(path):
    if not os.path.isdir(path):
        print(f"Warning: directory not found: {path}")
        return []
    return sorted([f for f in os.listdir(path) if f.endswith(".mat")])


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _normalise_sample_name(name: str) -> str:
    base = os.path.splitext(os.path.basename(name))[0].lower()
    base = base.replace("data_", "").replace("data", "")
    base = re.sub(r"[^a-z0-9]+", "", base)
    return base


def _to_trials_channels_time(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape={x.shape}")
    # Local Track4 files store (time, channels, trials).
    if x.shape[0] > x.shape[2]:
        x = np.transpose(x, (2, 1, 0))
    return x


def _prepare_segments(x: np.ndarray) -> np.ndarray:
    x = _to_trials_channels_time(x)
    if x.shape[-1] >= 768:
        x = x[:, :, -768:]
    else:
        pad = 768 - x.shape[-1]
        x = np.pad(x, ((0, 0), (0, 0), (0, pad)), mode="edge")
    x = signal.resample(x, 600, axis=2)
    x = x.reshape(x.shape[0], x.shape[1], 3, 200)
    return x.astype(np.float32)


def _load_epo(path: str):
    data = scipy.io.loadmat(path)
    if "epo" not in data:
        raise KeyError(f"Missing 'epo' in {path}")
    return data["epo"][0][0]


def _parse_mat_train_val(mat_path: str):
    epo = _load_epo(mat_path)
    x = _prepare_segments(epo["x"])
    y = np.asarray(epo["y"])
    if y.ndim != 2:
        raise ValueError(f"Unexpected y shape in {mat_path}: {y.shape}")
    labels = np.argmax(y, axis=0).astype(np.uint8)
    return x, labels


def _load_shared_strings(zf: zipfile.ZipFile):
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    strings = []
    for si in root.findall("a:si", XML_NS):
        texts = [node.text or "" for node in si.findall(".//a:t", XML_NS)]
        strings.append("".join(texts))
    return strings


def _cell_value(cell, shared_strings):
    value_node = cell.find("a:v", XML_NS)
    if value_node is None:
        return ""
    value = value_node.text or ""
    if cell.attrib.get("t") == "s":
        return shared_strings[int(value)]
    return value


def _col_to_index(col: str) -> int:
    index = 0
    for ch in col:
        index = index * 26 + (ord(ch.upper()) - ord("A") + 1)
    return index


def _index_to_col(index: int) -> str:
    out = []
    while index > 0:
        index, rem = divmod(index - 1, 26)
        out.append(chr(ord("A") + rem))
    return "".join(reversed(out))


def _parse_track4_labels_from_xlsx(path: str):
    with zipfile.ZipFile(path) as zf:
        shared_strings = _load_shared_strings(zf)
        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        sheet_name = workbook.find(".//a:sheets/a:sheet", XML_NS)
        if sheet_name is None:
            return {}
        sheet = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))

    rows = {}
    for row in sheet.findall(".//a:row", XML_NS):
        row_num = int(row.attrib["r"])
        rows[row_num] = {}
        for cell in row.findall("a:c", XML_NS):
            rows[row_num][cell.attrib["r"]] = _cell_value(cell, shared_strings)

    sample_cols = []
    for ref, value in rows.get(2, {}).items():
        if "sample" in value.lower():
            col = re.match(r"([A-Z]+)", ref).group(1)
            sample_cols.append((col, value))

    labels_map = {}
    for sample_col, sample_header in sample_cols:
        sample_name = _normalise_sample_name(sample_header)
        label_col = _index_to_col(_col_to_index(sample_col) + 1)
        labels = []
        for row_num in sorted(k for k in rows.keys() if k >= 4):
            val = rows[row_num].get(f"{label_col}{row_num}", "")
            if val == "":
                continue
            try:
                labels.append(int(float(val)))
            except Exception:
                continue
        if labels:
            labels_map[sample_name] = np.array(labels, dtype=np.uint8)
    return labels_map


def _try_read_test_labels(base_dir: str):
    candidates = [
        os.path.join(base_dir, "Test set (new)", "Track4_Answer Sheet_Test.xlsx"),
        os.path.join(base_dir, "Track4_Answer Sheet_Test.xlsx"),
        os.path.join(base_dir, "answer_sheet_track4.xlsx"),
        os.path.join(base_dir, "Test set", "Track4_Answer Sheet_Test.xlsx"),
    ]
    for path in candidates:
        if os.path.exists(path):
            labels_map = _parse_track4_labels_from_xlsx(path)
            if labels_map:
                print(f"Loaded true test labels from {path} for {len(labels_map)} files")
                return labels_map
    print("Warning: could not parse true labels for Track4 test set. Test labels will default to 0.")
    return {}


def _parse_mat_test(mat_path: str, sample_name: str, true_labels_dict):
    epo = _load_epo(mat_path)
    x = _prepare_segments(epo["x"])
    norm_name = _normalise_sample_name(sample_name)
    if norm_name in true_labels_dict and len(true_labels_dict[norm_name]) == x.shape[0]:
        y = true_labels_dict[norm_name]
    else:
        y = np.zeros((x.shape[0],), dtype=np.uint8)
    return x, y


def main():
    files_dict = {
        "train": _safe_listdir(TRAIN_DIR),
        "val": _safe_listdir(VAL_DIR),
        "test": _safe_listdir(TEST_DIR),
    }
    print(files_dict)

    dataset = {"train": [], "val": [], "test": []}
    _ensure_dir(PROCESSED_DIR)
    db = lmdb.open(PROCESSED_DIR, map_size=6 * 1024 * 1024 * 1024)

    for split, split_dir in (("train", TRAIN_DIR), ("val", VAL_DIR)):
        for file in files_dict[split]:
            x, y = _parse_mat_train_val(os.path.join(split_dir, file))
            for i, (sample, label) in enumerate(zip(x, y)):
                sample_key = f"{split}-{file[:-4]}-{i}"
                data_dict = {"sample": sample, "label": int(label), "subject": file[:-4]}
                with db.begin(write=True) as txn:
                    txn.put(sample_key.encode(), pickle.dumps(data_dict))
                dataset[split].append(sample_key)
            print(f"[{split}] {file}: {x.shape} labels={y.shape} classes={sorted(set(y.tolist()))}")

    true_labels = _try_read_test_labels(BASE_DIR)
    for file in files_dict["test"]:
        x, y = _parse_mat_test(os.path.join(TEST_DIR, file), file[:-4], true_labels)
        for i, (sample, label) in enumerate(zip(x, y)):
            sample_key = f"test-{file[:-4]}-{i}"
            data_dict = {"sample": sample, "label": int(label), "subject": file[:-4]}
            with db.begin(write=True) as txn:
                txn.put(sample_key.encode(), pickle.dumps(data_dict))
            dataset["test"].append(sample_key)
        print(f"[test] {file}: {x.shape} labels={y.shape} classes={sorted(set(y.tolist()))}")

    with db.begin(write=True) as txn:
        txn.put(b"__keys__", pickle.dumps(dataset))
    db.close()
    print(f"Done. LMDB saved at: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
