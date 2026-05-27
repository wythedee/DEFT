from collections import Counter

import torch
import torch.nn as nn

from .CSBrain import CSBrain


DATASET_CHANNEL_SPECS = {
    "SEED": {
        "channels": (
            "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4",
            "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "T7",
            "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3", "CP1",
            "CPZ", "CP2", "CP4", "CP6", "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4",
            "P6", "P8", "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "O1", "OZ", "O2",
        ),
        "seq_len": 4,
        "task_kind": "binary",
    },
    "SEED-IV": {
        "channels": (
            "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4",
            "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "T7",
            "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3", "CP1",
            "CPZ", "CP2", "CP4", "CP6", "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4",
            "P6", "P8", "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "O1", "OZ", "O2",
        ),
        "seq_len": 4,
        "task_kind": "multiclass",
    },
    "HeBin2021-LR": {
        "channels": (
            "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4",
            "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "T7",
            "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3", "CP1",
            "CPZ", "CP2", "CP4", "CP6", "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4",
            "P6", "P8", "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "O1", "OZ", "O2",
        ),
        "seq_len": 4,
        "task_kind": "binary",
    },
    "HeBin2021-UD": {
        "channels": (
            "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4",
            "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "T7",
            "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3", "CP1",
            "CPZ", "CP2", "CP4", "CP6", "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4",
            "P6", "P8", "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "O1", "OZ", "O2",
        ),
        "seq_len": 4,
        "task_kind": "binary",
    },
    "MI-KoreaU": {
        "channels": (
            "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "FC5", "FC1", "FC2", "FC6", "T7", "C3",
            "CZ", "C4", "T8", "TP9", "CP5", "CP1", "CP2", "CP6", "TP10", "P7", "P3", "PZ", "P4",
            "P8", "PO9", "O1", "OZ", "O2", "PO10", "FC3", "FC4", "C5", "C1", "C2", "C6", "CP3",
            "CPZ", "CP4", "P1", "P2", "POZ", "FT9", "FTT9H", "TTP7H", "TP7", "TPP9H", "FT10",
            "FTT10H", "TPP8H", "TP8", "TPP10H", "F9", "F10", "AF7", "AF3", "AF4", "AF8", "PO3", "PO4",
        ),
        "seq_len": 4,
        "task_kind": "binary",
    },
    "MI-Cho2017": {
        "channels": (
            "FP1", "AF7", "AF3", "F1", "F3", "F5", "F7", "FT7", "FC5", "FC3", "FC1", "C1", "C3",
            "C5", "T7", "TP7", "CP5", "CP3", "CP1", "P1", "P3", "P5", "P7", "P9", "PO7", "PO3", "O1",
            "IZ", "OZ", "POZ", "PZ", "CPZ", "FPZ", "FP2", "AF8", "AF4", "AFZ", "FZ", "F2", "F4", "F6",
            "F8", "FT8", "FC6", "FC4", "FC2", "FCZ", "CZ", "C2", "C4", "C6", "T8", "TP8", "CP6", "CP4",
            "CP2", "P2", "P4", "P6", "P8", "P10", "PO8", "PO4", "O2",
        ),
        "seq_len": 4,
        "task_kind": "binary",
    },
    "Schirrmeister2017": {
        "channels": (
            "Fp1", "Fp2", "Fpz", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6",
            "M1", "T7", "C3", "Cz", "C4", "T8", "M2", "CP5", "CP1", "CP2", "CP6", "P7", "P3",
            "Pz", "P4", "P8", "POz", "O1", "Oz", "O2", "AF7", "AF3", "AF4", "AF8", "F5", "F1",
            "F2", "F6", "FC3", "FCz", "FC4", "C5", "C1", "C2", "C6", "CP3", "CPz", "CP4", "P5",
            "P1", "P2", "P6", "PO5", "PO3", "PO4", "PO6", "FT7", "FT8", "TP7", "TP8", "PO7", "PO8",
            "FT9", "FT10", "TPP9h", "TPP10h", "PO9", "PO10", "P9", "P10", "AFF1", "AFz", "AFF2",
            "FFC5h", "FFC3h", "FFC4h", "FFC6h", "FCC5h", "FCC3h", "FCC4h", "FCC6h", "CCP5h", "CCP3h",
            "CCP4h", "CCP6h", "CPP5h", "CPP3h", "CPP4h", "CPP6h", "PPO1", "PPO2", "I1", "Iz", "I2", "AFp3h",
            "AFp4h", "AFF5h", "AFF6h", "FFT7h", "FFC1h", "FFC2h", "FFT8h", "FTT9h", "FTT7h", "FCC1h",
            "FCC2h", "FTT8h", "FTT10h", "TTP7h", "CCP1h", "CCP2h", "TTP8h", "TPP7h", "CPP1h", "CPP2h",
            "TPP8h", "PPO9h", "PPO5h", "PPO6h", "PPO10h", "POO9h", "POO3h", "POO4h", "POO10h", "OI1h", "OI2h",
        ),
        "seq_len": 4,
        "task_kind": "binary",
    },
    "CS-BCIC-Track4": {
        "channels": (
            "FP1", "AF7", "AF3", "AFZ", "F7", "F5", "F3", "F1", "FZ", "FT7", "FC5", "FC3", "FC1", "T7",
            "C5", "C3", "C1", "CZ", "TP7", "CP5", "CP3", "CP1", "CPZ", "P7", "P5", "P3", "P1", "PZ",
            "PO7", "PO3", "POZ", "FP2", "AF4", "AF8", "F2", "F4", "F6", "F8", "FC2", "FC4", "FC6", "FT8",
            "C2", "C4", "C6", "T8", "CP2", "CP4", "CP6", "TP8", "P2", "P4", "P6", "P8", "PO4", "PO8",
            "O1", "OZ", "O2", "IZ",
        ),
        "seq_len": 3,
        "task_kind": "multiclass",
    },
}


def _normalize_channel_name(name: str) -> str:
    value = str(name).strip().upper()
    if value.startswith("EEG "):
        value = value[4:]
    return value.replace("-REF", "").replace("-LE", "").replace(" ", "").replace(".", "")


def _region_id(channel_name: str) -> int:
    key = _normalize_channel_name(channel_name)
    if key in {"A1", "A2", "M1", "M2"}:
        return 2
    temporal_prefixes = ("FFT", "FTT", "FT", "TTP", "TPP", "TP", "T")
    central_prefixes = ("FCC", "CFC", "FFC", "FC", "CCP", "CPP", "CP", "C")
    occipital_prefixes = ("PPO", "POO", "PO", "CB", "OI", "IZ", "O", "I")
    frontal_prefixes = ("AFP", "AFF", "AF", "FP", "F")
    if key.startswith(temporal_prefixes):
        return 2
    if key.startswith(central_prefixes):
        return 4
    if key.startswith(occipital_prefixes):
        return 3
    if key.startswith("P"):
        return 1
    if key.startswith(frontal_prefixes):
        return 0
    return 4


def _build_region_layout(channel_names):
    brain_regions = [_region_id(ch) for ch in channel_names]
    region_groups = {region: [] for region in range(5)}
    for index, region in enumerate(brain_regions):
        region_groups[region].append(index)
    sorted_indices = []
    for region in range(5):
        sorted_indices.extend(region_groups[region])
    return brain_regions, sorted_indices


class GenericCSBrainFlatModel(nn.Module):
    def __init__(self, param):
        super().__init__()
        if param.downstream_dataset not in DATASET_CHANNEL_SPECS:
            raise ValueError(f"Unsupported generic downstream dataset: {param.downstream_dataset}")

        spec = DATASET_CHANNEL_SPECS[param.downstream_dataset]
        self.dataset_name = param.downstream_dataset
        self.seq_len = spec["seq_len"]
        self.task_kind = spec["task_kind"]
        self.channel_names = list(spec["channels"])
        self.channel_num = len(self.channel_names)
        self.flat_dim = self.channel_num * self.seq_len * 200
        self.output_dim = 1 if self.task_kind == "binary" else param.num_of_classes

        brain_regions, sorted_indices = _build_region_layout(self.channel_names)
        region_counts = Counter(brain_regions)
        print(
            f"{self.dataset_name} generic CSBrain config: "
            f"channels={self.channel_num}, seq_len={self.seq_len}, region_counts={dict(sorted(region_counts.items()))}"
        )
        print(f"{self.dataset_name} sorted indices: {sorted_indices}")

        if param.model != "CSBrain":
            raise ValueError(f"GenericCSBrainFlatModel only supports model=CSBrain, got {param.model}")

        self.backbone = CSBrain(
            in_dim=200,
            out_dim=200,
            d_model=200,
            dim_feedforward=800,
            seq_len=30,
            n_layer=param.n_layer,
            nhead=8,
            brain_regions=brain_regions,
            sorted_indices=sorted_indices,
        )

        if param.use_pretrained_weights:
            map_location = torch.device(f"cuda:{param.cuda}")
            state_dict = torch.load(param.foundation_dir, map_location=map_location)
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key.replace("module.", "").replace("backbone.", "")
                new_state_dict[new_key] = value
            model_state_dict = self.backbone.state_dict()
            matching_dict = {
                key: value
                for key, value in new_state_dict.items()
                if key in model_state_dict and value.size() == model_state_dict[key].size()
            }
            model_state_dict.update(matching_dict)
            self.backbone.load_state_dict(model_state_dict)

        self.backbone.proj_out = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(self.flat_dim, 4 * 200),
            nn.ELU(),
            nn.Dropout(param.dropout),
            nn.Linear(4 * 200, 200),
            nn.ELU(),
            nn.Dropout(param.dropout),
            nn.Linear(200, self.output_dim),
        )

    def forward(self, x):
        feats = self.backbone(x)
        out = self.classifier(feats.contiguous().view(x.shape[0], self.flat_dim))
        if self.task_kind == "binary":
            out = out[:, 0]
        return out


Model = GenericCSBrainFlatModel
