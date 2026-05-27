from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

from .standard_1020 import STANDARD_1020 as standard_1020


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    aliases: tuple[str, ...]
    ch_names: tuple[str, ...]


def _norm(name: str) -> str:
    value = str(name).strip().upper()
    if value.startswith("EEG "):
        value = value[4:]
    value = value.replace("-REF", "").replace("-LE", "")
    value = value.replace(" ", "").replace(".", "")
    return value


_STANDARD_MAP = {_norm(ch): ch for ch in standard_1020}
_CHANNEL_ALIASES = {
    "HEOR": "A2",
    "HEOL": "A1",
    "HEOG": "A2",
    "VEOG": "A1",
    "A2-A1": "A2",
    "A2A1": "A2",
    "AFF1": "AF1",
    "AFF2": "AF2",
    "AFF5": "AF5",
    "AFF6": "AF6",
    "AFF5H": "AF5",
    "AFF6H": "AF6",
    "AFP1": "AF1",
    "AFP2": "AF2",
    "AFP3H": "AF3",
    "AFP4H": "AF4",
    "FFT7H": "FT7",
    "FFT8H": "FT8",
    "FTT7H": "FT7",
    "FTT8H": "FT8",
    "FFC1H": "FC1",
    "FFC2H": "FC2",
    "FFC3H": "FC3",
    "FFC4H": "FC4",
    "FFC5H": "FC5",
    "FFC6H": "FC6",
    "FCC1H": "CFC3",
    "FCC2H": "CFC4",
    "FCC3H": "CFC5",
    "FCC4H": "CFC6",
    "FCC5H": "CFC7",
    "FCC6H": "CFC8",
    "CCP1H": "CCP3",
    "CCP2H": "CCP4",
    "CCP3H": "CCP5",
    "CCP4H": "CCP6",
    "CCP5H": "CCP7",
    "CCP6H": "CCP8",
    "CPP1H": "CP1",
    "CPP2H": "CP2",
    "CPP3H": "CP3",
    "CPP4H": "CP4",
    "CPP5H": "CP5",
    "CPP6H": "CP6",
    "PPO1": "PO1",
    "PPO2": "PO2",
    "PPO1H": "PO1",
    "PPO2H": "PO2",
    "PPO5H": "PO5",
    "PPO6H": "PO6",
    "PPO9H": "PO9",
    "PPO10H": "PO10",
    "POO1": "PO1",
    "POO2": "PO2",
    "POO3H": "PO3",
    "POO4H": "PO4",
    "POO9H": "PO9",
    "POO10H": "PO10",
    "OI1H": "O1",
    "OI2H": "O2",
    "I1": "O1",
    "I2": "O2",
    "TTP8H": "TPP8H",
    "TPP7H": "TP7",
}
_ALIAS_MAP = {_norm(src): dst for src, dst in _CHANNEL_ALIASES.items()}

_SEED_60_CH_NAMES = (
    "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4",
    "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "T7",
    "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3", "CP1",
    "CPZ", "CP2", "CP4", "CP6", "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4",
    "P6", "P8", "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "O1", "OZ", "O2",
)

_SEED_V_62_CH_NAMES = (
    "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4",
    "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "T7",
    "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3", "CP1",
    "CPZ", "CP2", "CP4", "CP6", "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4",
    "P6", "P8", "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "CB1", "O1", "OZ", "O2", "CB2",
)

_FACED_CH_NAMES = (
    "FP1", "FP2", "FZ", "F3", "F4", "F7", "F8", "FC1", "FC2", "FC5", "FC6", "CZ",
    "C3", "C4", "T7", "T8", "CP1", "CP2", "CP5", "CP6", "PZ", "P3", "P4", "P7",
    "P8", "PO3", "PO4", "OZ", "O1", "O2", "A2", "A1",
)

_MUMTAZ_CH_NAMES = (
    "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8", "T3", "T4",
    "T5", "T6", "FZ", "CZ", "PZ",
)

_SHU_CH_NAMES = (
    "FP1", "FP2", "FZ", "F3", "F4", "F7", "F8", "FC1", "FC2", "FC5", "FC6", "CZ", "C3", "C4",
    "T3", "T4", "A1", "A2", "CP1", "CP2", "CP5", "CP6", "PZ", "P3", "P4", "T5", "T6", "PO3",
    "PO4", "OZ", "O1", "O2",
)

_SCHIRRMEISTER_RAW_CH_NAMES = (
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
)

_MENTAL_ARITHMETIC_CH_NAMES = (
    "FP1", "FP2", "F3", "F4", "F7", "F8", "T3", "T4", "C3", "C4", "T5", "T6", "P3", "P4",
    "O1", "O2", "FZ", "CZ", "PZ", "A2",
)

_ATTEN_CH_NAMES = (
    "FP1", "AF5", "AFZ", "F1", "FC5", "FC1", "T7", "C3", "CZ", "CP5", "CP1", "P7", "P3", "PZ", "POZ", "O1",
    "FP2", "AF6", "F2", "FC2", "FC6", "C4", "T8", "CP2", "CP6", "P4", "P8", "O2", "A2", "A1",
)

_MI_TRACK1_FEWSHOT_CH_NAMES = (
    "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "FC5", "FC1", "FC2", "FC6", "T7", "C3", "CZ", "C4", "T8",
    "TP9", "CP5", "CP1", "CP2", "CP6", "TP10", "P7", "P3", "PZ", "P4", "P8", "PO9", "O1", "OZ", "O2", "PO10",
    "FC3", "FC4", "C5", "C1", "C2", "C6", "CP3", "CPZ", "CP4", "P1", "P2", "POZ", "FT9", "FTT9H", "TTP7H", "TP7",
    "TPP9H", "FT10", "FTT10H", "TPP8H", "TP8", "TPP10H", "F9", "F10", "AF7", "AF3", "AF4", "AF8", "PO3", "PO4",
)

_CS_BCIC_TRACK4_CH_NAMES = (
    "FP1", "AF7", "AF3", "AFZ", "F7", "F5", "F3", "F1", "FZ", "FT7", "FC5", "FC3", "FC1", "T7",
    "C5", "C3", "C1", "CZ", "TP7", "CP5", "CP3", "CP1", "CPZ", "P7", "P5", "P3", "P1", "PZ",
    "PO7", "PO3", "POZ", "FP2", "AF4", "AF8", "F2", "F4", "F6", "F8", "FC2", "FC4", "FC6", "FT8",
    "C2", "C4", "C6", "T8", "CP2", "CP4", "CP6", "TP8", "P2", "P4", "P6", "P8", "PO4", "PO8",
    "O1", "OZ", "O2", "IZ",
)

PRESET_SPECS: Dict[str, DatasetSpec] = {
    "BCIC-IV-2a": DatasetSpec(
        name="BCIC-IV-2a",
        aliases=("BCIC-IV-2a", "MI_BCI_IV_2a", "BCIC_IV_2a"),
        ch_names=(
            "FZ", "FC3", "FC1", "FCZ", "FC2", "FC4", "C5", "C3", "C1", "CZ", "C2", "C4",
            "C6", "CP3", "CP1", "CPZ", "CP2", "CP4", "P1", "PZ", "P2", "POZ",
        ),
    ),
    "SEED-IV": DatasetSpec(
        name="SEED-IV",
        aliases=("SEED-IV", "EMO_SEED_IV", "SEED_IV"),
        ch_names=_SEED_60_CH_NAMES,
    ),
    "SEED": DatasetSpec(
        name="SEED",
        aliases=("SEED", "EMO_SEED"),
        ch_names=_SEED_60_CH_NAMES,
    ),
    "SEED-V": DatasetSpec(
        name="SEED-V",
        aliases=("SEED-V", "EMO_SEED_V", "SEED_V"),
        ch_names=_SEED_V_62_CH_NAMES,
    ),
    "FACED": DatasetSpec(
        name="FACED",
        aliases=("FACED", "EMO_FACED"),
        ch_names=_FACED_CH_NAMES,
    ),
    "HeBin2021-LR": DatasetSpec(
        name="HeBin2021-LR",
        aliases=("HeBin2021-LR", "MI_HeBin2021_LR", "HeBin2021_LR"),
        ch_names=_SEED_60_CH_NAMES,
    ),
    "HeBin2021-UD": DatasetSpec(
        name="HeBin2021-UD",
        aliases=("HeBin2021-UD", "MI_HeBin2021_UD", "HeBin2021_UD"),
        ch_names=_SEED_60_CH_NAMES,
    ),
    "MI-KoreaU": DatasetSpec(
        name="MI-KoreaU",
        aliases=("MI-KoreaU", "MI_KoreaU"),
        ch_names=(
            "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "FC5", "FC1", "FC2", "FC6", "T7", "C3",
            "CZ", "C4", "T8", "TP9", "CP5", "CP1", "CP2", "CP6", "TP10", "P7", "P3", "PZ", "P4",
            "P8", "PO9", "O1", "OZ", "O2", "PO10", "FC3", "FC4", "C5", "C1", "C2", "C6", "CP3",
            "CPZ", "CP4", "P1", "P2", "POZ", "FT9", "FTT9H", "TTP7H", "TP7", "TPP9H", "FT10",
            "FTT10H", "TPP8H", "TP8", "TPP10H", "F9", "F10", "AF7", "AF3", "AF4", "AF8", "PO3", "PO4",
        ),
    ),
    "MI-Cho2017": DatasetSpec(
        name="MI-Cho2017",
        aliases=("MI-Cho2017", "MI_Cho2017", "Cho2017"),
        ch_names=(
            "FP1", "AF7", "AF3", "F1", "F3", "F5", "F7", "FT7", "FC5", "FC3", "FC1", "C1", "C3",
            "C5", "T7", "TP7", "CP5", "CP3", "CP1", "P1", "P3", "P5", "P7", "P9", "PO7", "PO3", "O1",
            "IZ", "OZ", "POZ", "PZ", "CPZ", "FPZ", "FP2", "AF8", "AF4", "AFZ", "FZ", "F2", "F4", "F6",
            "F8", "FT8", "FC6", "FC4", "FC2", "FCZ", "CZ", "C2", "C4", "C6", "T8", "TP8", "CP6", "CP4",
            "CP2", "P2", "P4", "P6", "P8", "P10", "PO8", "PO4", "O2",
        ),
    ),
    "PhysioNet-MI": DatasetSpec(
        name="PhysioNet-MI",
        aliases=("PhysioNet-MI", "MI_PhysioNet", "PhysioNet"),
        ch_names=(
            "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "C5", "C3", "C1", "CZ", "C2", "C4", "C6",
            "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "FP1", "FPZ", "FP2", "AF7", "AF3", "AFZ",
            "AF4", "AF8", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8", "FT7", "FT8", "T7",
            "T8", "T9", "T10", "TP7", "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",
            "PO7", "PO3", "POZ", "PO4", "PO8", "O1", "OZ", "O2", "IZ",
        ),
    ),
    "Mumtaz2016": DatasetSpec(
        name="Mumtaz2016",
        aliases=("Mumtaz2016", "MDD_Mumtaz", "Mumtaz"),
        ch_names=_MUMTAZ_CH_NAMES,
    ),
    "SHU-MI": DatasetSpec(
        name="SHU-MI",
        aliases=("SHU-MI", "MI_SHU", "SHU"),
        ch_names=_SHU_CH_NAMES,
    ),
    "Schirrmeister2017": DatasetSpec(
        name="Schirrmeister2017",
        aliases=("Schirrmeister2017", "MI_Schirrmeister2017", "Schirrmeister"),
        ch_names=_SCHIRRMEISTER_RAW_CH_NAMES,
    ),
    "MentalArithmetic": DatasetSpec(
        name="MentalArithmetic",
        aliases=("MentalArithmetic", "STR_MentalArithmetic", "Mental-Arithmetic"),
        ch_names=_MENTAL_ARITHMETIC_CH_NAMES,
    ),
    "ATTEN": DatasetSpec(
        name="ATTEN",
        aliases=("ATTEN", "ATTEN_GoNogo"),
        ch_names=_ATTEN_CH_NAMES,
    ),
    "MI-Track1-FewShot": DatasetSpec(
        name="MI-Track1-FewShot",
        aliases=("MI-Track1-FewShot", "MI_08_Track1_Few_shot", "Track1-FewShot"),
        ch_names=_MI_TRACK1_FEWSHOT_CH_NAMES,
    ),
    "ISRUC": DatasetSpec(
        name="ISRUC",
        aliases=("ISRUC", "SLEEP_05_isruc", "ISRUC-S1"),
        ch_names=("F3", "C3", "O1", "F4", "C4", "O2"),
    ),
    "ISRUC-S3": DatasetSpec(
        name="ISRUC-S3",
        aliases=("ISRUC-S3", "ISRUC_S3", "SLEEP_05_isruc_S3"),
        ch_names=("F3", "C3", "O1", "F4", "C4", "O2"),
    ),
    "CS-BCIC-Track4": DatasetSpec(
        name="CS-BCIC-Track4",
        aliases=("CS-BCIC-Track4", "CS_04_BCIC_Track3", "BCIC_Track4", "Track4"),
        ch_names=_CS_BCIC_TRACK4_CH_NAMES,
    ),
}


def parse_channel_names(raw: str) -> list[str]:
    value = str(raw or "").strip()
    if not value:
        return []
    path = Path(value)
    if path.exists():
        text = path.read_text(encoding="utf-8").strip()
        try:
            loaded = json.loads(text)
            if isinstance(loaded, list):
                return [str(item) for item in loaded]
        except json.JSONDecodeError:
            pass
        value = text
    value = value.replace("\n", ",")
    parts = [part.strip() for part in value.split(",") if part.strip()]
    return parts


def canonicalize_channel_names(ch_names: Sequence[str]) -> list[str]:
    resolved = []
    missing = []
    for ch in ch_names:
        key = _norm(ch)
        canonical = _STANDARD_MAP.get(key)
        if canonical is None:
            alias = _ALIAS_MAP.get(key)
            if alias is not None:
                canonical = alias
        if canonical is None:
            missing.append(str(ch))
        else:
            resolved.append(canonical)
    if missing:
        raise ValueError(
            "Some channels are not in LaBraM standard_1020 and need manual remapping: "
            + ", ".join(missing)
        )
    return resolved


def resolve_preset(dataset_name: str = "", lmdb_dir: str = "") -> Optional[DatasetSpec]:
    dataset_name = str(dataset_name or "").strip()
    if dataset_name:
        for spec in PRESET_SPECS.values():
            if dataset_name == spec.name or dataset_name in spec.aliases:
                return spec
    lowered = str(lmdb_dir or "").lower()
    for spec in PRESET_SPECS.values():
        for alias in spec.aliases:
            if alias.lower() in lowered:
                return spec
    return None


def resolve_channel_names(dataset_name: str = "", lmdb_dir: str = "", channel_names_override: str = "") -> tuple[str, list[str]]:
    override = parse_channel_names(channel_names_override)
    if override:
        preset = resolve_preset(dataset_name, lmdb_dir)
        resolved_name = dataset_name or (preset.name if preset else "custom")
        return resolved_name, canonicalize_channel_names(override)

    preset = resolve_preset(dataset_name, lmdb_dir)
    if preset is None:
        raise ValueError(
            "Could not infer channel names from dataset_name/lmdb_dir. "
            "Pass --channel_names with a comma-separated list or a JSON/text file."
        )
    return preset.name, canonicalize_channel_names(preset.ch_names)
