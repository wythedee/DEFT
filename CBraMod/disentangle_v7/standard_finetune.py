from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Dict, Literal, Tuple

import torch


TaskKind = Literal["multiclass", "binary", "regression"]


@dataclass(frozen=True)
class FinetuneSpec:
    dataset_module: str
    model_module: str
    task: TaskKind


FINETUNE_SPECS: Dict[str, FinetuneSpec] = {
    "FACED": FinetuneSpec("datasets.faced_dataset", "models.model_for_faced", "multiclass"),
    "SEED-V": FinetuneSpec("datasets.seedv_dataset", "models.model_for_seedv", "multiclass"),
    "PhysioNet-MI": FinetuneSpec("datasets.physio_dataset", "models.model_for_physio", "multiclass"),
    "SHU-MI": FinetuneSpec("datasets.shu_dataset", "models.model_for_shu", "binary"),
    "ISRUC": FinetuneSpec("datasets.isruc_dataset", "models.model_for_isruc", "multiclass"),
    "ISRUC-S3": FinetuneSpec("datasets.isruc_s3_dataset", "models.model_for_isruc", "multiclass"),
    "CHB-MIT": FinetuneSpec("datasets.chb_dataset", "models.model_for_chb", "binary"),
    "BCIC2020-3": FinetuneSpec("datasets.speech_dataset", "models.model_for_speech", "multiclass"),
    "CS-BCIC-Track4": FinetuneSpec(
        "datasets.cs_bcic_track4_dataset",
        "models.model_for_cs_bcic_track4",
        "multiclass",
    ),
    "Mumtaz2016": FinetuneSpec("datasets.mumtaz_dataset", "models.model_for_mumtaz", "binary"),
    "SEED-VIG": FinetuneSpec("datasets.seedvig_dataset", "models.model_for_seedvig", "regression"),
    "MentalArithmetic": FinetuneSpec("datasets.stress_dataset", "models.model_for_stress", "binary"),
    "TUEV": FinetuneSpec("datasets.tuev_dataset", "models.model_for_tuev", "multiclass"),
    "TUAB": FinetuneSpec("datasets.tuab_dataset", "models.model_for_tuab", "binary"),
    "BCIC-IV-2a": FinetuneSpec("datasets.bciciv2a_dataset", "models.model_for_bciciv2a", "multiclass"),
    "Schirrmeister2017": FinetuneSpec("datasets.sch2017_dataset", "models.model_for_sch2017", "binary"),
    "HeBin2021-LR": FinetuneSpec("datasets.hebin2021_dataset", "models.model_for_hebin", "binary"),
    "HeBin2021-UD": FinetuneSpec("datasets.hebin2021_dataset", "models.model_for_hebin", "binary"),
    "SEED-IV": FinetuneSpec("datasets.seediv_dataset", "models.model_for_seediv", "multiclass"),
    "SEED": FinetuneSpec("datasets.seed_dataset", "models.model_for_seed", "binary"),
    "ATTEN": FinetuneSpec("datasets.atten_dataset", "models.model_for_atten", "binary"),
    "MI-KoreaU": FinetuneSpec("datasets.mi_koreau_dataset", "models.model_for_mi_koreau", "binary"),
    "MI-Track1-FewShot": FinetuneSpec(
        "datasets.mi_track1_fewshot_dataset",
        "models.model_for_mi_track1_fewshot",
        "binary",
    ),
    "MI-Cho2017": FinetuneSpec("datasets.mi_cho2017_dataset", "models.model_for_mi_cho2017", "binary"),
    "MI-Shin2017A": FinetuneSpec("datasets.mi_shin2017a_dataset", "models.model_for_mi_shin2017a", "binary"),
    "MI-Weibo2014": FinetuneSpec("datasets.mi_weibo2014_dataset", "models.model_for_mi_weibo2014", "binary"),
}


def available_downstream_datasets() -> Tuple[str, ...]:
    return tuple(sorted(FINETUNE_SPECS.keys()))


def build_model_only(params) -> Tuple[torch.nn.Module, TaskKind]:
    """Instantiate the original per-dataset Model without importing the dataset module."""
    if params.downstream_dataset not in FINETUNE_SPECS:
        raise ValueError(
            f"Unknown downstream_dataset={params.downstream_dataset!r}. "
            f"Available: {', '.join(available_downstream_datasets())}"
        )
    spec = FINETUNE_SPECS[params.downstream_dataset]
    try:
        model_mod = importlib.import_module(spec.model_module)
    except ModuleNotFoundError as exc:
        if exc.name in {"einops"}:
            raise ModuleNotFoundError(
                f"Missing optional dependency {exc.name!r}. "
                "Install project deps first (e.g. `pip install -r requirements.txt`)."
            ) from exc
        raise
    return model_mod.Model(params), spec.task


def build_loader_and_model(params) -> Tuple[dict, torch.nn.Module, TaskKind]:
    """Build dataloaders + model using the original per-dataset implementations."""
    if params.downstream_dataset not in FINETUNE_SPECS:
        raise ValueError(
            f"Unknown downstream_dataset={params.downstream_dataset!r}. "
            f"Available: {', '.join(available_downstream_datasets())}"
        )
    spec = FINETUNE_SPECS[params.downstream_dataset]

    try:
        dataset_mod = importlib.import_module(spec.dataset_module)
    except ModuleNotFoundError as exc:
        if exc.name in {"lmdb"}:
            raise ModuleNotFoundError(
                f"Missing optional dependency {exc.name!r}. "
                "Install project deps first (e.g. `pip install -r requirements.txt`)."
            ) from exc
        raise

    model, task = build_model_only(params)

    loader = dataset_mod.LoadDataset(params).get_data_loader()
    return loader, model, task
