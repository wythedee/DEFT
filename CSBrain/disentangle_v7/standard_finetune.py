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
    "ISRUC_S1": FinetuneSpec("datasets.isruc_dataset", "models.model_for_isruc", "multiclass"),
    "ISRUC-S3": FinetuneSpec("datasets.isruc_s3_dataset", "models.model_for_isruc", "multiclass"),
    "ISRUC_S3": FinetuneSpec("datasets.isruc_s3_dataset", "models.model_for_isruc", "multiclass"),
    "BCIC2020-3": FinetuneSpec("datasets.speech_dataset", "models.model_for_speech", "multiclass"),
    "CS-BCIC-Track4": FinetuneSpec("datasets.generic_lmdb_dataset", "models.generic_csbrain_downstream", "multiclass"),
    "Mumtaz2016": FinetuneSpec("datasets.mumtaz_dataset", "models.model_for_mumtaz", "binary"),
    "SEED-VIG": FinetuneSpec("datasets.seedvig_dataset", "models.model_for_seedvig", "regression"),
    "MentalArithmetic": FinetuneSpec("datasets.stress_dataset", "models.model_for_stress", "binary"),
    "BCIC-IV-2a": FinetuneSpec("datasets.bciciv2a_dataset", "models.model_for_bciciv2a", "multiclass"),
    "Schirrmeister2017": FinetuneSpec("datasets.generic_lmdb_dataset", "models.generic_csbrain_downstream", "binary"),
    "HeBin2021-LR": FinetuneSpec("datasets.generic_lmdb_dataset", "models.generic_csbrain_downstream", "binary"),
    "HeBin2021-UD": FinetuneSpec("datasets.generic_lmdb_dataset", "models.generic_csbrain_downstream", "binary"),
    "SEED-IV": FinetuneSpec("datasets.generic_lmdb_dataset", "models.generic_csbrain_downstream", "multiclass"),
    "SEED": FinetuneSpec("datasets.generic_lmdb_dataset", "models.generic_csbrain_downstream", "binary"),
    "MI-KoreaU": FinetuneSpec("datasets.generic_lmdb_dataset", "models.generic_csbrain_downstream", "binary"),
    "MI-Cho2017": FinetuneSpec("datasets.generic_lmdb_dataset", "models.generic_csbrain_downstream", "binary"),
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
    model_module = spec.model_module
    if getattr(params, "for_disentangle", False) and params.downstream_dataset in {"ISRUC", "ISRUC_S1", "ISRUC-S3", "ISRUC_S3"}:
        model_module = "models.model_for_isruc_disentangle"
    try:
        model_mod = importlib.import_module(model_module)
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
