from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple

import torch

TaskKind = Literal['multiclass', 'binary']


@dataclass(frozen=True)
class FinetuneSpec:
    dataset_module: str
    model_module: str
    task: TaskKind
    num_classes: int
    defaults: Dict[str, object] = field(default_factory=dict)


FINETUNE_SPECS: Dict[str, FinetuneSpec] = {
    'FACED': FinetuneSpec('Datasets.faced_dataset', 'Models.model_for_faced', 'multiclass', 9),
    'BCIC': FinetuneSpec('Datasets.bciciv2a_dataset', 'Models.model_for_bciciv2a', 'multiclass', 4),
    'BCIC-IV-2a': FinetuneSpec('Datasets.bciciv2a_dataset', 'Models.model_for_bciciv2a', 'multiclass', 4),
    'SEED-V': FinetuneSpec('Datasets.seedv_dataset', 'Models.model_for_seedv', 'multiclass', 5),
    'SEED-IV': FinetuneSpec('Datasets.seediv_dataset', 'Models.model_for_seediv', 'multiclass', 4),
    'SEED': FinetuneSpec('Datasets.seed_dataset', 'Models.model_for_seed', 'multiclass', 3),
    'MI-KoreaU': FinetuneSpec('Datasets.mi_koreau_dataset', 'Models.model_for_mi_koreau', 'multiclass', 3),
    'PhysioNet-MI': FinetuneSpec('Datasets.physio_dataset', 'Models.model_for_physio', 'multiclass', 4),
    'SHU-MI': FinetuneSpec('Datasets.shu_dataset', 'Models.model_for_shu', 'binary', 1),
    'ISRUC_S1': FinetuneSpec('Datasets.isruc_dataset', 'Models.model_for_isruc', 'multiclass', 5, {'patch_size': 200, 'd_model': 200, 'seq_len': 30}),
    'ISRUC_S3': FinetuneSpec('Datasets.isruc_s3_dataset', 'Models.model_for_isruc', 'multiclass', 5, {'patch_size': 200, 'd_model': 200, 'seq_len': 30}),
    'ISRUC': FinetuneSpec('Datasets.isruc_dataset', 'Models.model_for_isruc', 'multiclass', 5, {'patch_size': 200, 'd_model': 200, 'seq_len': 30}),
    'ISRUC-S3': FinetuneSpec('Datasets.isruc_s3_dataset', 'Models.model_for_isruc', 'multiclass', 5, {'patch_size': 200, 'd_model': 200, 'seq_len': 30}),
    'CHB-MIT': FinetuneSpec('Datasets.chb_dataset', 'Models.model_for_chb', 'binary', 1),
    'BCIC2020-3': FinetuneSpec('Datasets.speech_dataset', 'Models.model_for_speech', 'multiclass', 6),
    'CS-BCIC-Track4': FinetuneSpec('Datasets.cs_bcic_track4_dataset', 'Models.model_for_cs_bcic_track4', 'multiclass', 3),
    'Mumtaz2016': FinetuneSpec('Datasets.mumtaz_dataset', 'Models.model_for_mumtaz', 'binary', 1),
    'MentalArithmetic': FinetuneSpec('Datasets.stress_dataset', 'Models.model_for_stress', 'binary', 1),
    'TUEV': FinetuneSpec('Datasets.tuev_dataset', 'Models.model_for_tuev', 'multiclass', 6),
    'TUAB': FinetuneSpec('Datasets.tuab_dataset', 'Models.model_for_tuab', 'binary', 1),
    'Schirrmeister2017': FinetuneSpec('Datasets.sch2017_dataset', 'Models.model_for_sch2017', 'binary', 1),
    'HeBin2021-LR': FinetuneSpec('Datasets.hebin2021_dataset', 'Models.model_for_hebin', 'binary', 1),
    'HeBin2021-UD': FinetuneSpec('Datasets.hebin2021_dataset', 'Models.model_for_hebin', 'binary', 1),
    'ATTEN': FinetuneSpec('Datasets.atten_dataset', 'Models.model_for_atten', 'binary', 1),
    'MI-Track1-FewShot': FinetuneSpec('Datasets.mi_track1_fewshot_dataset', 'Models.model_for_mi_track1_fewshot', 'binary', 1),
    'MI-Cho2017': FinetuneSpec('Datasets.mi_cho2017_dataset', 'Models.model_for_mi_cho2017', 'binary', 1),
    'MI-Shin2017A': FinetuneSpec('Datasets.mi_shin2017a_dataset', 'Models.model_for_mi_shin2017a', 'binary', 1),
    'MI-Weibo2014': FinetuneSpec('Datasets.mi_weibo2014_dataset', 'Models.model_for_mi_weibo2014', 'binary', 1),
}


def available_downstream_datasets() -> Tuple[str, ...]:
    return tuple(sorted(FINETUNE_SPECS.keys()))


def _apply_defaults(params, spec: FinetuneSpec) -> None:
    params.num_of_classes = int(spec.num_classes)
    params.task_kind = spec.task
    for key, value in (spec.defaults or {}).items():
        setattr(params, key, value)


def _build_codebrain_model_only(params, spec: FinetuneSpec) -> Tuple[torch.nn.Module, TaskKind]:
    _apply_defaults(params, spec)
    model_module = spec.model_module
    if getattr(params, 'for_disentangle', False) and params.downstream_dataset in {'ISRUC_S1', 'ISRUC_S3', 'ISRUC', 'ISRUC-S3'}:
        model_module = 'Models.model_for_isruc_disentangle'
    model_mod = importlib.import_module(model_module)
    return model_mod.Model(params), spec.task


def _build_disentangle_model_only(params) -> Tuple[torch.nn.Module, TaskKind]:
    mod = importlib.import_module('disentangle_v7.standard_finetune')
    return mod.build_model_only(params)


def build_model_only(params) -> Tuple[torch.nn.Module, TaskKind]:
    if params.downstream_dataset not in FINETUNE_SPECS:
        raise ValueError(
            f'Unknown downstream_dataset={params.downstream_dataset!r}. '
            f"Available: {', '.join(available_downstream_datasets())}"
        )

    spec = FINETUNE_SPECS[params.downstream_dataset]
    if getattr(params, 'model_logic', 'codebrain') == 'disentangle':
        model, task = _build_disentangle_model_only(params)
        params.task_kind = task
        return model, task
    return _build_codebrain_model_only(params, spec)


def build_loader_and_model(params) -> Tuple[dict, torch.nn.Module, TaskKind]:
    if params.downstream_dataset not in FINETUNE_SPECS:
        raise ValueError(
            f'Unknown downstream_dataset={params.downstream_dataset!r}. '
            f"Available: {', '.join(available_downstream_datasets())}"
        )

    spec = FINETUNE_SPECS[params.downstream_dataset]
    _apply_defaults(params, spec)
    dataset_mod = importlib.import_module(spec.dataset_module)
    model, task = build_model_only(params)
    params.task_kind = task
    loader = dataset_mod.LoadDataset(params).get_data_loader()
    return loader, model, task
