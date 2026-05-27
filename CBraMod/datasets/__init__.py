"""Dataset package.

This module intentionally avoids importing every dataset implementation at import time.
Some downstream dataset modules have optional dependencies (e.g., `lmdb`), and eager
imports make unrelated tooling (like `--help`) fail before those deps are installed.
"""

from __future__ import annotations

import importlib
from typing import Any


_SUBMODULES = {
    "atten_dataset",
    "bciciv2a_dataset",
    "chb_dataset",
    "cs_bcic_track4_dataset",
    "disentangle_dataset",
    "faced_dataset",
    "hebin2021_dataset",
    "isruc_dataset",
    "isruc_s3_dataset",
    "mi_cho2017_dataset",
    "mi_koreau_dataset",
    "mi_shin2017a_dataset",
    "mi_track1_fewshot_dataset",
    "mi_weibo2014_dataset",
    "mumtaz_dataset",
    "physio_dataset",
    "sch2017_dataset",
    "seed_dataset",
    "seediv_dataset",
    "seedv_dataset",
    "seedvig_dataset",
    "shu_dataset",
    "speech_dataset",
    "stress_dataset",
    "tuab_dataset",
    "tuev_dataset",
    "samplers",
}

__all__ = sorted(_SUBMODULES)


def __getattr__(name: str) -> Any:  # pragma: no cover
    if name in _SUBMODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

