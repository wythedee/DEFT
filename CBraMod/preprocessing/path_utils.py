from __future__ import annotations

import os
from pathlib import Path


def eeg_dataset_root() -> str:
    """Return the root directory that contains EEG dataset folders (MI/EMO/SLEEP/...).

    Resolution order:
      1) $EEG_DATASET_ROOT or $EEG_RAW_ROOT
      2) ./EEG_Dataset
    """

    env = os.environ.get("EEG_DATASET_ROOT") or os.environ.get("EEG_RAW_ROOT")
    if env:
        return os.path.expanduser(env)
    return str(Path.cwd() / "EEG_Dataset")


def processed_root() -> str:
    """Return the root directory where preprocessing outputs (LMDBs) are stored.

    Resolution order:
      1) $EEG_PROCESSED_ROOT
      2) ./cbramod_processed
    """

    env = os.environ.get("EEG_PROCESSED_ROOT")
    if env:
        return os.path.expanduser(env)
    return str(Path.cwd() / "cbramod_processed")


def first_existing(*candidates: str) -> str:
    """Return the first candidate path that exists; otherwise return the first candidate."""

    expanded = [os.path.expanduser(c) for c in candidates if c]
    for path in expanded:
        if os.path.exists(path):
            return path
    if not expanded:
        raise ValueError("No path candidates provided.")
    return expanded[0]
