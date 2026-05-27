"""Custom sampling utilities supporting subject-aware batching."""

from __future__ import annotations

from typing import Iterator, List, Sequence

import numpy as np
from torch.utils.data import Sampler


class SubjectBalancedBatchSampler(Sampler[List[int]]):
    """Batch sampler ensuring each batch draws equally from multiple subjects.

    This sampler groups trial indices per subject, draws ``samples_per_subject`` trials
    per subject, and assembles batches containing ``batch_size`` samples in total.
    Any leftover groups that cannot form a complete batch are discarded.
    """

    def __init__(
        self,
        subject_ids: Sequence[int],
        batch_size: int,
        samples_per_subject: int = 2,
        generator: np.random.Generator | None = None,
    ) -> None:
        if batch_size % samples_per_subject != 0:
            raise ValueError("batch_size must be divisible by samples_per_subject")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if samples_per_subject <= 0:
            raise ValueError("samples_per_subject must be positive")

        self.subject_ids = np.asarray(subject_ids)
        if self.subject_ids.ndim != 1:
            raise ValueError("subject_ids must be a 1-D sequence")
        self.batch_size = batch_size
        self.samples_per_subject = samples_per_subject
        subjects_per_batch = batch_size // samples_per_subject
        if subjects_per_batch <= 0:
            raise ValueError("batch_size must accommodate at least one subject")
        self._subjects_per_batch = subjects_per_batch

        unique_subjects = np.unique(self.subject_ids)
        self._subject_to_indices = {
            sub: np.where(self.subject_ids == sub)[0] for sub in unique_subjects
        }

        self.rng = generator if generator is not None else np.random.default_rng()

        self._num_batches = self._compute_num_batches()

    def __iter__(self) -> Iterator[List[int]]:
        groups: List[np.ndarray] = []
        for indices in self._subject_to_indices.values():
            shuffled = indices.copy()
            self.rng.shuffle(shuffled)
            n_full = len(shuffled) // self.samples_per_subject
            if n_full == 0:
                continue
            reshaped = shuffled[: n_full * self.samples_per_subject]
            groups.extend(
                np.split(reshaped, n_full)
            )

        if not groups:
            return

        self.rng.shuffle(groups)
        groups_per_batch = self._subjects_per_batch
        num_complete_batches = len(groups) // groups_per_batch

        for i in range(num_complete_batches):
            start = i * groups_per_batch
            batch_groups = groups[start : start + groups_per_batch]
            batch = [int(idx) for group in batch_groups for idx in group]
            yield batch

    def __len__(self) -> int:
        return self._num_batches

    def _compute_num_batches(self) -> int:
        total_groups = 0
        for indices in self._subject_to_indices.values():
            total_groups += len(indices) // self.samples_per_subject
        return total_groups // self._subjects_per_batch
