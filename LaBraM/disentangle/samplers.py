from __future__ import annotations

from typing import Iterator, List, Sequence

import numpy as np
from torch.utils.data import Sampler


class SubjectBalancedBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        subject_ids: Sequence[int],
        batch_size: int,
        samples_per_subject: int = 2,
        generator: np.random.Generator | None = None,
    ) -> None:
        if batch_size % samples_per_subject != 0:
            raise ValueError("batch_size must be divisible by samples_per_subject")
        if batch_size <= 0 or samples_per_subject <= 0:
            raise ValueError("batch_size and samples_per_subject must be positive")
        self.subject_ids = np.asarray(subject_ids)
        self.batch_size = int(batch_size)
        self.samples_per_subject = int(samples_per_subject)
        self._subjects_per_batch = self.batch_size // self.samples_per_subject
        self._subject_to_indices = {
            sub: np.where(self.subject_ids == sub)[0] for sub in np.unique(self.subject_ids)
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
            groups.extend(np.split(shuffled[: n_full * self.samples_per_subject], n_full))
        if not groups:
            return
        self.rng.shuffle(groups)
        num_complete_batches = len(groups) // self._subjects_per_batch
        for i in range(num_complete_batches):
            start = i * self._subjects_per_batch
            batch_groups = groups[start : start + self._subjects_per_batch]
            yield [int(idx) for group in batch_groups for idx in group]

    def __len__(self) -> int:
        return self._num_batches

    def _compute_num_batches(self) -> int:
        total_groups = sum(len(indices) // self.samples_per_subject for indices in self._subject_to_indices.values())
        return total_groups // self._subjects_per_batch
