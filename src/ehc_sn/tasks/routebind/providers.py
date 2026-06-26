"""Task-owned evaluation case providers for the Routebind task family.

These providers implement :class:`~ehc_sn.eval.contracts.EvaluationSourceProvider`
and supply batched Routebind replay cases to named evaluation regimes.

Ownership rules:
    - Providers live here, next to task evaluation code.
    - They know Routebind channel semantics and index conventions.
    - They do **not** own scheduling, metric naming, or figure rendering.
    - They do **not** make benchmark claims.

Batch format:
    Each provider yields batches ready for consumption through
    :meth:`execute_evaluation_batch` without adapter changes.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from ehc_sn.data.datasets import ProcessedDataset
from ehc_sn.data.index import filter_index, read_index
from ehc_sn.eval.contracts import EvaluationCaseBatch
from ehc_sn.tasks.routebind.traces import RoutebindEvaluationSourceContext


# =============================================================================
class RoutebindReplayProvider:
    """Routebind-task-owned provider for processed replay evaluation.

    Loads from a versioned processed Routebind task corpus and yields batched
    cases for named evaluation regimes.

    ``provider_settings`` keys:

    - ``dataset_path`` (*str*, required): Path to the processed Routebind task
      corpus root (directory containing ``index.jsonl`` and per-split channel
      directories).
    - ``split`` (*str*, default ``"val"``): Which dataset split to load from.
    - ``batch_size`` (*int*, default ``1``): Cases per evaluation batch.
    - ``n_cases`` (*int*, default ``0``): Maximum number of cases to use;
      ``0`` means use all entries in the split.
    """

    def __init__(
        self,
        *,
        dataset_path: str,
        split: str = "val",
        batch_size: int = 1,
        n_cases: int = 0,
    ) -> None:
        self._dataset_path = Path(dataset_path)
        self._split = split
        self._batch_size = batch_size
        self._n_cases = n_cases

        if not (self._dataset_path / "index.jsonl").exists():
            raise FileNotFoundError(
                f"No index.jsonl found at {self._dataset_path}. "
                f"Ensure dataset_path points to a versioned processed "
                f"Routebind corpus root (e.g. data/processed/routebind/default/v1)."
            )

    def provide_cases(
        self,
        max_batches: int = 0,
        max_samples: int | None = None,
    ) -> Iterator[EvaluationCaseBatch]:
        """Yield batched Routebind replay cases from the configured split.

        Args:
            max_batches: If > 0, yield at most this many batches;
                otherwise yield all.
            max_samples: If not None, yield at most this many samples (summed
                across batches).  ``max_batches`` takes precedence.

        Yields:
            :class:`~ehc_sn.eval.contracts.EvaluationCaseBatch` items with
            task-native batches, a deterministic ``case_id``, and split metadata.
        """
        data_root = self._dataset_path
        all_entries = read_index(data_root / "index.jsonl")
        entries = filter_index(all_entries, split=self._split)

        if self._n_cases > 0:
            entries = entries[: self._n_cases]

        if not entries:
            raise ValueError(
                f"No Routebind entries found for split={self._split!r} "
                f"at {data_root}."
            )

        dataset = ProcessedDataset(entries, data_root / self._split)
        loader: DataLoader[dict[str, Any]] = DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )

        samples_yielded = 0
        for batch_idx, batch in enumerate(loader):
            if max_batches > 0 and batch_idx >= max_batches:
                break

            first_key = next(iter(batch))
            n_in_batch = batch[first_key].shape[0]

            if max_samples is not None and samples_yielded >= max_samples:
                break

            # Trim batch if it would exceed max_samples.
            if (
                max_samples is not None
                and samples_yielded + n_in_batch > max_samples
            ):
                remaining = max_samples - samples_yielded
                batch = {k: v[:remaining] for k, v in batch.items()}
                n_in_batch = remaining

            samples_yielded += n_in_batch
            ids_in_batch = [
                entries[batch_idx * self._batch_size + i].id
                for i in range(n_in_batch)
            ]
            yield EvaluationCaseBatch(
                batch=batch,
                case_id=f"routebind-{self._split}-{batch_idx:04d}",
                n_samples=n_in_batch,
                source_context=RoutebindEvaluationSourceContext(
                    dataset_path=self._dataset_path,
                    split=self._split,
                    sample_ids=tuple(ids_in_batch),
                ),
            )

    def description(self) -> str:
        """Return a human-readable description for logging."""
        return (
            f"RoutebindReplayProvider("
            f"path={self._dataset_path}, "
            f"split={self._split!r}, "
            f"batch_size={self._batch_size}, "
            f"n_cases={self._n_cases})"
        )


__all__ = ["RoutebindReplayProvider"]
