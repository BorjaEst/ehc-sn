"""Task-owned evaluation case providers for the MazeHard task family.

These providers implement :class:`~ehc_sn.lightning.eval.contracts.EvaluationSourceProvider`
and supply batched MazeHard replay cases to named evaluation regimes.

Ownership rules:
    - Providers live here, next to task evaluation code.
    - They know MazeHard channel semantics and index conventions.
    - They do **not** own scheduling, metric naming, or figure rendering.
    - They do **not** make benchmark claims.

Batch format:
    Each provider yields batches already transformed through
    :func:`~ehc_sn.tasks.mazehard.runtime.coerce_maze_hard_batch`, so
    ``hrm_v1.py`` and ``hrm_v2.py`` can consume them unchanged through
    :meth:`execute_evaluation_batch` without any adapter change.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from ehc_sn.data.datasets import ProcessedDataset
from ehc_sn.data.index import filter_index, read_index
from ehc_sn.lightning.eval.contracts import EvaluationCaseBatch, EvaluationSourceProvider
from ehc_sn.tasks.mazehard.runtime import coerce_maze_hard_batch
from ehc_sn.tasks.mazehard.traces import MazeHardEvaluationSourceContext


# =================================================================================================
class MazeHardReplayDiagnosticProvider:
    """MazeHard-task-owned provider for processed replay diagnostics.

    Loads from a versioned processed MazeHard task corpus and yields batched
    cases for named evaluation regimes.  Each case batch is already transformed
    through :func:`~ehc_sn.tasks.mazehard.runtime.coerce_maze_hard_batch`
    (i.e. ``{"input_ids": Tensor, "labels": Tensor}``) so any HRM Lightning
    family can consume it unchanged through :meth:`execute_evaluation_batch`.

    This provider is diagnostic-only: it does not affect fit-path validation
    behavior and makes no benchmark claims.

    ``provider_settings`` keys:

    - ``dataset_path`` (*str*, required): Path to the processed MazeHard task
      corpus root (directory containing ``index.jsonl`` and per-split channel
      directories).
    - ``split`` (*str*, default ``"val"``): Which dataset split to load from.
    - ``batch_size`` (*int*, default ``1``): Mazes per evaluation case batch.
    - ``n_cases`` (*int*, default ``0``): Maximum number of individual mazes to use;
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

    def provide_cases(self, max_batches: int = 0) -> Iterator[EvaluationCaseBatch]:
        """Yield batched MazeHard replay cases from the configured split.

        Args:
            max_batches: If > 0, yield at most this many batches; otherwise yield all.

        Yields:
            :class:`~ehc_sn.lightning.eval.contracts.EvaluationCaseBatch` items with
            ``{"input_ids": Tensor, "labels": Tensor}`` batches, a deterministic
            ``case_id``, and split metadata.
        """
        data_root = self._dataset_path
        all_entries = read_index(data_root / "index.jsonl")
        entries = filter_index(all_entries, split=self._split)

        if self._n_cases > 0:
            entries = entries[: self._n_cases]

        dataset = ProcessedDataset(entries, data_root / self._split, transform=coerce_maze_hard_batch)
        loader: DataLoader[dict[str, Any]] = DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )

        for batch_idx, batch in enumerate(loader):
            if max_batches > 0 and batch_idx >= max_batches:
                break

            first_key = next(iter(batch))
            n_in_batch = batch[first_key].shape[0]
            ids_in_batch = [entries[batch_idx * self._batch_size + i].id for i in range(n_in_batch)]
            yield EvaluationCaseBatch(
                batch=batch,
                case_id=f"mazehard-{self._split}-{batch_idx:04d}",
                source_context=MazeHardEvaluationSourceContext(
                    task_family="mazehard",
                    dataset_path=self._dataset_path,
                    split=self._split,
                    sample_ids=tuple(ids_in_batch),
                ),
            )

    def description(self) -> str:
        """Return a human-readable description for logging."""
        return (
            f"MazeHardReplayDiagnosticProvider("
            f"path={self._dataset_path}, "
            f"split={self._split!r}, "
            f"batch_size={self._batch_size}, "
            f"n_cases={self._n_cases})"
        )


# =================================================================================================
class MazeHardFixedProbeProvider:
    """MazeHard-task-owned provider that evaluates a fixed, explicitly-named set of mazes.

    Use this for diagnostic regimes that must be reproducible across runs and checkpoints:
    you supply the exact :attr:`~ehc_sn.data.index.DatasetIndexEntry.id` values to load,
    and the provider yields them in stable order every time.

    This provider is diagnostic-only.  It does not make benchmark claims and does not
    affect fit-path validation behavior.

    ``provider_settings`` keys:

    - ``dataset_path`` (*str*, required): Path to the processed MazeHard task corpus root.
    - ``split`` (*str*, default ``"val"``): Dataset split that contains the samples.
    - ``sample_ids`` (*list[str]*, required): Ordered list of
      :attr:`~ehc_sn.data.index.DatasetIndexEntry.id` values to load.  Must be
      non-empty, contain no duplicates, and all ids must exist in the index.
    - ``batch_size`` (*int*, default ``1``): Mazes per evaluation case batch.
      When ``batch_size == 1`` the single sample id is used as ``case_id`` directly.
      When ``batch_size > 1`` a deterministic ``"mazehard-probe-NNNN"`` id is used and
      the ordered sample ids are recorded in ``metadata["sample_ids"]``.

    Raises:
        ValueError: On construction if ``sample_ids`` is empty or contains duplicates.
        KeyError: From :meth:`provide_cases` if any requested id is absent from the index.
    """

    def __init__(
        self,
        *,
        dataset_path: str,
        split: str = "val",
        sample_ids: list[str],
        batch_size: int = 1,
    ) -> None:
        if not sample_ids:
            raise ValueError("MazeHardFixedProbeProvider: sample_ids must not be empty")
        if len(sample_ids) != len(set(sample_ids)):
            duplicates = [s for s in sample_ids if sample_ids.count(s) > 1]
            raise ValueError(f"MazeHardFixedProbeProvider: sample_ids contains duplicates: {sorted(set(duplicates))}")
        self._dataset_path = Path(dataset_path)
        self._split = split
        self._sample_ids = sample_ids
        self._batch_size = batch_size

    def provide_cases(self, max_batches: int = 0) -> Iterator[EvaluationCaseBatch]:
        """Yield deterministic evaluation case batches for the configured sample ids.

        Samples are loaded in the exact order of ``sample_ids``.  If a requested id
        is absent from the index a :exc:`KeyError` is raised immediately.

        Args:
            max_batches: If > 0, stop after this many batches; otherwise yield all.

        Yields:
            :class:`~ehc_sn.lightning.eval.contracts.EvaluationCaseBatch` items with
            ``{"input_ids": Tensor, "labels": Tensor}`` batches, a deterministic
            ``case_id``, and probe metadata.

        Raises:
            KeyError: If any ``sample_id`` is not found in the dataset index.
        """
        all_entries = read_index(self._dataset_path / "index.jsonl")
        split_entries = filter_index(all_entries, split=self._split)
        by_id = {e.id: e for e in split_entries}

        missing = [sid for sid in self._sample_ids if sid not in by_id]
        if missing:
            raise KeyError(f"MazeHardFixedProbeProvider: sample_ids not found in split {self._split!r}: {missing}")

        ordered_entries = [by_id[sid] for sid in self._sample_ids]
        dataset = ProcessedDataset(ordered_entries, self._dataset_path / self._split, transform=coerce_maze_hard_batch)
        loader: DataLoader[dict[str, Any]] = DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )

        for batch_idx, batch in enumerate(loader):
            if max_batches > 0 and batch_idx >= max_batches:
                break

            first_key = next(iter(batch))
            n_in_batch = batch[first_key].shape[0]
            ids_in_batch = self._sample_ids[batch_idx * self._batch_size : batch_idx * self._batch_size + n_in_batch]

            if self._batch_size == 1:
                case_id = ids_in_batch[0]
            else:
                case_id = f"mazehard-probe-{batch_idx:04d}"

            yield EvaluationCaseBatch(
                batch=batch,
                case_id=case_id,
                source_context=MazeHardEvaluationSourceContext(
                    task_family="mazehard",
                    dataset_path=self._dataset_path,
                    split=self._split,
                    sample_ids=tuple(ids_in_batch),
                ),
            )

    def description(self) -> str:
        """Return a human-readable description for logging."""
        n = len(self._sample_ids)
        preview = self._sample_ids[:3]
        preview_str = ", ".join(repr(s) for s in preview)
        if n > 3:
            preview_str += f", … ({n} total)"
        return (
            f"MazeHardFixedProbeProvider("
            f"path={self._dataset_path}, "
            f"split={self._split!r}, "
            f"sample_ids=[{preview_str}], "
            f"batch_size={self._batch_size})"
        )


# =================================================================================================
__all__ = ["MazeHardReplayDiagnosticProvider", "MazeHardFixedProbeProvider"]
