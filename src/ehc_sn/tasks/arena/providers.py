"""Task-owned evaluation case providers for the arena task family.

These providers implement :class:`~ehc_sn.lightning.eval.contracts.EvaluationSourceProvider`
and supply batched arena replay cases to named evaluation regimes.

Ownership rules:
    - Providers live here, next to task evaluation code.
    - They know arena channel semantics and index conventions.
    - They do **not** own scheduling, metric naming, or figure rendering.
    - They do **not** make benchmark claims.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from ehc_sn.data.datasets import ProcessedDataset
from ehc_sn.data.index import filter_index, read_index
from ehc_sn.lightning.eval.contracts import EvaluationCaseBatch
from ehc_sn.tasks.arena.traces import ArenaEvaluationSourceContext


# =============================================================================
class ArenaReplayDiagnosticProvider:
    """Arena-task-owned provider for processed replay diagnostics.

    Loads from a versioned processed arena split and yields batched replay cases
    for named evaluation regimes. Case identity is derived from the dataset index.

    This provider is diagnostic-only: it yields raw arena channel tensors in the
    same format as the fit-path DataLoader, so any Lightning family with an arena
    adapter can consume the cases through :meth:`execute_evaluation_batch` without
    modification.

    ``provider_settings`` keys:

    - ``dataset_path`` (*str*, required): Path to the processed dataset root
      (directory containing ``index.jsonl`` and per-split channel arrays).
    - ``split`` (*str*, default ``"val"``): Which dataset split to load from.
    - ``batch_size`` (*int*, default ``1``): Arena episodes per evaluation case batch.
    - ``n_cases`` (*int*, default ``0``): Maximum number of individual episodes to use;
      ``0`` means use all episodes in the split.
    """

    def __init__(  # ----------------------------------------------------------
        self,
        *,
        dataset_path: str,
        split: str = "val",
        batch_size: int = 1,
        n_cases: int = 0,
    ) -> None:
        """Initialize with dataset path, split, batch size, and case limit."""
        self._dataset_path = Path(dataset_path)
        self._split = split
        self._batch_size = batch_size
        self._n_cases = n_cases

    def provide_cases(  # -----------------------------------------------------
        self,
        max_batches: int = 0,
    ) -> Iterator[EvaluationCaseBatch]:
        """Yield batched arena replay cases from the configured split.

        Args:
            max_batches: If > 0, yield at most this many batches; otherwise yield all.

        Yields:
            :class:`~ehc_sn.lightning.eval.contracts.EvaluationCaseBatch` items with
            arena channel tensors, a deterministic ``case_id``, and split metadata.
        """
        data_root = self._dataset_path
        all_entries = read_index(data_root / "index.jsonl")
        entries = filter_index(all_entries, split=self._split)

        if self._n_cases > 0:
            entries = entries[: self._n_cases]

        dataset = ProcessedDataset(entries, data_root / self._split)
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

            start = batch_idx * self._batch_size
            n_episodes = batch[next(iter(batch))].shape[0]
            ids_in_batch = [e.id for e in entries[start : start + n_episodes]]
            yield EvaluationCaseBatch(
                batch=batch,
                case_id=f"arena-{self._split}-{batch_idx:04d}",
                source_context=ArenaEvaluationSourceContext(
                    task_family="arena",
                    dataset_path=self._dataset_path,
                    split=self._split,
                    sample_ids=tuple(ids_in_batch),
                ),
            )

    def description(  # -------------------------------------------------------
        self,
    ) -> str:
        """Return a human-readable description for logging."""
        return (
            f"ArenaReplayDiagnosticProvider("
            f"path={self._dataset_path}, "
            f"split={self._split!r}, "
            f"batch_size={self._batch_size}, "
            f"n_cases={self._n_cases})"
        )


# =============================================================================
class ArenaFixedProbeProvider:
    """Arena-task-owned provider that evaluates a fixed, explicitly-named set of samples.

    Use this for diagnostic regimes that must be reproducible across runs and checkpoints:
    you supply the exact :attr:`~ehc_sn.data.index.DatasetIndexEntry.id` values to load,
    and the provider yields them in stable order every time.

    This provider is diagnostic-only.  It does not make benchmark claims and
    does not affect fit-path validation behavior.

    ``provider_settings`` keys:

    - ``dataset_path`` (*str*, required): Path to the processed dataset root.
    - ``split`` (*str*, default ``"val"``): Dataset split that contains the samples.
    - ``sample_ids`` (*list[str]*, required): Ordered list of
      :attr:`~ehc_sn.data.index.DatasetIndexEntry.id` values to load.  Must be
      non-empty, contain no duplicates, and all ids must exist in the index.
    - ``batch_size`` (*int*, default ``1``): Episodes per evaluation case batch.
      When ``batch_size == 1`` the single sample id is used as ``case_id`` directly.
      When ``batch_size > 1`` a deterministic ``"arena-probe-NNNN"`` id is used and
      the ordered sample ids are recorded in ``metadata["sample_ids"]``.

    Raises:
        ValueError: On construction if ``sample_ids`` is empty or contains duplicates.
        KeyError: From :meth:`provide_cases` if any requested id is absent from the index.
    """

    def __init__(  # ----------------------------------------------------------
        self,
        *,
        dataset_path: str,
        split: str = "val",
        sample_ids: list[str],
        batch_size: int = 1,
    ) -> None:
        """Initialize the provider with dataset path, split, sample ids, and batch size."""
        if not sample_ids:
            raise ValueError(
                "ArenaFixedProbeProvider: sample_ids must not be empty",
            )
        if len(sample_ids) != len(set(sample_ids)):
            duplicates = [s for s in sample_ids if sample_ids.count(s) > 1]
            raise ValueError(
                "ArenaFixedProbeProvider: sample_ids contains duplicates: "
                f"{sorted(set(duplicates))}"
            )
        self._dataset_path = Path(dataset_path)
        self._split = split
        self._sample_ids = sample_ids
        self._batch_size = batch_size

    def provide_cases(  # -----------------------------------------------------
        self,
        max_batches: int = 0,
    ) -> Iterator[EvaluationCaseBatch]:
        """Yield deterministic evaluation case batches for the configured sample ids.

        Samples are loaded in the exact order of ``sample_ids``.  If a requested id
        is absent from the index a :exc:`KeyError` is raised immediately so the caller
        gets a fast, actionable error rather than a silent empty result.

        Args:
            max_batches: If > 0, stop after this many batches; otherwise yield all.

        Yields:
            :class:`~ehc_sn.lightning.eval.contracts.EvaluationCaseBatch` items with
            arena channel tensors, a deterministic ``case_id``, and probe metadata.

        Raises:
            KeyError: If any ``sample_id`` is not found in the dataset index.
        """
        all_entries = read_index(self._dataset_path / "index.jsonl")
        split_entries = filter_index(all_entries, split=self._split)
        by_id = {e.id: e for e in split_entries}

        missing = [sid for sid in self._sample_ids if sid not in by_id]
        if missing:
            raise KeyError(
                "ArenaFixedProbeProvider: sample_ids not found in split "
                f"{self._split!r}: {missing}"
            )

        ordered_entries = [by_id[sid] for sid in self._sample_ids]
        dataset = ProcessedDataset(
            ordered_entries, self._dataset_path / self._split
        )
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
            ids_in_batch = self._sample_ids[
                batch_idx * self._batch_size : batch_idx * self._batch_size
                + n_in_batch
            ]

            if self._batch_size == 1:
                case_id = ids_in_batch[0]
            else:
                case_id = f"arena-probe-{batch_idx:04d}"

            yield EvaluationCaseBatch(
                batch=batch,
                case_id=case_id,
                source_context=ArenaEvaluationSourceContext(
                    task_family="arena",
                    dataset_path=self._dataset_path,
                    split=self._split,
                    sample_ids=tuple(ids_in_batch),
                ),
            )

    def description(  # -------------------------------------------------------
        self,
    ) -> str:
        """Return a human-readable description for logging."""
        n = len(self._sample_ids)
        preview = self._sample_ids[:3]
        preview_str = ", ".join(repr(s) for s in preview)
        if n > 3:
            preview_str += f", … ({n} total)"
        return (
            f"ArenaFixedProbeProvider("
            f"path={self._dataset_path}, "
            f"split={self._split!r}, "
            f"sample_ids=[{preview_str}], "
            f"batch_size={self._batch_size})"
        )


# =============================================================================
__all__ = ["ArenaReplayDiagnosticProvider", "ArenaFixedProbeProvider"]
