"""Trace reader abstraction — load per-case bounded TraceTree views.

Provides a protocol and two implementations:

- ``ZarrEvaluationTraceReader`` — reads from ``traces/behavioral.zarr`` with
  ``traces/trace_index.json`` (canonical new format).
- ``LegacyNpzEvaluationTraceReader`` — reads from ``cases/*.dense.npz`` /
  ``cases/*.meta.json`` (old per-case format).
- ``open_trace_reader`` — resolves a reader for an evaluation artifact root.

Behavoral rules:
    - Prefer Zarr over NPZ when both exist.
    - Return ``None`` when neither format exists.
    - ``ZarrEvaluationTraceReader`` slices arrays lazily and reconstructs a
      ``TraceTree`` from sliced NumPy arrays.
    - ``LegacyNpzEvaluationTraceReader`` delegates to the existing
      ``_rehydrate_trace_tree`` helper.
"""

from __future__ import annotations

import json
from collections.abc import Collection
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ehc_sn.evaluation.artifacts import _rehydrate_trace_tree
from ehc_sn.traces.trace_tree import TraceTree

# =============================================================================
# Protocol
# =============================================================================


class EvaluationTraceReader(Protocol):
    """Reads per-case bounded trace views from a persisted archive."""

    def case_ids(self) -> tuple[str, ...]:
        """Return all case IDs available in this archive."""
        ...

    def has_case(self, case_id: str) -> bool:
        """Return True if *case_id* is available."""
        ...

    def load_case(
        self,
        case_id: str,
        *,
        fields: Collection[str] | None = None,
        max_steps: int | None = None,
    ) -> TraceTree:
        """Load one case as a ``TraceTree``.

        Args:
            case_id: Case identifier.
            fields: If provided, only these fields are loaded (dot-delimited
                names matching Zarr array keys).  ``None`` loads all fields.
            max_steps: If provided, only the first *max_steps* steps are
                loaded.  ``None`` loads all steps.

        Returns:
            A ``TraceTree`` with ``dense_leaves`` populated from the sliced
            arrays and metadata attached.
        """
        ...


# =============================================================================
# Zarr reader
# =============================================================================


def _zarr_open(path: Path) -> Any:
    """Open a Zarr v2 array at *path*, raising ImportError if zarr missing."""
    try:
        import zarr as _zarr
    except ImportError as exc:
        raise ImportError(
            "zarr is required to read Zarr traces. "
            "Install with: pip install 'ehp-sn[evaluation]'"
        ) from exc
    return _zarr.open_array(str(path), mode="r")


class ZarrEvaluationTraceReader:
    """Reads cases from a concatenated-step Zarr archive with a JSON index.

    The Zarr archive stores one array per trace field, with all cases
    concatenated along the leading axis.  The sidecar ``trace_index.json``
    records case boundaries (start offset and length per case).
    """

    def __init__(self, zarr_path: Path, index_path: Path) -> None:
        """Initialize a Zarr trace reader.

        Args:
            zarr_path: Path to ``behavioral.zarr`` directory (or parent
                directory containing per-field Zarr arrays).
            index_path: Path to ``trace_index.json`` sidecar.

        Raises:
            FileNotFoundError: If either path does not exist.
            ValueError: If the index schema is unrecognized.
        """
        if not zarr_path.exists():
            raise FileNotFoundError(f"Zarr archive not found: {zarr_path}")
        if not index_path.exists():
            raise FileNotFoundError(f"Trace index not found: {index_path}")

        raw = json.loads(index_path.read_text(encoding="utf-8"))
        if raw.get("schema") != "ehp_sn.evaluation.trace_index.v1":
            raise ValueError(
                f"Unsupported trace index schema: {raw.get('schema')!r}. "
                f"Expected 'ehp_sn.evaluation.trace_index.v1'."
            )

        self._zarr_path = zarr_path
        self._index = raw
        # Build case_id -> (start, length) + meta lookup.
        self._case_map: dict[str, tuple[int, int]] = {}
        self._case_meta: dict[str, dict[str, Any]] = {}
        for entry in raw.get("cases", []):
            cid = entry.get("case_id")
            if cid:
                self._case_map[cid] = (entry["start"], entry["length"])
                meta = entry.get("meta")
                if isinstance(meta, dict):
                    self._case_meta[cid] = meta

    # -- Public protocol ------------------------------------------------------

    def case_ids(self) -> tuple[str, ...]:
        return tuple(self._case_map.keys())

    def has_case(self, case_id: str) -> bool:
        return case_id in self._case_map

    def load_case(
        self,
        case_id: str,
        *,
        fields: Collection[str] | None = None,
        max_steps: int | None = None,
    ) -> TraceTree:
        """Load one case as a ``TraceTree`` by slicing Zarr arrays."""
        if case_id not in self._case_map:
            raise KeyError(f"Case {case_id!r} not found in trace index.")

        start, length = self._case_map[case_id]
        if max_steps is not None and max_steps > 0:
            length = min(length, max_steps)

        # Determine which Zarr arrays to load.
        zarr_keys: list[str] = list(self._index.get("fields", []))
        if fields is not None:
            zarr_keys = [k for k in zarr_keys if k in fields]

        # Slice each Zarr array and build a dense dict.
        # First pass: collect regular fields and detect multi-scale band arrays.
        dense: dict[str, np.ndarray] = {}
        # Track multi-scale parent keys → mapping of band names to arrays.
        multiscale_groups: dict[str, dict[str, np.ndarray]] = {}

        for key in zarr_keys:
            arr_name = key.replace("/", ".")
            arr_path = self._zarr_path / arr_name
            if not arr_path.exists():
                # Check if this is a multi-scale parent — look for _band sub-arrays.
                band_prefix = arr_name
                band_paths = sorted(
                    self._zarr_path.glob(f"{band_prefix}._band_*")
                )
                if not band_paths:
                    continue
                group: dict[str, np.ndarray] = {}
                for bp in band_paths:
                    # Extract band name from the trailing `_band_{name}` suffix.
                    band_tag = bp.name.rpartition("._band_")[-1]
                    zarr_arr = _zarr_open(bp)
                    sliced = np.asarray(zarr_arr[start : start + length])
                    group[band_tag] = sliced
                if group:
                    multiscale_groups[key] = group
                continue
            zarr_arr = _zarr_open(arr_path)
            sliced = zarr_arr[start : start + length]
            # Restore slash-delimited key for TraceTree.
            dense[key] = np.asarray(sliced)

        # Merge multi-scale groups into dense dict as separate slash-delimited
        # paths so figure templates can access per-band data via
        # ``trace.get("diagnostic/lec/cells/b0")``.
        for parent_key, bands in multiscale_groups.items():
            for band_tag, band_arr in bands.items():
                dense[f"{parent_key}/{band_tag}"] = band_arr

        # Build meta from index (per-case task metadata).
        meta: dict[str, Any] = dict(self._case_meta.get(case_id, {}))
        meta.update(
            {
                "trace_case_id": case_id,
                "trace_start": start,
                "trace_length": length,
            }
        )
        return _rehydrate_trace_tree(dense=dense, meta=meta)


# =============================================================================
# Legacy NPZ reader
# =============================================================================


def _load_legacy_npz_manifest(
    cases_dir: Path,
) -> list[dict[str, Any]]:
    """Build a case manifest from NPZ file discovery.

    Reads ``cases/`` directory for ``*.dense.npz`` files and constructs
    rows compatible with ``_rehydrate_trace_tree``.
    """
    entries: list[dict[str, Any]] = []
    if not cases_dir.exists():
        return entries
    for npz_path in sorted(cases_dir.glob("*.dense.npz")):
        meta_path = npz_path.with_suffix("").with_suffix(".meta.json")
        stem = npz_path.stem  # e.g. "0000-routebind-test-0000"
        # Parse case_id from stem: first two tokens are index and case_id.
        parts = stem.split("-", 2)
        case_id = parts[-1] if len(parts) > 2 else stem
        meta: dict[str, Any] = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                meta = {}
        entries.append(
            {
                "case_id": case_id,
                "dense_path": npz_path,
                "meta": meta,
            }
        )
    return entries


class LegacyNpzEvaluationTraceReader:
    """Reads cases from per-case ``.dense.npz`` / ``.meta.json`` files.

    Delegates trace reconstruction to the existing ``_rehydrate_trace_tree``
    helper in ``evaluation.artifacts``.
    """

    def __init__(self, cases_dir: Path) -> None:
        """Initialize a legacy NPZ trace reader.

        Args:
            cases_dir: Path to the ``cases/`` directory containing
                ``*.dense.npz`` and ``*.meta.json`` files.
        """
        self._cases_dir = cases_dir
        self._entries: list[dict[str, Any]] = _load_legacy_npz_manifest(
            cases_dir
        )
        self._case_index: dict[str, int] = {}
        for i, entry in enumerate(self._entries):
            cid = entry["case_id"]
            if cid not in self._case_index:
                self._case_index[cid] = i

    # -- Public protocol ------------------------------------------------------

    def case_ids(self) -> tuple[str, ...]:
        return tuple(e["case_id"] for e in self._entries)

    def has_case(self, case_id: str) -> bool:
        return case_id in self._case_index

    def load_case(
        self,
        case_id: str,
        *,
        fields: Collection[str] | None = None,
        max_steps: int | None = None,
    ) -> TraceTree:
        """Load one case from NPZ files and return a ``TraceTree``."""
        if case_id not in self._case_index:
            raise KeyError(f"Case {case_id!r} not found in NPZ cases.")
        entry = self._entries[self._case_index[case_id]]
        npz_path = entry["dense_path"]

        # Load NPZ data.
        import numpy as _np

        data = dict(_np.load(npz_path))

        # Apply field selection if requested.
        if fields is not None:
            # Convert dot-delimited fields to slash-delimited keys.
            slash_keys = {f.replace(".", "/") for f in fields}
            data = {k: v for k, v in data.items() if k in slash_keys}

        # Apply max_steps if requested.
        if max_steps is not None and max_steps > 0:
            for k in list(data.keys()):
                arr = data[k]
                if arr.ndim >= 1:
                    data[k] = arr[:max_steps]

        meta = entry["meta"]
        return _rehydrate_trace_tree(dense=data, meta=meta)


# =============================================================================
# Resolver
# =============================================================================


def open_trace_reader(artifact_root: Path) -> EvaluationTraceReader | None:
    """Resolve a trace reader for an evaluation artifact root directory.

    Checks for Zarr archive first (``traces/behavioral.zarr`` +
    ``traces/trace_index.json``).  Falls back to legacy NPZ
    (``cases/*.dense.npz``).  Returns ``None`` when neither exists.

    Args:
        artifact_root: Root directory of a completed evaluation artifact
            (contains ``manifest.json`` and ``_SUCCESS``).

    Returns:
        A reader instance, or ``None`` if no trace data is available.
    """
    # Check Zarr first.
    traces_dir = artifact_root / "traces"
    index_path = traces_dir / "trace_index.json"
    if traces_dir.exists() and index_path.exists():
        # The Zarr arrays are stored directly inside traces/ (one subdirectory
        # per field, e.g. traces/act.halted/).  Pass the traces dir as the
        # Zarr archive root.
        return ZarrEvaluationTraceReader(traces_dir, index_path)

    # Fall back to legacy NPZ.
    cases_dir = artifact_root / "cases"
    if cases_dir.exists() and list(cases_dir.glob("*.dense.npz")):
        return LegacyNpzEvaluationTraceReader(cases_dir)

    return None


# =============================================================================
__all__ = [
    "EvaluationTraceReader",
    "ZarrEvaluationTraceReader",
    "LegacyNpzEvaluationTraceReader",
    "open_trace_reader",
]
