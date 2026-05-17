"""Trace Tree implementation for batched simulation rollouts.

This module keeps a strict numeric-only dynamic channel for stacking/batching
and a separate metadata channel, while validating container structure via
PyTorch's pytree TreeSpec (first-observed structure wins).
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from ehc_sn.utils import torch_pytree


# =============================================================================
@dataclass
class TraceConfig:
    """Configuration for trace construction and validation.

    Attributes:
        global_paths: Segment paths treated as global (no batch axis enforced).
        metadata_paths: Segment paths forced into the metadata lane even when numeric.
        rebase_time_on_slice: Whether to rebase time to zero on slices.
    """

    global_paths: set[tuple[str, ...]] = field(default_factory=set)
    metadata_paths: set[tuple[str, ...]] = field(default_factory=set)
    rebase_time_on_slice: bool = True


# =============================================================================
@dataclass
class TraceTree:
    """Trace tree builder and container for a rollout."""

    config: TraceConfig = field(default_factory=TraceConfig)
    batch_size: Optional[int] = None
    length: int = 0
    spec: Optional[Any] = None
    paths: list[tuple[str, ...]] = field(default_factory=list)
    path_strs: list[str] = field(default_factory=list)
    path_to_index: dict[str, int] = field(default_factory=dict)
    leaf_is_numeric: list[bool] = field(default_factory=list)
    buffers: list[Optional[list[np.ndarray]]] = field(default_factory=list)
    meta_first: list[Optional[Any]] = field(default_factory=list)
    attached_meta: dict[str, Any] = field(default_factory=dict)
    leaf_signatures: list[Optional[tuple[tuple[int, ...], np.dtype]]] = field(default_factory=list)  # fmt: skip
    dense_leaves: Optional[list[Any]] = None

    def append(  # ------------------------------------------------------------
        self,
        state: Any,
    ) -> None:
        """Append a state snapshot into the trace tree."""
        leaves, spec, paths = torch_pytree.flatten_with_paths(state)
        if self.spec is None:
            self._init_from_first(spec, paths, leaves)
        elif not torch_pytree.spec_equal(self.spec, spec):
            raise ValueError(
                "TraceTree structure mismatch: TreeSpec changed between steps",
            )
        elif paths != self.paths:
            raise ValueError(
                "TraceTree leaf path order mismatch between steps",
            )

        for idx, leaf in enumerate(leaves):
            path = self.paths[idx]
            arr = torch_pytree.to_numeric_array(leaf)
            if self.leaf_is_numeric[idx]:
                if arr is None:
                    raise ValueError(
                        f"Leaf changed from numeric to meta at {_path_str(path)}",
                    )
                self._append_numeric(idx, arr, path)
            else:
                if self.meta_first[idx] is None:
                    self.meta_first[idx] = leaf
        self.length += 1
        self.dense_leaves = None

    def finalize(  # ----------------------------------------------------------
        self,
    ) -> None:
        """Finalize the trace by stacking buffered data into arrays."""
        self.dense_leaves = self._build_dense_leaves(clear=True)

    def slice_time(  # --------------------------------------------------------
        self,
        t0: int,
        t1: int,
    ) -> "TraceTree":
        """Return a time-sliced copy of the trace."""
        t0 = max(0, t0)
        t1 = min(self.length, t1)
        sliced = TraceTree(config=self.config)
        sliced.spec = self.spec
        sliced.paths = list(self.paths)
        sliced.path_strs = list(self.path_strs)
        sliced.path_to_index = dict(self.path_to_index)
        sliced.leaf_is_numeric = list(self.leaf_is_numeric)
        sliced.meta_first = list(self.meta_first)
        sliced.attached_meta = deepcopy(self.attached_meta)
        sliced.leaf_signatures = list(self.leaf_signatures)
        sliced.batch_size = self.batch_size
        sliced.length = max(0, t1 - t0)
        if self.dense_leaves is None:
            dense_leaves = self._build_dense_leaves(clear=False)
        else:
            dense_leaves = self.dense_leaves
        sliced.dense_leaves = [
            None if leaf is None else leaf[t0:t1] for leaf in dense_leaves
        ]
        return sliced

    def get(  # ---------------------------------------------------------------
        self,
        path: str,
    ) -> np.ndarray:
        """Return a dense array at a slash-delimited path."""
        if not self.path_to_index:
            raise ValueError("TraceTree has no data")
        idx = self.path_to_index.get(path)
        if idx is None:
            raise ValueError(f"Dense key '{path}' missing")
        if not self.leaf_is_numeric[idx]:
            raise ValueError(f"Dense key '{path}' refers to metadata leaf")
        dense_leaves = (
            self._build_dense_leaves(clear=False)
            if self.dense_leaves is None
            else self.dense_leaves
        )
        leaf = dense_leaves[idx]
        if leaf is None:
            raise ValueError(f"Dense key '{path}' is missing data")
        return leaf

    def export_dense_tree(  # -------------------------------------------------
        self,
    ) -> Any:
        """Export dense trace data as a nested pytree."""
        if self.spec is None:
            return {}
        dense_leaves = (
            self._build_dense_leaves(clear=False)
            if self.dense_leaves is None
            else self.dense_leaves
        )
        return torch_pytree.tree_unflatten(dense_leaves, self.spec)

    def export_meta_tree(  # --------------------------------------------------
        self,
    ) -> Any:
        """Export static metadata as a nested pytree."""
        if self.spec is None:
            meta_tree: Any = {}
        else:
            meta_leaves = [
                None if is_numeric else self.meta_first[idx]
                for idx, is_numeric in enumerate(self.leaf_is_numeric)
            ]
            meta_tree = torch_pytree.tree_unflatten(meta_leaves, self.spec)
        if not self.attached_meta:
            return meta_tree
        if not isinstance(meta_tree, Mapping):
            meta_tree = {}
        merged = deepcopy(dict(meta_tree))
        _merge_meta_mapping(merged, self.attached_meta, overwrite=True)
        return merged

    def attach_meta(  # -------------------------------------------------------
        self,
        meta: Mapping[str, Any],
        *,
        overwrite: bool = False,
    ) -> None:
        """Attach out-of-band metadata that is not part of the observed-step spec.

        Args:
            meta: Nested metadata mapping keyed by trace-style path segments.
            overwrite: Whether to replace existing metadata at conflicting paths.
        """
        if not isinstance(meta, Mapping):
            raise TypeError(
                f"TraceTree.attach_meta expected a mapping, got {type(meta).__name__}."
            )
        _merge_meta_mapping(self.attached_meta, meta, overwrite=overwrite)

    def attach_dense(  # ------------------------------------------------------
        self,
        path: str,
        array: "np.ndarray",
        *,
        overwrite: bool = True,
    ) -> None:
        """Attach a pre-built dense numeric array at a slash-delimited path.

        Used for post-trace injection of dataset-derived figure inputs (e.g.
        geometry-join outputs for Arena figure rendering). The array is inserted
        as if it had been part of the original trace so that ``get(path)`` and
        ``_trace_has_numeric_path(trace, path)`` work correctly.

        Args:
            path: Slash-delimited path string (e.g. ``"world_step/location_ids"``).
            array: Dense numeric array to attach.
            overwrite: If ``True`` (default), replace an existing dense leaf at
                ``path``. If ``False``, raise ``ValueError`` on collision.
        """
        if self.dense_leaves is None:
            self.finalize()
        arr = np.asarray(array)
        if path in self.path_to_index:
            idx = self.path_to_index[path]
            if not self.leaf_is_numeric[idx]:
                raise ValueError(
                    f"Cannot attach_dense at '{path}': existing leaf is metadata, not numeric."
                )
            if not overwrite:
                raise ValueError(
                    f"attach_dense collision at '{path}' and overwrite=False."
                )
            assert self.dense_leaves is not None
            self.dense_leaves[idx] = arr
            return
        path_tuple = tuple(path.split("/"))
        idx = len(self.leaf_is_numeric)
        self.paths.append(path_tuple)
        self.path_strs.append(path)
        self.path_to_index[path] = idx
        self.leaf_is_numeric.append(True)
        self.buffers.append(None)
        self.meta_first.append(None)
        self.leaf_signatures.append(None)
        assert self.dense_leaves is not None
        self.dense_leaves.append(arr)

    def export(  # ------------------------------------------------------------
        self,
        *,
        flatten: bool = False,
        sep: str = "/",
    ) -> Any:
        """Export dense trace data, optionally flattening to a path map."""
        dense = self.export_dense_tree()
        if not flatten:
            return dense
        return self._flatten_dense_map(sep=sep)

    def get_meta(self) -> dict[str, Any]:
        """Return root metadata dictionary, if available."""
        meta = self.export_meta_tree()
        return dict(meta) if isinstance(meta, dict) else {}

    def get_meta_path(self, path: str) -> Any:
        """Return the metadata leaf stored at ``path``."""
        value = self._meta_at_path(path)
        if value is None:
            raise ValueError(f"TraceTree metadata key '{path}' missing")
        return value

    def has_meta_path(self, path: str) -> bool:
        """Return whether a metadata leaf exists at ``path``."""
        return self._meta_at_path(path) is not None

    def get_environments(self) -> list[Any]:
        """Return environments stored in the trace metadata."""
        envs = self._meta_at_path("environments")
        return list(envs) if envs is not None else []

    def get_visited(self) -> Any:
        """Return visited masks stored in the trace metadata."""
        return self._meta_at_path("visited")

    def get_world(self, env_idx: int) -> Any:
        """Return the World object for a selected environment index."""
        envs = self.get_environments()
        if not envs:
            raise ValueError("TraceTree has no environments")
        if not (0 <= env_idx < len(envs)):
            raise IndexError(f"env_idx {env_idx} out of range [0, {len(envs)})")
        return envs[env_idx]

    def n_freq(self, base_path: str) -> int:
        """Return number of indexed children at a multiscale path."""
        if not self.path_strs:
            return 0

        prefix = base_path.strip("/")
        child_prefix = f"{prefix}/"
        indices: set[int] = set()
        for path_str in self.path_strs:
            if not path_str.startswith(child_prefix):
                continue
            suffix = path_str[len(child_prefix) :]
            if not suffix:
                continue
            next_seg = suffix.split("/", maxsplit=1)[0]
            if next_seg.isdigit():
                indices.add(int(next_seg))
        return len(indices)

    def validate_env_idx(  # --------------------------------------------------
        self,
        env_idx: int,
    ) -> int:
        """Validate and return environment index."""
        batch_size = int(self.batch_size or 0)
        if not (0 <= env_idx < batch_size):
            raise IndexError(
                f"env_idx {env_idx} out of range [0, {batch_size})"
            )
        return env_idx

    def validate_freq_idx(  # -------------------------------------------------
        self,
        base_path: str,
        freq_idx: int,
    ) -> int:
        """Validate and return frequency index."""
        n_freq = self.n_freq(base_path)
        if not (0 <= freq_idx < n_freq):
            raise IndexError(f"freq_idx {freq_idx} out of range [0, {n_freq})")
        return freq_idx

    def _init_from_first(  # --------------------------------------------------
        self,
        spec: Any,
        paths: list[tuple[str, ...]],
        leaves: list[Any],
    ) -> None:
        self.spec = spec
        self.paths = list(paths)
        self.path_strs = ["/".join(path) for path in self.paths]
        self.path_to_index = {
            path_str: idx for idx, path_str in enumerate(self.path_strs)
        }
        self.leaf_is_numeric = []
        self.buffers = []
        self.meta_first = []
        self.leaf_signatures = []
        for path, leaf in zip(self.paths, leaves, strict=False):
            if _is_global_path(path, global_paths=self.config.metadata_paths):
                self.leaf_is_numeric.append(False)
                self.buffers.append(None)
                self.meta_first.append(None)
                self.leaf_signatures.append(None)
                continue
            arr = torch_pytree.to_numeric_array(leaf)
            if arr is None:
                self.leaf_is_numeric.append(False)
                self.buffers.append(None)
                self.meta_first.append(None)
            else:
                self.leaf_is_numeric.append(True)
                self.buffers.append([])
                self.meta_first.append(None)
            self.leaf_signatures.append(None)

    def _append_numeric(  # ---------------------------------------------------
        self,
        idx: int,
        arr: np.ndarray,
        path: tuple[str, ...],
    ) -> None:
        """Append a numeric leaf array into the trace, enforcing batch axis."""
        if arr.dtype == object:
            raise ValueError(f"Irregular value at {_path_str(path)}")
        is_global = _is_global_path(path, global_paths=self.config.global_paths)
        if not is_global:
            self._enforce_batch_axis(arr, path)
        self._enforce_signature(idx, arr, is_global=is_global, path=path)
        buffer = self.buffers[idx]
        if buffer is None:
            raise ValueError(f"Numeric buffer missing at {_path_str(path)}")
        buffer.append(arr)

    def _enforce_batch_axis(  # -----------------------------------------------
        self,
        arr: np.ndarray,
        path: tuple[str, ...],
    ) -> None:
        """Enforce consistent batch axis for a numeric leaf, setting batch_size on first."""
        if arr.ndim == 0:
            return
        if self.batch_size is None:
            self.batch_size = int(arr.shape[0])
            return
        if int(arr.shape[0]) != self.batch_size:
            raise ValueError(
                f"Batch size mismatch at {_path_str(path)}: "
                f"expected {self.batch_size}, got {arr.shape[0]}"
            )

    def _enforce_signature(  # ------------------------------------------------
        self,
        idx: int,
        arr: np.ndarray,
        *,
        is_global: bool,
        path: tuple[str, ...],
    ) -> None:
        """Enforce consistent shape/dtype signature for a numeric leaf."""
        if is_global:
            shape_sig = tuple(arr.shape)
        elif arr.ndim == 0:
            shape_sig = ()
        else:
            shape_sig = tuple(arr.shape[1:])
        sig = (shape_sig, arr.dtype)
        prev = self.leaf_signatures[idx]
        if prev is None:
            self.leaf_signatures[idx] = sig
            return
        if prev != sig:
            raise ValueError(
                f"Shape/dtype mismatch at {_path_str(path)}: "
                f"expected {prev}, got {sig}"
            )

    def _build_dense_leaves(  # -----------------------------------------------
        self,
        *,
        clear: bool,
    ) -> list[Any]:
        """Stack buffered numeric leaves into arrays, optionally clearing buffers."""
        if self.spec is None:
            return []
        dense_leaves: list[Any] = [None] * len(self.leaf_is_numeric)
        for idx, is_numeric in enumerate(self.leaf_is_numeric):
            if not is_numeric:
                continue
            buffer = self.buffers[idx]
            if buffer is None:
                raise ValueError("Numeric buffer missing")
            if len(buffer) != self.length:
                raise ValueError(
                    f"Incomplete data buffer at {self.path_strs[idx]}: "
                    f"expected {self.length}, got {len(buffer)}"
                )
            dense_leaves[idx] = np.stack(buffer, axis=0)
            if clear:
                buffer.clear()
        return dense_leaves

    def _flatten_dense_map(  # ------------------------------------------------
        self,
        *,
        sep: str,
    ) -> dict[str, np.ndarray]:
        """Return a flat mapping of path strings to dense leaf arrays."""
        dense_leaves = (
            self._build_dense_leaves(clear=False)
            if self.dense_leaves is None
            else self.dense_leaves
        )
        flattened: dict[str, np.ndarray] = {}
        for idx, is_numeric in enumerate(self.leaf_is_numeric):
            if not is_numeric:
                continue
            leaf = dense_leaves[idx]
            if leaf is None:
                continue
            path = self.path_strs[idx]
            key = path if path else "<root>"
            flattened[key.replace("/", sep)] = leaf
        return flattened

    def _meta_at_path(  # -----------------------------------------------------
        self,
        path: str,
    ) -> Any:
        """Return the metadata value at ``path``, checking both attached meta and meta_first."""
        attached = _lookup_meta_path(self.attached_meta, path)
        if attached is not None:
            return attached
        if not self.path_to_index:
            return None
        idx = self.path_to_index.get(path)
        if idx is None or self.leaf_is_numeric[idx]:
            return None
        return self.meta_first[idx]


# =============================================================================
def _lookup_meta_path(  # -----------------------------------------------------
    meta: Mapping[str, Any],
    path: str,
) -> Any:
    """Return the attached metadata value at ``path`` or ``None`` when missing."""
    if not path:
        return meta
    current: Any = meta
    for segment in path.split("/"):
        if not isinstance(current, Mapping) or segment not in current:
            return None
        current = current[segment]
    return current


# =============================================================================
def _merge_meta_mapping(  # ----------------------------------------------------
    destination: dict[str, Any],
    source: Mapping[str, Any],
    *,
    overwrite: bool,
) -> None:
    """Merge nested metadata mappings into ``destination``.

    Raises:
        ValueError: If a source path conflicts with an existing destination path and
            ``overwrite`` is ``False``.
    """
    for key, value in source.items():
        existing = destination.get(key)
        if isinstance(existing, dict) and isinstance(value, Mapping):
            _merge_meta_mapping(existing, value, overwrite=overwrite)
            continue
        if key in destination and not overwrite:
            raise ValueError(f"TraceTree metadata path '{key}' already exists.")
        destination[key] = deepcopy(value)


# =============================================================================
def _is_global_path(  # -------------------------------------------------------
    path: tuple[str, ...],
    *,
    global_paths: set[tuple[str, ...]],
) -> bool:
    """Return True if any prefix of path is marked as global."""
    if not global_paths:
        return False
    for i in range(1, len(path) + 1):
        if path[:i] in global_paths:
            return True
    return False


# =============================================================================
def _path_str(  # -------------------------------------------------------------
    path: tuple[str, ...],
) -> str:
    """Return a readable path string for error messages."""
    return "/".join(path) or "<root>"


# =============================================================================
__all__ = [
    "TraceConfig",
    "TraceTree",
    "TraceField",
    "TraceSpec",
    "TraceObserver",
]
