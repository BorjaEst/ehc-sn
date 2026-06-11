"""Working-memory probes for HRM MazeHard evaluation artifacts.

This module implements compact derived-evidence probes that read hidden-state
traces transiently and output small summary arrays.  The results are meant to
be persisted as probe artifacts (``.npz`` + ``.json``) alongside eval artifacts,
so that report figures can render without loading dense ``pfc/z_H`` / ``pfc/z_L``
tensors.

Current probes:

- ``pfc_path_memory_probe``: linear readout of target-path information from
  z_H and z_L across recurrent steps.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold


# =============================================================================
@dataclass(frozen=True)
class ProbeArtifact:
    """Compact result of one evaluation probe.

    ``arrays`` holds named NumPy arrays (time-series metrics, scalars).
    ``metadata`` holds JSON-serialisable key-value pairs that describe
    the probe method, target, and configuration.
    """

    probe_id: str
    arrays: dict[str, NDArray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


# =============================================================================
def compute_pfc_path_memory_probe(
    *,
    z_h: NDArray,
    z_l: NDArray,
    target_overlay: NDArray,
    slot_alignment: str = "exclude_first",
    split: str = "stratified_token_kfold",
    n_splits: int = 5,
    random_state: int = 0,
) -> ProbeArtifact:
    """Compute per-step linear readout of target-path information from z_H/z_L.

    Parameters
    ----------
    z_h:
        High-level state, shape ``(T, B, S, D)``.
    z_l:
        Low-level state, shape ``(T, B, S, D)``.
    target_overlay:
        Binary (or multi-class) target overlay, shape ``(B, N)``.
    slot_alignment:
        How to select grid-token slots from the S dimension.
        ``"exclude_first"`` → ``z[:, :, 1:, :]`` (slots 1..S-1, S-1 == N).
        ``"exclude_last"`` → ``z[:, :, :-1, :]`` (slots 0..S-2, S-1 == N).
    split:
        Cross-validation strategy: ``"stratified_token_kfold"``.
    n_splits:
        Number of stratified folds (must be >= 2).
    random_state:
        RNG seed for reproducibility.

    Returns
    -------
    ProbeArtifact
        With arrays for balanced accuracy, ROC AUC, and average precision
        per step, plus best-step and final-score scalars.

    Raises
    ------
    ValueError
        If ``slot_alignment`` is unknown, selected slot count does not match
        ``N``, or ``n_splits < 2``.
    """
    if slot_alignment not in ("exclude_first", "exclude_last"):
        raise ValueError(
            f"Unknown slot_alignment: {slot_alignment!r}. "
            f"Expected 'exclude_first' or 'exclude_last'."
        )
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}.")

    T, B, S, D = z_h.shape
    if z_l.shape != z_h.shape:
        raise ValueError(
            f"z_H and z_L must have the same shape; "
            f"got z_H {z_h.shape}, z_L {z_l.shape}"
        )

    if target_overlay.ndim == 1:
        target_overlay = target_overlay[np.newaxis, :]
    N = target_overlay.shape[-1]

    # Select grid-token slots.
    if slot_alignment == "exclude_first":
        grid_z = z_h[:, :, 1:, :]  # (T, B, S-1, D)
    else:
        grid_z = z_h[:, :, :-1, :]  # (T, B, S-1, D)

    if grid_z.shape[2] != N:
        raise ValueError(
            f"Selected {grid_z.shape[2]} slots with {slot_alignment=}, "
            f"but target_overlay has N={N}. Check slot alignment."
        )

    # Binary target mask: background = mode, foreground = rest.
    values, counts = np.unique(target_overlay, return_counts=True)
    background = values[np.argmax(counts)]
    y: NDArray = (target_overlay != background).astype(int).ravel()  # (N,)
    n_positive = int(y.sum())

    # Use only first batch element.
    z_h_b0 = grid_z[:, 0, :, :]  # (T, N, D)
    z_l_b0 = z_l[:, 0, :, :] if slot_alignment == "exclude_first" else z_l[:, 0, :-1, :]
    # Re-align for L (same slot_alignment logic).
    if slot_alignment == "exclude_first":
        z_l_b0 = z_l[:, 0, 1:, :]
    else:
        z_l_b0 = z_l[:, 0, :-1, :]

    # Prepare cross-validator.
    cv = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )

    probe = LogisticRegression(
        class_weight="balanced",
        C=1.0,
        max_iter=1000,
        random_state=random_state,
    )

    metrics: dict[str, list[np.ndarray]] = {
        "h_balanced_accuracy": [],
        "l_balanced_accuracy": [],
        "h_roc_auc": [],
        "l_roc_auc": [],
        "h_average_precision": [],
        "l_average_precision": [],
    }

    for t in range(T):
        X_h = z_h_b0[t]  # (N, D)
        X_l = z_l_b0[t]  # (N, D)

        for prefix, X in [("h_", X_h), ("l_", X_l)]:
            ba_folds: list[float] = []
            ra_folds: list[float] = []
            ap_folds: list[float] = []
            for train_idx, test_idx in cv.split(X, y):
                probe.fit(X[train_idx], y[train_idx])
                y_pred = probe.predict(X[test_idx])
                y_prob = probe.predict_proba(X[test_idx])[:, 1]
                ba_folds.append(balanced_accuracy_score(y[test_idx], y_pred))
                ra_folds.append(roc_auc_score(y[test_idx], y_prob))
                ap_folds.append(
                    average_precision_score(y[test_idx], y_prob)
                )
            metrics[f"{prefix}balanced_accuracy"].append(np.array(ba_folds))
            metrics[f"{prefix}roc_auc"].append(np.array(ra_folds))
            metrics[f"{prefix}average_precision"].append(np.array(ap_folds))

    # Build arrays.
    arrays: dict[str, NDArray] = {}
    for key, fold_list in metrics.items():
        stacked = np.stack(fold_list, axis=0)  # (T, n_splits)
        arrays[f"{key}_mean"] = stacked.mean(axis=1).astype(np.float32)
        arrays[f"{key}_std"] = stacked.std(axis=1, ddof=1).astype(np.float32)

    # Scalars.
    h_best = int(np.argmax(arrays["h_balanced_accuracy_mean"]))
    l_best = int(np.argmax(arrays["l_balanced_accuracy_mean"]))
    arrays["h_best_step"] = np.array(h_best, dtype=np.int32)
    arrays["l_best_step"] = np.array(l_best, dtype=np.int32)
    arrays["h_final_score"] = arrays["h_balanced_accuracy_mean"][-1]
    arrays["l_final_score"] = arrays["l_balanced_accuracy_mean"][-1]

    metadata: dict[str, Any] = {
        "probe_id": "pfc_path_memory_probe",
        "probe_version": 1,
        "method": "logistic_regression_balanced",
        "target": "target/solution_overlay",
        "states": ["pfc/z_H", "pfc/z_L"],
        "slot_alignment": slot_alignment,
        "split": split,
        "n_splits": n_splits,
        "n_tokens": N,
        "n_positive": n_positive,
        "n_features": D,
        "steps": T,
        "primary_metric": "balanced_accuracy",
        "random_state": random_state,
    }

    return ProbeArtifact(
        probe_id="pfc_path_memory_probe",
        arrays=arrays,
        metadata=metadata,
    )


def compute_pfc_path_memory_probe_from_artifact(
    artifact_path: str,
    *,
    slot_alignment: str = "exclude_first",
    n_splits: int = 5,
    random_state: int = 0,
) -> ProbeArtifact:
    """Load an eval artifact and compute the path-memory probe.

    Convenience function for script/notebook validation.
    """
    from ehc_sn.eval.artifacts import load_artifact_run_cases

    cases = load_artifact_run_cases(artifact_path)
    t = cases[0].trace

    z_h: NDArray = t.get("pfc/z_H")
    z_l: NDArray = t.get("pfc/z_L")
    target_raw: NDArray = np.asarray(
        t.get_meta_path("target/solution_overlay")
    )

    return compute_pfc_path_memory_probe(
        z_h=z_h,
        z_l=z_l,
        target_overlay=target_raw,
        slot_alignment=slot_alignment,
        n_splits=n_splits,
        random_state=random_state,
    )


# =============================================================================
# Persistence helpers
# =============================================================================


def save_probe_artifact(
    artifact: ProbeArtifact,
    *,
    npz_path: str | Path,
    json_path: str | Path | None = None,
) -> None:
    """Persist a probe artifact to NPZ + JSON files.

    Parameters
    ----------
    artifact:
        The probe artifact to persist.
    npz_path:
        Path for the arrays NPZ file (e.g. ``"probes/pfc_path_memory.npz"``).
    json_path:
        Optional path for the metadata JSON file. If ``None``, inferred from
        ``npz_path`` by replacing the suffix with ``.json``.
    """
    npz_path = Path(npz_path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **artifact.arrays)

    if json_path is None:
        json_path = npz_path.with_suffix(".json")
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(artifact.metadata, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def load_probe_artifact(
    *,
    npz_path: str | Path,
    json_path: str | Path | None = None,
) -> ProbeArtifact:
    """Load a probe artifact from NPZ + JSON files.

    Parameters
    ----------
    npz_path:
        Path to the arrays NPZ file.
    json_path:
        Optional path to the metadata JSON file. If ``None``, inferred from
        ``npz_path`` by replacing the suffix with ``.json``.

    Returns
    -------
    ProbeArtifact
        The deserialized probe artifact.
    """
    npz_path = Path(npz_path)
    if json_path is None:
        json_path = npz_path.with_suffix(".json")
    json_path = Path(json_path)

    arrays: dict[str, NDArray] = dict(np.load(npz_path, allow_pickle=True))
    metadata: dict[str, Any] = json.loads(json_path.read_text(encoding="utf-8"))
    probe_id: str = metadata.get("probe_id", "unknown")
    return ProbeArtifact(probe_id=probe_id, arrays=arrays, metadata=metadata)


def compute_and_persist_probes_for_artifact(
    artifact_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    slot_alignment: str = "exclude_first",
    n_splits: int = 5,
    random_state: int = 0,
) -> Path:
    """Compute the PFC path-memory probe for an eval artifact and persist it.

    Writes ``probes/pfc_path_memory_probe.npz`` and
    ``probes/pfc_path_memory_probe.json`` inside *output_dir* (or inside
    the artifact directory if *output_dir* is ``None``).

    Parameters
    ----------
    artifact_path:
        Path to an eval artifact directory (containing ``_SUCCESS`` and
        ``manifest.json``).
    output_dir:
        Directory for the probe output. If ``None``, a ``probes/``
        subdirectory is created inside *artifact_path*.
    slot_alignment:
        Passed through to :func:`compute_pfc_path_memory_probe`.
    n_splits:
        Passed through to :func:`compute_pfc_path_memory_probe`.
    random_state:
        Passed through to :func:`compute_pfc_path_memory_probe`.

    Returns
    -------
    Path
        The resolved output directory.

    Raises
    ------
    FileNotFoundError
        If the eval artifact does not exist or lacks required traces.
    """
    from ehc_sn.eval.artifacts import load_artifact_run_cases

    artifact_path = Path(artifact_path)
    cases = load_artifact_run_cases(artifact_path)
    if not cases:
        raise FileNotFoundError(
            f"No cases found in eval artifact: {artifact_path}"
        )

    t = cases[0].trace
    z_h: NDArray = t.get("pfc/z_H")
    z_l: NDArray = t.get("pfc/z_L")
    target_raw: NDArray = np.asarray(
        t.get_meta_path("target/solution_overlay")
    )

    probe = compute_pfc_path_memory_probe(
        z_h=z_h,
        z_l=z_l,
        target_overlay=target_raw,
        slot_alignment=slot_alignment,
        n_splits=n_splits,
        random_state=random_state,
    )

    if output_dir is None:
        output_dir = artifact_path / "probes"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_path = output_dir / f"{probe.probe_id}.npz"
    json_path = output_dir / f"{probe.probe_id}.json"
    save_probe_artifact(probe, npz_path=npz_path, json_path=json_path)

    return output_dir


# =============================================================================
def _probe_summary_report(artifact: ProbeArtifact) -> str:
    """Return a human-readable summary of probe results."""
    m = artifact.metadata
    arr = artifact.arrays
    lines = [
        f"Probe: {m['probe_id']}",
        f"  Method: {m['method']}, Slot alignment: {m['slot_alignment']}",
        f"  Split: {m['split']}, n_splits={m['n_splits']}",
        f"  Tokens: {m['n_tokens']}, Positive: {m['n_positive']}, Features: {m['n_features']}",
        f"  Steps: {m['steps']}",
        "",
        "  Balanced accuracy (z_H):",
        f"    Mean over steps: {arr['h_balanced_accuracy_mean'].mean():.4f}",
        f"    Best step: {int(arr['h_best_step'])} ({arr['h_balanced_accuracy_mean'].max():.4f})",
        f"    Final step: {arr['h_final_score']:.4f}",
        "",
        "  Balanced accuracy (z_L):",
        f"    Mean over steps: {arr['l_balanced_accuracy_mean'].mean():.4f}",
        f"    Best step: {int(arr['l_best_step'])} ({arr['l_balanced_accuracy_mean'].max():.4f})",
        f"    Final step: {arr['l_final_score']:.4f}",
    ]
    return "\n".join(lines)
