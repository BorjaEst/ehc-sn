"""Shared regime runner for named evaluation regimes.

The runner is the engine that:

1. Builds per-regime metric collections (fresh, namespaced, never ``val/``).
2. Iterates provider case batches.
3. Calls :meth:`~ehc_sn.lightning.eval.contracts.SupportsEvaluationRegimes.execute_evaluation_batch`
   on the Lightning surface.
4. Updates the namespaced metric collection from each batch via
   :attr:`~ehc_sn.lightning.eval.contracts.EvaluationBatchArtifacts.apply_to_metrics`.
5. Caches the last trace-bearing artifact per regime for figure consumers.
6. Returns collected metrics and the cached artifact.

This module is stateless: the caller (callback) owns state between calls.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from typing import Any, Optional

from torchmetrics import MetricCollection

from ehc_sn.lightning.eval.contracts import (
    EvaluationBatchArtifacts,
    EvaluationRegimeSettings,
    EvaluationSourceProvider,
    SupportsEvaluationRegimes,
)

log = logging.getLogger(__name__)


# =================================================================================================
@dataclass
class RegimeRunResult:
    """Outcome of running one named evaluation regime.

    Attributes:
        regime_id: The regime that was run.
        metric_namespace: The namespace used for logging (e.g. ``"diag/probe/"``).
        metrics: Computed metric values, keyed by full metric name (with prefix).
        latest_artifact: Last artifact produced; ``None`` if the provider was empty.
        n_batches: Number of case batches consumed.
        provider_description: Human-readable provider description for logging.
    """

    regime_id: str
    metric_namespace: str
    metrics: dict[str, Any]
    latest_artifact: Optional[EvaluationBatchArtifacts]
    n_batches: int
    provider_description: str


# =================================================================================================
def load_provider(regime: EvaluationRegimeSettings) -> EvaluationSourceProvider:
    """Import and instantiate the provider class referenced by ``regime.provider_ref``.

    ``provider_ref`` must be a dotted Python import path to a class or callable
    that returns an :class:`~ehc_sn.lightning.eval.contracts.EvaluationSourceProvider`.
    ``provider_settings`` are forwarded as keyword arguments.

    Args:
        regime: The regime configuration owning the provider reference.

    Returns:
        An instantiated :class:`~ehc_sn.lightning.eval.contracts.EvaluationSourceProvider`.

    Raises:
        ImportError: If the module or attribute cannot be found.
        TypeError: If the instantiated object does not satisfy :class:`EvaluationSourceProvider`.
    """
    module_path, _, attr = regime.provider_ref.rpartition(".")
    if not module_path:
        raise ImportError(f"provider_ref '{regime.provider_ref}' must be a dotted path (module.ClassName).")
    module = importlib.import_module(module_path)
    factory = getattr(module, attr)
    provider: object = factory(**regime.provider_settings)
    if not isinstance(provider, EvaluationSourceProvider):
        raise TypeError(
            f"provider_ref '{regime.provider_ref}' produced an object that does not satisfy "
            "EvaluationSourceProvider. Ensure it implements provide_cases() and description()."
        )
    return provider


# =================================================================================================
def run_evaluation_regime(
    *,
    pl_module: SupportsEvaluationRegimes,
    regime: EvaluationRegimeSettings,
    provider: EvaluationSourceProvider,
) -> RegimeRunResult:
    """Execute one named evaluation regime against a Lightning module.

    Steps:
    1. Build a fresh namespaced metric collection via ``pl_module.build_evaluation_metrics``.
    2. Iterate case batches from ``provider.provide_cases``.
    3. Call ``pl_module.execute_evaluation_batch`` for each batch.
    4. Stamp regime identity onto the artifact.
    5. Update the metric collection via ``artifact.apply_to_metrics`` if present.
    6. Cache the last trace-bearing artifact.
    7. Return computed metrics and the latest artifact.

    This function **never touches** ``pl_module.val_metrics`` or logs to ``val/``.

    Args:
        pl_module: A Lightning module satisfying :class:`SupportsEvaluationRegimes`.
        regime: The regime configuration (namespace, schedule, trace request).
        provider: An instantiated provider yielding :class:`EvaluationCaseBatch` items.

    Returns:
        :class:`RegimeRunResult` with computed metrics and the latest artifact.
    """
    namespace = regime.metric_namespace  # e.g. "diag/my_probe/"
    metric_collection: MetricCollection = pl_module.build_evaluation_metrics(namespace)
    metric_collection.reset()

    # Mirror Lightning's batch transfer sequence so providers can stay device-agnostic.
    # Providers yield CPU tensors by design; the runner owns the invariant that batches
    # passed to execute_evaluation_batch are in Lightning runtime form (correct device +
    # precision).  See Lightning evaluation_loop.py:414, module.py:355, strategy.py:278.
    trainer = getattr(pl_module, "trainer", None)

    def _transfer_batch(batch: Any) -> Any:
        if trainer is None:
            return batch
        try:
            batch = trainer.precision_plugin.convert_input(batch)
            batch = pl_module._on_before_batch_transfer(batch, dataloader_idx=0)  # type: ignore[union-attr]
            batch = trainer.strategy.batch_to_device(batch, dataloader_idx=0)
        except Exception:
            log.debug("Regime runner: batch transfer failed; using batch as-is.", exc_info=True)
        return batch

    # Move metric collection to the strategy root device once so metric state
    # does not sit on CPU while receiving device tensors.
    if trainer is not None:
        try:
            root_device = trainer.strategy.root_device
            metric_collection = metric_collection.to(root_device)
        except Exception:
            log.debug("Regime runner: metric collection device transfer failed; leaving on default device.", exc_info=True)

    trace_request = regime.trace_request if regime.trace_request.enabled else None
    latest_artifact: Optional[EvaluationBatchArtifacts] = None
    n_batches = 0

    for case_batch in provider.provide_cases(max_batches=regime.schedule.max_batches):
        batch = _transfer_batch(case_batch.batch)
        artifact = pl_module.execute_evaluation_batch(batch, trace_request)

        # Stamp regime identity and forwarded metadata onto the artifact.
        artifact.regime_id = regime.regime_id
        artifact.metric_namespace = namespace
        artifact.case_id = case_batch.case_id
        artifact.source_metadata = case_batch.metadata

        # Update the regime metric collection via the family-owned closure.
        if artifact.apply_to_metrics is not None:
            try:
                artifact.apply_to_metrics(metric_collection)
            except Exception:
                log.debug("Regime runner: metric update raised an error for regime '%s'.", regime.regime_id, exc_info=True)

        # Cache the most recent trace-bearing artifact; fall back to any artifact.
        if artifact.trace is not None:
            latest_artifact = artifact
        elif latest_artifact is None:
            latest_artifact = artifact

        n_batches += 1

    computed: dict[str, Any] = {}
    if n_batches > 0:
        try:
            raw = metric_collection.compute()
            computed = {k: v.item() if hasattr(v, "item") else v for k, v in raw.items()}
        except Exception:
            log.debug("Regime runner: metric compute raised an error for regime '%s'.", regime.regime_id, exc_info=True)

    return RegimeRunResult(
        regime_id=regime.regime_id,
        metric_namespace=namespace,
        metrics=computed,
        latest_artifact=latest_artifact,
        n_batches=n_batches,
        provider_description=provider.description(),
    )


# =============================================================================
__all__ = ["RegimeRunResult", "load_provider", "run_evaluation_regime"]
