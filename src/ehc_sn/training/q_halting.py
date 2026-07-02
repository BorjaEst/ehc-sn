"""Capability-generic value-control training helpers.

These primitives implement TD(0) batch assembly and zero-bootstrap validation
scoring for any model that emits a compatible value-control output.  They are
model-agnostic, task-agnostic, and controller-family-agnostic: no concrete
controller state type is imported here.

Task-specific field extraction is delegated to an injected
:class:`HybridValueObjectiveBinding`.

Training path — online (TD(0) with live bootstrap)::

    source -> controller.step() -> QHaltingInteractionRecord
    -> TD0ActorCriticBatchBuilder.build_ac_batch()  (TD(0) + V(s_{t+1}) bootstrap)
    -> HybridRLObjective.compute_step(batch) -> loss

Training path — deliberation (optional bootstrap)::

    source -> controller.step() -> QHaltingInteractionRecord
    -> TD0ActorCriticBatchBuilder.build_deliberation_ac_batch(
           next_obs=..., carry=...  # optional
       )
       (TD(0), bootstrap = V(s_{t+1}) when provided)
    -> HybridRLObjective.compute_step(batch) -> loss

Validation path (zero bootstrap)::

    RolloutChunk -> ZeroBootstrapActorCriticValidationScorer(chunk)
    -> HybridRLObjective.compute_step(zero-bootstrap batch) -> HybridRLObjectiveStep

Both zero-bootstrap paths share :func:`_zero_bootstrap_batch`.
"""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch import Tensor

from ehc_sn.controllers.contracts.actor_critic import (
    OnlineBootstrapCarry,
    OnlineBootstrapRuntime,
    QHaltingExecutionSnapshot,
    QHaltingInteractionRecord,
    QHaltingRolloutBackbone,
)
from ehc_sn.objectives.composites.hybrid_rl import (
    HybridRLObjective,
    HybridRLObjectiveStep,
    HybridValueBatch,
)
from ehc_sn.rollouts.materialization import EvaluatedChunk
from ehc_sn.rollouts.runtime import RolloutChunk, StepRecord
from ehc_sn.rollouts.scoring import score_rollout_chunk
from ehc_sn.types import Batch


# =============================================================================
class TD0ActorCriticBatchBuilder:
    """Single-step TD(0) post-processor for actor-critic training.

    Owns:
        - bootstrap value computation via ``V(s_{t+1})`` (online-specific,
          requires backbone and task runtime carry).
        - generic TD(0) batch assembly via :meth:`assemble_batch` (no RL types).
        - online orchestrator :meth:`build_ac_batch` that calls both.

    Does **not** own optimizer stepping, scheduler stepping, or loss computation.
    Does **not** inherit from ``nn.Module``; it is a plain stateless helper.
    """

    def __init__(  # ----------------------------------------------------------
        self,
        backbone: QHaltingRolloutBackbone,
        runtime: OnlineBootstrapRuntime | None,
        gamma: float,
        supervision_builder: Callable[[Any], object],
        token_weight_builder: Callable[[Any], Tensor] | None = None,
    ) -> None:
        """Initialize the batch builder with a backbone for bootstrap value
        computation, a runtime for bootstrap value extraction, a discount
        factor gamma, a supervision builder for task-specific field extraction,
        and an optional token weight builder.
        """
        self._backbone = backbone
        self._runtime = runtime
        self._gamma = gamma
        self._supervision_builder = supervision_builder
        self._token_weight_builder = token_weight_builder

    @torch.no_grad()
    def compute_bootstrap_value(  # -------------------------------------------
        self,
        carry: OnlineBootstrapCarry,
    ) -> Tensor:
        """Run the backbone on ``s_{t+1}`` and return ``V(s_{t+1})``.

        Online-specific: requires an :class:`OnlineBootstrapRuntime` (set at
        construction time) and an :class:`OnlineBootstrapCarry` (from the
        post-step controller state).

        Done-mask zeroing (``* (1 - done)``) is applied in :meth:`assemble_batch`,
        not here, so the raw critic value for all slots is returned.

        Raises:
            RuntimeError: If ``runtime`` was ``None`` at construction (deliberation
                path; online bootstrap requires a runtime with
                ``extract_next_step_obs``).

        Returns:
            Bootstrap critic value tensor of shape ``(B,)``, detached.
        """
        if self._runtime is None:
            raise RuntimeError(
                "TD0ActorCriticBatchBuilder.compute_bootstrap_value requires a "
                "non-None runtime that implements "
                "OnlineBootstrapRuntime.extract_next_step_obs. Pass "
                "runtime=None only when using the zero-bootstrap deliberation "
                "path."
            )
        next_obs = self._runtime.extract_next_step_obs(carry)
        output, _ = self._backbone(next_obs, carry.model_state)
        return output.critic.state_value.squeeze(-1)

    @torch.no_grad()
    def compute_deliberation_bootstrap_value(  # ------------------------------
        self,
        next_obs: Batch,
        carry: OnlineBootstrapCarry,
    ) -> Tensor:
        """Run the backbone on deliberation next-step observations.

        Deliberation path: callers provide the next-step observations directly
        (no runtime), along with the post-step carry holding the recurrent
        model state.
        """
        output, _ = self._backbone(next_obs, carry.model_state)
        return output.critic.state_value.squeeze(-1)

    def assemble_batch(  # ----------------------------------------------------
        self,
        record: QHaltingInteractionRecord,
        snapshot: QHaltingExecutionSnapshot,
        bootstrap_value: Tensor,
        *,
        use_token_weights: bool = False,
    ) -> HybridValueBatch:
        """Assemble a fully materialised TD(0) value-control batch.

        Generic: depends only on the neutral Q-halting record, a minimal
        execution snapshot (``steps`` and ``halted``), and a pre-computed
        bootstrap value.  No RL runtime types are accessed here.

        TD(0) formulas::

            return = r + gamma * bootstrap_value * (1 - done)
        """
        reward = record.reward.squeeze(-1)
        done = record.done.float()
        state_values = record.state_value.squeeze(-1)

        returns = reward + self._gamma * bootstrap_value * (1.0 - done)

        token_weights = None
        if use_token_weights and self._token_weight_builder is not None:
            token_weights = self._token_weight_builder(record)

        supervision = self._supervision_builder(record)
        task_logits = getattr(supervision, "task_logits", None)
        if task_logits is None:
            raise RuntimeError(
                "TD0ActorCriticBatchBuilder.assemble_batch: supervision "
                "object has no 'task_logits' attribute."
            )

        return HybridValueBatch(
            actions=record.sampled_action,
            q_values=record.q_values,
            rewards=reward,
            done=record.done,
            terminated=record.terminated,
            truncated=record.truncated,
            state_values=state_values,
            task_logits=task_logits,
            labels=supervision.labels,
            token_weights=token_weights,
            bootstrap_value=bootstrap_value,
            returns=returns,
            steps=snapshot.steps,
            halted=snapshot.halted,
        )

    def build_ac_batch(  # ----------------------------------------------------
        self,
        record: QHaltingInteractionRecord,
        carry: OnlineBootstrapCarry,
        *,
        use_token_weights: bool = False,
    ) -> HybridValueBatch:
        """Build a fully materialised value-control batch from an online rollout carry.

        Online-specific orchestrator: calls :meth:`compute_bootstrap_value`
        (which requires a non-None :class:`OnlineBootstrapRuntime`) then
        delegates to the generic :meth:`assemble_batch`.

        Raises:
            RuntimeError: If ``runtime`` is ``None``, ``record.task_output`` is
                ``None``, or required policy_decision fields are ``None``.
        """
        if record.task_output is None:
            raise RuntimeError(
                "TD0ActorCriticBatchBuilder requires a non-None task_output on "
                "QHaltingInteractionRecord."
            )
        bootstrap_value = self.compute_bootstrap_value(carry)
        return self.assemble_batch(
            record,
            carry,
            bootstrap_value,
            use_token_weights=use_token_weights,
        )

    def build_deliberation_ac_batch(  # ---------------------------------------
        self,
        record: QHaltingInteractionRecord,
        snapshot: QHaltingExecutionSnapshot,
        next_obs: Batch | None = None,
        carry: OnlineBootstrapCarry | None = None,
        *,
        use_token_weights: bool = False,
    ) -> HybridValueBatch:
        """Build a TD(0) batch for deliberation rollouts.

        Canonical path for any deliberation or static-observation training
        context where no live environment provides ``V(s_{t+1})``. When
        ``next_obs`` and ``carry`` are provided, the next-step critic value
        is computed from the backbone and used for bootstrap.
        Accepts any :class:`~ehp_sn.controllers.contracts.actor_critic.QHaltingExecutionSnapshot`.

        Args:
            record: Interaction record from the controller step.
            snapshot: Post-step execution snapshot (must expose ``steps`` and ``halted``).

        Raises:
            RuntimeError: If ``record.task_output`` is ``None``.
        """
        if record.task_output is None:
            raise RuntimeError(
                "TD0ActorCriticBatchBuilder.build_deliberation_ac_batch "
                "requires a non-None task_output on QHaltingInteractionRecord."
            )
        if next_obs is not None and carry is not None:
            bootstrap_value = self.compute_deliberation_bootstrap_value(
                next_obs, carry
            )
            return self.assemble_batch(
                record,
                snapshot,
                bootstrap_value,
                use_token_weights=use_token_weights,
            )
        return _zero_bootstrap_batch(
            record,
            snapshot.steps,
            snapshot.halted,
            self._supervision_builder,
            token_weight_builder=self._token_weight_builder,
            use_token_weights=use_token_weights,
        )


# =============================================================================
def _zero_bootstrap_batch(  # -------------------------------------------------
    ir: QHaltingInteractionRecord,
    steps: Tensor,
    halted: Tensor,
    supervision_builder: Callable[[Any], object],
    token_weight_builder: Callable[[Any], Tensor] | None = None,
    *,
    use_token_weights: bool = False,
) -> HybridValueBatch:
    """Canonical zero-bootstrap TD(0) batch assembly.

    Shared by :meth:`TD0ActorCriticBatchBuilder.build_deliberation_ac_batch`
    and :class:`ZeroBootstrapActorCriticValidationScorer`.

    With zero bootstrap: ``returns = reward + gamma * 0 * (1 - done) = reward``.
    Only the bootstrap term is zeroed.
    """
    reward = ir.reward.squeeze(-1)
    state_values = ir.state_value.squeeze(-1)
    returns = reward  # zero bootstrap collapses the discount term
    token_weights = None
    if use_token_weights and token_weight_builder is not None:
        token_weights = token_weight_builder(ir)

    supervision = supervision_builder(ir)
    task_logits = getattr(supervision, "task_logits", None)
    if task_logits is None:
        raise RuntimeError(
            "_zero_bootstrap_batch: supervision object has no 'task_logits' attribute."
        )

    return HybridValueBatch(
        actions=ir.sampled_action,
        q_values=ir.q_values,
        rewards=reward,
        done=ir.done,
        terminated=ir.terminated,
        truncated=ir.truncated,
        state_values=state_values,
        task_logits=task_logits,
        labels=supervision.labels,
        token_weights=token_weights,
        bootstrap_value=torch.zeros_like(reward),
        returns=returns,
        steps=steps,
        halted=halted,
    )


# =============================================================================
class ZeroBootstrapActorCriticValidationScorer:
    """Zero-bootstrap validation scorer for value-control models.

    Adapts a :class:`~ehp_sn.rollouts.StepRecord` (whose ``outputs`` field
    must be a :class:`~ehp_sn.controllers.contracts.actor_critic.QHaltingInteractionRecord`)
    into a :class:`~ehp_sn.objectives.hybrid_rl.HybridValueBatch` with zero
    bootstrap values via :func:`_zero_bootstrap_batch`, then calls
    :meth:`HybridRLObjective.compute_step`.

    Bootstrap value is zeroed; returns equal the immediate reward only.
    This is the same canonical construction path used by
    :meth:`TD0ActorCriticBatchBuilder.build_deliberation_ac_batch`.

    Satisfies the :class:`~ehp_sn.objectives.rollout.RolloutScorer` protocol.
    """

    def __init__(  # ----------------------------------------------------------
        self,
        objective: HybridRLObjective,
        supervision_builder: Callable[[Any], object],
        token_weight_builder: Callable[[Any], Tensor] | None = None,
    ) -> None:
        """Initialize the scorer with a loss head for step-wise loss
        computation, a supervision builder for task-specific field extraction,
        and an optional token weight builder.
        """
        self._objective = objective
        self._supervision_builder = supervision_builder
        self._token_weight_builder = token_weight_builder

    def __call__(self, chunk: RolloutChunk, **options: Any) -> EvaluatedChunk:
        """Score all records in a rollout chunk and return an evaluated chunk."""
        return score_rollout_chunk(chunk, self, **options)

    def evaluate_step(  # -----------------------------------------------------
        self,
        record: StepRecord,
        **options: Any,
    ) -> HybridRLObjectiveStep:
        """Score one executed rollout step with zero bootstrap.

        Delegates to :func:`_zero_bootstrap_batch` — the same canonical
        construction path used by
        :meth:`TD0ActorCriticBatchBuilder.build_deliberation_ac_batch`.

        Returns:
            :class:`~ehp_sn.objectives.hybrid_rl.HybridRLObjectiveStep` with
            zero-bootstrap RL diagnostics.

        Raises:
            TypeError: If ``record.outputs`` is not a
                ``QHaltingInteractionRecord``.
            RuntimeError: If ``record.outputs.task_output`` is ``None``.
        """
        ir: QHaltingInteractionRecord = record.outputs
        if not isinstance(ir, QHaltingInteractionRecord):
            raise TypeError(
                f"ZeroBootstrapActorCriticValidationScorer expects QHaltingInteractionRecord, "
                f"got {type(ir).__name__}."
            )
        if ir.task_output is None:
            raise RuntimeError(
                "ZeroBootstrapActorCriticValidationScorer requires a non-None "
                "task_output on QHaltingInteractionRecord."
            )

        steps = record.snapshot.steps
        if steps is None:
            steps = torch.zeros(
                ir.reward.shape[0], dtype=torch.long, device=ir.reward.device
            )

        ac_batch = _zero_bootstrap_batch(
            ir,
            steps,
            record.snapshot.halted,
            self._supervision_builder,
            token_weight_builder=self._token_weight_builder,
            use_token_weights=bool(options.get("use_token_weights", False)),
        )
        is_warmup = bool(options.get("is_warmup", False))
        return self._objective.compute_step(ac_batch, is_warmup=is_warmup)


# =============================================================================
__all__ = [
    "OnlineBootstrapCarry",
    "OnlineBootstrapRuntime",
    "TD0ActorCriticBatchBuilder",
    "ZeroBootstrapActorCriticValidationScorer",
]
