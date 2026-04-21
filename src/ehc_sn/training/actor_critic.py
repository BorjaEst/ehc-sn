"""Capability-generic actor-critic training helpers.

These primitives implement TD(0) batch assembly and zero-bootstrap validation
scoring for any model that emits a compatible actor-critic output.  They are
model-agnostic and task-agnostic; task-specific field extraction is delegated
to an injected :class:`HybridActorCriticTaskBinding`.

Training path (use :class:`TD0ActorCriticBatchBuilder`)::

    source -> controller.step() -> InteractionRecord
    -> TD0ActorCriticBatchBuilder.build_ac_batch()  (TD(0) post-processing)
    -> HybridRLLossHead.compute_step(batch) -> loss

Validation path (use :class:`ZeroBootstrapActorCriticValidationScorer`)::

    RolloutChunk -> ZeroBootstrapActorCriticValidationScorer(chunk)
    -> HybridRLLossHead.compute_step(zero-bootstrap batch) -> HybridRLLossStep
"""

from __future__ import annotations

from typing import Any, Protocol

import torch
from torch import Tensor

from ehc_sn.controllers.rl import InteractionRecord, RLRolloutBackbone, RLRolloutState, RLTaskRuntime
from ehc_sn.objectives.hybrid_rl import HybridActorCriticBatch, HybridRLLossHead, HybridRLLossStep
from ehc_sn.rollouts import EvaluatedChunk, ObservedStep, RolloutChunk, StepRecord


# =================================================================================================
class HybridActorCriticTaskBinding(Protocol):
    """Adapter-owned extraction of task-specific fields from an :class:`InteractionRecord`.

    Implement this protocol in the adapter layer so that the generic
    :class:`TD0ActorCriticBatchBuilder` and
    :class:`ZeroBootstrapActorCriticValidationScorer` remain task-agnostic.

    The binding owns:
    - extraction of token-prediction logits for the LM loss component
    - extraction of supervision labels for the LM loss component
    """

    def extract_task_logits(self, record: InteractionRecord) -> Tensor:
        """Return token-prediction logits from the task output on ``record``."""
        ...

    def extract_labels(self, record: InteractionRecord) -> Tensor:
        """Return supervision labels from the interaction record."""
        ...


# =================================================================================================
class TD0ActorCriticBatchBuilder:
    """Single-step TD(0) post-processor for actor-critic training.

    Owns:
        - bootstrap value computation via ``V(s_{t+1})``.
        - TD(0) return: ``r + gamma * V(s_{t+1}) * (1 - done)``
        - advantage: ``(return - V(s_t)).detach()``
        - :class:`~ehc_sn.objectives.hybrid_rl.HybridActorCriticBatch` assembly.

    Does **not** own optimizer stepping, scheduler stepping, or loss computation.
    Does **not** inherit from ``nn.Module``; it is a plain stateless helper.
    """

    def __init__(
        self,
        backbone: RLRolloutBackbone,
        runtime: RLTaskRuntime,
        gamma: float,
        task_binding: HybridActorCriticTaskBinding,
    ) -> None:
        self._backbone = backbone
        self._runtime = runtime
        self._gamma = gamma
        self._task_binding = task_binding

    # -- Bootstrap value computation ------------------------------------------------------------

    @torch.no_grad()
    def compute_bootstrap_value(self, new_state: RLRolloutState) -> Tensor:
        """Run the backbone on ``s_{t+1}`` and return ``V(s_{t+1})``.

        Done-mask zeroing (``* (1 - done)``) is applied in :meth:`build_ac_batch`,
        not here, so the raw critic value for all slots is returned.

        Returns:
            Bootstrap critic value tensor of shape ``(B,)``, detached.
        """
        next_obs = self._runtime.extract_next_step_obs(new_state)
        output, _ = self._backbone(next_obs, new_state.model_state)
        if output.critic is None:
            B = new_state.halted.shape[0]
            return new_state.halted.new_zeros(B, dtype=torch.float32)
        return output.critic.state_value.squeeze(-1)

    # -- Batch assembly -------------------------------------------------------------------------

    def build_ac_batch(
        self,
        record: InteractionRecord,
        new_state: RLRolloutState,
    ) -> HybridActorCriticBatch:
        """Build a fully materialized actor-critic batch with TD(0) targets.

        TD(0) formulas::

            bootstrap_value = V(s_{t+1})          # via backbone, no_grad
            return          = r + gamma * bootstrap_value * (1 - done)
            advantage       = (return - V(s_t)).detach()

        Raises:
            RuntimeError: If ``record.task_output`` is ``None`` or required
                policy_decision fields are ``None``.
        """
        if record.task_output is None:
            raise RuntimeError(
                "TD0ActorCriticBatchBuilder requires a non-None task_output on InteractionRecord."
            )
        if record.policy_decision.log_prob is None:
            raise RuntimeError(
                "TD0ActorCriticBatchBuilder.build_ac_batch: policy_decision.log_prob is None."
            )
        if record.policy_decision.entropy is None:
            raise RuntimeError(
                "TD0ActorCriticBatchBuilder.build_ac_batch: policy_decision.entropy is None."
            )

        reward = record.reward.squeeze(-1)
        done = record.done.float()
        value_est = record.value_estimate.squeeze(-1)

        bootstrap_value = self.compute_bootstrap_value(new_state)
        returns = reward + self._gamma * bootstrap_value * (1.0 - done)
        advantages = (returns - value_est).detach()

        return HybridActorCriticBatch(
            actions=record.sampled_action,
            policy_logits=record.policy_logits,
            rewards=reward,
            done=record.done,
            terminated=record.terminated,
            truncated=record.truncated,
            value_estimates=value_est,
            action_log_prob=record.policy_decision.log_prob,
            action_entropy=record.policy_decision.entropy,
            task_logits=self._task_binding.extract_task_logits(record),
            labels=self._task_binding.extract_labels(record),
            bootstrap_value=bootstrap_value,
            returns=returns,
            advantages=advantages,
            steps=new_state.steps,
            halted=new_state.halted,
        )


# =================================================================================================
class ZeroBootstrapActorCriticValidationScorer:
    """Zero-bootstrap validation scorer for actor-critic models.

    Adapts a :class:`~ehc_sn.rollouts.StepRecord` (whose ``outputs`` field
    must be an :class:`~ehc_sn.controllers.rl.InteractionRecord`) into a
    :class:`~ehc_sn.objectives.hybrid_rl.HybridActorCriticBatch` with zero
    bootstrap values, then calls :meth:`HybridRLLossHead.compute_step`.

    Bootstrap value is zeroed; returns equal the immediate reward only.
    This is conservative and correct for validation — the focus is token
    accuracy and episode metrics, not precise TD target estimation.

    Satisfies the :class:`~ehc_sn.lightning._rollout.RolloutObjective` protocol.
    """

    def __init__(
        self,
        objective: HybridRLLossHead,
        task_binding: HybridActorCriticTaskBinding,
    ) -> None:
        self._objective = objective
        self._task_binding = task_binding

    def __call__(self, chunk: RolloutChunk, **options: Any) -> EvaluatedChunk:
        """Score all records in a rollout chunk and return an evaluated chunk."""
        observed_steps: list[ObservedStep] = []
        total_loss: Tensor | None = None

        for record in chunk.records:
            step_output = self.evaluate_step(record, **options)
            observed_steps.append(
                ObservedStep(
                    index=record.index,
                    batch=record.batch,
                    snapshot=record.snapshot,
                    outputs=step_output,
                )
            )
            total_loss = step_output.loss if total_loss is None else total_loss + step_output.loss

        if total_loss is None:
            raise ValueError("ZeroBootstrapActorCriticValidationScorer received an empty chunk.")

        return EvaluatedChunk(
            steps=tuple(observed_steps),
            loss=total_loss,
            final_carry=chunk.final_carry,
            source_exhausted=chunk.source_exhausted,
        )

    def evaluate_step(self, record: StepRecord, **options: Any) -> HybridRLLossStep:
        """Score one executed rollout step with zero bootstrap.

        Returns:
            :class:`~ehc_sn.objectives.hybrid_rl.HybridRLLossStep` with
            approximate (zero-bootstrap) RL diagnostics.

        Raises:
            TypeError: If ``record.outputs`` is not an ``InteractionRecord``.
            RuntimeError: If ``record.outputs.task_output`` is ``None`` or
                required ``policy_decision`` fields are ``None``.
        """
        ir: InteractionRecord = record.outputs
        if not isinstance(ir, InteractionRecord):
            raise TypeError(
                f"ZeroBootstrapActorCriticValidationScorer expects InteractionRecord, "
                f"got {type(ir).__name__}."
            )
        if ir.task_output is None:
            raise RuntimeError(
                "ZeroBootstrapActorCriticValidationScorer requires a non-None "
                "task_output on InteractionRecord."
            )
        if ir.policy_decision.log_prob is None:
            raise RuntimeError(
                "ZeroBootstrapActorCriticValidationScorer: policy_decision.log_prob is None."
            )
        if ir.policy_decision.entropy is None:
            raise RuntimeError(
                "ZeroBootstrapActorCriticValidationScorer: policy_decision.entropy is None."
            )

        reward = ir.reward.squeeze(-1)
        zero = torch.zeros_like(reward)

        steps = record.snapshot.steps
        if steps is None:
            steps = torch.zeros(reward.shape[0], dtype=torch.long, device=reward.device)

        ac_batch = HybridActorCriticBatch(
            actions=ir.sampled_action,
            policy_logits=ir.policy_logits,
            rewards=reward,
            done=ir.done,
            terminated=ir.terminated,
            truncated=ir.truncated,
            value_estimates=ir.value_estimate.squeeze(-1),
            action_log_prob=ir.policy_decision.log_prob,
            action_entropy=ir.policy_decision.entropy,
            task_logits=self._task_binding.extract_task_logits(ir),
            labels=self._task_binding.extract_labels(ir),
            bootstrap_value=zero,
            returns=reward,
            advantages=zero,
            steps=steps,
            halted=record.snapshot.halted,
        )
        is_warmup = bool(options.get("is_warmup", False))
        return self._objective.compute_step(ac_batch, is_warmup=is_warmup)


# =================================================================================================
__all__ = [
    "HybridActorCriticTaskBinding",
    "TD0ActorCriticBatchBuilder",
    "ZeroBootstrapActorCriticValidationScorer",
]
