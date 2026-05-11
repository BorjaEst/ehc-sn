"""Capability-generic actor-critic training helpers.

These primitives implement TD(0) batch assembly and zero-bootstrap validation
scoring for any model that emits a compatible actor-critic output.  They are
model-agnostic, task-agnostic, and controller-family-agnostic: no concrete
controller state type is imported here.

Task-specific field extraction is delegated to an injected
:class:`HybridActorCriticTaskBinding`.

Training path — online (TD(0) with live bootstrap)::

    source -> controller.step() -> ActorCriticInteractionRecord
    -> TD0ActorCriticBatchBuilder.build_ac_batch()  (TD(0) + V(s_{t+1}) bootstrap)
    -> HybridRLLossHead.compute_step(batch) -> loss

Training path — deliberation (zero bootstrap)::

    source -> controller.step() -> ActorCriticInteractionRecord
    -> TD0ActorCriticBatchBuilder.build_deliberation_ac_batch()  (TD(0), bootstrap=0)
    -> HybridRLLossHead.compute_step(batch) -> loss

Validation path (zero bootstrap)::

    RolloutChunk -> ZeroBootstrapActorCriticValidationScorer(chunk)
    -> HybridRLLossHead.compute_step(zero-bootstrap batch) -> HybridRLLossStep

Both zero-bootstrap paths share :func:`_zero_bootstrap_batch`.
"""

from __future__ import annotations

from typing import Any, Protocol

import torch
from torch import Tensor

from ehc_sn.controllers.contracts.actor_critic import (
    ActorCriticExecutionSnapshot,
    ActorCriticInteractionRecord,
    ActorCriticRolloutBackbone,
    OnlineBootstrapCarry,
    OnlineBootstrapRuntime,
)
from ehc_sn.objectives.hybrid_rl import HybridActorCriticBatch, HybridRLLossHead, HybridRLLossStep
from ehc_sn.rollouts import EvaluatedChunk, ObservedStep, RolloutChunk, StepRecord
from ehc_sn.types import Batch


# =================================================================================================
class HybridActorCriticTaskBinding(Protocol):
    """Adapter-owned extraction of task-specific fields from an :class:`ActorCriticInteractionRecord`.

    Implement this protocol in the adapter layer so that the generic
    :class:`TD0ActorCriticBatchBuilder` and
    :class:`ZeroBootstrapActorCriticValidationScorer` remain task-agnostic.

    The binding owns:
    - extraction of token-prediction logits for the LM loss component
    - extraction of supervision labels for the LM loss component
    """

    def extract_task_logits(self, record: ActorCriticInteractionRecord) -> Tensor:
        """Return token-prediction logits from the task output on ``record``."""
        ...

    def extract_labels(self, record: ActorCriticInteractionRecord) -> Tensor:
        """Return supervision labels from the interaction record."""
        ...


# =================================================================================================
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

    def __init__(
        self,
        backbone: ActorCriticRolloutBackbone,
        runtime: OnlineBootstrapRuntime | None,
        gamma: float,
        task_binding: HybridActorCriticTaskBinding,
    ) -> None:
        self._backbone = backbone
        self._runtime = runtime
        self._gamma = gamma
        self._task_binding = task_binding

    # -- Bootstrap value computation (online-specific) ------------------------------------------

    @torch.no_grad()
    def compute_bootstrap_value(self, carry: OnlineBootstrapCarry) -> Tensor:
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
                "TD0ActorCriticBatchBuilder.compute_bootstrap_value requires a non-None runtime "
                "that implements OnlineBootstrapRuntime.extract_next_step_obs. "
                "Pass runtime=None only when using the zero-bootstrap deliberation path."
            )
        next_obs = self._runtime.extract_next_step_obs(carry)
        output, _ = self._backbone(next_obs, carry.model_state)
        return output.critic.state_value.squeeze(-1)

    # -- Generic batch assembly -----------------------------------------------------------------

    def assemble_batch(
        self,
        record: ActorCriticInteractionRecord,
        snapshot: ActorCriticExecutionSnapshot,
        bootstrap_value: Tensor,
    ) -> HybridActorCriticBatch:
        """Assemble a fully materialised TD(0) actor-critic batch.

        Generic: depends only on the neutral actor-critic record, a minimal
        execution snapshot (``steps`` and ``halted``), and a pre-computed
        bootstrap value.  No RL runtime types are accessed here.

        TD(0) formulas::

            return    = r + gamma * bootstrap_value * (1 - done)
            advantage = (return - V(s_t)).detach()

        Raises:
            RuntimeError: If required ``policy_decision`` fields are ``None``.
        """
        if record.policy_decision.log_prob is None:
            raise RuntimeError("TD0ActorCriticBatchBuilder.assemble_batch: policy_decision.log_prob is None.")
        if record.policy_decision.entropy is None:
            raise RuntimeError("TD0ActorCriticBatchBuilder.assemble_batch: policy_decision.entropy is None.")

        reward = record.reward.squeeze(-1)
        done = record.done.float()
        value_est = record.value_estimate.squeeze(-1)

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
            steps=snapshot.steps,
            halted=snapshot.halted,
        )

    # -- Online-specific orchestrator -----------------------------------------------------------

    def build_ac_batch(
        self,
        record: ActorCriticInteractionRecord,
        carry: OnlineBootstrapCarry,
    ) -> HybridActorCriticBatch:
        """Build a fully materialised actor-critic batch from an online rollout carry.

        Online-specific orchestrator: calls :meth:`compute_bootstrap_value`
        (which requires a non-None :class:`OnlineBootstrapRuntime`) then
        delegates to the generic :meth:`assemble_batch`.

        Raises:
            RuntimeError: If ``runtime`` is ``None``, ``record.task_output`` is
                ``None``, or required policy_decision fields are ``None``.
        """
        if record.task_output is None:
            raise RuntimeError("TD0ActorCriticBatchBuilder requires a non-None task_output on ActorCriticInteractionRecord.")
        bootstrap_value = self.compute_bootstrap_value(carry)
        return self.assemble_batch(record, carry, bootstrap_value)

    # -- Zero-bootstrap orchestrator (deliberation / static-obs training) ----------------------

    def build_deliberation_ac_batch(
        self,
        record: ActorCriticInteractionRecord,
        snapshot: ActorCriticExecutionSnapshot,
    ) -> HybridActorCriticBatch:
        """Build a TD(0) batch with zero bootstrap value.

        Canonical path for any deliberation or static-observation training
        context where no live environment provides ``V(s_{t+1})``.
        Accepts any :class:`~ehc_sn.controllers.contracts.actor_critic.ActorCriticExecutionSnapshot`
        (concrete controller states such as ``DeliberationACRolloutState`` satisfy
        this protocol automatically).

        Args:
            record: Interaction record from the controller step.
            snapshot: Post-step execution snapshot (must expose ``steps`` and ``halted``).

        Raises:
            RuntimeError: If ``record.task_output`` is ``None``.
        """
        if record.task_output is None:
            raise RuntimeError(
                "TD0ActorCriticBatchBuilder.build_deliberation_ac_batch requires a non-None "
                "task_output on ActorCriticInteractionRecord."
            )
        return _zero_bootstrap_batch(record, snapshot.steps, snapshot.halted, self._task_binding)


# =================================================================================================
def _zero_bootstrap_batch(
    ir: ActorCriticInteractionRecord,
    steps: Tensor,
    halted: Tensor,
    task_binding: HybridActorCriticTaskBinding,
) -> HybridActorCriticBatch:
    """Canonical zero-bootstrap TD(0) batch assembly.

    Shared by :meth:`TD0ActorCriticBatchBuilder.build_deliberation_ac_batch`
    and :class:`ZeroBootstrapActorCriticValidationScorer`.

    With zero bootstrap: ``returns = reward + gamma * 0 * (1 - done) = reward``.
    Advantages are ``(reward - V(s_t)).detach()`` — consistent with the online TD(0)
    formula; only the bootstrap term is zeroed.
    """
    reward = ir.reward.squeeze(-1)
    value_est = ir.value_estimate.squeeze(-1)
    returns = reward  # zero bootstrap collapses the discount term
    advantages = (returns - value_est).detach()
    return HybridActorCriticBatch(
        actions=ir.sampled_action,
        policy_logits=ir.policy_logits,
        rewards=reward,
        done=ir.done,
        terminated=ir.terminated,
        truncated=ir.truncated,
        value_estimates=value_est,
        action_log_prob=ir.policy_decision.log_prob,
        action_entropy=ir.policy_decision.entropy,
        task_logits=task_binding.extract_task_logits(ir),
        labels=task_binding.extract_labels(ir),
        bootstrap_value=torch.zeros_like(reward),
        returns=returns,
        advantages=advantages,
        steps=steps,
        halted=halted,
    )


# =================================================================================================
class ZeroBootstrapActorCriticValidationScorer:
    """Zero-bootstrap validation scorer for actor-critic models.

    Adapts a :class:`~ehc_sn.rollouts.StepRecord` (whose ``outputs`` field
    must be an :class:`~ehc_sn.controllers.contracts.actor_critic.ActorCriticInteractionRecord`)
    into a :class:`~ehc_sn.objectives.hybrid_rl.HybridActorCriticBatch` with zero
    bootstrap values via :func:`_zero_bootstrap_batch`, then calls
    :meth:`HybridRLLossHead.compute_step`.

    Bootstrap value is zeroed; returns equal the immediate reward only.
    This is the same canonical construction path used by
    :meth:`TD0ActorCriticBatchBuilder.build_deliberation_ac_batch`.

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

        Delegates to :func:`_zero_bootstrap_batch` — the same canonical
        construction path used by
        :meth:`TD0ActorCriticBatchBuilder.build_deliberation_ac_batch`.

        Returns:
            :class:`~ehc_sn.objectives.hybrid_rl.HybridRLLossStep` with
            zero-bootstrap RL diagnostics.

        Raises:
            TypeError: If ``record.outputs`` is not an ``ActorCriticInteractionRecord``.
            RuntimeError: If ``record.outputs.task_output`` is ``None`` or
                required ``policy_decision`` fields are ``None``.
        """
        ir: ActorCriticInteractionRecord = record.outputs
        if not isinstance(ir, ActorCriticInteractionRecord):
            raise TypeError(
                f"ZeroBootstrapActorCriticValidationScorer expects ActorCriticInteractionRecord, "
                f"got {type(ir).__name__}."
            )
        if ir.task_output is None:
            raise RuntimeError(
                "ZeroBootstrapActorCriticValidationScorer requires a non-None "
                "task_output on ActorCriticInteractionRecord."
            )
        if ir.policy_decision.log_prob is None:
            raise RuntimeError("ZeroBootstrapActorCriticValidationScorer: policy_decision.log_prob is None.")
        if ir.policy_decision.entropy is None:
            raise RuntimeError("ZeroBootstrapActorCriticValidationScorer: policy_decision.entropy is None.")

        steps = record.snapshot.steps
        if steps is None:
            steps = torch.zeros(ir.reward.shape[0], dtype=torch.long, device=ir.reward.device)

        ac_batch = _zero_bootstrap_batch(ir, steps, record.snapshot.halted, self._task_binding)
        is_warmup = bool(options.get("is_warmup", False))
        return self._objective.compute_step(ac_batch, is_warmup=is_warmup)


# =================================================================================================
__all__ = [
    "HybridActorCriticTaskBinding",
    "OnlineBootstrapCarry",
    "OnlineBootstrapRuntime",
    "TD0ActorCriticBatchBuilder",
    "ZeroBootstrapActorCriticValidationScorer",
]
