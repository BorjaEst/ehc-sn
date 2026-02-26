""" """

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol, Tuple

import torch
from pydantic import BaseModel, Field
from torch import Tensor

Batch = Dict[str, Tensor]  # Generic batch type, can be specialized as needed


# ==================================================================================================
class ACTControllerConfig(BaseModel, extra="forbid"):
    """Configuration for :class:`ACTController`.

    Attributes:
        exploration_prob: Probability of enabling exploration for a slot on a
            given step. Exploration delays halting by enforcing a random minimum
            number of steps before halting is allowed.
        halt_max_steps: Hard cap on the number of ACT steps per slot.
    """

    exploration_prob: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Exploration probability for ACT halting.",
    )
    halt_max_steps: int = Field(
        ...,
        ge=1,
        description="Maximum number of ACT steps.",
    )


# =================================================================================================
# NOTE: ACTBackbone and HaltingHead protocols are STR's public contract and
# will move to modules/str/ (see spec-architecture.md §9 "STR protocol
# migration"). After the move, act_controller.py will import them from there.
# =================================================================================================
class ACTBackbone(Protocol):
    """Minimal interface required by the ACT controller backbone.

    The controller treats the backbone as a recurrent function:

    - Input: ``inputs`` for the current slot step and an optional recurrent
        ``state``.
    - Output: next recurrent state, main prediction logits, and differentiable
        features used by the halting head.
    """

    def init_state(  # ----------------------------------------------------------------------------
        self, batch_size: int
    ) -> Any:  # fmt: skip
        ...  # fmt: skip

    def reset_state(  # ---------------------------------------------------------------------------
        self, reset_flag: Tensor, state: Any
    ) -> Any:   # fmt: skip
        ...  # fmt: skip

    def __call__(  # ------------------------------------------------------------------------------
        self, inputs: Tensor, state: Any | None = None
    ) -> Tuple[Any, Tensor, Tensor]:  # fmt: skip
        ...  # fmt: skip


# =================================================================================================
class HaltingHead(Protocol):
    """Minimal interface for converting features into halting logits."""

    def __call__(  # ------------------------------------------------------------------------------
        self, features: Tensor
    ) -> Tuple[Tensor, Tensor]:  # fmt: skip
        ...  # fmt: skip


# =================================================================================================
@dataclass
class ACTState:
    """Per-slot controller state.

    Notes:
        ``halted`` acts as a per-slot "needs reset" flag: when True, that slot
        will be refreshed with a new sample from the incoming batch and its
        recurrent state will be reset before the next model step.
    """

    model_state: Any  # Recurrent state of the model (e.g. LSTM hidden states)
    steps: Tensor  # Per-slot step counter, shape: (B,)
    halted: Tensor  # Per-slot reset/done flag, shape: (B,)
    data: Dict[str, Tensor]  # Per-slot buffers that persist across steps until reset


# =================================================================================================
@dataclass
class ACTOutput:
    """Outputs produced by a single controller step.

    Notes:
        The ``target_continue`` is only populated during training when TD(0)
        targets are computed.
    """

    logits: Tensor  # Main task logits (consumed by the supervised loss/metrics)
    halt_logits: Tensor  # Per-slot Q-logits for halting, shape: (B,)
    continue_logits: Tensor  # Per-slot Q-logits for continuing, shape: (B,)
    action: Tensor  # Selected greedy action: 0=halt, 1=continue
    target_continue: Tensor | None = None  # TD(0) bootstrap target for continue head

    def detach(self) -> ACTOutput:
        """Return a new ACTOutput with all tensors detached from the computation graph."""
        return ACTOutput(
            logits=self.logits.detach(),
            halt_logits=self.halt_logits.detach(),
            continue_logits=self.continue_logits.detach(),
            action=self.action.detach(),
            target_continue=self.target_continue.detach() if self.target_continue is not None else None,
        )


# =================================================================================================
class ACTController:
    """Controller that runs multiple ACT steps and manages per-slot resets.

    The controller is responsible for:

    - Resetting recurrent state for slots that finished an episode.
    - Swapping in fresh batch elements for finished slots (without changing
      batch size).
    - Selecting halt/continue actions from halting-head Q-logits.
    - Optionally computing TD(0) targets for the continue head during training.

    Important:
        The controller does not read ``nn.Module.training``. Callers must pass
        explicit flags to select deterministic vs exploratory behavior and
        whether TD targets are produced.
    """

    HALT_ACTION = 0
    CONTINUE_ACTION = 1

    def __init__(  # ------------------------------------------------------------------------------
        self, backbone: ACTBackbone, halt_head: HaltingHead, config: ACTControllerConfig
    ) -> None:  # fmt: skip
        """Initialize the ACT controller.

        Args:
            backbone: The recurrent backbone to control, which must implement the
                :class:`ACTBackbone` protocol.
            halt_head: Module that converts backbone features into halting logits.
            config: Configuration for the controller behavior.
        """
        self._backbone = backbone
        self._halt_head = halt_head
        self._config = config

    @property
    def backbone(self) -> ACTBackbone:
        return self._backbone

    @property
    def halt_head(self) -> HaltingHead:
        return self._halt_head

    @property
    def config(self) -> ACTControllerConfig:
        return self._config

    def initial_state(  # -------------------------------------------------------------------------
        self, batch_sample: Batch
    ) -> ACTState:  # fmt: skip
        """Create an initial :class:`ACTState` for a new loop.

        The initial ``halted=True`` for all slots forces the first call to
        :meth:`refresh_slot_data` to populate per-slot buffers from ``batch``
        (since buffers start empty).
        """
        batch_size, device = batch_sample["inputs"].shape[0], batch_sample["inputs"].device
        return ACTState(  # FIXME: We need to replace batch_dict by observations and labels
            model_state=self.backbone.init_state(batch_size),
            steps=torch.zeros((batch_size,), dtype=torch.int32, device=device),
            halted=torch.ones((batch_size,), dtype=torch.bool, device=device),
            data={k: torch.empty_like(v) for k, v in batch_sample.items()},
        )

    def step(  # ----------------------------------------------------------------------------------
        self, state: ACTState, batch: Batch,
        allow_halt: bool = True, explore: bool = True,
    ) -> Tuple[ACTState, ACTOutput]:  # fmt: skip
        """Run one ACT step, this performs, in order:

        1) Refreshes per-slot data for slots marked done.
        2) Resets recurrent state for those slots.
        3) Runs a single model step.
        4) Updates step counters and selects halt/continue actions.
        5) Emits outputs for loss and metrics.

        Args:
            state: Current controller state.
            batch: Incoming batch of new samples (same shapes as buffers).
            allow_halt: Whether a greedy halt action can end the episode early.
            explore: Whether to apply the minimum halting step constraint to a
                random subset of slots (per-step exploration rule).
        """
        data = self.refresh_slot_data(batch, state)
        model_state = self.backbone.reset_state(state.halted, state.model_state)
        model_state, logits, features = self.backbone(data["inputs"], model_state)
        q_halt, q_continue = self.halt_head(features)

        # Reset the step counter when a slot starts a fresh episode.
        steps = torch.where(state.halted, 0, state.steps) + 1
        action, done = self._select_action_and_done(q_halt, q_continue, steps, allow_halt, explore)

        state = ACTState(model_state=model_state, steps=steps, halted=done, data=data)
        output = ACTOutput(logits=logits, halt_logits=q_halt, continue_logits=q_continue, action=action)

        # TD(0) bootstrap target for the continue head.
        with torch.no_grad():
            _, _, next_features = self.backbone(data["inputs"], model_state)
            next_q_halt, next_q_continue = self.halt_head(next_features)
        is_last_step = steps >= self._config.halt_max_steps
        next_q = torch.where(is_last_step, next_q_halt, torch.maximum(next_q_halt, next_q_continue))
        output.target_continue = torch.sigmoid(next_q)  # Sigmoid to convert logits to probabilities

        return state, output

    def refresh_slot_data(  # ---------------------------------------------------------------------
        self, batch: Batch, state: ACTState
    ) -> Dict[str, Tensor]:  # fmt: skip
        """Replace finished slots with new batch data (episode reset).

        Args:
            batch: New batch data.
            state: Current ACT state with done/halted flags.

        Returns:
            A dict of tensors with the same keys/shapes as ``batch`` where
            slots marked ``halted=True`` are replaced by the new incoming
            ``batch`` values and the rest keep their previous buffered values.
        """
        data, halted = state.data, state.halted
        return {
            k: torch.where(halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], data[k]) for k in batch
        }

    def _select_action_and_done(  # ---------------------------------------------------------------
        self, q_halt: Tensor, q_continue: Tensor, steps: Tensor, allow_halt: bool, explore: bool,
    ) -> Tuple[Tensor, Tensor]:  # fmt: skip
        """Select halt/continue action and compute done mask.

        Action selection is greedy from the two Q-logit heads.

        Done conditions:
            - Always: ``steps >= halt_max_steps``.
            - If ``allow_halt``: selecting the halt action may also end the episode,
                optionally delayed by exploration.

        Exploration:
            When ``explore=True``, with probability ``exploration_prob`` per slot,
            enforce a random minimum number of steps (in ``[2, halt_max_steps]``)
            before halting is allowed. This biases episodes away from trivially
            halting at step 1 and encourages multi-step rollouts.
        """
        config = self.config  # convenience alias
        q_halt, q_continue = q_halt.detach(), q_continue.detach()  # No gradients flow
        action = torch.where(q_halt > q_continue, self.HALT_ACTION, self.CONTINUE_ACTION)
        done = steps >= config.halt_max_steps

        if allow_halt:
            done = done | (action == self.HALT_ACTION)

        if explore and (config.halt_max_steps > 1):
            exploration_flag = torch.rand_like(q_halt) < config.exploration_prob
            min_halt_steps = exploration_flag * torch.randint_like(
                steps, low=2, high=config.halt_max_steps + 1
            )
            done = done & (steps >= min_halt_steps)

        return action, done


__all__ = ["ACTBackbone", "ACTControllerConfig", "ACTController", "ACTState", "HaltingHead"]
