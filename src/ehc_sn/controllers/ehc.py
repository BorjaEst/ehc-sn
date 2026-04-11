"""Thin EHC rollout controller with controller-owned commit semantics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, cast

import torch
from pydantic import BaseModel, Field
from tensordict import TensorDict, TensorDictBase
from torch import Tensor

from ehc_sn.controllers._base import BaseController, RolloutState
from ehc_sn.envs.dungeon_walk import DungeonWalk
from ehc_sn.policies import CategoricalPolicy, CategoricalPolicyConfig, PolicyInput
from ehc_sn.types import Batch
from ehc_sn.utils.detach import DetachMixin


# ==================================================================================================
class EHCControllerConfig(BaseModel, extra="forbid"):
    """Configuration for :class:`EHCController`."""

    internal_policy: CategoricalPolicyConfig = Field(
        default_factory=CategoricalPolicyConfig,
        description="Sampling configuration for internal control actions.",
    )
    motor_policy: CategoricalPolicyConfig = Field(
        default_factory=CategoricalPolicyConfig,
        description="Sampling configuration for environment motor actions.",
    )
    max_internal_steps: int = Field(
        default=1,
        ge=1,
        description="Maximum number of internal cycles before a forced commit.",
    )
    max_steps: int = Field(
        default=32,
        ge=1,
        description="Maximum number of committed environment steps before halting a slot.",
    )
    commit_action: int = Field(
        default=0,
        ge=0,
        description="Internal action index that commits the current controller step.",
    )


# =================================================================================================
class EHCControlOutput(Protocol):
    """Structural protocol for the control payload emitted by the EHC backbone."""

    theta_summary: Tensor
    internal_control_logits: Tensor
    motor_logits: Tensor
    reward_logits: Tensor


# =================================================================================================
class EHCModelOutput[ModelState](Protocol):
    """Structural protocol for the model output consumed by :class:`EHCController`."""

    state: ModelState
    obs_logits: tuple[Tensor, Tensor, Tensor]
    control: EHCControlOutput


# =================================================================================================
class EHCRolloutBackbone[ModelState](Protocol):
    """Backbone protocol required by :class:`EHCController`."""

    def init_state(self, batch_size: int, *, device: Any | None = None) -> ModelState:
        """Return a fresh recurrent state for ``batch_size`` slots."""

    def reset_state(self, reset_flag: Tensor, state: ModelState) -> ModelState:
        """Reset flagged rows of the recurrent state."""

    def __call__(self, batch: Batch, state: ModelState | None = None) -> EHCModelOutput[ModelState]:
        """Run one EHC forward step from ``batch`` and ``state``."""


# =================================================================================================
@dataclass
class EHCRolloutState[ModelState](RolloutState[ModelState]):
    """Controller carry/state for EHC rollouts.

    ``data`` stays aligned with the payload scored by the final internal cycle
    of the current controller step. ``env_td`` is already advanced to the next
    environment state and seeds the next controller iteration.
    ``conditioning_data`` stores static optional cortical inputs such as
    serialized ``input_ids`` or exogenous ``external_context`` that do not come
    from the environment step payload itself.
    """

    env_td: TensorDictBase
    static_data: dict[str, Tensor]
    conditioning_data: dict[str, Tensor]


# =================================================================================================
@dataclass(frozen=True)
class EHCControllerOutput[ModelState](DetachMixin):
    """Outputs produced by one EHC controller step."""

    model_output: EHCModelOutput[ModelState]
    internal_action: Tensor
    commit_mask: Tensor
    motor_action: Tensor
    reward: Tensor
    cycle_count: int
    forced_commit: bool

    @property
    def obs_logits(self) -> tuple[Tensor, Tensor, Tensor]:
        """Return the observation logits produced by the final internal cycle."""
        return self.model_output.obs_logits

    @property
    def theta_summary(self) -> Tensor:
        """Return the final control-summary vector emitted by the backbone."""
        return self.model_output.control.theta_summary


# =================================================================================================
class EHCController[ModelState](BaseController[ModelState, EHCControllerConfig]):
    """Thin batch-synchronized EHC controller with commit-only environment stepping."""

    def __init__(  # ------------------------------------------------------------------------------
        self, backbone: EHCRolloutBackbone[ModelState], env: DungeonWalk, config: EHCControllerConfig,
    ) -> None:  # fmt: skip
        super().__init__(backbone=cast(Any, backbone), config=config)
        self._env = env
        self._internal_policy = CategoricalPolicy(config.internal_policy)
        self._motor_policy = CategoricalPolicy(config.motor_policy)
        self._validate_commit_action(backbone)

    @property
    def backbone(self) -> EHCRolloutBackbone[ModelState]:
        """Return the wrapped EHC backbone typed to the local controller protocol."""
        return cast(EHCRolloutBackbone[ModelState], super().backbone)

    @property
    def environment(self) -> DungeonWalk:
        """Return the environment stepped by the controller."""
        return self._env

    def initial_state(  # -------------------------------------------------------------------------
        self, batch_sample: Batch,
    ) -> EHCRolloutState[ModelState]:  # fmt: skip
        """Build an initial rollout state from a static maze batch."""
        reset_td = self._build_reset_td(batch_sample)
        env_td = self.environment.reset(reset_td)
        batch_size = int(reset_td.batch_size[0])
        return EHCRolloutState(
            model_state=self.backbone.init_state(batch_size, device=env_td.device),
            steps=self._zeros(batch_size, dtype="int32", device=env_td.device),
            halted=self._zeros(batch_size, dtype="bool", device=env_td.device),
            data=self._extract_step_data(env_td, conditioning_data=self._extract_conditioning_data(batch_sample)),
            env_td=env_td,
            static_data={key: value.clone() for key, value in reset_td.items()},
            conditioning_data=self._extract_conditioning_data(batch_sample),
        )

    def step(  # ----------------------------------------------------------------------------------
        self, state: EHCRolloutState[ModelState], batch: Batch, *,
        allow_halt: bool = True, explore: bool = True, **_: Any,
    ) -> tuple[EHCRolloutState[ModelState], EHCControllerOutput[ModelState]]:  # fmt: skip
        """Advance one controller step with repeated internal cycles and one env transition."""
        static_data, conditioning_data, env_td = self._refresh_halted_slots(batch, state)
        current_data = self._extract_step_data(env_td, conditioning_data=conditioning_data)
        model_state = self.backbone.reset_state(state.halted, state.model_state)

        model_output, internal_action, commit_mask, cycle_count, forced_commit = self._run_internal_cycles(
            current_data,
            model_state,
            explore=explore,
        )
        motor_action = self._select_motor_action(env_td, model_output.control.motor_logits, explore=explore)

        next_env_td = env_td.clone()
        next_env_td["action"] = motor_action
        next_env_td = self.environment.step(next_env_td)["next"]

        steps = self.advance_steps(state)
        done = next_env_td["done"].squeeze(-1)
        if allow_halt:
            done = done | (steps >= self.config.max_steps)

        new_state = EHCRolloutState(
            model_state=model_output.state,
            steps=steps,
            halted=done,
            data=current_data,
            env_td=next_env_td,
            static_data=static_data,
            conditioning_data=conditioning_data,
        )
        output = EHCControllerOutput(
            model_output=model_output,
            internal_action=internal_action,
            commit_mask=commit_mask,
            motor_action=motor_action,
            reward=next_env_td["reward"],
            cycle_count=cycle_count,
            forced_commit=forced_commit,
        )
        return new_state, output

    def _validate_commit_action(self, backbone: EHCRolloutBackbone[ModelState]) -> None:
        """Validate the configured commit action when the backbone advertises its action count."""
        backbone_config = getattr(backbone, "config", None)
        internal_action_count = getattr(backbone_config, "internal_action_count", None)
        if internal_action_count is None:
            return
        if self.config.commit_action >= int(internal_action_count):
            raise ValueError(
                "commit_action must be smaller than the backbone internal action count. "
                f"Got commit_action={self.config.commit_action}, internal_action_count={internal_action_count}."
            )

    def _run_internal_cycles(  # -----------------------------------------------------------------
        self, current_data: Batch, model_state: ModelState, *, explore: bool,
    ) -> tuple[EHCModelOutput[ModelState], Tensor, Tensor, int, bool]:  # fmt: skip
        """Loop internal reasoning cycles until batch-wide commit or the configured budget."""
        model_output: EHCModelOutput[ModelState] | None = None
        internal_action: Tensor | None = None
        commit_mask: Tensor | None = None
        cycle_count = 0

        for cycle_count in range(1, self.config.max_internal_steps + 1):
            model_output = self.backbone(current_data, model_state)
            model_state = model_output.state
            internal_action = self._select_internal_action(model_output.control.internal_control_logits, explore=explore)
            commit_mask = internal_action == self.config.commit_action
            if bool(torch.all(commit_mask).item()):
                return model_output, internal_action, commit_mask, cycle_count, False

        if model_output is None or internal_action is None or commit_mask is None:
            raise RuntimeError("EHCController failed to execute an internal cycle.")
        return model_output, internal_action, commit_mask, cycle_count, True

    def _select_internal_action(self, logits: Tensor, *, explore: bool) -> Tensor:
        """Sample one internal action per row from the backbone internal-control logits."""
        decision = self._internal_policy(
            PolicyInput(valid_action_mask=torch.ones_like(logits, dtype=torch.bool), logits=logits),
            explore=explore,
        )
        return decision.action.to(device=logits.device, dtype=torch.int64).view(-1)

    def _select_motor_action(self, env_td: TensorDictBase, logits: Tensor, *, explore: bool) -> Tensor:
        """Sample one motor action per row from the environment-valid action subset."""
        decision = self._motor_policy(self._make_motor_policy_input(env_td, logits), explore=explore)
        action = decision.action.to(device=env_td.device, dtype=torch.int64)
        if action.ndim == 1:
            action = action.unsqueeze(-1)
        return action

    def _make_motor_policy_input(self, env_td: TensorDictBase, logits: Tensor) -> PolicyInput:
        """Adapt the live environment state to the reusable categorical-policy contract."""
        return PolicyInput(
            valid_action_mask=env_td["valid_action_mask"],
            location_id=env_td["location_id"] if "location_id" in env_td.keys() else None,
            region_id=env_td["region_id"] if "region_id" in env_td.keys() else None,
            step_count=env_td["step_count"] if "step_count" in env_td.keys() else None,
            logits=logits,
        )

    def _build_reset_td(  # ----------------------------------------------------------------------
        self, batch: Batch,
    ) -> TensorDict:  # fmt: skip
        """Return the static maze tensors required by ``DungeonWalk.reset``."""
        required = ("topology", "observations", "mask_valid")
        optional = ("regions", "start", "goals", "landmarks")
        missing = [key for key in required if key not in batch]
        if missing:
            raise KeyError(f"EHCController reset batch is missing required maze keys: {', '.join(missing)}.")

        reset_data = {key: batch[key] for key in required}
        for key in optional:
            if key in batch:
                reset_data[key] = batch[key]

        batch_size = int(next(iter(reset_data.values())).shape[0])
        device = next(iter(reset_data.values())).device
        return TensorDict(reset_data, batch_size=[batch_size], device=device)

    def _extract_step_data(  # -------------------------------------------------------------------
        self, env_td: TensorDictBase, *, conditioning_data: dict[str, Tensor] | None = None,
    ) -> dict[str, Tensor]:  # fmt: skip
        """Extract the current-step EHC payload from a dungeon environment state."""
        keys = (
            "observation", "observation_id", "previous_action", "location_id", "region_id",
            "landmark_id", "valid_action_mask", "step_count",
        )  # fmt: skip
        payload = {key: env_td[key] for key in keys if key in env_td.keys()}
        if "step_count" in payload:
            payload["episode_start"] = payload["step_count"].squeeze(-1).to(torch.int32) == 0
        if conditioning_data is not None:
            payload.update(conditioning_data)
        return payload

    def _extract_conditioning_data(self, batch: Batch) -> dict[str, Tensor]:
        """Return static optional conditioning tensors carried outside the environment."""
        keys = ("input_ids", "external_context")
        return {key: batch[key].clone() for key in keys if key in batch}

    @staticmethod
    def _merge_static_rows(
        current: dict[str, Tensor],
        incoming: dict[str, Tensor],
        reset_mask: Tensor,
    ) -> dict[str, Tensor]:
        """Merge static batch rows for halted slots while preserving active rows."""
        if not incoming:
            return current
        merged: dict[str, Tensor] = {}
        for key in current.keys() | incoming.keys():
            if key not in current:
                merged[key] = incoming[key].clone()
                continue
            if key not in incoming:
                merged[key] = current[key]
                continue
            value = incoming[key]
            merged[key] = torch.where(reset_mask.view((-1,) + (1,) * (value.ndim - 1)), value, current[key])
        return merged

    def _refresh_halted_slots(  # ---------------------------------------------------------------
        self, batch: Batch, state: EHCRolloutState[ModelState],
    ) -> tuple[dict[str, Tensor], dict[str, Tensor], TensorDictBase]:  # fmt: skip
        """Reset halted slots from the incoming maze batch and keep active slots unchanged."""
        if not torch.any(state.halted):
            return state.static_data, state.conditioning_data, state.env_td

        new_reset_td = self._build_reset_td(batch)
        new_static = {key: new_reset_td[key] for key in new_reset_td.keys()}
        static_data = self._merge_static_rows(state.static_data, new_static, state.halted)
        conditioning_data = self._merge_static_rows(
            state.conditioning_data,
            self._extract_conditioning_data(batch),
            state.halted,
        )
        reset_td = TensorDict(static_data, batch_size=state.env_td.batch_size, device=state.env_td.device)
        env_td = self.environment.reset_slots(state.halted, reset_td, state.env_td)
        return static_data, conditioning_data, env_td

    @staticmethod
    def _zeros(batch_size: int, *, dtype: str, device: Any) -> Tensor:
        """Allocate a per-slot zero tensor with the requested dtype."""
        if dtype == "int32":
            return torch.zeros((batch_size,), dtype=torch.int32, device=device)
        if dtype == "bool":
            return torch.zeros((batch_size,), dtype=torch.bool, device=device)
        raise ValueError(f"Unsupported zero dtype request: {dtype}.")

    def refresh_slot_data(  # ---------------------------------------------------------------------
        self, batch: Batch, state: EHCRolloutState[ModelState],
    ) -> dict[str, Tensor]:  # fmt: skip
        """Return the current-step payload cached in the controller carry."""
        _ = batch
        return state.data


# =================================================================================================
__all__ = [
    "EHCController",
    "EHCControllerConfig",
    "EHCControllerOutput",
    "EHCControlOutput",
    "EHCModelOutput",
    "EHCRolloutBackbone",
    "EHCRolloutState",
]
