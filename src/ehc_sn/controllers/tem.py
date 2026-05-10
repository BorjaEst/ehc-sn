"""Canonical TEM controller-facing contracts.

This module defines the rollout contract consumed by TEM loss heads. The
controller-facing output is intentionally semantic: loss heads read named TEM
properties rather than tuple positions or legacy model-internal structures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import torch
from pydantic import BaseModel, Field
from tensordict import TensorDict, TensorDictBase
from torch import Tensor

from ehc_sn.controllers._base import BaseController, RolloutBackbone, RolloutState
from ehc_sn.envs.dungeon_walk import ACTION_STAY, DungeonWalk
from ehc_sn.loss.consistency import LatentCode, LatentRelation
from ehc_sn.policies import ActionPolicy, PolicyInput, ScriptedPolicyConfig
from ehc_sn.policies.random_walk import RandomWalkPolicy, RandomWalkPolicyConfig
from ehc_sn.policies.stay import StayPolicy, StayPolicyConfig
from ehc_sn.types import Batch
from ehc_sn.utils.detach import DetachMixin

if TYPE_CHECKING:
    from ehc_sn.tasks.arena.capabilities.replay import ArenaReplayCapability

GRID_TRANSITION_RELATION: str = "grid_transition"
PLACE_TRANSITION_RELATION: str = "place_transition"
PLACE_SENSORY_RELATION: str = "place_sensory"
GRID_REG_TERM: str = "grid"
PLACE_REG_TERM: str = "place"


# ==================================================================================================
class TEMControllerConfig(BaseModel, extra="forbid"):
    """Configuration for :class:`TEMController`."""

    policy: ScriptedPolicyConfig = Field(
        default_factory=StayPolicyConfig,
        description="Configuration for the scripted TEM walk policy.",
    )
    max_steps: int = Field(
        default=1,
        ge=1,
        description="Number of rollout steps to execute before halting a slot.",
    )


# =================================================================================================
class TEMRolloutBackbone[ModelState, ModelOutput](RolloutBackbone[ModelState, ModelOutput], Protocol):
    """Backbone protocol expected by :class:`TEMController`."""


# =================================================================================================
@dataclass
class TEMRolloutState[ModelState](RolloutState[ModelState]):
    """Controller carry/state for TEM rollouts.

    ``data`` stores the payload aligned with the most recent TEM forward pass:
    current observation, previous action, and derived per-step metadata.
    Before the first step, it stores the initial current-step payload.

    ``env_td`` stores the mutable environment state that seeds the next TEM
    iteration (dungeon mode only). ``None`` in arena replay mode.

    ``resident`` and ``cursor`` are used in arena replay mode. ``resident``
    holds the per-slot admitted trajectory arrays; ``cursor`` is the per-slot
    step index ``(B,)`` int64.
    """

    env_td: TensorDictBase | None
    static_data: dict[str, Tensor]
    visit_counts: Tensor
    resident: dict[str, Tensor] | None = None
    cursor: Tensor | None = None


# =================================================================================================
@dataclass(frozen=True)
class TEMOutput(DetachMixin):
    """Canonical controller-facing TEM step output.

    Attributes:
        obs_logits:
            Three observation-logit tensors ordered as inference, retrieved,
            and ancestral pathways.
        latent_relations:
            Named TEM latent comparison terms. Expected keys are
            ``grid_transition``, ``place_transition``, and optional
            ``place_sensory``.
        reg_terms:
            Optional named regularization targets. Expected keys are ``grid``
            and ``place`` when present.
        theta_cls:
            Optional summary features for auxiliary diagnostics.

    Notes:
        Heads should consume the named relation map rather than legacy
        tuple-position semantics.
    """

    obs_logits: tuple[Tensor, Tensor, Tensor]

    @property
    def logits_inference(self) -> Tensor:
        """Return observation logits from the inference pathway."""
        return self.obs_logits[0]

    @property
    def logits_retrieved(self) -> Tensor:
        """Return observation logits from the retrieved pathway."""
        return self.obs_logits[1]

    @property
    def logits_ancestral(self) -> Tensor:
        """Return observation logits from the ancestral pathway."""
        return self.obs_logits[2]

    latent_relations: dict[str, LatentRelation]

    @property
    def grid_post(self) -> Tensor:
        """Return post-transition grid code."""
        return self.latent_relations[GRID_TRANSITION_RELATION].lhs

    @property
    def grid_prior(self) -> Tensor:
        """Return pre-transition grid code."""
        return self.latent_relations[GRID_TRANSITION_RELATION].rhs

    @property
    def place_post(self) -> Tensor:
        """Return post-transition place code."""
        return self.latent_relations[PLACE_TRANSITION_RELATION].lhs

    @property
    def place_prior(self) -> Tensor:
        """Return pre-transition place code."""
        return self.latent_relations[PLACE_TRANSITION_RELATION].rhs

    @property
    def place_sensory(self) -> Tensor | None:
        """Return sensory place code if present, else None."""
        relation = self.latent_relations.get(PLACE_SENSORY_RELATION)
        return relation.lhs if relation is not None else None

    reg_terms: dict[str, LatentCode] | None = None

    @property
    def reg_grid(self) -> LatentCode | None:
        """Return grid regularization target if present, else None."""
        return self.reg_terms.get(GRID_REG_TERM) if self.reg_terms is not None else None

    @property
    def reg_place(self) -> LatentCode | None:
        """Return place regularization target if present, else None."""
        return self.reg_terms.get(PLACE_REG_TERM) if self.reg_terms is not None else None

    theta_cls: Tensor | None = None


# =================================================================================================
class TEMController[ModelState](BaseController[ModelState, TEMControllerConfig]):
    """TEM rollout controller supporting both DungeonWalk and Arena replay modes."""

    def __init__(  # ------------------------------------------------------------------------------
        self,
        backbone: TEMRolloutBackbone[ModelState],
        env: DungeonWalk | None,
        config: TEMControllerConfig,
        *,
        replay: ArenaReplayCapability | None = None,
        observation_dim: int | None = None,
    ) -> None:  # fmt: skip
        """Create a controller.

        Args:
            backbone: Model implementing :class:`TEMRolloutBackbone`.
            env: TorchRL environment for DungeonWalk mode. ``None`` in Arena replay mode.
            config: Controller configuration.
            replay: Arena replay capability. When provided, the controller operates in
                replay mode and ``env`` must be ``None``.
            observation_dim: Observation vocabulary size for one-hot conversion in
                replay mode. Required when ``replay`` is not ``None``.
        """
        super().__init__(backbone=backbone, config=config)
        self._env = env
        self._replay = replay
        self._observation_dim = observation_dim
        if replay is None:
            if isinstance(config.policy, StayPolicyConfig):
                self._policy: ActionPolicy | None = StayPolicy(action=ACTION_STAY)
            elif isinstance(config.policy, RandomWalkPolicyConfig):
                self._policy = RandomWalkPolicy(seed=config.policy.seed)
            else:
                raise TypeError(f"Unsupported TEM policy config: {type(config.policy).__name__}.")
        else:
            self._policy = None

    @property
    def environment(self) -> DungeonWalk:
        """Return the TorchRL environment used for stepping (DungeonWalk mode only)."""
        if self._env is None:
            raise AttributeError("TEMController is in arena replay mode; no DungeonWalk environment is available.")
        return self._env

    def set_evaluation_seed(self, seed: int | None) -> None:
        """Seed controller-owned stochastic evaluation surfaces.

        In arena replay mode this is a no-op: replay trajectories are
        deterministic from the dataset and require no env/policy seeding.
        """
        if self._replay is not None:
            return
        if seed is None:
            raise ValueError("TEM evaluation requires an explicit seed for reproducible sampling.")
        self.environment._set_seed(int(seed))
        set_seed = getattr(self._policy, "set_seed", None)
        if callable(set_seed):
            set_seed(int(seed))

    def initial_state(  # -------------------------------------------------------------------------
        self, batch_sample: Batch
    ) -> TEMRolloutState[ModelState]:  # fmt: skip
        """Build an initial rollout state from a batch.

        Dispatches to arena replay or DungeonWalk mode based on the replay
        capability configured at construction time.
        """
        if self._replay is not None:
            return self._initial_state_arena(batch_sample)
        return self._initial_state_dungeon(batch_sample)

    def _initial_state_dungeon(  # ----------------------------------------------------------------
        self, batch_sample: Batch
    ) -> TEMRolloutState[ModelState]:  # fmt: skip
        """Build an initial rollout state from a static maze batch."""
        reset_td = self._build_reset_td(batch_sample)
        env_td = self.environment.reset(reset_td)
        batch_size = int(reset_td.batch_size[0])
        return TEMRolloutState(
            model_state=self.backbone.init_state(batch_size, device=env_td.device),
            steps=self._zeros(batch_size, dtype="int32", device=env_td.device),
            halted=self._zeros(batch_size, dtype="bool", device=env_td.device),
            data=self._extract_step_data(env_td),
            env_td=env_td,
            static_data={key: value.clone() for key, value in reset_td.items()},
            visit_counts=self._new_visit_counts(reset_td, device=env_td.device),
        )

    def _initial_state_arena(  # ------------------------------------------------------------------
        self, batch_sample: Batch
    ) -> TEMRolloutState[ModelState]:  # fmt: skip
        """Build an initial rollout state from an Arena replay batch."""
        assert self._replay is not None
        device = batch_sample["trajectory_row"].device
        batch_size = int(batch_sample["trajectory_row"].shape[0])
        cursor = torch.zeros((batch_size,), dtype=torch.int64, device=device)
        resident = {key: batch_sample[key].clone() for key in batch_sample if key.startswith("trajectory_")}
        step_data, _ = self._replay.extract_step_per_slot(resident, cursor, {})
        step_data = self._inject_observation(step_data, device)
        step_data.update(self._trace_metadata())
        return TEMRolloutState(
            model_state=self.backbone.init_state(batch_size, device=device),
            steps=self._zeros(batch_size, dtype="int32", device=device),
            halted=self._zeros(batch_size, dtype="bool", device=device),
            data=step_data,
            env_td=None,
            static_data={},
            visit_counts=torch.zeros((batch_size, 1), dtype=torch.int32, device=device),
            resident=resident,
            cursor=cursor,
        )

    def step(  # ----------------------------------------------------------------------------------
        self, state: TEMRolloutState[ModelState], batch: Batch, *,
        allow_halt: bool = True, explore: bool = True, **_: Any,
    ) -> tuple[TEMRolloutState[ModelState], TEMOutput]:  # fmt: skip
        """Advance the controller by one variational step.

        Dispatches to arena replay or DungeonWalk mode based on the replay
        capability configured at construction time.
        """
        if self._replay is not None:
            return self._step_arena(state, batch, allow_halt=allow_halt)
        return self._step_dungeon(state, batch, allow_halt=allow_halt, explore=explore)

    def _step_dungeon(  # -------------------------------------------------------------------------
        self, state: TEMRolloutState[ModelState], batch: Batch, *,
        allow_halt: bool = True, explore: bool = True,
    ) -> tuple[TEMRolloutState[ModelState], TEMOutput]:  # fmt: skip
        """Advance the controller via DungeonWalk environment stepping.

        The returned carry keeps ``data`` aligned with the payload used for the
        forward pass so losses and traces supervise the current step. The
        environment state is still advanced and stored in ``env_td`` to seed the
        next controller iteration.
        """
        static_data, env_td, visit_counts = self._refresh_halted_slots(batch, state)
        current_data = self._annotate_revisit_state(self._extract_step_data(env_td), env_td, visit_counts)
        model_state = self.backbone.reset_state(state.halted, state.model_state)
        model_state, obs_logits, _, grid, place = self.backbone(current_data, model_state)

        latent_relations = self._coerce_latent(grid, place)
        reg_terms = self._coerce_regularization(grid, place)
        action = self._policy_action(env_td, explore=explore)

        next_env_td = env_td.clone()
        next_env_td["action"] = action
        next_env_td = self.environment.step(next_env_td)["next"]

        steps = self.advance_steps(state)
        done = next_env_td["done"].squeeze(-1)
        if allow_halt:
            done = done | (steps >= self.config.max_steps)

        state = TEMRolloutState(
            model_state=model_state,
            steps=steps,
            halted=done,
            data=current_data,
            env_td=next_env_td,
            static_data=static_data,
            visit_counts=self._record_visit(visit_counts, env_td["location_id"]),
        )  # fmt: skip
        output = TEMOutput(obs_logits=obs_logits, latent_relations=latent_relations, reg_terms=reg_terms)

        return state, output

    def _step_arena(  # ---------------------------------------------------------------------------
        self, state: TEMRolloutState[ModelState], batch: Batch, *,
        allow_halt: bool = True,
    ) -> tuple[TEMRolloutState[ModelState], TEMOutput]:  # fmt: skip
        """Advance the controller by one step from Arena replay trajectories.

        Admits a new trajectory row for each halted slot, advances the per-slot
        cursor, extracts the current-step payload, and halts when the cursor
        reaches ``trajectory_length``.
        """
        assert self._replay is not None
        assert state.resident is not None
        assert state.cursor is not None

        resident, cursor = self._refresh_halted_slots_arena(batch, state)
        step_data, _ = self._replay.extract_step_per_slot(resident, cursor, {})
        step_data = self._inject_observation(step_data, resident["trajectory_row"].device)
        step_data.update(self._trace_metadata())

        model_state = self.backbone.reset_state(state.halted, state.model_state)
        model_state, obs_logits, _, grid, place = self.backbone(step_data, model_state)

        latent_relations = self._coerce_latent(grid, place)
        reg_terms = self._coerce_regularization(grid, place)

        steps = self.advance_steps(state)
        next_cursor = cursor + 1
        trajectory_length = resident["trajectory_length"].reshape(-1)
        halted = next_cursor >= trajectory_length if allow_halt else torch.zeros_like(next_cursor, dtype=torch.bool)

        new_state = TEMRolloutState(
            model_state=model_state,
            steps=steps,
            halted=halted,
            data=step_data,
            env_td=None,
            static_data={},
            visit_counts=state.visit_counts,
            resident=resident,
            cursor=next_cursor,
        )
        output = TEMOutput(obs_logits=obs_logits, latent_relations=latent_relations, reg_terms=reg_terms)
        return new_state, output

    def _refresh_halted_slots_arena(  # -----------------------------------------------------------
        self, batch: Batch, state: TEMRolloutState[ModelState],
    ) -> tuple[dict[str, Tensor], Tensor]:  # fmt: skip
        """Admit new trajectory rows for halted slots; keep active slots unchanged."""
        assert state.resident is not None
        assert state.cursor is not None

        if not torch.any(state.halted):
            return state.resident, state.cursor

        new_rows = {key: batch[key].clone() for key in batch if key.startswith("trajectory_")}
        merged: dict[str, Tensor] = {}
        for key, old_val in state.resident.items():
            if key in new_rows:
                halted_view = state.halted.view((-1,) + (1,) * (old_val.ndim - 1))
                merged[key] = torch.where(halted_view, new_rows[key], old_val)
            else:
                merged[key] = old_val

        new_cursor = state.cursor.clone()
        new_cursor[state.halted] = 0
        return merged, new_cursor

    def _inject_observation(  # -------------------------------------------------------------------
        self, step_data: dict[str, Tensor], device: Any,
    ) -> dict[str, Tensor]:  # fmt: skip
        """Convert ``observation_id`` to a one-hot ``observation`` tensor.

        The TEM model expects ``inputs["observation"]`` as a float tensor of
        shape ``(B, observation_dim)``. Arena replay provides ``observation_id``
        as an integer index; this method converts it locally without touching
        the dataset or model contracts.
        """
        if "observation_id" not in step_data or self._observation_dim is None:
            return step_data
        obs_id = step_data["observation_id"].reshape(-1).to(dtype=torch.int64, device=device)
        obs = torch.zeros(obs_id.shape[0], self._observation_dim, dtype=torch.float32, device=device)
        obs.scatter_(1, obs_id.unsqueeze(-1), 1.0)
        out = {**step_data, "observation": obs}
        if "landmark_id" in out:
            lm = out["landmark_id"]
            out["landmark_id"] = lm.clamp(min=0)
        return out

    def _build_reset_td(  # ----------------------------------------------------------------------
        self, batch: Batch,
    ) -> TensorDict:  # fmt: skip
        """Return the static maze tensors required by ``EnvBase.reset``."""
        required = ("topology", "observations", "mask_valid")
        optional = ("regions", "start", "goals", "landmarks")
        missing = [key for key in required if key not in batch]
        if missing:
            raise KeyError(f"TEMController reset batch is missing required maze keys: {', '.join(missing)}.")

        reset_data = {key: batch[key] for key in required}
        for key in optional:
            if key in batch:
                reset_data[key] = batch[key]

        batch_size = int(next(iter(reset_data.values())).shape[0])
        device = next(iter(reset_data.values())).device
        return TensorDict(reset_data, batch_size=[batch_size], device=device)

    def _extract_step_data(  # -------------------------------------------------------------------
        self, env_td: TensorDictBase,
    ) -> dict[str, Tensor]:  # fmt: skip
        """Extract the current-step model payload from an environment state.

        The model-facing payload exposes the current observation and the action
        that produced it via ``previous_action``. It also adds explicit
        episode-start metadata derived from the environment step counter.
        """
        keys = (
            "observation", "observation_id", "previous_action", "location_id", "region_id",
            "landmark_id", "valid_action_mask", "step_count",
        )  # fmt: skip
        payload = {key: env_td[key] for key in keys if key in env_td.keys()}
        if "step_count" in payload:
            payload["episode_start"] = payload["step_count"].squeeze(-1).to(torch.int32) == 0
        payload.update(self._trace_metadata())
        return payload

    def _trace_metadata(  # ----------------------------------------------------------------------
        self,
    ) -> dict[str, Tensor]:  # fmt: skip
        """Return detached static TEM metadata needed by trace observers."""
        lec = getattr(self.backbone, "lec", None)
        filter_module = None if lec is None else getattr(lec, "filter", None)
        alpha = None if filter_module is None else getattr(filter_module, "alpha", None)
        w_f = None if lec is None else getattr(lec, "w_f", None)
        if alpha is None or w_f is None:
            return {}
        return {
            "lec_alpha_sigmoid": torch.stack([torch.sigmoid(alpha_item).detach() for alpha_item in alpha]),
            "lec_w_f_sigmoid": torch.stack([torch.sigmoid(weight).detach() for weight in w_f]),
        }

    def _annotate_revisit_state(
        self,
        payload: dict[str, Tensor],
        env_td: TensorDictBase,
        visit_counts: Tensor,
    ) -> dict[str, Tensor]:
        """Attach revisit eligibility aligned with the current-step payload."""
        location_id = env_td["location_id"].to(device=visit_counts.device, dtype=torch.int64)
        prior_counts = visit_counts.gather(dim=1, index=location_id).squeeze(-1)
        payload["is_revisit"] = prior_counts > 0
        return payload

    def _refresh_halted_slots(  # ---------------------------------------------------------------
        self, batch: Batch, state: TEMRolloutState[ModelState],
    ) -> tuple[dict[str, Tensor], TensorDictBase, Tensor]:  # fmt: skip
        """Reset halted slots from the incoming maze batch and keep active slots unchanged."""
        if not torch.any(state.halted):
            return state.static_data, state.env_td, state.visit_counts

        new_static = self._build_reset_td(batch)
        static_data = {
            key: torch.where(state.halted.view((-1,) + (1,) * (value.ndim - 1)), value, state.static_data[key])
            for key, value in new_static.items()
        }  # fmt: skip
        reset_td = TensorDict(static_data, batch_size=state.env_td.batch_size, device=state.env_td.device)
        env_td = self.environment.reset_slots(state.halted, reset_td, state.env_td)
        visit_counts = state.visit_counts.clone()
        visit_counts[state.halted] = 0
        return static_data, env_td, visit_counts

    def _policy_action(  # -----------------------------------------------------------------------
        self, env_td: TensorDictBase, *, explore: bool,
    ) -> Tensor:  # fmt: skip
        """Return an action tensor sampled by the configured scripted policy."""
        policy_input = self._make_policy_input(env_td)
        action = self._policy(policy_input, explore=explore).action
        if action.ndim == 1:
            action = action.unsqueeze(-1)
        return action.to(device=env_td.device, dtype=torch.int64)

    def _make_policy_input(self, env_td: TensorDictBase) -> PolicyInput:
        """Adapt environment state to the reusable policy input contract."""
        return PolicyInput(
            valid_action_mask=env_td["valid_action_mask"],
            location_id=env_td["location_id"] if "location_id" in env_td.keys() else None,
            region_id=env_td["region_id"] if "region_id" in env_td.keys() else None,
            step_count=env_td["step_count"] if "step_count" in env_td.keys() else None,
        )

    @staticmethod
    def _zeros(batch_size: int, *, dtype: str, device: Any) -> Tensor:
        """Allocate a per-slot zero tensor with the requested dtype."""
        if dtype == "int32":
            return torch.zeros((batch_size,), dtype=torch.int32, device=device)
        if dtype == "bool":
            return torch.zeros((batch_size,), dtype=torch.bool, device=device)
        raise ValueError(f"Unsupported zero dtype request: {dtype}.")

    @staticmethod
    def _new_visit_counts(reset_td: TensorDictBase, *, device: Any) -> Tensor:
        """Allocate per-slot location visit counters for the current maze shape."""
        topology = reset_td["topology"]
        batch_size = int(topology.shape[0])
        n_locations = int(topology.shape[-2] * topology.shape[-1])
        return torch.zeros((batch_size, n_locations), dtype=torch.int32, device=device)

    @staticmethod
    def _record_visit(visit_counts: Tensor, location_id: Tensor) -> Tensor:
        """Increment visit counters for the current-step locations."""
        updated = visit_counts.clone()
        index = location_id.to(device=updated.device, dtype=torch.int64)
        increments = torch.ones_like(index, dtype=updated.dtype, device=updated.device)
        updated.scatter_add_(1, index, increments)
        return updated

    def _coerce_latent(  # --------------------------------------------------------------
        self, grid: tuple[Tensor, Tensor], place: tuple[Tensor, Tensor, Tensor],
    ) -> dict[str, LatentRelation]:  # fmt: skip
        """ """
        grid_post, grid_prior = grid
        place_post, place_prior, place_sensory = place
        latent_relations = {
            GRID_TRANSITION_RELATION: LatentRelation(lhs=grid_post, rhs=grid_prior),
            PLACE_TRANSITION_RELATION: LatentRelation(lhs=place_post, rhs=place_prior),
        }
        if place_sensory is not None:
            latent_relations[PLACE_SENSORY_RELATION] = LatentRelation(lhs=place_post, rhs=place_sensory)

        return latent_relations

    def _coerce_regularization(  # --------------------------------------------------------------
        self, grid: tuple[Tensor, Tensor], place: tuple[Tensor, Tensor, Tensor],
    ) -> dict[str, LatentCode] | None:  # fmt: skip
        """ """
        reg_terms: dict[str, LatentCode] | None = None
        # TODO: add logic

        return reg_terms

    def refresh_slot_data(  # ---------------------------------------------------------------------
        self, batch: Batch, state: TEMRolloutState[ModelState]
    ) -> dict[str, Tensor]:  # fmt: skip
        """Return the current-step payload cached in the controller carry.

        This payload is aligned with the outputs from the most recent forward
        pass, not with the already-stepped environment state stored in
        ``state.env_td``.
        """
        return state.data


# =================================================================================================
__all__ = [
    "GRID_REG_TERM", "GRID_TRANSITION_RELATION", "PLACE_REG_TERM", "PLACE_SENSORY_RELATION",
    "PLACE_TRANSITION_RELATION",
    "TEMController", "TEMControllerConfig", "TEMOutput", "TEMRolloutBackbone",
    "TEMRolloutState",
]  # fmt: skip
