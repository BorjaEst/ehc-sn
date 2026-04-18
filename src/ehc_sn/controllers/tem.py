""" """

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import torch
from pydantic import BaseModel, Field
from tensordict import TensorDict, TensorDictBase
from torch import Tensor

from ehc_sn.controllers._base import BaseController, RolloutBackbone, RolloutState
from ehc_sn.envs.dungeon_walk import ACTION_STAY
from ehc_sn.loss.consistency import LatentCode, LatentRelation
from ehc_sn.policies import ActionPolicy, PolicyInput, ScriptedPolicyConfig
from ehc_sn.policies.random_walk import RandomWalkPolicy, RandomWalkPolicyConfig
from ehc_sn.policies.stay import StayPolicy, StayPolicyConfig
from ehc_sn.types import Batch
from ehc_sn.utils.detach import DetachMixin


# =================================================================================================
# Private structural protocols — keep the controller model-agnostic and adapter-agnostic.
# Concrete bridge outputs satisfy these structurally; no import of adapter types is needed.
# =================================================================================================
class _GridCodesProtocol(Protocol):
    """Minimal structural contract for TEM grid-code bundles."""

    post: LatentCode
    prior: LatentCode


class _PlaceCodesProtocol(Protocol):
    """Minimal structural contract for TEM place-code bundles."""

    inference: LatentCode
    ancestral: LatentCode
    retrieved: LatentCode | None
    sensory: LatentCode | None


class _TEMDiagnosticsProtocol(Protocol):
    """Minimal structural contract for TEM diagnostic bundles."""

    obs_logits: tuple[Tensor, Tensor, Tensor]
    grid_codes: _GridCodesProtocol
    place_codes: _PlaceCodesProtocol


class _TEMBridgeOutputProtocol(Protocol):
    """Minimal structural contract for TEM bridge outputs returned by the backbone."""

    tem: _TEMDiagnosticsProtocol


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
class TEMEnvironment(Protocol):
    """Minimal environment protocol required by :class:`TEMController` and :class:`TEMTaskRuntime`."""

    def reset(self, tensordict: TensorDictBase) -> TensorDictBase: ...

    def step(self, tensordict: TensorDictBase) -> TensorDictBase: ...

    def reset_slots(self, reset_mask: Tensor, tensordict: TensorDictBase, state: TensorDictBase) -> TensorDictBase: ...

    def _set_seed(self, seed: int | None) -> None: ...


# =================================================================================================
class TEMRolloutBackbone[ModelState](RolloutBackbone[ModelState, _TEMBridgeOutputProtocol], Protocol):
    """Backbone protocol expected by :class:`TEMController`.

    Narrows the generic ``RolloutBackbone`` contract to require that the forward
    call returns a :class:`_TEMBridgeOutputProtocol`-conformant bridge output, and
    that ``init_state`` accepts an optional ``device`` keyword argument.
    """

    def init_state(self, batch_size: int, *, device: Any = None) -> ModelState: ...  # type: ignore[override]

    def __call__(  # fmt: skip
        self,
        batch: Batch,
        state: ModelState | None = None,
    ) -> tuple[ModelState, _TEMBridgeOutputProtocol]: ...


# =================================================================================================
class TEMTaskRuntime(Protocol):
    """Task-owned rollout shaping injected into :class:`TEMController`."""

    def build_reset_td(self, batch: Batch) -> TensorDict: ...

    def extract_step_data(
        self,
        env_td: TensorDictBase,
        *,
        trace_metadata: Mapping[str, Tensor] | None = None,
    ) -> dict[str, Tensor]: ...

    def annotate_revisit_state(
        self,
        payload: dict[str, Tensor],
        env_td: TensorDictBase,
        visit_counts: Tensor,
    ) -> dict[str, Tensor]: ...

    def refresh_halted_slots(
        self,
        batch: Batch,
        halted: Tensor,
        static_data: dict[str, Tensor],
        env_td: TensorDictBase,
        visit_counts: Tensor,
        *,
        environment: TEMEnvironment,
    ) -> tuple[dict[str, Tensor], TensorDictBase, Tensor]: ...

    def new_visit_counts(self, reset_td: TensorDictBase, *, device: Any) -> Tensor: ...

    def record_visit(self, visit_counts: Tensor, location_id: Tensor) -> Tensor: ...

    def make_policy_input(self, env_td: TensorDictBase) -> PolicyInput: ...


# =================================================================================================
@dataclass
class TEMRolloutState[ModelState](RolloutState[ModelState]):
    """Controller carry/state for TEM rollouts.

    ``data`` stores the payload aligned with the most recent TEM forward pass:
    current observation, previous action, and derived per-step metadata.
    Before the first step, it stores the initial current-step payload.

    ``env_td`` stores the mutable environment state that seeds the next TEM
    iteration. After a controller step it has already advanced to the next
    current-state payload, but ``data`` remains aligned with the outputs that
    were just produced.
    """

    env_td: TensorDictBase
    static_data: dict[str, Tensor]
    visit_counts: Tensor


# =================================================================================================
@dataclass(frozen=True)
class TEMStepOutput(DetachMixin):
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
    def grid_post(self) -> LatentCode:
        """Return post-transition grid code."""
        return self.latent_relations[GRID_TRANSITION_RELATION].lhs

    @property
    def grid_prior(self) -> LatentCode:
        """Return pre-transition grid code."""
        return self.latent_relations[GRID_TRANSITION_RELATION].rhs

    @property
    def place_post(self) -> LatentCode:
        """Return post-transition place code."""
        return self.latent_relations[PLACE_TRANSITION_RELATION].lhs

    @property
    def place_prior(self) -> LatentCode:
        """Return pre-transition place code."""
        return self.latent_relations[PLACE_TRANSITION_RELATION].rhs

    @property
    def place_sensory(self) -> LatentCode | None:
        """Return sensory place code if present, else None."""
        relation = self.latent_relations.get(PLACE_SENSORY_RELATION)
        return relation.rhs if relation is not None else None

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
    """TEM rollout controller with controller-owned environment stepping."""

    def __init__(  # ------------------------------------------------------------------------------
        self,
        backbone: TEMRolloutBackbone[ModelState],
        env: TEMEnvironment,
        config: TEMControllerConfig,
        runtime: TEMTaskRuntime,
    ) -> None:  # fmt: skip
        """Create a controller.

        Args:
            backbone: Model implementing :class:`TEMRolloutBackbone`.
            env: TorchRL environment used to generate online walk steps.
            config: Controller configuration.
            runtime: Task-owned rollout shaping runtime.
        """
        super().__init__(backbone=backbone, config=config)
        self._env = env
        self._runtime = runtime
        if isinstance(config.policy, StayPolicyConfig):
            self._policy: ActionPolicy = StayPolicy(action=ACTION_STAY)
        elif isinstance(config.policy, RandomWalkPolicyConfig):
            self._policy = RandomWalkPolicy(seed=config.policy.seed)
        else:
            raise TypeError(f"Unsupported TEM policy config: {type(config.policy).__name__}.")

    @property
    def environment(self) -> TEMEnvironment:
        """Return the TorchRL environment used for stepping."""
        return self._env

    @property
    def runtime(self) -> TEMTaskRuntime:
        """Return the injected task-owned rollout runtime."""
        return self._runtime

    @property
    def backbone(self) -> TEMRolloutBackbone[ModelState]:  # type: ignore[override]
        """Return the TEM backbone narrowed to :class:`TEMRolloutBackbone`."""
        return self._backbone  # type: ignore[return-value]

    def set_evaluation_seed(self, seed: int | None) -> None:
        """Seed controller-owned stochastic evaluation surfaces.

        TEM evaluation is only reproducible when both the reset sampler and any
        stochastic scripted policy are explicitly seeded.
        """
        if seed is None:
            raise ValueError("TEM evaluation requires an explicit seed for reproducible sampling.")
        self.environment._set_seed(int(seed))
        set_seed = getattr(self._policy, "set_seed", None)
        if callable(set_seed):
            set_seed(int(seed))

    def initial_state(  # -------------------------------------------------------------------------
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

    def step(  # ----------------------------------------------------------------------------------
        self, state: TEMRolloutState[ModelState], batch: Batch, *,
        allow_halt: bool = True, explore: bool = True, **_: Any,
    ) -> tuple[TEMRolloutState[ModelState], TEMStepOutput]:  # fmt: skip
        """Advance the controller by one variational step.

        The returned carry keeps ``data`` aligned with the payload used for the
        forward pass so losses and traces supervise the current step. The
        environment state is still advanced and stored in ``env_td`` to seed the
        next controller iteration.
        """
        static_data, env_td, visit_counts = self._refresh_halted_slots(batch, state)
        current_data = self._annotate_revisit_state(self._extract_step_data(env_td), env_td, visit_counts)
        model_state = self.backbone.reset_state(state.halted, state.model_state)
        model_state, bridge_output = self.backbone(current_data, model_state)

        latent_relations = self._coerce_latent(bridge_output)
        reg_terms = self._coerce_regularization(bridge_output)
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
        output = TEMStepOutput(
            obs_logits=bridge_output.tem.obs_logits,
            latent_relations=latent_relations,
            reg_terms=reg_terms,
        )

        return state, output

    def _build_reset_td(  # ----------------------------------------------------------------------
        self, batch: Batch,
    ) -> TensorDict:  # fmt: skip
        """Return the static maze tensors required by ``EnvBase.reset``."""
        return self.runtime.build_reset_td(batch)

    def _extract_step_data(  # -------------------------------------------------------------------
        self, env_td: TensorDictBase,
    ) -> dict[str, Tensor]:  # fmt: skip
        """Extract the current-step model payload from an environment state.

        The model-facing payload exposes the current observation and the action
        that produced it via ``previous_action``. It also adds explicit
        episode-start metadata derived from the environment step counter.
        """
        return self.runtime.extract_step_data(env_td, trace_metadata=self._trace_metadata())

    def _trace_metadata(  # ----------------------------------------------------------------------
        self,
    ) -> dict[str, Tensor]:  # fmt: skip
        """Return detached static TEM metadata needed by trace observers."""
        lec = getattr(self.backbone, "lec", None)
        if lec is None:
            model = getattr(self.backbone, "model", None)
            lec = None if model is None else getattr(model, "lec", None)
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
        return self.runtime.annotate_revisit_state(payload, env_td, visit_counts)

    def _refresh_halted_slots(  # ---------------------------------------------------------------
        self, batch: Batch, state: TEMRolloutState[ModelState],
    ) -> tuple[dict[str, Tensor], TensorDictBase, Tensor]:  # fmt: skip
        """Reset halted slots from the incoming maze batch and keep active slots unchanged."""
        return self.runtime.refresh_halted_slots(
            batch,
            state.halted,
            state.static_data,
            state.env_td,
            state.visit_counts,
            environment=self.environment,
        )

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
        return self.runtime.make_policy_input(env_td)

    @staticmethod
    def _zeros(batch_size: int, *, dtype: str, device: Any) -> Tensor:
        """Allocate a per-slot zero tensor with the requested dtype."""
        if dtype == "int32":
            return torch.zeros((batch_size,), dtype=torch.int32, device=device)
        if dtype == "bool":
            return torch.zeros((batch_size,), dtype=torch.bool, device=device)
        raise ValueError(f"Unsupported zero dtype request: {dtype}.")

    def _new_visit_counts(self, reset_td: TensorDictBase, *, device: Any) -> Tensor:
        """Allocate per-slot location visit counters for the current maze shape."""
        return self.runtime.new_visit_counts(reset_td, device=device)

    def _record_visit(self, visit_counts: Tensor, location_id: Tensor) -> Tensor:
        """Increment visit counters for the current-step locations."""
        return self.runtime.record_visit(visit_counts, location_id)

    def _coerce_latent(  # --------------------------------------------------------------
        self, bridge_output: _TEMBridgeOutputProtocol,
    ) -> dict[str, LatentRelation]:  # fmt: skip
        """Assemble named TEM latent relations from the bridge output ``tem`` bundle."""
        tem = bridge_output.tem
        gc = tem.grid_codes
        pc = tem.place_codes
        latent_relations: dict[str, LatentRelation] = {
            GRID_TRANSITION_RELATION: LatentRelation(lhs=gc.post, rhs=gc.prior),
            PLACE_TRANSITION_RELATION: LatentRelation(lhs=pc.inference, rhs=pc.ancestral),
        }
        if pc.sensory is not None:
            latent_relations[PLACE_SENSORY_RELATION] = LatentRelation(lhs=pc.inference, rhs=pc.sensory)
        return latent_relations

    def _coerce_regularization(  # --------------------------------------------------------------
        self, bridge_output: _TEMBridgeOutputProtocol,
    ) -> dict[str, LatentCode] | None:  # fmt: skip
        """Assemble named TEM regularization targets from the bridge output."""
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
    "TEMController", "TEMControllerConfig", "TEMStepOutput", "TEMRolloutBackbone",
    "TEMRolloutState", "TEMTaskRuntime",
]  # fmt: skip
