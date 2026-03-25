from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Annotated, Literal, Optional, TypeAlias

import torch
from pydantic import BaseModel, Field
from torch import Tensor, nn

from ehc_sn import utils
from ehc_sn.types import Activation, Device, Dtype, RetrievalRole

MissingCueBehavior: TypeAlias = Literal["error", "use_available", "zeros"]


class RoleQueryPolicySettings(BaseModel, extra="forbid"):
    """Role-aware compatibility policy matching the legacy TEM query flow."""

    kind: Literal["by_role"] = Field(
        default="by_role",
        description="Use x for inference recall and g for generative recall, with optional fallback.",
    )
    missing_behavior: MissingCueBehavior = Field(
        default="use_available",
        description="Fallback behavior when the role-preferred cue is unavailable.",
    )


class XOnlyQueryPolicySettings(BaseModel, extra="forbid"):
    """Always use the sensory cue as the retrieval query."""

    kind: Literal["x_only"] = Field(
        default="x_only",
        description="Use only the observation-derived cue as the retrieval query.",
    )
    missing_behavior: MissingCueBehavior = Field(
        default="use_available",
        description="Fallback behavior when the sensory cue is unavailable.",
    )


class GOnlyQueryPolicySettings(BaseModel, extra="forbid"):
    """Always use the grid cue as the retrieval query."""

    kind: Literal["g_only"] = Field(
        default="g_only",
        description="Use only the grid-derived cue as the retrieval query.",
    )
    missing_behavior: MissingCueBehavior = Field(
        default="use_available",
        description="Fallback behavior when the grid cue is unavailable.",
    )


class GatedMixQueryPolicySettings(BaseModel, extra="forbid"):
    """Learn a per-frequency mixture between x and g cues."""

    kind: Literal["gated_mix"] = Field(
        default="gated_mix",
        description="Learn a per-frequency gate between sensory and grid cues.",
    )
    missing_behavior: MissingCueBehavior = Field(
        default="use_available",
        description="Fallback behavior when one cue is unavailable.",
    )


class ConjunctiveQueryPolicySettings(BaseModel, extra="forbid"):
    """Build a learned conjunctive query from sensory and grid cues."""

    kind: Literal["conjunctive"] = Field(
        default="conjunctive",
        description="Combine sensory and grid cues through a learned conjunctive projection.",
    )
    missing_behavior: MissingCueBehavior = Field(
        default="use_available",
        description="Fallback behavior when one cue is unavailable.",
    )
    activation: Activation = Field(
        default="leaky_relu",
        description="Activation applied after the conjunctive projection.",
    )
    clamp_min: float = Field(
        default=-1.0,
        description="Minimum clamp value applied before conjunctive activation.",
    )
    clamp_max: float = Field(
        default=1.0,
        description="Maximum clamp value applied before conjunctive activation.",
    )


QueryPolicySettings: TypeAlias = Annotated[
    RoleQueryPolicySettings
    | XOnlyQueryPolicySettings
    | GOnlyQueryPolicySettings
    | GatedMixQueryPolicySettings
    | ConjunctiveQueryPolicySettings,
    Field(discriminator="kind"),
]


class QueryPolicy(nn.Module, ABC):
    """Resolve a retrieval query from the available sensory and grid cues."""

    def __init__(  # ------------------------------------------------------------------------------
        self,
        shape: list[int],
        config: QueryPolicySettings,
        *,
        device: Optional[Device] = None,
        dtype: Optional[Dtype] = None,
    ) -> None:  # fmt: skip
        del device, dtype
        super().__init__()
        self._shape = list(shape)
        self._config = config

    @property
    def shape(self) -> list[int]:
        """Return the multi-frequency query widths expected by this policy."""
        return self._shape

    @property
    def config(self) -> QueryPolicySettings:
        """Return query-policy settings."""
        return self._config

    def forward(  # -------------------------------------------------------------------------------
        self,
        *,
        x_query: Optional[list[Tensor]],
        g_query: Optional[list[Tensor]],
        role: RetrievalRole,
    ) -> list[Tensor]:  # fmt: skip
        """Return a resolved multi-scale query for memory retrieval."""
        self._validate_query(x_query, name="x_query")
        self._validate_query(g_query, name="g_query")
        return self._forward(x_query=x_query, g_query=g_query, role=role)

    @abstractmethod
    def _forward(  # ------------------------------------------------------------------------------
        self,
        *,
        x_query: Optional[list[Tensor]],
        g_query: Optional[list[Tensor]],
        role: RetrievalRole,
    ) -> list[Tensor]:  # fmt: skip
        """Implement policy-specific query resolution."""

    def _validate_query(  # -----------------------------------------------------------------------
        self, query: Optional[list[Tensor]], *, name: str,
    ) -> None:  # fmt: skip
        """Validate a multi-scale query when one is provided."""
        if query is None:
            return

        if len(query) != len(self.shape):
            raise ValueError(f"{name} must contain {len(self.shape)} frequency tensors, got {len(query)}.")

        batch_size: int | None = None
        for index, (tensor, width) in enumerate(zip(query, self.shape, strict=True)):
            if tensor.ndim != 2:
                raise ValueError(f"{name}[{index}] must be rank-2 `(B, {width})`, got shape {tuple(tensor.shape)}.")
            if int(tensor.shape[1]) != width:
                raise ValueError(f"{name}[{index}] must have width {width}, got {int(tensor.shape[1])}.")
            if batch_size is None:
                batch_size = int(tensor.shape[0])
            elif int(tensor.shape[0]) != batch_size:
                raise ValueError(f"All tensors in {name} must share the same batch size.")

    def _fallback(  # -----------------------------------------------------------------------------
        self,
        *,
        preferred: Optional[list[Tensor]],
        other: Optional[list[Tensor]],
        missing_behavior: MissingCueBehavior,
        preferred_name: str,
    ) -> list[Tensor]:  # fmt: skip
        """Return the configured fallback query when the preferred cue is missing."""
        if preferred is not None:
            return preferred
        if missing_behavior == "use_available" and other is not None:
            return other
        if missing_behavior == "zeros" and other is not None:
            return [torch.zeros_like(tensor) for tensor in other]
        raise ValueError(f"{preferred_name} is required for query policy '{self.config.kind}'.")


class RoleQueryPolicy(QueryPolicy):
    """Compatibility policy that preserves the legacy TEM role-based query selection."""

    @property
    def config(self) -> RoleQueryPolicySettings:
        return super().config  # type: ignore[return-value]

    def _forward(  # ------------------------------------------------------------------------------
        self,
        *,
        x_query: Optional[list[Tensor]],
        g_query: Optional[list[Tensor]],
        role: RetrievalRole,
    ) -> list[Tensor]:  # fmt: skip
        if role == "inference":
            return self._fallback(
                preferred=x_query,
                other=g_query,
                missing_behavior=self.config.missing_behavior,
                preferred_name="x_query",
            )
        return self._fallback(
            preferred=g_query,
            other=x_query,
            missing_behavior=self.config.missing_behavior,
            preferred_name="g_query",
        )


class XOnlyQueryPolicy(QueryPolicy):
    """Policy that always prefers the sensory cue."""

    @property
    def config(self) -> XOnlyQueryPolicySettings:
        return super().config  # type: ignore[return-value]

    def _forward(  # ------------------------------------------------------------------------------
        self,
        *,
        x_query: Optional[list[Tensor]],
        g_query: Optional[list[Tensor]],
        role: RetrievalRole,
    ) -> list[Tensor]:  # fmt: skip
        del role
        return self._fallback(
            preferred=x_query,
            other=g_query,
            missing_behavior=self.config.missing_behavior,
            preferred_name="x_query",
        )


class GOnlyQueryPolicy(QueryPolicy):
    """Policy that always prefers the grid cue."""

    @property
    def config(self) -> GOnlyQueryPolicySettings:
        return super().config  # type: ignore[return-value]

    def _forward(  # ------------------------------------------------------------------------------
        self,
        *,
        x_query: Optional[list[Tensor]],
        g_query: Optional[list[Tensor]],
        role: RetrievalRole,
    ) -> list[Tensor]:  # fmt: skip
        del role
        return self._fallback(
            preferred=g_query,
            other=x_query,
            missing_behavior=self.config.missing_behavior,
            preferred_name="g_query",
        )


class GatedMixQueryPolicy(QueryPolicy):
    """Learn a per-frequency convex interpolation between x and g cues."""

    def __init__(  # ------------------------------------------------------------------------------
        self,
        shape: list[int],
        config: GatedMixQueryPolicySettings,
        *,
        device: Optional[Device] = None,
        dtype: Optional[Dtype] = None,
    ) -> None:  # fmt: skip
        super().__init__(shape, config, device=device, dtype=dtype)
        self._gates = nn.ModuleList([nn.Linear(2 * width, width, device=device, dtype=dtype) for width in shape])
        self._reset_parameters()

    @property
    def config(self) -> GatedMixQueryPolicySettings:
        return super().config  # type: ignore[return-value]

    def _forward(  # ------------------------------------------------------------------------------
        self,
        *,
        x_query: Optional[list[Tensor]],
        g_query: Optional[list[Tensor]],
        role: RetrievalRole,
    ) -> list[Tensor]:  # fmt: skip
        del role
        if x_query is None or g_query is None:
            return self._fallback(
                preferred=x_query,
                other=g_query,
                missing_behavior=self.config.missing_behavior,
                preferred_name="x_query and g_query",
            )

        mixed: list[Tensor] = []
        for gate_layer, x_tensor, g_tensor in zip(self._gates, x_query, g_query, strict=True):
            gate = torch.sigmoid(gate_layer(torch.cat((x_tensor, g_tensor), dim=1)))
            mixed.append(gate * x_tensor + (1.0 - gate) * g_tensor)
        return mixed

    def _reset_parameters(  # ---------------------------------------------------------------------
        self,
    ) -> None:  # fmt: skip
        """Initialize gates to an even mixture before training."""
        for layer in self._gates:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)


class ConjunctiveQueryPolicy(QueryPolicy):
    """Build a learned conjunctive query from sensory and grid cues."""

    def __init__(  # ------------------------------------------------------------------------------
        self,
        shape: list[int],
        config: ConjunctiveQueryPolicySettings,
        *,
        device: Optional[Device] = None,
        dtype: Optional[Dtype] = None,
    ) -> None:  # fmt: skip
        super().__init__(shape, config, device=device, dtype=dtype)
        self._activation_fn = utils.activation_from_str(config.activation)
        self._projections = nn.ModuleList([nn.Linear(3 * width, width, device=device, dtype=dtype) for width in shape])
        self._reset_parameters()

    @property
    def config(self) -> ConjunctiveQueryPolicySettings:
        return super().config  # type: ignore[return-value]

    def _forward(  # ------------------------------------------------------------------------------
        self,
        *,
        x_query: Optional[list[Tensor]],
        g_query: Optional[list[Tensor]],
        role: RetrievalRole,
    ) -> list[Tensor]:  # fmt: skip
        del role
        if x_query is None or g_query is None:
            return self._fallback(
                preferred=x_query,
                other=g_query,
                missing_behavior=self.config.missing_behavior,
                preferred_name="x_query and g_query",
            )

        conjunctive: list[Tensor] = []
        for projection, x_tensor, g_tensor in zip(self._projections, x_query, g_query, strict=True):
            joined = torch.cat((x_tensor, g_tensor, x_tensor * g_tensor), dim=1)
            tensor = projection(joined)
            tensor = torch.clamp(tensor, min=self.config.clamp_min, max=self.config.clamp_max)
            conjunctive.append(self._activation_fn(tensor))
        return conjunctive

    def _reset_parameters(  # ---------------------------------------------------------------------
        self,
    ) -> None:  # fmt: skip
        """Initialize projections to start from the classic x*g conjunction."""
        for width, projection in zip(self.shape, self._projections, strict=True):
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)
            identity = torch.eye(width, dtype=projection.weight.dtype, device=projection.weight.device)
            projection.weight.data[:, 2 * width : 3 * width] = identity


def build_query_policy(  # ------------------------------------------------------------------------
    shape: list[int],
    config: QueryPolicySettings,
    *,
    device: Optional[Device] = None,
    dtype: Optional[Dtype] = None,
) -> QueryPolicy:  # fmt: skip
    """Construct the configured query policy."""
    if config.kind == "by_role":
        return RoleQueryPolicy(shape, config, device=device, dtype=dtype)
    if config.kind == "x_only":
        return XOnlyQueryPolicy(shape, config, device=device, dtype=dtype)
    if config.kind == "g_only":
        return GOnlyQueryPolicy(shape, config, device=device, dtype=dtype)
    if config.kind == "gated_mix":
        return GatedMixQueryPolicy(shape, config, device=device, dtype=dtype)
    if config.kind == "conjunctive":
        return ConjunctiveQueryPolicy(shape, config, device=device, dtype=dtype)
    raise ValueError(f"Unsupported query policy '{config.kind}'.")


__all__ = [
    "ConjunctiveQueryPolicySettings",
    "GOnlyQueryPolicySettings",
    "GatedMixQueryPolicySettings",
    "MissingCueBehavior",
    "QueryPolicy",
    "QueryPolicySettings",
    "RoleQueryPolicySettings",
    "XOnlyQueryPolicySettings",
    "build_query_policy",
]
