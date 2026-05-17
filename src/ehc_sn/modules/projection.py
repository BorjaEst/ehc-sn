"""Typed projection endpoints and edge builders.

This module separates endpoint structure from projection behavior. A projection
endpoint may be flat, multiscale, token-sequence shaped, or a workspace-shaped
structured tensor bank.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import ItemsView, KeysView, Mapping, Sequence, ValuesView
from typing import Annotated, Literal, Optional, Protocol, TypeAlias

import torch
from pydantic import AliasChoices, BaseModel, Field, model_validator
from torch import Tensor
from torch import device as Device
from torch import dtype as Dtype
from torch import nn

from ehc_sn import utils
from ehc_sn.types import InitStrategy, ProjectionBridge, ProjectionKind


# =============================================================================
class ProjectionSettings(BaseModel, extra="forbid"):
    """Settings for one projection edge.

    ``kind`` selects the projector family. ``bridge`` is optional and defaults
    to ``auto``, allowing the projection owner to infer how source and target
    endpoint structures should be reconciled.
    """

    kind: ProjectionKind = Field(
        ...,
        validation_alias=AliasChoices("kind", "mode"),
        description="Projection family used after endpoint reconciliation.",
    )
    bridge: ProjectionBridge = Field(
        default="auto",
        description="Optional bridge override used to reconcile mismatched "
        "endpoint structures.",
    )
    init: InitStrategy = Field(
        default="identity",
        description="Initialization strategy: 'identity' for structured "
        "identity-like initialization or 'random'.",
    )
    learnable: bool = Field(
        default=False,
        description="If True, projection parameters remain trainable after "
        "construction and reset.",
    )
    rank: Optional[int | list[int]] = Field(
        default=None,
        description=(
            "Low-rank bottleneck size for low_rank projections. "
            "Provide one integer to reuse the same rank across bands, "
            "or a per-band rank list. Auto-derived via GCD when omitted."
        ),
    )

    @model_validator(mode="after")
    def validate_settings(self) -> "ProjectionSettings":
        if self.kind == "identity" and self.learnable:
            raise ValueError("identity projections must not be learnable.")
        if self.kind != "low_rank" and self.rank is not None:
            raise ValueError("rank is only valid for low_rank projections.")
        return self


# =============================================================================
class TEMComponent(Protocol):
    shape: Sequence[int]


# =============================================================================
class StructuredEndpointSpec(BaseModel, extra="forbid"):
    """Base class for non-flat endpoints with internal structure."""


# =============================================================================
class FlatEndpointSpec(BaseModel, extra="forbid"):
    """One flat feature-vector endpoint."""

    kind: Literal["flat"] = "flat"
    width: int = Field(..., ge=1)


class MultiScaleEndpointSpec(StructuredEndpointSpec):
    """One multiscale endpoint exposing one width per band."""

    kind: Literal["multiscale"] = "multiscale"
    shape: list[int] = Field(..., min_length=1)


class TokenSequenceEndpointSpec(StructuredEndpointSpec):
    """One homogeneous token-sequence endpoint with a fixed hidden width."""

    kind: Literal["token_sequence"] = "token_sequence"
    width: int = Field(..., ge=1)
    seq_len: Optional[int] = Field(default=None, ge=1)


class WorkspaceFamilySpec(BaseModel, extra="forbid"):
    """One named repeated family within a workspace-shaped endpoint."""

    name: str = Field(..., min_length=1)
    size: int = Field(..., ge=1)


# =============================================================================
class WorkspaceEndpointSpec(StructuredEndpointSpec):
    """One workspace-shaped endpoint with fixed slots and repeated families.

    Unlike :class:`TokenSequenceEndpointSpec`, a workspace endpoint preserves
    structural meaning for named fixed roles and exchangeable slot families.
    """

    kind: Literal["workspace"] = Field(default="workspace", const=True)
    width: int = Field(..., ge=1)
    fixed: tuple[str, ...] = Field(default_factory=tuple)
    families: tuple[WorkspaceFamilySpec, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def validate_workspace(self) -> "WorkspaceEndpointSpec":
        names = [*self.fixed, *(family.name for family in self.families)]
        if any(not name for name in self.fixed):
            raise ValueError("workspace fixed-slot names must be non-empty.")
        if len(set(names)) != len(names):
            raise ValueError(
                f"workspace endpoint names must be unique, got {names!r}.",
            )
        return self

    @property
    def size(self) -> int:
        """Total slot count: len(fixed) + sum(family sizes)."""
        return len(self.fixed) + sum(family.size for family in self.families)


# =============================================================================
ProjectionEndpointSpec: TypeAlias = Annotated[
    FlatEndpointSpec
    | MultiScaleEndpointSpec
    | TokenSequenceEndpointSpec
    | WorkspaceEndpointSpec,
    Field(discriminator="kind"),
]
ProjectionEndpointValue: TypeAlias = (
    TEMComponent | Sequence[int] | ProjectionEndpointSpec
)
ProjectionModuleEdge: TypeAlias = tuple[
    ProjectionEndpointValue, ProjectionEndpointValue, ProjectionSettings
]
ProjectionNamedEdge: TypeAlias = tuple[str, str, ProjectionSettings]


# =============================================================================
def flat_endpoint(  # ---------------------------------------------------------
    width: int,
) -> FlatEndpointSpec:
    """Return a typed flat endpoint descriptor."""
    return FlatEndpointSpec(width=int(width))


def multiscale_endpoint(  # ---------------------------------------------------
    component_or_shape: TEMComponent | Sequence[int],
) -> MultiScaleEndpointSpec:
    """Return a typed multiscale endpoint descriptor."""
    return MultiScaleEndpointSpec(shape=_coerce_shape(component_or_shape))


def token_sequence_endpoint(  # -----------------------------------------------
    width: int, *, seq_len: Optional[int] = None
) -> TokenSequenceEndpointSpec:
    """Return a typed token-sequence endpoint descriptor."""
    return TokenSequenceEndpointSpec(width=int(width), seq_len=seq_len)


def workspace_endpoint(  # ----------------------------------------------------
    width: int,
    *,
    fixed: Sequence[str] = (),
    families: Mapping[str, int] | Sequence[tuple[str, int]] = (),
) -> WorkspaceEndpointSpec:
    """Return a typed workspace endpoint descriptor."""
    if isinstance(families, Mapping):
        family_items = families.items()
    else:
        family_items = families
    return WorkspaceEndpointSpec(
        width=int(width),
        fixed=tuple(str(name) for name in fixed),
        families=tuple(
            WorkspaceFamilySpec(name=str(name), size=int(size))
            for name, size in family_items
        ),
    )


# =============================================================================
class ProjectionEdgeSpec(BaseModel, extra="forbid"):
    """One named projection edge between two typed endpoints."""

    source: str = Field(..., min_length=1)
    target: str = Field(..., min_length=1)
    settings: ProjectionSettings = Field(...)


# =============================================================================
class ProjectionModule(nn.Module):
    """Aligned projection between multiscale source and target codes."""

    def __init__(  # ----------------------------------------------------------
        self,
        z_from: TEMComponent | Sequence[int],
        z_to: TEMComponent | Sequence[int],
        settings: ProjectionSettings,
    ):
        """Initialize the projection module with specified shapes."""
        super().__init__()
        self._shape_from = _coerce_shape(z_from)
        self._shape_to = _coerce_shape(z_to)
        if len(self._shape_from) != len(self._shape_to):
            raise ValueError(
                "Aligned ProjectionModule requires the same number of bands on "
                "source and target."
            )

        self._settings = settings
        self._module = self.create_module(
            settings, self._shape_from, self._shape_to
        )
        self.set_learning()

    @property
    def shape_from(self) -> list[int]:
        """Input width per multiscale band."""
        return list(self._shape_from)

    @property
    def shape_to(self) -> list[int]:
        """Output width per multiscale band."""
        return list(self._shape_to)

    @staticmethod
    def create_module(  # -----------------------------------------------------
        settings: ProjectionSettings,
        shape_from: list[int],
        shape_to: list[int],
    ) -> "AbstractProjectionModule":
        """Create the projection module based on settings and shapes."""
        if settings.kind == "identity":
            module = IdentityModule(shape_from, shape_to)
        elif settings.kind == "linear":
            module = LinearModule(shape_from, shape_to, settings)
        elif settings.kind == "tiling":
            module = TileModule(shape_from, shape_to, settings)
        elif settings.kind == "low_rank":
            module = LowRankModule(shape_from, shape_to, settings)
        else:
            raise ValueError(f"Unknown projection kind: {settings.kind}")
        return module

    def set_learning(self) -> None:
        """Enable or disable learning of projection weights."""
        for param in self._module.parameters():
            param.requires_grad = self._settings.learnable

    def reset_parameters(self) -> None:
        """Reset projection parameters according to the configured init strategy."""
        self._module.reset_parameters()
        self.set_learning()

    def forward(self, z: Sequence[Tensor]) -> list[Tensor]:
        return self._module.forward(z)

    def inverse(self, p: Sequence[Tensor]) -> list[Tensor]:
        return self._module.inverse(p)


# =============================================================================
class PerBandLinear(nn.Module):
    """Apply one learned linear surface per aligned multiscale band."""

    def __init__(  # ----------------------------------------------------------
        self,
        shape_from: Sequence[int],
        shape_to: int | Sequence[int],
        *,
        bias: bool = True,
        device: Optional[Device] = None,
        dtype: Optional[Dtype] = None,
    ) -> None:
        """Initialize the per-band linear projection module."""
        super().__init__()
        self._shape_from = [int(width) for width in shape_from]
        if len(self._shape_from) < 1:
            raise ValueError("shape_from must contain at least one band.")

        if isinstance(shape_to, int):
            self._shape_to = [int(shape_to) for _ in self._shape_from]
        else:
            self._shape_to = [int(width) for width in shape_to]
            if len(self._shape_to) != len(self._shape_from):
                raise ValueError(
                    f"shape_to must have length {len(self._shape_from)}, "
                    f"got {len(self._shape_to)}."
                )

        self.layers = nn.ModuleList(
            [
                nn.Linear(
                    in_features,
                    out_features,
                    bias=bias,
                    device=device,
                    dtype=dtype,
                )
                for in_features, out_features in zip(
                    self._shape_from, self._shape_to, strict=True
                )
            ]
        )

    @property
    def shape_from(self) -> list[int]:
        """Input width per band."""
        return list(self._shape_from)

    @property
    def shape_to(self) -> list[int]:
        """Output width per band."""
        return list(self._shape_to)

    def reset_parameters(self, *, init_std: float) -> None:
        """Initialize the per-band linear surfaces with truncated normal weights."""
        for layer in self.layers:
            utils.trunc_normal_init_(layer.weight, std=init_std)
            if layer.bias is not None:
                layer.bias.data.zero_()

    def forward(  # -----------------------------------------------------------
        self,
        codes: Sequence[Tensor],
        *,
        dtype: Optional[torch.dtype] = None,
    ) -> list[Tensor]:
        """Project one tensor per band after validating aligned widths."""
        if len(codes) != len(self.layers):
            raise ValueError(
                f"codes must have length {len(self.layers)}, got {len(codes)}."
            )

        outputs: list[Tensor] = []
        for index, (code, layer, width) in enumerate(
            zip(codes, self.layers, self._shape_from, strict=True)
        ):
            if code.ndim != 2 or int(code.shape[1]) != width:
                raise ValueError(
                    f"codes[{index}] must have shape (B, {width}), got {tuple(code.shape)}."
                )
            if dtype is not None:
                code = code.to(dtype)
            outputs.append(layer(code))
        return outputs


# =============================================================================
class ProjectionBundle(nn.Module):
    """Named collection of projection edges stored in a ModuleDict."""

    def __init__(  # ----------------------------------------------------------
        self,
        *,
        endpoints: Mapping[str, ProjectionEndpointSpec],
        edges: Mapping[str, ProjectionEdgeSpec],
    ) -> None:
        super().__init__()
        self._endpoints = dict(endpoints)
        self._edge_specs = dict(edges)
        modules: dict[str, nn.Module] = {}
        for name, edge in self._edge_specs.items():
            if edge.source not in self._endpoints:
                raise KeyError(
                    f"Unknown projection source endpoint: {edge.source!r}."
                )
            if edge.target not in self._endpoints:
                raise KeyError(
                    f"Unknown projection target endpoint: {edge.target!r}."
                )
            modules[name] = build_projection_edge(
                source=self._endpoints[edge.source],
                target=self._endpoints[edge.target],
                settings=edge.settings,
            )
        self.edges = nn.ModuleDict(modules)

    @classmethod
    def from_modules(  # ------------------------------------------------------
        cls,
        edges: Mapping[str, ProjectionModuleEdge] | None = None,
        /,
        **more_edges: ProjectionModuleEdge,
    ) -> "ProjectionBundle":
        """Build a bundle from direct source-target endpoint pairs.

        This is the compact authoring API for small call sites. Each edge is
        specified as ``(source_endpoint, target_endpoint, settings)`` and is
        normalized into explicit endpoint and edge specs before construction.
        """
        resolved_edges = _merge_projection_edges(edges, more_edges)
        modules: dict[str, ProjectionEndpointValue] = {}
        named_edges: dict[str, ProjectionNamedEdge] = {}
        normalized_names: dict[int, str] = {}

        for edge_name, (source, target, settings) in resolved_edges.items():
            source_name = _intern_builder_endpoint_name(
                normalized_names,
                source,
                fallback_name=f"__{edge_name}_source",
            )
            target_name = _intern_builder_endpoint_name(
                normalized_names,
                target,
                fallback_name=f"__{edge_name}_target",
            )
            modules.setdefault(source_name, source)
            modules.setdefault(target_name, target)
            named_edges[edge_name] = (source_name, target_name, settings)

        return cls.from_named_modules(modules=modules, edges=named_edges)

    @classmethod
    def from_named_modules(  # ------------------------------------------------
        cls,
        *,
        modules: Mapping[str, ProjectionEndpointValue],
        edges: Mapping[str, ProjectionNamedEdge],
    ) -> "ProjectionBundle":
        """Build a bundle from reusable named endpoints and named edges.

        This is the scalable public API. Endpoint names stay stable, while edge
        tuples remain explicit about source, target, and settings.
        """
        endpoint_specs = {
            name: _coerce_endpoint_spec(value)
            for name, value in modules.items()
        }
        edge_specs = {
            name: ProjectionEdgeSpec(
                source=source, target=target, settings=settings
            )
            for name, (source, target, settings) in edges.items()
        }
        return cls(endpoints=endpoint_specs, edges=edge_specs)

    @property
    def endpoint_specs(self) -> dict[str, ProjectionEndpointSpec]:
        """Return the normalized endpoint specs used to construct the bundle."""
        return dict(self._endpoints)

    @property
    def edge_specs(self) -> dict[str, ProjectionEdgeSpec]:
        """Return the normalized edge specs used to construct the bundle."""
        return dict(self._edge_specs)

    def reset_parameters(self) -> None:
        """Reset all named projection edges."""
        for edge in self.edges.values():
            if hasattr(edge, "reset_parameters"):
                edge.reset_parameters()

    def __getitem__(self, name: str) -> nn.Module:
        return self.edges[name]

    def __contains__(self, name: object) -> bool:
        return name in self.edges

    def __iter__(self):
        return iter(self.edges)

    def keys(  # --------------------------------------------------------------
        self,
    ) -> KeysView[str]:
        """Return the registered edge names."""
        return self.edges.keys()

    def items(  # -------------------------------------------------------------
        self,
    ) -> ItemsView[str, nn.Module]:
        """Return the registered edge-name to module mapping."""
        return self.edges.items()

    def values(  # ------------------------------------------------------------
        self,
    ) -> ValuesView[nn.Module]:
        """Return the registered edge modules."""
        return self.edges.values()

    def __getattr__(  # -------------------------------------------------------
        self,
        name: str,
    ) -> object:
        """Allow direct attribute access to registered edges."""
        try:
            return super().__getattr__(name)
        except AttributeError as exc:
            modules = self.__dict__.get("_modules")
            edges = None if modules is None else modules.get("edges")
            if isinstance(edges, nn.ModuleDict) and name in edges:
                return edges[name]
            raise exc


# =============================================================================
class AbstractProjectionModule(nn.Module, ABC):
    """Abstract base class for projection modules between aligned multiscale bands."""

    @abstractmethod
    def reset_parameters(  # --------------------------------------------------
        self,
    ) -> None:
        """Reset projection parameters according to the configured init strategy."""

    @abstractmethod
    def forward(  # -----------------------------------------------------------
        self,
        z: Sequence[Tensor],
    ) -> list[Tensor]:
        """Project from source to target multiscale codes."""

    @abstractmethod
    def inverse(  # -----------------------------------------------------------
        self,
        p: Sequence[Tensor],
    ) -> list[Tensor]:
        """Project from target back to source multiscale codes."""


# =============================================================================
class IdentityModule(AbstractProjectionModule):
    """Identity map over aligned multiscale bands."""

    def __init__(  # ----------------------------------------------------------
        self,
        shape_from: list[int],
        shape_to: list[int],
    ) -> None:
        """Initialize the identity projection module with specified shapes."""
        super().__init__()
        if shape_from != shape_to:
            raise ValueError(
                "identity projections require matching source and target shapes."
            )
        self._shape = list(shape_from)

    def reset_parameters(self) -> None:
        """Identity projections have no parameters to reset."""

    def forward(  # -----------------------------------------------------------
        self,
        z: Sequence[Tensor],
    ) -> list[Tensor]:
        return [z_f for z_f in z]

    def inverse(  # -----------------------------------------------------------
        self,
        p: Sequence[Tensor],
    ) -> list[Tensor]:
        return [p_f for p_f in p]


# =============================================================================
class _BaseMatrixProjectionModule(AbstractProjectionModule):
    """Shared matrix-based projection logic for aligned multiscale bands."""

    def __init__(  # ----------------------------------------------------------
        self,
        matrices: list[Tensor],
        *,
        learnable: bool,
    ) -> None:
        """Initialize the matrix projection module with specified projection matrices."""
        super().__init__()
        self.w = nn.ParameterList(
            [
                nn.Parameter(matrix, requires_grad=learnable)
                for matrix in matrices
            ]
        )

    def forward(  # -----------------------------------------------------------
        self,
        z_from: Sequence[Tensor],
    ) -> list[Tensor]:
        return [torch.matmul(z_from[f], self.w[f]) for f in range(len(z_from))]

    def inverse(  # -----------------------------------------------------------
        self,
        z_to: Sequence[Tensor],
    ) -> list[Tensor]:
        return [torch.matmul(z_to[f], self.w[f].T) for f in range(len(z_to))]

    def _copy_matrices(  # ----------------------------------------------------
        self,
        matrices: list[Tensor],
    ) -> None:
        if len(matrices) != len(self.w):
            raise ValueError(
                f"Expected {len(self.w)} matrices, got {len(matrices)}."
            )
        for param, matrix in zip(self.w, matrices, strict=True):
            param.data.copy_(matrix.to(device=param.device, dtype=param.dtype))


# =============================================================================
class LinearModule(_BaseMatrixProjectionModule):
    """Dense learned projection with configurable identity-like or random initialization."""

    def __init__(  # ----------------------------------------------------------
        self,
        shape_from: list[int],
        shape_to: list[int],
        settings: ProjectionSettings,
    ) -> None:
        """Initialize the linear projection module with specified shapes and settings."""
        self._shape_from = list(shape_from)
        self._shape_to = list(shape_to)
        self._settings = settings
        super().__init__(
            self.create_matrices(shape_from, shape_to, settings),
            learnable=settings.learnable,
        )

    @staticmethod
    def create_matrices(  # ---------------------------------------------------
        shape_from: list[int],
        shape_to: list[int],
        settings: ProjectionSettings,
    ) -> list[Tensor]:
        """Create the projection matrices based on settings and shapes."""
        if settings.init == "identity":
            return [
                _create_linear_identity_matrix(n_in, n_out)
                for n_in, n_out in zip(shape_from, shape_to, strict=True)
            ]
        if settings.init == "random":
            return utils.create_random_projection(shape_from, shape_to)
        raise ValueError(f"Unknown init strategy: {settings.init}")

    def reset_parameters(self) -> None:
        """Reset projection parameters according to the configured init strategy."""
        self._copy_matrices(
            self.create_matrices(
                self._shape_from, self._shape_to, self._settings
            )
        )


# =============================================================================
class TileModule(_BaseMatrixProjectionModule):
    """Structured tiling/repetition projection over aligned multiscale bands."""

    def __init__(  # ----------------------------------------------------------
        self,
        shape_from: list[int],
        shape_to: list[int],
        settings: ProjectionSettings,
    ) -> None:
        """Initialize the tiling projection module with specified shapes and settings."""
        self._shape_from = list(shape_from)
        self._shape_to = list(shape_to)
        self._settings = settings
        super().__init__(
            self.create_matrices(shape_from, shape_to, settings),
            learnable=settings.learnable,
        )

    @staticmethod
    def create_matrices(  # ---------------------------------------------------
        shape_from: list[int],
        shape_to: list[int],
        settings: ProjectionSettings,
    ) -> list[Tensor]:
        """Create the projection matrices based on settings and shapes."""
        if settings.init == "identity":
            return utils.create_tiling_matrices(shape_from, shape_to)
        if settings.init == "random":
            return utils.create_random_projection(shape_from, shape_to)
        raise ValueError(f"Unknown init strategy: {settings.init}")

    def reset_parameters(self) -> None:
        self._copy_matrices(
            self.create_matrices(
                self._shape_from, self._shape_to, self._settings
            )
        )


# =============================================================================
class LowRankModule(AbstractProjectionModule):
    """Low-rank factorized projection over aligned multiscale bands."""

    def __init__(  # ----------------------------------------------------------
        self,
        shape_from: list[int],
        shape_to: list[int],
        settings: ProjectionSettings,
    ) -> None:
        """Initialize the low-rank projection module with specified shapes and settings."""
        super().__init__()
        self._shape_from = list(shape_from)
        self._shape_to = list(shape_to)
        self._settings = settings
        w_down, w_repeat = self.create_matrices(shape_from, shape_to, settings)
        self.w_down = nn.ParameterList(
            [nn.Parameter(w, requires_grad=settings.learnable) for w in w_down]
        )
        self.w_repeat = nn.ParameterList(
            [
                nn.Parameter(w, requires_grad=settings.learnable)
                for w in w_repeat
            ]
        )

    @staticmethod
    def create_matrices(  # ---------------------------------------------------
        shape_from: list[int],
        shape_to: list[int],
        settings: ProjectionSettings,
    ) -> tuple[list[Tensor], list[Tensor]]:
        """Create the down-projection and repeat matrices based on settings and shapes."""
        rank_list = _coerce_rank_list(settings.rank, shape_from, shape_to)
        if settings.init == "identity":
            w_down = utils.create_downsample_matrix(shape_from, rank_list)
            w_repeat = utils.create_repeat_matrices(rank_list, shape_to)
        elif settings.init == "random":
            w_down = utils.create_random_projection(shape_from, rank_list)
            w_repeat = utils.create_random_projection(rank_list, shape_to)
        else:
            raise ValueError(f"Unknown init strategy: {settings.init}")
        return w_down, w_repeat

    def reset_parameters(self) -> None:
        """Reset projection parameters according to the configured init strategy."""
        w_down, w_repeat = self.create_matrices(
            self._shape_from, self._shape_to, self._settings
        )
        for param, matrix in zip(self.w_down, w_down, strict=True):
            param.data.copy_(matrix.to(device=param.device, dtype=param.dtype))
        for param, matrix in zip(self.w_repeat, w_repeat, strict=True):
            param.data.copy_(matrix.to(device=param.device, dtype=param.dtype))

    def forward(  # -----------------------------------------------------------
        self,
        z_from: Sequence[Tensor],
    ) -> list[Tensor]:
        """Project from source to target multiscale codes using low-rank factorization."""
        z_down = [
            torch.matmul(z_from[f], self.w_down[f]) for f in range(len(z_from))
        ]
        return [
            torch.matmul(z_down[f], self.w_repeat[f])
            for f in range(len(z_down))
        ]

    def inverse(  # -----------------------------------------------------------
        self,
        z_to: Sequence[Tensor],
    ) -> list[Tensor]:
        """Project from target back to source multiscale codes using low-rank factorization."""
        z_down = [
            torch.matmul(z_to[f], self.w_repeat[f].T) for f in range(len(z_to))
        ]
        return [
            torch.matmul(z_down[f], self.w_down[f].T)
            for f in range(len(z_down))
        ]


# =============================================================================
class _FlatProjectionEdge(nn.Module):
    """Apply an aligned projection to one flat feature vector per row."""

    def __init__(  # ----------------------------------------------------------
        self,
        width_from: int,
        width_to: int,
        settings: ProjectionSettings,
    ) -> None:
        """Initialize the flat projection edge with specified widths and settings."""
        super().__init__()
        self._width_from = int(width_from)
        self._width_to = int(width_to)
        self.projection = ProjectionModule(
            [self._width_from], [self._width_to], settings
        )

    def reset_parameters(self) -> None:
        """Reset projection parameters according to the configured init strategy."""
        self.projection.reset_parameters()

    def forward(  # -----------------------------------------------------------
        self,
        x: Tensor,
    ) -> Tensor:
        """Project from source to target flat features after validating input shape."""
        _validate_flat_tensor(x, self._width_from, name="x")
        return self.projection([x])[0]

    def inverse(  # -----------------------------------------------------------
        self,
        y: Tensor,
    ) -> Tensor:
        """Project from target back to source flat features after validating input shape."""
        _validate_flat_tensor(y, self._width_to, name="y")
        return self.projection.inverse([y])[0]


# =============================================================================
class _TokenSequenceProjectionEdge(nn.Module):
    """Apply a flat projection independently to each token in a sequence."""

    def __init__(  # ----------------------------------------------------------
        self,
        width_from: int,
        width_to: int,
        settings: ProjectionSettings,
    ) -> None:
        """Initialize the token-sequence projection edge with specified widths and settings."""
        super().__init__()
        self._width_from = int(width_from)
        self._width_to = int(width_to)
        self.projection = _FlatProjectionEdge(
            self._width_from, self._width_to, settings
        )

    def reset_parameters(self) -> None:
        """Reset projection parameters according to the configured init strategy."""
        self.projection.reset_parameters()

    def forward(  # -----------------------------------------------------------
        self,
        x: Tensor,
    ) -> Tensor:
        """Project from source to target token features after validating input shape."""
        _validate_token_sequence_tensor(x, self._width_from, name="x")
        batch_size, seq_len, _ = x.shape
        y = self.projection(x.reshape(batch_size * seq_len, self._width_from))
        return y.view(batch_size, seq_len, self._width_to)

    def inverse(  # -----------------------------------------------------------
        self,
        y: Tensor,
    ) -> Tensor:
        """Project from target back to source token features after validating input shape."""
        _validate_token_sequence_tensor(y, self._width_to, name="y")
        batch_size, seq_len, _ = y.shape
        x = self.projection.inverse(
            y.reshape(batch_size * seq_len, self._width_to)
        )
        return x.view(batch_size, seq_len, self._width_from)


# =============================================================================
class _WorkspaceProjectionEdge(nn.Module):
    """Apply an aligned projection to each fixed slot in a workspace-shaped tensor.

    **Private — not part of the public projection API.**  Callers should use
    :func:`build_projection_edge` with two :class:`WorkspaceEndpointSpec` values.

    Each fixed role gets its own independent :class:`ProjectionModule`, so
    gradients and weight updates are fully isolated across slots by default.
    Weight tying across roles is deliberately not supported in this version;
    shared weights would collapse the role-specific projection surfaces that the
    fixed-slot contract requires.  If a tying ablation is ever needed, it should
    be expressed as an explicit opt-in option rather than changing the default.

    Scope: fixed-slot-only workspace endpoints.  Workspace families are not
    yet supported — both source and target must have an empty ``families``
    tuple.

    Constraints:
        - Both source and target must have no families.
        - Fixed slot names must match in order (same names, same positions).
        - At least one fixed slot is required.

    Args:
        source:   Source workspace endpoint descriptor.
        target:   Target workspace endpoint descriptor.
        settings: Projection settings applied to every per-slot projector.
    """

    def __init__(  # ----------------------------------------------------------
        self,
        source: WorkspaceEndpointSpec,
        target: WorkspaceEndpointSpec,
        settings: ProjectionSettings,
    ) -> None:
        """Initialize the workspace projection edge with specified source and target endpoint specs and settings."""
        super().__init__()
        if source.families:
            raise ValueError(
                f"workspace->workspace projection does not support source families yet; "
                f"source has families {[f.name for f in source.families]!r}."
            )
        if target.families:
            raise ValueError(
                f"workspace->workspace projection does not support target families yet; "
                f"target has families {[f.name for f in target.families]!r}."
            )
        if source.fixed != target.fixed:
            raise ValueError(
                f"workspace->workspace projection requires fixed slot names to match in order. "
                f"source fixed: {list(source.fixed)!r}, target fixed: {list(target.fixed)!r}."
            )
        if not source.fixed:
            raise ValueError(
                "workspace->workspace projection requires at least one fixed slot."
            )
        self._source = source
        self._target = target
        self._settings = settings
        # One independent ProjectionModule per fixed slot — distinct parameter block per role.
        self.projectors = nn.ModuleList(
            [
                ProjectionModule([source.width], [target.width], settings)
                for _ in source.fixed
            ]
        )

    def reset_parameters(self) -> None:
        """Reset all per-slot projection parameters."""
        for proj in self.projectors:
            proj.reset_parameters()

    def forward(  # -----------------------------------------------------------
        self,
        x: Tensor,
    ) -> Tensor:
        """Project each fixed slot independently.

        Args:
            x: Workspace tensor of shape ``(B, S, W_from)`` where
               ``S == len(source.fixed)`` and ``W_from == source.width``.

        Returns:
            Projected tensor of shape ``(B, S, W_to)`` where ``W_to == target.width``.
        """
        n_slots = len(self._source.fixed)
        w_from = self._source.width
        if (
            x.ndim != 3
            or int(x.shape[1]) != n_slots
            or int(x.shape[2]) != w_from
        ):
            raise ValueError(
                f"x must have shape (B, {n_slots}, {w_from}), got {tuple(x.shape)}."
            )
        slots = [self.projectors[i]([x[:, i, :]])[0] for i in range(n_slots)]
        return torch.stack(slots, dim=1)  # (B, S, W_to)


# =============================================================================
class _BroadcastProjectionEdge(nn.Module):
    """Broadcast one flat feature vector across multiscale target bands."""

    def __init__(  # ----------------------------------------------------------
        self,
        width_from: int,
        shape_to: Sequence[int],
        settings: ProjectionSettings,
    ) -> None:
        """Initialize the broadcast projection edge with specified input width, target shape, and settings."""
        super().__init__()
        self._width_from = int(width_from)
        self._shape_to = [int(width) for width in shape_to]
        self._n_bands = len(self._shape_to)
        self.projection = ProjectionModule(
            [self._width_from] * self._n_bands, self._shape_to, settings
        )

    def reset_parameters(self) -> None:
        self.projection.reset_parameters()

    def forward(  # -----------------------------------------------------------
        self,
        x: Tensor,
    ) -> list[Tensor]:
        """Project from source flat features to target multiscale bands after validating input shape."""
        _validate_flat_tensor(x, self._width_from, name="x")
        return self.projection([x for _ in range(self._n_bands)])

    def inverse(  # -----------------------------------------------------------
        self,
        y: Sequence[Tensor],
    ) -> Tensor:
        """Project from target multiscale bands back to source flat features after validating input shapes."""
        raise NotImplementedError(
            "broadcast flat->multiscale projections do not define a canonical inverse."
        )


# =============================================================================
def build_projection_edge(  # -------------------------------------------------
    *,
    source: ProjectionEndpointSpec,
    target: ProjectionEndpointSpec,
    settings: ProjectionSettings,
) -> nn.Module:
    """Build one projection edge from typed endpoint specs and edge settings."""
    bridge = _resolve_bridge(source, target, override=settings.bridge)
    if bridge == "aligned":
        if source.kind == "multiscale" and target.kind == "multiscale":
            return ProjectionModule(source.shape, target.shape, settings)
        if source.kind == "flat" and target.kind == "flat":
            return _FlatProjectionEdge(source.width, target.width, settings)
        if source.kind == "token_sequence" and target.kind == "token_sequence":
            return _TokenSequenceProjectionEdge(
                source.width, target.width, settings
            )
        if source.kind == "workspace" and target.kind == "workspace":
            return _WorkspaceProjectionEdge(source, target, settings)  # type: ignore[arg-type]
        raise ValueError(
            f"aligned bridge requires matching endpoint kinds, got {source.kind!r} -> {target.kind!r}."
        )

    if bridge == "broadcast":
        if source.kind == "flat" and target.kind == "multiscale":
            return _BroadcastProjectionEdge(
                source.width, target.shape, settings
            )
        raise ValueError(
            f"broadcast bridge supports flat -> multiscale only, got {source.kind!r} -> {target.kind!r}."
        )

    raise ValueError(f"Unknown projection bridge: {bridge}")


# =============================================================================
def _coerce_rank_list(  # -----------------------------------------------------
    rank: int | Sequence[int] | None,
    shape_from: Sequence[int],
    shape_to: Sequence[int],
) -> list[int]:
    """Normalize low-rank config into one per-band rank list and validate it."""
    n_freq = len(shape_from)
    if rank is None:
        rank_list = [
            math.gcd(int(n_in), int(n_out))
            for n_in, n_out in zip(shape_from, shape_to, strict=True)
        ]
    elif isinstance(rank, int):
        rank_list = [int(rank) for _ in range(n_freq)]
    else:
        rank_list = [int(r) for r in list(rank)]
        if len(rank_list) != n_freq:
            raise ValueError(
                f"rank must have length {n_freq}, got {len(rank_list)}."
            )

    for f, (n_in, n_out, r) in enumerate(
        zip(shape_from, shape_to, rank_list, strict=True)
    ):
        if r <= 0:
            raise ValueError(f"rank[{f}] must be > 0, got {r}.")
        if r > n_in:
            raise ValueError(f"rank[{f}]={r} cannot exceed input dim {n_in}.")
        if n_out % r != 0:
            raise ValueError(f"rank[{f}]={r} must divide output dim {n_out}.")
    return rank_list


# =============================================================================
def _coerce_shape(  # ---------------------------------------------------------
    component_or_shape: TEMComponent | Sequence[int],
) -> list[int]:
    """Return a validated multiscale shape from either a component or a raw shape list."""
    if isinstance(component_or_shape, Sequence) and all(
        isinstance(width, int) for width in component_or_shape
    ):
        shape = [int(width) for width in component_or_shape]
    else:
        shape = [int(width) for width in component_or_shape.shape]
    if len(shape) < 1:
        raise ValueError("projection shapes must contain at least one band.")
    if any(width <= 0 for width in shape):
        raise ValueError(
            f"projection shapes must contain positive widths, got {shape}."
        )
    return shape


# =============================================================================
def _coerce_endpoint_spec(  # -------------------------------------------------
    value: ProjectionEndpointValue,
) -> ProjectionEndpointSpec:
    """Normalize one endpoint value into an explicit endpoint spec."""
    if isinstance(
        value,
        (
            FlatEndpointSpec,
            MultiScaleEndpointSpec,
            TokenSequenceEndpointSpec,
            WorkspaceEndpointSpec,
        ),
    ):
        return value
    return multiscale_endpoint(value)


# =============================================================================
def _merge_projection_edges(  # -----------------------------------------------
    edges: Mapping[str, ProjectionModuleEdge] | None,
    more_edges: Mapping[str, ProjectionModuleEdge],
) -> dict[str, ProjectionModuleEdge]:
    """Merge positional and keyword edge definitions into one mapping."""
    merged = {} if edges is None else dict(edges)
    for name, edge in more_edges.items():
        if name in merged:
            raise ValueError(f"Duplicate projection edge name: {name!r}.")
        merged[name] = edge
    return merged


# =============================================================================
def _intern_builder_endpoint_name(  # -----------------------------------------
    normalized_names: dict[int, str],
    endpoint: ProjectionEndpointValue,
    *,
    fallback_name: str,
) -> str:
    """Return one stable synthetic endpoint name for a builder-supplied endpoint."""
    cache_key = id(endpoint)
    if cache_key not in normalized_names:
        normalized_names[cache_key] = fallback_name
    return normalized_names[cache_key]


# =============================================================================
def _resolve_bridge(  # -------------------------------------------------------
    source: ProjectionEndpointSpec,
    target: ProjectionEndpointSpec,
    *,
    override: ProjectionBridge,
) -> ProjectionBridge:
    """Resolve the bridge used to reconcile one source-target endpoint pair."""
    if override != "auto":
        return override
    if source.kind == "workspace" and target.kind == "workspace":
        return "aligned"
    if source.kind == "workspace" or target.kind == "workspace":
        raise ValueError(
            "mixed workspace/non-workspace projection is not supported. "
            "Use aligned workspace->workspace edges or explicit role-specific projectors."
        )
    if source.kind == target.kind and source.kind in {
        "flat",
        "multiscale",
        "token_sequence",
    }:
        return "aligned"
    if source.kind == "flat" and target.kind == "multiscale":
        return "broadcast"
    raise ValueError(
        f"Could not resolve projection bridge automatically for {source.kind!r} -> {target.kind!r}."
    )


# =============================================================================
def _create_linear_identity_matrix(  # ----------------------------------------
    n_in: int,
    n_out: int,
) -> Tensor:
    """Return one dense identity-like initialization matrix."""
    matrix = torch.zeros((n_in, n_out), dtype=torch.float32)
    for index in range(min(n_in, n_out)):
        matrix[index, index] = 1.0
    return matrix


# =============================================================================
def _validate_flat_tensor(  # -------------------------------------------------
    tensor: Tensor,
    width: int,
    *,
    name: str,
) -> None:
    """Validate one flat feature tensor with shape ``(B, width)``."""
    if tensor.ndim != 2 or int(tensor.shape[1]) != width:
        raise ValueError(
            f"{name} must have shape (B, {width}), got {tuple(tensor.shape)}."
        )


# =============================================================================
def _validate_token_sequence_tensor(  # ---------------------------------------
    tensor: Tensor,
    width: int,
    *,
    name: str,
) -> None:
    """Validate one token-sequence tensor with shape ``(B, S, width)``."""
    if tensor.ndim != 3 or int(tensor.shape[2]) != width:
        raise ValueError(
            f"{name} must have shape (B, S, {width}), got {tuple(tensor.shape)}."
        )


# =============================================================================
__all__ = [
    "FlatEndpointSpec",
    "MultiScaleEndpointSpec",
    "PerBandLinear",
    "ProjectionBundle",
    "ProjectionEdgeSpec",
    "ProjectionEndpointSpec",
    "ProjectionModule",
    "ProjectionSettings",
    "StructuredEndpointSpec",
    "TEMComponent",
    "TokenSequenceEndpointSpec",
    "WorkspaceEndpointSpec",
    "WorkspaceFamilySpec",
    "build_projection_edge",
    "flat_endpoint",
    "multiscale_endpoint",
    "token_sequence_endpoint",
    "workspace_endpoint",
]
