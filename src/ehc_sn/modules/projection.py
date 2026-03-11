from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import List, Literal, Optional, Protocol, Sequence, Tuple

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor, nn

from ehc_sn import utils
from ehc_sn.modules.mlp import MLP
from ehc_sn.types import InitStrategy, ProjectionMode


# =================================================================================================
class ProjectionSettings(BaseModel, extra="forbid"):
    """Settings for projection modules."""

    mode: ProjectionMode = Field(
        ...,
        description="Projection mode strategy",
    )
    init: InitStrategy = Field(
        default="identity",
        description="Initialization strategy: 'identity' (structured) or 'random'",
    )
    learnable: bool = Field(
        default=False,
        description="If True, projection matrices are learnable",
    )
    rank: Optional[int] = Field(
        default=None,
        description="Low_rank rank for low_rank mode (auto-derived via GCD if None)",
    )


# =================================================================================================
class TEMComponent(Protocol):
    shape: Sequence[int]


# =================================================================================================
class ProjectionModule(nn.Module):

    def __init__(self, z_from: TEMComponent, z_to: TEMComponent, settings: ProjectionSettings):
        self._shape_from, self._shape_to = z_from.shape, z_to.shape
        if len(self._shape_from) != len(self._shape_to):
            raise ValueError("Components must have the same number of frequency modules.")

        super().__init__()
        self._module: Optional[AbstractModule] = None
        self._settings = settings
        self._module = self.create_module(settings, self._shape_from, self._shape_to)
        self.set_learning()

    @staticmethod
    def create_module(settings: ProjectionSettings, shape_from: list[int], shape_to: list[int]) -> nn.Module:
        if settings.mode == "tiling":
            module = TileModule(shape_from, shape_to, settings)
        elif settings.mode == "low_rank":
            module = LowRankModule(shape_from, shape_to, settings)
        else:
            raise ValueError(f"Unknown projection mode: {settings.mode}")
        return module

    def set_learning(self) -> None:
        """Enable or disable learning of projection weights."""
        for param in self._module.parameters():
            param.requires_grad = self._settings.learnable

    def forward(self, z: list[Tensor]) -> list[Tensor]:
        return self._module.forward(z)

    def inverse(self, p: list[Tensor]) -> list[Tensor]:
        return self._module.inverse(p)


# =================================================================================================
class AbstractModule(nn.Module, ABC):
    @abstractmethod
    def forward(self, z: list[Tensor]) -> list[Tensor]:
        pass

    @abstractmethod
    def inverse(self, p: list[Tensor]) -> list[Tensor]:
        pass


# =================================================================================================
class TileModule(AbstractModule):

    def __init__(self, shape_from: list[int], shape_to: list[int], settings: ProjectionSettings):
        super(TileModule, self).__init__()
        self._settings = settings
        w_list = self.create_matrices(shape_from, shape_to, settings)
        self.w = nn.ParameterList([nn.Parameter(w, requires_grad=settings.learnable) for w in w_list])

    @staticmethod
    def create_matrices(
        shape_from: list[int], shape_to: list[int], settings: ProjectionSettings
    ) -> list[Tensor]:
        if settings.init == "identity":
            w_list = utils.create_tiling_matrices(shape_from, shape_to)
        elif settings.init == "random":
            w_list = utils.create_random_projection(shape_from, shape_to)
        else:
            raise ValueError(f"Unknown init strategy: {settings.init}")
        return w_list

    def forward(self, z_from: list[Tensor]) -> list[Tensor]:
        # z[f]: [B, n_in] -> [B, n_out]
        return [torch.matmul(z_from[f], self.w[f]) for f in range(len(z_from))]

    def inverse(self, z_to: list[Tensor]) -> list[Tensor]:
        # z[f]: [B, n_out] -> [B, n_in]
        return [torch.matmul(z_to[f], self.w[f].T) for f in range(len(z_to))]


# =================================================================================================
class LowRankModule(AbstractModule):

    def __init__(self, shape_from: list[int], shape_to: list[int], settings: ProjectionSettings):
        super(LowRankModule, self).__init__()
        self._settings = settings
        w_down, w_repeat = self.create_matrices(shape_from, shape_to, settings)
        self.w_down = nn.ParameterList([nn.Parameter(w, requires_grad=settings.learnable) for w in w_down])
        self.w_repeat = nn.ParameterList(
            [nn.Parameter(w, requires_grad=settings.learnable) for w in w_repeat]
        )

    @staticmethod
    def create_matrices(
        shape_from: list[int], shape_to: list[int], settings: ProjectionSettings
    ) -> tuple[list[Tensor], list[Tensor]]:
        rank_list = _coerce_rank_list(settings.rank, shape_from, shape_to)
        if settings.init == "identity":  # Legacy-equivalent: downsample to r dims, then repeat to n_out
            w_down = utils.create_downsample_matrix(shape_from, rank_list)
            w_repeat = utils.create_repeat_matrices(rank_list, shape_to)
        elif settings.init == "random":
            w_down = utils.create_random_projection(shape_from, rank_list)
            w_repeat = utils.create_random_projection(rank_list, shape_to)
        else:
            raise ValueError(f"Unknown init strategy: {settings.init}")
        return w_down, w_repeat

    def forward(self, z_from: list[Tensor]) -> list[Tensor]:
        # z[f]: [B, n_in] -> [B, r] -> [B, n_out]
        z_down = [torch.matmul(z_from[f], self.w_down[f]) for f in range(len(z_from))]
        return [torch.matmul(z_down[f], self.w_repeat[f]) for f in range(len(z_down))]

    def inverse(self, z_to: list[Tensor]) -> list[Tensor]:
        # z[f]: [B, n_out] -> [B, r] -> [B, n_in]
        z_down = [torch.matmul(z_to[f], self.w_repeat[f].T) for f in range(len(z_to))]
        return [torch.matmul(z_down[f], self.w_down[f].T) for f in range(len(z_down))]


# =================================================================================================
def _coerce_rank_list(rank, shape_from: Sequence[int], shape_to: Sequence[int]) -> list[int]:
    """Normalize rank config into per-frequency list and validate."""
    n_freq = len(shape_from)
    if rank is None:
        rank_list = [math.gcd(int(n_in), int(n_out)) for n_in, n_out in zip(shape_from, shape_to)]
    elif isinstance(rank, int):
        rank_list = [int(rank) for _ in range(n_freq)]
    else:
        rank_list = [int(r) for r in list(rank)]
        if len(rank_list) != n_freq:
            raise ValueError(f"rank must have length {n_freq}, got {len(rank_list)}.")

    for f, (n_in, n_out, r) in enumerate(zip(shape_from, shape_to, rank_list)):
        if r <= 0:
            raise ValueError(f"rank[{f}] must be > 0, got {r}.")
        if r > n_in:
            raise ValueError(f"rank[{f}]={r} cannot exceed input dim {n_in}.")
        if n_out % r != 0:
            raise ValueError(f"rank[{f}]={r} must divide output dim {n_out}.")
    return rank_list
