from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Mapping, Sequence, TypeVar

import torch
from torch import Tensor, nn

from ehc_sn.utils import trunc_normal_init_

WorkspaceContextT = TypeVar("WorkspaceContextT")


@dataclass(frozen=True)
class WorkspaceSpec:
    """Declare a fixed ordered set of workspace slot names."""

    names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.names:
            raise ValueError("WorkspaceSpec requires at least one slot name.")
        if any(not name for name in self.names):
            raise ValueError("WorkspaceSpec slot names must be non-empty.")
        if len(set(self.names)) != len(self.names):
            raise ValueError(f"WorkspaceSpec slot names must be unique, got {self.names!r}.")

    @classmethod
    def from_names(cls, names: Sequence[str]) -> "WorkspaceSpec":
        """Build a spec from any ordered sequence of slot names."""
        return cls(tuple(str(name) for name in names))

    @property
    def size(self) -> int:
        """Return the declared number of slots."""
        return len(self.names)

    def index(self, name: str) -> int:
        """Return the slot index for a named slot."""
        try:
            return self.names.index(name)
        except ValueError as exc:
            raise KeyError(f"Unknown workspace slot: {name!r}.") from exc

    def compose(self, *others: "WorkspaceSpec") -> "WorkspaceSpec":
        """Concatenate multiple specs while preserving declared order."""
        names = list(self.names)
        for other in others:
            names.extend(other.names)
        return WorkspaceSpec(tuple(names))


@dataclass(frozen=True)
class NamedWorkspace:
    """Tensor-backed workspace with named and indexed slot access."""

    spec: WorkspaceSpec
    tokens: Tensor

    def __post_init__(self) -> None:
        if self.tokens.ndim != 3:
            raise ValueError(f"workspace tokens must have shape (B, S, D), got {tuple(self.tokens.shape)}.")
        if int(self.tokens.shape[1]) != self.spec.size:
            raise ValueError(f"workspace token count must match spec size {self.spec.size}, got {tuple(self.tokens.shape)}.")

    @property
    def names(self) -> tuple[str, ...]:
        """Return the declared slot names."""
        return self.spec.names

    def __len__(self) -> int:
        return self.spec.size

    def index(self, name: str) -> int:
        """Return the integer position of a named slot."""
        return self.spec.index(name)

    def __getitem__(self, key: int | str) -> Tensor:
        """Return one workspace slot by integer index or slot name."""
        if isinstance(key, int):
            return self.tokens[:, key]
        return self.tokens[:, self.spec.index(key)]

    def select(self, names: Sequence[str]) -> Tensor:
        """Return a stacked token view following the requested slot order."""
        indices = [self.spec.index(name) for name in names]
        return self.tokens[:, indices]

    def compose(self, *others: "NamedWorkspace") -> "NamedWorkspace":
        """Concatenate multiple named workspaces along the slot axis."""
        if not others:
            return self

        batch_size = int(self.tokens.shape[0])
        hidden_size = int(self.tokens.shape[2])
        for other in others:
            if int(other.tokens.shape[0]) != batch_size or int(other.tokens.shape[2]) != hidden_size:
                raise ValueError("All composed workspaces must share batch size and hidden size.")

        return NamedWorkspace(
            spec=self.spec.compose(*(other.spec for other in others)),
            tokens=torch.cat([self.tokens, *(other.tokens for other in others)], dim=1),
        )


class WorkspaceSlotWriter(nn.Module, Generic[WorkspaceContextT]):
    """Base class for modules that produce one workspace slot token."""

    def forward(self, context: WorkspaceContextT) -> Tensor:
        raise NotImplementedError


class ComposedWorkspaceWriter(nn.Module, Generic[WorkspaceContextT]):
    """Compose named slot writers into one ordered workspace tensor."""

    def __init__(
        self,
        *,
        spec: WorkspaceSpec,
        writers: Mapping[str, WorkspaceSlotWriter[WorkspaceContextT]],
        hidden_size: int,
        use_slot_ids: bool = True,
    ) -> None:
        super().__init__()
        writer_names = set(writers)
        spec_names = set(spec.names)
        if writer_names != spec_names:
            missing = [name for name in spec.names if name not in writers]
            extra = [name for name in writers if name not in spec_names]
            raise ValueError(f"writer names must match spec names; missing={missing}, extra={extra}.")

        self._spec = spec
        self._hidden_size = int(hidden_size)
        self.writers = nn.ModuleDict({name: writers[name] for name in spec.names})
        self.slot_id = nn.Embedding(spec.size, hidden_size) if use_slot_ids else None
        self.reset_parameters()

    @property
    def spec(self) -> WorkspaceSpec:
        """Return the declared output workspace spec."""
        return self._spec

    @property
    def hidden_size(self) -> int:
        """Return the output hidden size per slot."""
        return self._hidden_size

    def reset_parameters(self) -> None:
        """Reset slot ids and delegate resets to child slot writers when available."""
        if self.slot_id is not None:
            trunc_normal_init_(self.slot_id.weight, std=0.02)
        for writer in self.writers.values():
            if hasattr(writer, "reset_parameters"):
                writer.reset_parameters()

    def forward(self, context: WorkspaceContextT) -> NamedWorkspace:
        tokens: list[Tensor] = []
        for name in self.spec.names:
            token = self.writers[name](context)
            if token.ndim != 2 or int(token.shape[1]) != self.hidden_size:
                raise ValueError(f"slot writer {name!r} must return shape (B, {self.hidden_size}), got {tuple(token.shape)}.")
            tokens.append(token)

        workspace_tokens = torch.stack(tokens, dim=1)
        if self.slot_id is not None:
            slot_ids = torch.arange(self.spec.size, device=workspace_tokens.device)
            workspace_tokens = workspace_tokens + self.slot_id(slot_ids).unsqueeze(0)
        return NamedWorkspace(spec=self.spec, tokens=workspace_tokens)


def workspace_from_prefixed_tokens(
    tokens: Tensor,
    spec: WorkspaceSpec,
    *,
    prefix_tokens: int = 1,
) -> NamedWorkspace:
    """Recover a named workspace from a prefixed PFC token sequence."""
    if tokens.ndim != 3:
        raise ValueError(f"tokens must have shape (B, S, D), got {tuple(tokens.shape)}.")
    if prefix_tokens < 0:
        raise ValueError(f"prefix_tokens must be non-negative, got {prefix_tokens}.")

    start = int(prefix_tokens)
    stop = start + spec.size
    if int(tokens.shape[1]) < stop:
        raise ValueError(f"tokens sequence length must be at least {stop} to recover {spec.size} workspace slots.")
    return NamedWorkspace(spec=spec, tokens=tokens[:, start:stop])


__all__ = [
    "ComposedWorkspaceWriter",
    "NamedWorkspace",
    "WorkspaceSlotWriter",
    "WorkspaceSpec",
    "workspace_from_prefixed_tokens",
]
