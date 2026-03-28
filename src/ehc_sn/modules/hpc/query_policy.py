"""Operator-style read requests and prepared read inputs for HPC memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Literal, Optional, TypeAlias

import torch
from pydantic import BaseModel, Field
from torch import Tensor, nn

from ehc_sn.types import DEFAULT_FACTOR_BANK_NAME, Device, Dtype

CueFamily: TypeAlias = str


# =================================================================================================
@dataclass(frozen=True)
class ReadCues:
    """Named multi-scale cues available to one memory read."""

    families: dict[CueFamily, list[Tensor]] = field(default_factory=dict)

    def get(self, family: Optional[CueFamily]) -> Optional[list[Tensor]]:
        """Return one named cue family when present."""
        if family is None:
            return None
        return self.families.get(family)

    def require(self, family: CueFamily) -> list[Tensor]:
        """Return one named cue family or raise when it is absent."""
        query = self.get(family)
        if query is None:
            raise ValueError(f"Cue family '{family}' is required.")
        return query

    def with_family(self, family: CueFamily, query: list[Tensor]) -> "ReadCues":
        """Return a new cue mapping with one family inserted or replaced."""
        families = dict(self.families)
        families[family] = query
        return ReadCues(families=families)


# =================================================================================================
class CueRead(BaseModel, extra="forbid"):
    """Read memory from exactly one cue family."""

    kind: Literal["cue"] = "cue"
    cue: CueFamily
    read_bank: str = DEFAULT_FACTOR_BANK_NAME


# =================================================================================================
class TargetRead(BaseModel, extra="forbid"):
    """Read one target family from one or more source families."""

    kind: Literal["target"] = "target"
    sources: tuple[CueFamily, ...]
    target: CueFamily
    target_init: Optional[CueFamily] = None
    read_bank: str = DEFAULT_FACTOR_BANK_NAME


# =================================================================================================
MemoryRead: TypeAlias = Annotated[CueRead | TargetRead, Field(discriminator="kind")]


# =================================================================================================
@dataclass(frozen=True)
class PreparedCueRead:
    """Prepared backend input for one cue read."""

    query: Tensor
    read_bank: str = DEFAULT_FACTOR_BANK_NAME


# =================================================================================================
@dataclass(frozen=True)
class PreparedTargetRead:
    """Prepared backend input for one target read."""

    source_queries: dict[CueFamily, Tensor]
    target: CueFamily
    fallback_query: Tensor
    read_bank: str = DEFAULT_FACTOR_BANK_NAME
    initial_target_query: Optional[Tensor] = None


# =================================================================================================
PreparedRead: TypeAlias = PreparedCueRead | PreparedTargetRead


class _BaseQueryComposer(nn.Module):
    """Base class for internal cue composers."""

    @property
    def shape(self) -> list[int]:
        """Return the multi-frequency query widths expected by this policy."""
        raise NotImplementedError

    @property
    def compose(  # -------------------------------------------------------------------------------
        self, *, read_cues: ReadCues, read: MemoryRead,
    ) -> PreparedRead:  # fmt: skip
        """Compose backend-ready evidence from one typed read request."""
        raise NotImplementedError

    def _flatten_query(self, query: list[Tensor]) -> Tensor:
        """Flatten a validated multi-scale query into shape ``(B, S)``."""
        return torch.cat(query, dim=1)

    def _validate_cues(self, read_cues: ReadCues) -> None:
        """Validate all populated cue families in one cue bundle."""
        for family, query in read_cues.families.items():
            self._validate_query(query, name=f"cues[{family!r}]")

    def _validate_query(self, query: Optional[list[Tensor]], *, name: str) -> None:
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

    def _validate_family(self, family: CueFamily, *, label: str) -> None:
        """Validate one cue-family name consumed by a read request."""
        if family == "":
            raise ValueError(f"{label} must not be empty.")

    def _validate_families(self, families: tuple[CueFamily, ...], *, label: str) -> None:
        """Validate one explicit family tuple consumed by a read request."""
        if len(families) < 1:
            raise ValueError(f"{label} must contain at least one cue family.")
        if len(set(families)) != len(families):
            raise ValueError(f"{label} must be distinct.")
        for index, family in enumerate(families):
            self._validate_family(family, label=f"{label}[{index}]")

    def _validate_request(self, read: MemoryRead) -> None:
        """Validate one typed read request before evidence composition."""
        if isinstance(read, CueRead):
            self._validate_family(read.cue, label="read.cue")
            return

        self._validate_families(read.sources, label="read.sources")
        self._validate_family(read.target, label="read.target")
        if read.target in read.sources:
            raise ValueError("read.sources must not include the target family.")
        if read.target_init is not None:
            self._validate_family(read.target_init, label="read.target_init")


# =================================================================================================
class ReadComposer(_BaseQueryComposer):
    """Prepare backend-ready retrieval inputs from typed read operators."""

    def __init__(  # ------------------------------------------------------------------------------
        self, shape: list[int], *,
        device: Optional[Device] = None, dtype: Optional[Dtype] = None,
    ) -> None:  # fmt: skip
        """Initialize the internal cue composer."""
        del device, dtype
        super().__init__()
        self._shape = list(shape)

    @property
    def shape(self) -> list[int]:
        """Return the multi-frequency query widths expected by this composer."""
        return self._shape

    def compose(  # -------------------------------------------------------------------------------
        self, *, read_cues: ReadCues, read: MemoryRead,
    ) -> PreparedRead:  # fmt: skip
        """Compose backend-ready evidence from one typed read request."""
        self._validate_cues(read_cues)
        self._validate_request(read)
        if isinstance(read, CueRead):
            return PreparedCueRead(
                query=self._flatten_query(read_cues.require(read.cue)),
                read_bank=read.read_bank,
            )

        source_queries = {family: self._flatten_query(read_cues.require(family)) for family in read.sources}
        initial_target_query: Optional[Tensor] = None
        if read.target_init is not None:
            initial_target_query = self._flatten_query(read_cues.require(read.target_init))

        fallback_query = initial_target_query
        if fallback_query is None:
            fallback_query = self._compose_source_fallback(tuple(source_queries.values()))

        return PreparedTargetRead(
            source_queries=source_queries,
            target=read.target,
            fallback_query=fallback_query,
            read_bank=read.read_bank,
            initial_target_query=initial_target_query,
        )

    def _compose_source_fallback(self, source_queries: tuple[Tensor, ...]) -> Tensor:
        """Return the fallback query used when a targeted read has no target-side seed."""
        if len(source_queries) == 1:
            return source_queries[0]
        return torch.stack(list(source_queries), dim=0).mean(dim=0)


# =================================================================================================
def build_read_composer(  # -----------------------------------------------------------------------
    shape: list[int], *,
    device: Optional[Device] = None, dtype: Optional[Dtype] = None,
) -> ReadComposer:  # fmt: skip
    """Construct the internal cue composer."""
    return ReadComposer(shape, device=device, dtype=dtype)


# =================================================================================================
__all__ = [
    "ReadCues", "ReadComposer", "PreparedRead", "MemoryRead", "TargetRead", "CueRead",
    "PreparedCueRead", "PreparedTargetRead", "build_read_composer",
]  # fmt: skip
