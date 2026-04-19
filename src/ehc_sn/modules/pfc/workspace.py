"""Workspace schema, compiled layout, and runtime slot-bank view for the PFC module.

Three layers of abstraction:

1. :class:`WorkspaceSchema` — declarative structure: named fixed slots and exchangeable
   slot families.  No tensor involvement.

2. :class:`WorkspaceLayout` — compiled view of the schema: maps each fixed slot and
   family name to a concrete position or slice within a z_H tensor.

3. :class:`Workspace` — runtime view: a :class:`WorkspaceLayout` paired with one live
   z_H token tensor.  Exposes object-based ``slot()`` and ``family()`` accessors.

Ownership rule: this module owns only the addressing scheme over a slot bank.
It does not define working-memory dynamics (``reasoning.py`` owns those).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


# =============================================================================
@dataclass(frozen=True)
class FixedSlot:
    """Declaration of a singleton named slot in a workspace schema.

    Attributes:
        name: Unique semantic role name (non-empty).
    """

    name: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("FixedSlot name must be non-empty.")


# =============================================================================
@dataclass(frozen=True)
class SlotFamily:
    """Declaration of a named, exchangeable repeated-slot family.

    All slots within a family are semantically interchangeable (e.g. schema
    hypothesis slots).  Runtime access is via :meth:`Workspace.family`.

    Attributes:
        name: Unique family name (non-empty).
        size: Number of slots in the family (>= 1).
    """

    name: str
    size: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("SlotFamily name must be non-empty.")
        if self.size < 1:
            raise ValueError(f"SlotFamily size must be at least 1, got {self.size}.")


# =============================================================================
@dataclass(frozen=True)
class WorkspaceSchema:
    """Declarative workspace structure: ordered fixed slots then ordered families.

    All names across ``fixed`` and ``families`` must be unique.

    Attributes:
        fixed: Ordered singleton slot declarations.
        families: Ordered repeating slot-family declarations.
    """

    fixed: tuple[FixedSlot, ...]
    families: tuple[SlotFamily, ...]

    def __post_init__(self) -> None:
        names = [s.name for s in self.fixed] + [f.name for f in self.families]
        if len(set(names)) != len(names):
            raise ValueError(f"WorkspaceSchema names must be unique, got {names!r}.")

    @property
    def size(self) -> int:
        """Total slot count: len(fixed) + sum of family sizes."""
        return len(self.fixed) + sum(f.size for f in self.families)


# =============================================================================
class WorkspaceLayout:
    """Compiled slot and family offsets derived from a :class:`WorkspaceSchema`.

    Fixed slots are placed first (in declaration order), then each family's
    slots are placed contiguously (in family declaration order).

    Use :meth:`from_schema` to construct.

    Attributes:
        schema: The originating declarative schema.
    """

    def __init__(self, schema: WorkspaceSchema) -> None:
        self.schema = schema
        offset = 0
        self._slot_offsets: dict[str, int] = {}
        self._family_slices: dict[str, slice] = {}
        for fixed in schema.fixed:
            self._slot_offsets[fixed.name] = offset
            offset += 1
        for family in schema.families:
            self._family_slices[family.name] = slice(offset, offset + family.size)
            offset += family.size
        self._size = offset

    @classmethod
    def from_schema(cls, schema: WorkspaceSchema) -> "WorkspaceLayout":
        """Compile a layout from a declarative schema."""
        return cls(schema)

    @property
    def size(self) -> int:
        """Total number of slots in this layout."""
        return self._size

    def slot(self, name: str) -> int:
        """Return the integer position of a named fixed slot.

        Raises:
            KeyError: If ``name`` is not a declared fixed slot.
        """
        try:
            return self._slot_offsets[name]
        except KeyError:
            raise KeyError(f"Unknown fixed slot: {name!r}.") from None

    def family(self, name: str) -> slice:
        """Return the contiguous slice for a named slot family.

        Raises:
            KeyError: If ``name`` is not a declared slot family.
        """
        try:
            return self._family_slices[name]
        except KeyError:
            raise KeyError(f"Unknown slot family: {name!r}.") from None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, WorkspaceLayout):
            return NotImplemented
        return self.schema == other.schema

    def __hash__(self) -> int:
        return hash(self.schema)

    def bind(self, tokens: Tensor) -> "Workspace":
        """Bind a live token tensor to this layout, returning a runtime :class:`Workspace`.

        This is the canonical way to create a :class:`Workspace` from a tensor; prefer
        ``layout.bind(tokens)`` over direct ``Workspace(layout=layout, tokens=tokens)``
        construction.

        Args:
            tokens: Token tensor of shape ``(B, self.size, D)``.

        Returns:
            :class:`Workspace` bound to this layout and the supplied tokens.
        """
        return Workspace(layout=self, tokens=tokens)

    def __repr__(self) -> str:
        return f"WorkspaceLayout(schema={self.schema!r})"


# =============================================================================
@dataclass(frozen=True)
class Workspace:
    """Runtime-bound view: a :class:`WorkspaceLayout` over one z_H token tensor.

    Attributes:
        layout: Compiled slot/family layout.
        tokens: Token tensor of shape ``(B, layout.size, D)``.
    """

    layout: WorkspaceLayout
    tokens: Tensor

    def __post_init__(self) -> None:
        if self.tokens.ndim != 3:
            raise ValueError(f"Workspace tokens must have shape (B, S, D), got {tuple(self.tokens.shape)}.")
        if int(self.tokens.shape[1]) != self.layout.size:
            raise ValueError(f"Workspace token count must match layout size {self.layout.size}, " f"got {tuple(self.tokens.shape)}.")

    def slot(self, name: str) -> Tensor:
        """Return one fixed-slot tensor.

        Args:
            name: Declared fixed-slot name.

        Returns:
            Tensor of shape ``(B, D)``.
        """
        return self.tokens[:, self.layout.slot(name)]

    def family(self, name: str) -> Tensor:
        """Return all slot tensors for a named family.

        Args:
            name: Declared slot-family name.

        Returns:
            Tensor of shape ``(B, family_size, D)``.
        """
        return self.tokens[:, self.layout.family(name)]


# =============================================================================
__all__ = ["FixedSlot", "SlotFamily", "Workspace", "WorkspaceLayout", "WorkspaceSchema"]
