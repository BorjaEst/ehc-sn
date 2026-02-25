"""Decorators for figure panel methods."""

from __future__ import annotations

from typing import Any, Callable, Sequence


def colorbar(*, group: str, label: str | None = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Attach colorbar grouping metadata to a panel method.

    Args:
        group: Shared colorbar group name.
        label: Optional colorbar label.

    Returns:
        A decorator that attaches metadata to the function.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        setattr(fn, "_tem_colorbar", {"group": group, "label": label})
        return fn

    return decorator


def panel(
    *,
    slots: Sequence[str] | None = None,
    primary: str | None = None,
    order: int | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Mark a method as a panel renderer.

    Args:
        slots: Slots owned by this panel. Defaults to the method name.
        primary: Slot passed as the primary Axes. Defaults to the first slot.
        order: Optional explicit ordering key (lower renders first).

    Returns:
        A decorator that attaches panel metadata to the function.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        default_slots = [fn.__name__]
        owned_slots = list(slots) if slots is not None else default_slots
        if not owned_slots:
            owned_slots = default_slots
        primary_slot = primary or owned_slots[0]
        setattr(
            fn,
            "_tem_panel",
            {"slots": tuple(owned_slots), "primary": primary_slot, "order": order},
        )
        return fn

    return decorator
