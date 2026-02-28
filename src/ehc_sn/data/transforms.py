"""Load-time transforms for maze NPZ channels.

Transforms operate on ``dict[str, np.ndarray]`` channel dicts and return the
same type. They are composable via :class:`Compose` and follow the
torchvision-style callable convention.

Provided transforms:
- :class:`RandomDihedral` — randomly applies one of the 8 dihedral symmetries
  (4 rotations × 2 flips) consistently across all channels.
- :func:`channels_to_grid` — merges ``topology``, ``start``, and ``goals``
  channels into a single ``int32`` grid using canonical SEM IDs. Utility for
  gymnasium environments and model adapters; not called by the dataset or
  DataModule directly.
- :class:`Compose` — chains multiple transforms sequentially.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ehc_sn.data.schema import CHANNEL_GOALS, CHANNEL_START, CHANNEL_TOPOLOGY
from ehc_sn.data.vocabulary import EMPTY_ID, GOAL_ID, START_ID, WALL_ID
from ehc_sn.types import Channels
from ehc_sn.utils.symmetry import dihedral_transform


# =================================================================================================
class Compose:
    """Chain multiple channel transforms sequentially.

    Each transform receives the output of the previous one.

    Args:
        transforms: Ordered list of callables ``Channels -> Channels``.

    Example::

        transform = Compose([RandomDihedral(), channels_to_grid])
        sample = transform(raw_channels)
    """

    def __init__(  # ------------------------------------------------------------------------------
        self, transforms: list[Callable[[Channels], Channels]],
    ) -> None:  # fmt: skip
        self.transforms = transforms

    def __call__(  # -----------------------------------------------------------------------------
        self, channels: Channels,
    ) -> Channels:  # fmt: skip
        for t in self.transforms:
            channels = t(channels)
        return channels

    def __repr__(  # ------------------------------------------------------------------------------
        self,
    ) -> str:  # fmt: skip
        steps = ", ".join(repr(t) for t in self.transforms)
        return f"{type(self).__name__}([{steps}])"


# =================================================================================================
class RandomDihedral:
    """Apply a random dihedral symmetry to all channels consistently.

    The dihedral group D4 has 8 elements (4 rotations × 2 reflections). Applying
    the same transformation to every channel ensures spatial consistency across
    ``topology``, ``start``, ``goals``, ``solution``, etc.

    Args:
        rng: Optional numpy random generator for reproducibility. If ``None``,
            a new generator is created from the global numpy random state.

    Note:
        This transform is applied at load time (per :class:`~ehc_sn.data.datasets.MazeDataset`
        ``__getitem__`` call), not baked into on-disk files.
    """

    def __init__(  # ------------------------------------------------------------------------------
        self, rng: np.random.Generator | None = None,
    ) -> None:  # fmt: skip
        self._rng = rng if rng is not None else np.random.default_rng()

    def __call__(  # ------------------------------------------------------------------------------
        self, channels: Channels
    ) -> Channels:  # fmt: skip
        tid = int(self._rng.integers(8))
        return {name: dihedral_transform(arr, tid) for name, arr in channels.items()}

    def __repr__(  # ------------------------------------------------------------------------------
        self,
    ) -> str:  # fmt: skip
        return f"{type(self).__name__}()"


# =================================================================================================
def channels_to_grid(  # --------------------------------------------------------------------------
    channels: Channels,
) -> Channels:  # fmt: skip
    """Merge structural channels into a single canonical semantic grid.

    Combines ``topology``, ``start``, and ``goals`` into a single ``int32``
    array ``"grid"`` of shape ``(H, W)`` using canonical SEM IDs.  The result
    is returned non-destructively alongside the original channels.

    Priority (later assignments win): ``WALL < EMPTY < START < GOAL``.

    .. note::
        ``"grid"`` is a **derived synthetic key** — it is not present on disk.
        Channels with data-dependent vocabularies (``landmarks``,
        ``observations``, ``solution``, ``regions``) are deliberately excluded;
        they remain accessible via the original channel keys.

    Args:
        channels: Dict of canonical NPZ channel arrays.  Must contain
            ``"topology"`` (bool, H×W).  Optional: ``"start"``, ``"goals"``.

    Returns:
        Input dict extended with ``"grid": int32 array of shape (H, W)``.
    """
    topology = channels[CHANNEL_TOPOLOGY]
    grid = np.where(topology, EMPTY_ID, WALL_ID).astype(np.int32)
    if CHANNEL_START in channels:
        grid = np.where(channels[CHANNEL_START], START_ID, grid)
    if CHANNEL_GOALS in channels:
        grid = np.where(channels[CHANNEL_GOALS], GOAL_ID, grid)
    return {**channels, "grid": grid}


# =================================================================================================
__all__ = ["Channels", "Compose", "RandomDihedral", "channels_to_grid"]
