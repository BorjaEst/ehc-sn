"""Load-time transforms for processed channel dicts.

Transforms operate on ``dict[str, np.ndarray]`` channel dicts loaded from
versioned processed split roots and return the same type. They are composable
via :class:`Compose` and follow the torchvision-style callable convention.

Provided transforms:
- :class:`RandomDihedral` — randomly applies one of the 8 dihedral symmetries
  (4 rotations × 2 flips) consistently across all channels.
- :class:`Compose` — chains multiple transforms sequentially.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from ehc_sn.types import Channels
from ehc_sn.utils.symmetry import dihedral_transform


# =================================================================================================
class Compose:
    """Chain multiple channel transforms sequentially.

    Each transform receives the output of the previous one.

    Args:
        transforms: Ordered list of callables ``Channels -> Channels``.

    Example::

        transform = Compose([RandomDihedral()])
        sample = transform(channels)

    Note:
        Task-specific transforms (such as grid-projection or action encoding)
        live outside :mod:`ehc_sn.data.transforms` in their respective task
        adapter packages.
    """

    def __init__(  # ------------------------------------------------------------------------------
        self, transforms: Sequence[Callable[[Channels], Channels] | None],
    ) -> None:  # fmt: skip
        self.transforms = []
        for transform in transforms:
            if transform is None:
                continue
            if not callable(transform):
                raise TypeError(f"Compose transforms must be callable or None, got {type(transform).__name__}.")  # fmt: skip
            self.transforms.append(transform)

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

        Samples that mix spatial grids with non-spatial task fields (for
        example replay trajectories or scalar metadata) are returned unchanged.
        Generic dihedral augmentation is only valid for uniform 2D spatial
        channel sets.
    """

    def __init__(  # ------------------------------------------------------------------------------
        self, rng: np.random.Generator | None = None,
    ) -> None:  # fmt: skip
        self._rng = rng if rng is not None else np.random.default_rng()

    def __call__(  # ------------------------------------------------------------------------------
        self, channels: Channels
    ) -> Channels:  # fmt: skip
        first_channel = next(iter(channels.values()))
        if first_channel.ndim != 2:
            return channels

        h, w = first_channel.shape
        if any(channel.ndim != 2 or channel.shape != (h, w) for channel in channels.values()):
            return channels

        valid_tids = (0, 1, 2, 3, 4, 5, 6, 7) if h == w else (0, 2, 4, 5)
        tid = int(valid_tids[int(self._rng.integers(len(valid_tids)))])
        return {name: dihedral_transform(arr, tid) for name, arr in channels.items()}

    def __repr__(  # ------------------------------------------------------------------------------
        self,
    ) -> str:  # fmt: skip
        return f"{type(self).__name__}()"


# =================================================================================================
__all__ = ["Channels", "Compose", "RandomDihedral"]
