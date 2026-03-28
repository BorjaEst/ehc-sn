"""Concrete hippocampal memory implementations composed from shared HPC layers.

The module exposes the two supported HPC realizations in the repo: an
attractor-style dense Hebbian memory and an attention-style explicit
factor-memory system.
"""

from __future__ import annotations

from typing import Optional

from pydantic import Field
from torch import Tensor

from ehc_sn import utils
from ehc_sn.modules.hpc._base import HPCBase, HPCCommonSettings, HPCState
from ehc_sn.modules.hpc.memory import DenseHebbianStoreBackend, FactorAppendStoreBackend, FactorStoreSettings, LinearStoreSettings
from ehc_sn.modules.hpc.query import AttractorRead, AttractorReadSettings, FactorRead, FactorReadSettings
from ehc_sn.modules.hpc.query_policy import MemoryRead, ReadCues
from ehc_sn.modules.hpc.update import EpisodicWrite, EpisodicWriteSettings, HebbianWrite, HebbianWriteSettings, build_hebbian_layout
from ehc_sn.types import Device, Dtype, MemoryEntry, MemoryState, RetrievalRole


# =================================================================================================
class HPCAttractorSettings(HPCCommonSettings):
    """Configuration for dense Hebbian memory with attractor recall."""

    read: AttractorReadSettings = Field(
        default_factory=AttractorReadSettings,
        description="Settings for attractor retrieval dynamics.",
    )
    write: HebbianWriteSettings = Field(
        default_factory=HebbianWriteSettings,
        description="Settings for Hebbian memory write.",
    )
    store: LinearStoreSettings = Field(
        default_factory=LinearStoreSettings,
        description="Settings for the Hebbian memory store.",
    )


# =================================================================================================
class HPCAttentionSettings(HPCCommonSettings):
    """Configuration for explicit factor memory with attention-style recall."""

    read: FactorReadSettings = Field(
        default_factory=FactorReadSettings,
        description="Settings for factor-memory retrieval.",
    )
    write: EpisodicWriteSettings = Field(
        default_factory=EpisodicWriteSettings,
        description="Settings for factor-memory write.",
    )
    store: FactorStoreSettings = Field(
        default_factory=FactorStoreSettings,
        description="Settings for the factor memory store.",
    )


# =================================================================================================
class HPCAttractor(HPCBase):
    """Attractor-based hippocampal memory with dense Hebbian write dynamics.

    This implementation stores cue-indexed memories as dense linear operators
    and retrieves place codes via staged attractor dynamics.
    """

    def __init__(  # ------------------------------------------------------------------------------
        self, n_stages: int, f_initial: list[float], config: HPCAttractorSettings, *,
        device: Optional[Device] = None, dtype: Optional[Dtype] = None,
    ) -> None:  # fmt: skip
        """Initialize attractor retrieval, Hebbian writes, and update masks.

        Args:
            n_stages: Number of hierarchical stages used by the retrieval and
                write masks.
            f_initial: Frequency ordering used to build hierarchical Hebbian
                connectivity.
            config: Attractor-memory configuration.
            device: Optional device used when allocating submodules.
            dtype: Optional dtype used when allocating submodules.
        """
        super().__init__(config, device=device, dtype=dtype)

        masks = utils.update_to_masks(self.shape, update=utils.make_update_hierarchical(n_stages, self.n_freq))
        self.register_buffer("masks_hierarchical", masks, persistent=False)
        masks = utils.update_to_masks(self.shape, update=utils.make_update_full(n_stages, self.n_freq))
        self.register_buffer("masks_full", masks, persistent=False)

        hebbian_layout = build_hebbian_layout(n_stages, self.shape, f_initial)

        self.retrieval_module = AttractorRead(config.read)
        self.write_module = HebbianWrite(config.write, device=device, dtype=dtype)
        self.store_backend = DenseHebbianStoreBackend(self.write_module, layout=hebbian_layout)

    @property
    def config(self) -> HPCAttractorSettings:
        """Return the typed config for the attractor memory module."""
        return super().config  # type: ignore[return-value]

    def _init_memory_impl(  # --------------------------------------------------------------------
        self, batch_size: int, *,
        device: Optional[Device] = None,
    ) -> MemoryState:  # fmt: skip
        """Initialize linear Hebbian stores for g-cued and optional x-cued memory."""
        g_cued = self.store_backend.init_store(batch_size, device=device)
        x_cued = g_cued if self.config.common_memory else g_cued.clone()
        return MemoryState(g_cued=g_cued, x_cued=x_cued)

    def _set_runtime_impl(  # --------------------------------------------------------------------
        self, *, eta: float, hebbian_decay: float,
    ) -> None:  # fmt: skip
        """Apply runtime Hebbian learning rate and decay parameters."""
        self.write_module.runtime.eta = float(eta)
        self.write_module.runtime.hebbian_decay = float(hebbian_decay)

    def _recall_flat_impl(  # --------------------------------------------------------------------
        self, query: Tensor, memory: MemoryEntry, *, role: RetrievalRole,
    ) -> Tensor:  # fmt: skip
        """Recall a flattened code using role-specific attractor update masks."""
        masks = self.masks_hierarchical if role == "generative" else self.masks_full
        return self.retrieval_module(query, memory.as_linear_view(), masks=masks)

    def _update_memory_impl(  # -------------------------------------------------------------------
        self, memory: MemoryState, key: Tensor, g_value: Tensor, x_value: Optional[Tensor],
        named_writes: dict[str, Tensor],
    ) -> MemoryState:  # fmt: skip
        """Write flattened values into the Hebbian store for the active memory entries."""
        if named_writes:
            names = ", ".join(sorted(named_writes))
            raise TypeError(f"{type(self).__name__} does not support named bank writes: {names}.")
        g_cued = self.store_backend.apply_write(memory.g_cued, key, g_value, masked=True)
        x_cued = g_cued if self.config.common_memory else memory.x_cued
        if not self.config.common_memory and x_value is not None:
            x_cued = self.store_backend.apply_write(memory.x_cued, key, x_value, masked=False)
        return MemoryState(g_cued=g_cued, x_cued=x_cued)

    def _merge_memory_rows_impl(  # ---------------------------------------------------------------
        self, flag: Tensor, current: MemoryEntry, fresh: MemoryEntry,
    ) -> MemoryEntry:  # fmt: skip
        """Merge representation rows during partial reset using the Hebbian reset strategy."""
        return self.store_backend.merge_rows(flag, current, fresh)


# =================================================================================================
class HPCAttention(HPCBase):
    """Attention-based hippocampal memory with explicit factor slots.

    This implementation stores episodes as explicit key-value factors and
    supports both resolved cue reads and targeted source-to-target retrieval.
    """

    def __init__(  # ------------------------------------------------------------------------------
        self, n_stages: int, f_initial: list[float], config: HPCAttentionSettings, *,
        device: Optional[Device] = None, dtype: Optional[Dtype] = None,
    ) -> None:  # fmt: skip
        """Initialize factor-slot retrieval and append-only write modules.

        ``n_stages`` and ``f_initial`` are accepted for constructor parity with
        ``HPCAttractor`` but are not currently used by the attention backend.
        """
        del n_stages, f_initial
        super().__init__(config, device=device, dtype=dtype)
        self.retrieval_module = FactorRead(config.read)
        self.write_module = EpisodicWrite(config.write)
        self.store_backend = FactorAppendStoreBackend(self.write_module, feature_dim=sum(self.shape), settings=config.store)

    @property
    def config(self) -> HPCAttentionSettings:
        """Return the typed config for the attention memory module."""
        return super().config  # type: ignore[return-value]

    def _init_memory_impl(  # --------------------------------------------------------------------
        self, batch_size: int, *,
        device: Optional[Device] = None,
    ) -> MemoryState:  # fmt: skip
        """Initialize factor stores for g-cued memory and optional separate x-cued memory."""
        g_cued = self.store_backend.init_store(batch_size, device=device)
        x_cued = g_cued if self.config.common_memory else self.store_backend.init_store(batch_size, device=device)
        return MemoryState(g_cued=g_cued, x_cued=x_cued)

    def _set_runtime_impl(  # --------------------------------------------------------------------
        self, *, eta: float, hebbian_decay: float,
    ) -> None:  # fmt: skip
        """Ignore Hebbian runtime parameters for slot-based attention memory."""
        del eta, hebbian_decay

    def _recall_flat_impl(  # --------------------------------------------------------------------
        self, query: Tensor, memory: MemoryEntry, *, role: RetrievalRole,
    ) -> Tensor:  # fmt: skip
        """Recall a flattened code from factor memory without role-specific masking."""
        del role
        return self.retrieval_module(query, memory.as_factor_view())

    def _update_memory_impl(  # -------------------------------------------------------------------
        self, memory: MemoryState, key: Tensor, g_value: Tensor, x_value: Optional[Tensor],
        named_writes: dict[str, Tensor],
    ) -> MemoryState:  # fmt: skip
        """Write flattened values into factor slots for the active memory entries."""
        g_cued = self.store_backend.append_write(memory.g_cued, key, g_value)
        x_cued = g_cued if self.config.common_memory else memory.x_cued
        if self.config.common_memory:
            x_cued = g_cued
        elif x_value is not None:
            x_cued = self.store_backend.append_write(memory.x_cued, key, x_value)

        if named_writes:
            available_bank_names = set(memory.g_cued.bank_names) | set(memory.x_cued.bank_names)
            missing_bank_names = sorted(set(named_writes) - available_bank_names)
            if missing_bank_names:
                names = ", ".join(repr(name) for name in missing_bank_names)
                raise ValueError(f"Named factor-memory banks {names} are not available for writing.")

            for bank_name, bank_value in named_writes.items():
                if bank_name in g_cued.bank_names:
                    g_cued = self.store_backend.append_write(g_cued, key, bank_value, bank_name=bank_name)
                if self.config.common_memory:
                    x_cued = g_cued
                elif bank_name in x_cued.bank_names:
                    x_cued = self.store_backend.append_write(x_cued, key, bank_value, bank_name=bank_name)

        return MemoryState(g_cued=g_cued, x_cued=x_cued)

    def _merge_memory_rows_impl(  # ---------------------------------------------------------------
        self, flag: Tensor, current: MemoryEntry, fresh: MemoryEntry,
    ) -> MemoryEntry:  # fmt: skip
        """Merge representation rows during partial reset using the factor-store strategy."""
        return self.store_backend.merge_rows(flag, current, fresh)

    def recall(  # --------------------------------------------------------------------------------
        self, *, read_cues: ReadCues, state: HPCState, role: RetrievalRole, read: MemoryRead,
    ) -> list[Tensor]:  # fmt: skip
        """Recall from factor memory using composer-produced retrieval evidence.

        Unlike the base implementation, this method accepts both resolved cue
        reads and targeted read operators because the factor-memory reader can
        consume either evidence form directly.
        """
        memory = state.memory.for_role(role)
        evidence = self.prepare_read(read_cues=read_cues, read=read)
        recalled = self.retrieval_module.recall_from_evidence(evidence, memory.as_factor_view())
        return self._unflatten_memory_code(recalled)


# =================================================================================================
__all__ = [
    "HPCAttention", "HPCAttentionSettings", "HPCAttractor", "HPCAttractorSettings",
]  # fmt: skip
