"""Concrete hippocampal memory modules composed from shared HPC layers."""

from __future__ import annotations

from typing import Optional

from pydantic import Field
from torch import Tensor

from ehc_sn import utils
from ehc_sn.modules.hpc import update
from ehc_sn.modules.hpc._base import HPCBase, HPCCommonSettings, HPCState
from ehc_sn.modules.hpc.query import AttractorRead, AttractorReadSettings, FactorRead, FactorReadSettings
from ehc_sn.modules.hpc.query_policy import MemoryRead, ReadCues
from ehc_sn.modules.hpc.update import EpisodicWrite, EpisodicWriteSettings, HebbianWrite, HebbianWriteSettings
from ehc_sn.types import Device, Dtype, MemoryEntry, MemoryState, RetrievalRole


# =================================================================================================
class HPCAttractorSettings(HPCCommonSettings):
    """Settings for the attractor-based hippocampal implementation."""

    read: AttractorReadSettings = Field(
        default_factory=AttractorReadSettings,
        description="Settings for attractor retrieval dynamics.",
    )
    write: HebbianWriteSettings = Field(
        default_factory=HebbianWriteSettings,
        description="Settings for Hebbian memory write.",
    )


# =================================================================================================
class HPCAttentionSettings(HPCCommonSettings):
    """Settings for the attention-based hippocampal implementation."""

    read: FactorReadSettings = Field(
        default_factory=FactorReadSettings,
        description="Settings for factor-memory retrieval.",
    )
    write: EpisodicWriteSettings = Field(
        default_factory=EpisodicWriteSettings,
        description="Settings for factor-memory write.",
    )


# =================================================================================================
class HPCAttractor(HPCBase):
    """Attractor-based hippocampal memory with Hebbian write dynamics."""

    def __init__(  # ------------------------------------------------------------------------------
        self, n_stages: int, f_initial: list[float], config: HPCAttractorSettings, *,
        device: Optional[Device] = None, dtype: Optional[Dtype] = None,
    ) -> None:  # fmt: skip
        """Initialize attractor retrieval, Hebbian writes, and masked update helpers."""
        super().__init__(config, device=device, dtype=dtype)

        masks = utils.update_to_masks(self.shape, update=utils.make_update_hierarchical(n_stages, self.n_freq))
        self.register_buffer("masks_hierarchical", masks, persistent=False)
        masks = utils.update_to_masks(self.shape, update=utils.make_update_full(n_stages, self.n_freq))
        self.register_buffer("masks_full", masks, persistent=False)

        mask = utils.make_hebbian_write_mask(n_stages, self.shape, f_initial)
        self.register_buffer("update_mask", mask, persistent=False)

        self.retrieval_module = AttractorRead(config.read)
        self.write_module = HebbianWrite(config.write, device=device, dtype=dtype)
        store_components = update.build_hebbian_store_components(
            self.write_module,
            emit_store=config.write.emit_store,
            feature_dim=sum(self.shape),
            update_mask=self.update_mask,
            n_stages=n_stages,
            shape=self.shape,
            f_initial=f_initial,
        )
        self._store_factory = store_components.store_factory
        self._store_applier = store_components.store_applier
        self._reset_strategy = store_components.reset_strategy

    @property
    def config(self) -> HPCAttractorSettings:
        """Return the typed config for the attractor memory module."""
        return super().config  # type: ignore[return-value]

    def _init_memory_impl(  # --------------------------------------------------------------------
        self, batch_size: int, *,
        device: Optional[Device] = None,
    ) -> MemoryState:  # fmt: skip
        """Initialize linear Hebbian stores for g-cued and optional x-cued memory."""
        g_cued = self._store_factory.init_store(batch_size, device=device)
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
        g_cued = self._store_applier.apply(memory.g_cued, key, g_value, masked=True)
        x_cued = g_cued if self.config.common_memory else memory.x_cued
        if not self.config.common_memory and x_value is not None:
            x_cued = self._store_applier.apply(memory.x_cued, key, x_value, masked=False)
        return MemoryState(g_cued=g_cued, x_cued=x_cued)

    def _merge_memory_rows_impl(  # ---------------------------------------------------------------
        self, flag: Tensor, current: MemoryEntry, fresh: MemoryEntry,
    ) -> MemoryEntry:  # fmt: skip
        """Merge representation rows during partial reset using the Hebbian reset strategy."""
        return self._reset_strategy.merge_rows(flag, current, fresh)


# =================================================================================================
class HPCAttention(HPCBase):
    """Attention-based hippocampal memory with explicit factor slots."""

    def __init__(  # ------------------------------------------------------------------------------
        self, n_stages: int, f_initial: list[float], config: HPCAttentionSettings, *,
        device: Optional[Device] = None, dtype: Optional[Dtype] = None,
    ) -> None:  # fmt: skip
        """Initialize factor-slot retrieval and write modules."""
        del n_stages, f_initial
        super().__init__(config, device=device, dtype=dtype)
        self.retrieval_module = FactorRead(config.read)
        self.write_module = EpisodicWrite(self.shape, config.write)
        store_components = update.build_factor_store_components(self.write_module)
        self._store_factory = store_components.store_factory
        self._store_applier = store_components.store_applier
        self._reset_strategy = store_components.reset_strategy

    @property
    def config(self) -> HPCAttentionSettings:
        """Return the typed config for the attention memory module."""
        return super().config  # type: ignore[return-value]

    def _init_memory_impl(  # --------------------------------------------------------------------
        self, batch_size: int, *,
        device: Optional[Device] = None,
    ) -> MemoryState:  # fmt: skip
        """Initialize factor stores for g-cued memory and optional separate x-cued memory."""
        g_cued = self._store_factory.init_store(batch_size, device=device)
        x_cued = g_cued if self.config.common_memory else self._store_factory.init_store(batch_size, device=device)
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
        g_cued = self._store_applier.apply(memory.g_cued, key, g_value)
        x_cued = g_cued if self.config.common_memory else memory.x_cued
        if self.config.common_memory:
            x_cued = g_cued
        elif x_value is not None:
            x_cued = self._store_applier.apply(memory.x_cued, key, x_value)

        if named_writes:
            available_bank_names = set(memory.g_cued.bank_names) | set(memory.x_cued.bank_names)
            missing_bank_names = sorted(set(named_writes) - available_bank_names)
            if missing_bank_names:
                names = ", ".join(repr(name) for name in missing_bank_names)
                raise ValueError(f"Named factor-memory banks {names} are not available for writing.")

            for bank_name, bank_value in named_writes.items():
                if bank_name in g_cued.bank_names:
                    g_cued = self._store_applier.apply(g_cued, key, bank_value, bank_name=bank_name)
                if self.config.common_memory:
                    x_cued = g_cued
                elif bank_name in x_cued.bank_names:
                    x_cued = self._store_applier.apply(x_cued, key, bank_value, bank_name=bank_name)

        return MemoryState(g_cued=g_cued, x_cued=x_cued)

    def _merge_memory_rows_impl(  # ---------------------------------------------------------------
        self, flag: Tensor, current: MemoryEntry, fresh: MemoryEntry,
    ) -> MemoryEntry:  # fmt: skip
        """Merge representation rows during partial reset using the factor-store strategy."""
        return self._reset_strategy.merge_rows(flag, current, fresh)

    def recall(  # --------------------------------------------------------------------------------
        self, *, read_cues: ReadCues, state: HPCState, role: RetrievalRole, read: MemoryRead,
    ):  # fmt: skip
        """Recall from factor memory through composer-produced retrieval evidence."""
        memory = state.memory.for_role(role)
        evidence = self.prepare_read(read_cues=read_cues, read=read)
        recalled = self.retrieval_module.recall_from_evidence(evidence, memory.as_factor_view())
        return self._unflatten_memory_code(recalled)


# =================================================================================================
__all__ = [
    "HPCAttention", "HPCAttentionSettings", "HPCAttractor", "HPCAttractorSettings",
]  # fmt: skip
