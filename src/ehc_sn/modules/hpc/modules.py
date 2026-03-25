"""HPC memory package composed from shared base, store, update, and query layers."""

from __future__ import annotations

from typing import Optional

from pydantic import Field
from torch import Tensor

from ehc_sn import utils
from ehc_sn.modules.hpc import update
from ehc_sn.modules.hpc._base import HPCBase, HPCCommonSettings, HPCState
from ehc_sn.modules.hpc.query import AttentionSettings, AttractorNetwork, AttractorSettings, FactorRetrieval
from ehc_sn.modules.hpc.update import FactorMemoryWrite, FactorMemoryWriteSettings, HebbianMemoryWrite, HebbianMemoryWriteSettings
from ehc_sn.types import Device, Dtype, MemoryEntry, MemoryState, RetrievalRole


# =================================================================================================
class HPCAttractorSettings(HPCCommonSettings):
    retrieval: AttractorSettings = Field(
        default_factory=AttractorSettings,
        description="Settings for attractor retrieval dynamics.",
    )
    write: HebbianMemoryWriteSettings = Field(
        default_factory=HebbianMemoryWriteSettings,
        description="Settings for Hebbian memory write.",
    )


# =================================================================================================
class HPCAttentionSettings(HPCCommonSettings):
    retrieval: AttentionSettings = Field(
        default_factory=AttentionSettings,
        description="Settings for attention retrieval.",
    )
    write: FactorMemoryWriteSettings = Field(
        default_factory=FactorMemoryWriteSettings,
        description="Settings for factor-memory write.",
    )


class HPCAttractor(HPCBase):
    def __init__(
        self,
        n_stages: int,
        f_initial: list[float],
        config: HPCAttractorSettings,
        *,
        device: Optional[Device] = None,
        dtype: Optional[Dtype] = None,
    ) -> None:
        super().__init__(config, device=device, dtype=dtype)

        masks = utils.update_to_masks(self.shape, update=utils.make_update_hierarchical(n_stages, self.n_freq))
        self.register_buffer("masks_hierarchical", masks, persistent=False)
        masks = utils.update_to_masks(self.shape, update=utils.make_update_full(n_stages, self.n_freq))
        self.register_buffer("masks_full", masks, persistent=False)

        mask = utils.make_hebbian_write_mask(n_stages, self.shape, f_initial)
        self.register_buffer("update_mask", mask, persistent=False)

        self.retrieval_module = AttractorNetwork(config.retrieval)
        self.write_module = HebbianMemoryWrite(config.write, device=device, dtype=dtype)
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
        return super().config  # type: ignore[return-value]

    def _init_memory_impl(self, batch_size: int, *, device: Optional[Device] = None) -> MemoryState:
        g_cued = self._store_factory.init_store(batch_size, device=device)
        x_cued = g_cued if self.config.common_memory else g_cued.clone()
        return MemoryState(g_cued=g_cued, x_cued=x_cued)

    def _set_runtime_impl(self, *, eta: float, hebbian_decay: float) -> None:
        self.write_module.runtime.eta = float(eta)
        self.write_module.runtime.hebbian_decay = float(hebbian_decay)

    def _recall_flat_impl(self, query: Tensor, memory: MemoryEntry, *, role: RetrievalRole) -> Tensor:
        masks = self.masks_hierarchical if role == "generative" else self.masks_full
        return self.retrieval_module(query, memory.as_linear_view(), masks=masks)

    def _update_memory_impl(
        self,
        memory: MemoryState,
        key: Tensor,
        g_value: Tensor,
        x_value: Optional[Tensor],
    ) -> MemoryState:
        g_cued = self._store_applier.apply(memory.g_cued, key, g_value, masked=True)
        x_cued = g_cued if self.config.common_memory else memory.x_cued
        if not self.config.common_memory and x_value is not None:
            x_cued = self._store_applier.apply(memory.x_cued, key, x_value, masked=False)
        return MemoryState(g_cued=g_cued, x_cued=x_cued)

    def _merge_memory_rows_impl(self, flag: Tensor, current: MemoryEntry, fresh: MemoryEntry) -> MemoryEntry:
        return self._reset_strategy.merge_rows(flag, current, fresh)


class HPCAttention(HPCBase):
    def __init__(
        self,
        n_stages: int,
        f_initial: list[float],
        config: HPCAttentionSettings,
        *,
        device: Optional[Device] = None,
        dtype: Optional[Dtype] = None,
    ) -> None:
        del n_stages, f_initial
        super().__init__(config, device=device, dtype=dtype)
        self.retrieval_module = FactorRetrieval(config.retrieval)
        self.write_module = FactorMemoryWrite(self.shape, config.write)
        store_components = update.build_factor_store_components(self.write_module)
        self._store_factory = store_components.store_factory
        self._store_applier = store_components.store_applier
        self._reset_strategy = store_components.reset_strategy

    @property
    def config(self) -> HPCAttentionSettings:
        return super().config  # type: ignore[return-value]

    def _init_memory_impl(self, batch_size: int, *, device: Optional[Device] = None) -> MemoryState:
        g_cued = self._store_factory.init_store(batch_size, device=device)
        x_cued = g_cued if self.config.common_memory else self._store_factory.init_store(batch_size, device=device)
        return MemoryState(g_cued=g_cued, x_cued=x_cued)

    def _set_runtime_impl(self, *, eta: float, hebbian_decay: float) -> None:
        del eta, hebbian_decay

    def _recall_flat_impl(self, query: Tensor, memory: MemoryEntry, *, role: RetrievalRole) -> Tensor:
        del role
        return self.retrieval_module(query, memory.as_factor_view())

    def _update_memory_impl(
        self,
        memory: MemoryState,
        key: Tensor,
        g_value: Tensor,
        x_value: Optional[Tensor],
    ) -> MemoryState:
        g_cued = self._store_applier.apply(memory.g_cued, key, g_value)
        x_cued = g_cued if self.config.common_memory else memory.x_cued
        if self.config.common_memory:
            x_cued = g_cued
        elif x_value is not None:
            x_cued = self._store_applier.apply(memory.x_cued, key, x_value)
        return MemoryState(g_cued=g_cued, x_cued=x_cued)

    def _merge_memory_rows_impl(self, flag: Tensor, current: MemoryEntry, fresh: MemoryEntry) -> MemoryEntry:
        return self._reset_strategy.merge_rows(flag, current, fresh)

    def recall(self, *, x_query, g_query, state: HPCState, role: RetrievalRole):
        p_query = self.query_policy(x_query=x_query, g_query=g_query, role=role)
        view = state.memory.for_role(role).as_factor_view()
        flat_query = self._flatten_memory_code(p_query)

        if self.config.retrieval.iterations == 1 or self.config.retrieval.recurrence == "none":
            recalled = self.retrieval_module(flat_query, view)
            return self._unflatten_memory_code(recalled)

        anchor_query = self._flatten_memory_code(g_query) if g_query is not None else flat_query
        anchor_logits = self.retrieval_module.compute_logits(anchor_query, view.keys)
        recalled = self.retrieval_module.recall_from_logits(
            anchor_logits,
            view.values,
            valid_mask=view.valid_mask,
            fallback_query=flat_query,
        )

        for _ in range(self.config.retrieval.iterations - 1):
            value_logits = self.retrieval_module.compute_logits(recalled, view.values)
            recurrent_logits = anchor_logits * value_logits
            recalled = self.retrieval_module.recall_from_logits(
                recurrent_logits,
                view.values,
                valid_mask=view.valid_mask,
                fallback_query=flat_query,
            )

        return self._unflatten_memory_code(recalled)


__all__ = [
    "HPCAttention",
    "HPCAttentionSettings",
    "HPCAttractor",
    "HPCAttractorSettings",
    "HPCBase",
    "HPCCommonSettings",
    "HPCState",
]  # fmt: skip
