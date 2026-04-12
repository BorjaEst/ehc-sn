"""EHC v1 model preserving the staged TEM backbone with explicit connection classes."""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, TypeAlias

import torch
from pydantic import AliasChoices, BaseModel, Field, computed_field, model_validator
from torch import Tensor, nn

from ehc_sn.models.ehc.core.ehc_base import EHCControlStep, EHCMemoryStep, EHCOutput, EHCTransitionPlan
from ehc_sn.modules.autoencoder import Autoencoder, AutoencoderSettings
from ehc_sn.modules.hpc import HPCAttention, HPCAttentionSettings, HPCState
from ehc_sn.modules.hpc import SensoryRead as HPCSensoryRead
from ehc_sn.modules.hpc.query_policy import ReadCues, TargetRead
from ehc_sn.modules.lec import LECModel, LECSettings, LECState
from ehc_sn.modules.mec import MECModel, MECSettings, MECState
from ehc_sn.modules.pfc import PFCModel, PFCSettings, PFCState
from ehc_sn.modules.projection import ProjectionModule, ProjectionSettings
from ehc_sn.modules.str import STRModelLinear, STRSettings, STRState
from ehc_sn.modules.transformer import TransformerSequenceSummary, TransformerSequenceSummaryConfig
from ehc_sn.types import Device, Dtype, MemoryState, MultiScaleCode
from ehc_sn.utils import trunc_normal_init_
from ehc_sn.utils.detach import DetachMixin
from ehc_sn.utils.tensor_ops import (
    apply_per_band,
    as_batch_column,
    as_optional_batch_column,
    as_optional_feature_matrix,
    multiscale_mean_abs,
    multiscale_row_mse,
)

# Community-standard map-style batch: plain dict returned by MazeDataset / DataLoader.
Batch: TypeAlias = Dict[str, Tensor]
ObsLogits = tuple[Tensor, Tensor, Tensor]  # (inference, retrieved, ancestral)
GridCodes = tuple[MultiScaleCode, MultiScaleCode]  # (posterior, prior)
PlaceCodes = tuple[MultiScaleCode, MultiScaleCode, Optional[MultiScaleCode]]  # (posterior, prior, sensory-cued retrieval)

EHC_BAND_TOKEN_GROUPS = 6
EHC_SCALAR_TOKEN_COUNT = 6


# =================================================================================================
class ModelSettings_V1(BaseModel, extra="forbid", strict=False):
    """Canonical EHC v1 model settings.

    EHC v1 preserves the staged TEM memory path while adding a fixed semantic
    cortical workspace, a dedicated cortical sequence summary path, and a
    target-bank cortical cue pathway.
    """

    # ~~~~ Core dimensions and hyperparameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    observation_dim: int = Field(
        ...,
        ge=1,
        description="Dimensionality of raw observations from the environment.",
    )
    action_count: int = Field(
        ...,
        ge=1,
        description="Number of discrete motor actions in the environment.",
    )
    internal_action_count: int = Field(
        ...,
        ge=1,
        description="Number of internal control actions scored by the PFC/STR pathway.",
    )
    external_context_dim: int = Field(
        default=1,
        ge=1,
        description="Width of the optional continuous exogenous context payload consumed by the cortical cue interface.",
    )
    sequence_vocab_size: int = Field(
        ...,
        ge=1,
        description="Vocabulary size for optional serialized discrete input_ids consumed by the cortical sequence summary path.",
    )
    sequence_summary_layers: int = Field(
        default=1,
        ge=1,
        validation_alias=AliasChoices("sequence_summary_layers", "sequence_encoder_layers"),
        description="Number of transformer blocks in the dedicated cortical sequence summary stack.",
    )
    pfc_hpc_cortical_cue_summary_layers: int = Field(
        default=1,
        ge=1,
        validation_alias=AliasChoices("pfc_hpc_cortical_cue_summary_layers", "cue_conditioning_layers"),
        description="Number of transformer blocks used to summarize pre-HPC cortical evidence for PFCHPCCorticalCuePathway.",
    )
    f_initial: list[float] = Field(
        default_factory=lambda: [0.99, 0.3, 0.09, 0.5, 0.4],
        min_length=1,
        description="Initial feature frequencies resolved across MEC/HPC modules.",
    )
    enable_sensory_recall: bool = Field(
        default=True,
        description="Whether phase-1 sensory-cued recall is enabled in the preserved TEM path.",
    )

    # ~~~~
    hpc: HPCAttentionSettings = Field(
        ...,
        description="Settings for the attention-based hippocampal module.",
    )
    lec: LECSettings = Field(
        ...,
        description="Settings for the LEC module.",
    )
    mec: MECSettings = Field(
        ...,
        description="Settings for the MEC module.",
    )
    pfc: PFCSettings = Field(
        ...,
        description="Settings for the recurrent cortical workspace module.",
    )
    str: STRSettings = Field(
        ...,
        description="Settings for the narrow striatal reward/value head.",
    )
    autoencoder: AutoencoderSettings = Field(
        ...,
        description="Settings for the observation codec / sensory transducer.",
    )
    lec_to_hpc_x_projection: ProjectionSettings = Field(
        default_factory=lambda: ProjectionSettings(mode="tiling", learnable=False),
        validation_alias=AliasChoices("lec_to_hpc_x_projection", "projection_lec"),
        description="Inter-region code projection settings mapping LEC features into HPC query space.",
    )
    mec_to_hpc_g_projection: ProjectionSettings = Field(
        default_factory=lambda: ProjectionSettings(mode="low_rank", learnable=False),
        validation_alias=AliasChoices("mec_to_hpc_g_projection", "projection_mec"),
        description="Inter-region code projection settings mapping MEC codes into HPC query space.",
    )

    @model_validator(mode="after")
    def validate_model(self) -> "ModelSettings_V1":
        if any(not (0.0 < value < 1.0) for value in self.f_initial):
            raise ValueError("f_initial values must be strictly between 0 and 1.")
        if self.pfc.cortex.pos_encodings != "rope":
            raise ValueError("EHC v1 cortical sequence summary path currently requires pfc.cortex.pos_encodings='rope'.")
        if len(self.mec_shape) != len(self.f_initial):
            raise ValueError("The resolved MEC stage count must match len(f_initial).")
        if len(self.hpc.shape) != self.n_total_freq:
            raise ValueError("len(hpc.shape) must equal the derived total MEC frequency count.")
        if self.pfc.seq_length != self.seq_length:
            raise ValueError(f"pfc.seq_length must equal the derived semantic layout length {self.seq_length}.")
        if self.pfc.hidden_size != self.str.n_features:
            raise ValueError("str.n_features must equal the PFC hidden size.")
        if self.pfc.value_head.hidden_size != self.hidden_size:
            raise ValueError("pfc.value_head.hidden_size must equal the cortical hidden size.")
        if self.pfc.value_head.n_actions != self.internal_action_count:
            raise ValueError("pfc.value_head.n_actions must equal internal_action_count.")
        if self.str.n_actions != self.internal_action_count:
            raise ValueError("str.n_actions must equal internal_action_count.")
        if "c" not in self.hpc.store.bank_names:
            raise ValueError("hpc.store.bank_names must include 'c' for the cortical target bank.")
        return self

    @computed_field
    @property
    def lec_shape(self) -> list[int]:
        """Return the resolved LEC feature shape across all frequencies."""
        return [self.lec.feature_dim] * self.n_total_freq

    @computed_field
    @property
    def mec_shape(self) -> list[int]:
        """Return the resolved MEC shape including any OVC expansion."""
        return self.mec.mec_shape

    @computed_field
    @property
    def n_total_freq(self) -> int:
        """Return the total number of resolved MEC/HPC frequencies."""
        return self.mec.n_total_freq

    @computed_field
    @property
    def seq_length(self) -> int:
        """Return the fixed semantic cortical workspace length."""
        return EHC_BAND_TOKEN_GROUPS * self.n_total_freq + EHC_SCALAR_TOKEN_COUNT

    @computed_field
    @property
    def hidden_size(self) -> int:
        """Return the shared cortical hidden size."""
        return self.pfc.hidden_size

    @computed_field
    @property
    def init_std(self) -> float:
        """Return the default initialization scale for local EHC projections."""
        return 1.0 / math.sqrt(self.hidden_size)

    @classmethod
    def from_config(cls, path: Path) -> "ModelSettings_V1":
        """Load model settings from a TOML configuration file."""
        return cls.model_validate(tomllib.load(Path(path).open("rb")))

    @computed_field
    @property
    def sequence_summary_config(self) -> TransformerSequenceSummaryConfig:
        """Return the config for the dedicated cortical sequence summary stack."""
        return TransformerSequenceSummaryConfig(block=self.pfc.cortex, n_layers=self.sequence_summary_layers)

    @computed_field
    @property
    def pfc_hpc_cortical_cue_summary_config(self) -> TransformerSequenceSummaryConfig:
        """Return the config for the PFCHPCCorticalCuePathway summarizer."""
        return TransformerSequenceSummaryConfig(
            block=self.pfc.cortex,
            n_layers=self.pfc_hpc_cortical_cue_summary_layers,
        )


# =================================================================================================
@dataclass
class EHCState(DetachMixin):
    """Container for the recurrent state owned by EHC region modules only."""

    pfc: PFCState  # State of the Prefrontal Cortex
    str: STRState  # State of the Striatum
    lec: LECState  # State of the Lateral Entorhinal Cortex
    mec: MECState  # State of the Medial Entorhinal Cortex
    hpc: HPCState  # State of the Hippocampus


def _build_band_projectors(
    shape: list[int],
    hidden_size: int,
    *,
    device: Optional[Device],
    dtype: Optional[Dtype],
    bias: bool = True,
    inverse: bool = False,
) -> nn.ModuleList:
    """Return one linear projection per frequency band."""
    if inverse:
        return nn.ModuleList([nn.Linear(hidden_size, width, bias=bias, device=device, dtype=dtype) for width in shape])
    return nn.ModuleList([nn.Linear(width, hidden_size, bias=bias, device=device, dtype=dtype) for width in shape])


def _reset_linear_module(module: nn.Linear, *, init_std: float) -> None:
    """Initialize one linear module using the local EHC convention."""
    trunc_normal_init_(module.weight, std=init_std)
    if module.bias is not None:
        module.bias.data.zero_()


def _reset_linear_modules(modules: nn.ModuleList, *, init_std: float) -> None:
    """Initialize every linear module in one registered module list."""
    for module in modules:
        _reset_linear_module(module, init_std=init_std)


@dataclass(frozen=True)
class _EHCReadPathway:
    """Frozen container for one typed HPC retrieval operator."""

    read: TargetRead


class _EHCProjectionBundle(nn.Module):
    """Own the thin stateless code transforms composed by EHC v1."""

    def __init__(
        self,
        *,
        mec: MECModel,
        lec: LECModel,
        hpc: HPCAttention,
        config: ModelSettings_V1,
    ) -> None:
        super().__init__()
        self.mec_to_hpc_g = ProjectionModule(mec, hpc, config.mec_to_hpc_g_projection)
        self.lec_to_hpc_x = ProjectionModule(lec, hpc, config.lec_to_hpc_x_projection)


class _EHCSequencePathway(nn.Module):
    """Own the optional serialized-sequence conditioning path."""

    def __init__(
        self,
        config: ModelSettings_V1,
        *,
        device: Optional[Device] = None,
        dtype: Optional[Dtype] = None,
    ) -> None:
        super().__init__()
        self.embedding_scale = math.sqrt(float(config.hidden_size))
        self.token_embedding = nn.Embedding(
            config.sequence_vocab_size,
            config.hidden_size,
            padding_idx=0,
            device=device,
            dtype=dtype,
        )
        self.summarizer = TransformerSequenceSummary(config.sequence_summary_config, device=device, dtype=dtype)

    def reset_parameters(self, *, init_std: float) -> None:
        """Initialize the sequence-pathway modules owned by EHC."""
        self.summarizer.reset_parameters()
        trunc_normal_init_(self.token_embedding.weight, std=init_std)
        if self.token_embedding.padding_idx is not None:
            self.token_embedding.weight.data[self.token_embedding.padding_idx].zero_()


class _EHCPFCHPCCuePathway(nn.Module):
    """Own the learnable parts of the PFC-to-HPC cortical cue pathway."""

    def __init__(
        self,
        config: ModelSettings_V1,
        *,
        device: Optional[Device] = None,
        dtype: Optional[Dtype] = None,
    ) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        n_freq = config.n_total_freq
        self.summarizer = TransformerSequenceSummary(
            config.pfc_hpc_cortical_cue_summary_config,
            device=device,
            dtype=dtype,
        )
        self.query = _build_band_projectors(
            [hidden_size] * n_freq,
            hidden_size,
            device=device,
            dtype=dtype,
            bias=False,
        )
        self.key = _build_band_projectors(
            [hidden_size] * n_freq,
            hidden_size,
            device=device,
            dtype=dtype,
            bias=False,
        )
        self.output = _build_band_projectors(
            config.hpc.shape,
            hidden_size,
            device=device,
            dtype=dtype,
            inverse=True,
        )
        self.gate = nn.Linear(config.external_context_dim + 4, 1, device=device, dtype=dtype)

    def reset_parameters(self, *, init_std: float) -> None:
        """Initialize the learnable cue-pathway surfaces owned by EHC."""
        self.summarizer.reset_parameters()
        _reset_linear_modules(self.query, init_std=init_std)
        _reset_linear_modules(self.key, init_std=init_std)
        _reset_linear_modules(self.output, init_std=init_std)
        _reset_linear_module(self.gate, init_std=init_std)


class _EHCPathwayBundle(nn.Module):
    """Group EHC pathways by circuit semantics rather than by primitive type."""

    def __init__(
        self,
        config: ModelSettings_V1,
        *,
        device: Optional[Device] = None,
        dtype: Optional[Dtype] = None,
    ) -> None:
        super().__init__()
        self.sequence = _EHCSequencePathway(config, device=device, dtype=dtype)
        self.pfc_hpc_cue = _EHCPFCHPCCuePathway(config, device=device, dtype=dtype)
        self.hpc_pfc_cortical = _EHCReadPathway(
            read=TargetRead(
                kind="target",
                sources=("g", "x"),
                target="c",
                target_init="c",
                read_bank="c",
            )
        )
        self.hpc_pfc_sensory = _EHCReadPathway(
            read=TargetRead(
                kind="target",
                sources=("g",),
                target="x",
                target_init="x",
            )
        )
        self.pfc_hpc_replay = _EHCReadPathway(
            read=TargetRead(
                kind="target",
                sources=("g",),
                target="x",
                target_init="c",
            )
        )

    def reset_parameters(self, *, init_std: float) -> None:
        """Initialize the learnable pathway-owned surfaces."""
        self.sequence.reset_parameters(init_std=init_std)
        self.pfc_hpc_cue.reset_parameters(init_std=init_std)


class _EHCWorkspaceBundle(nn.Module):
    """Own the tokenization surfaces that pack semantic cortical workspaces."""

    def __init__(
        self,
        config: ModelSettings_V1,
        *,
        device: Optional[Device] = None,
        dtype: Optional[Dtype] = None,
    ) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        self.external_context = nn.Linear(
            config.external_context_dim,
            hidden_size,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.previous_action = nn.Embedding(config.action_count, hidden_size, device=device, dtype=dtype)
        self.episode_start = nn.Linear(1, hidden_size, bias=False, device=device, dtype=dtype)
        self.landmark = nn.Linear(1, hidden_size, bias=False, device=device, dtype=dtype)
        self.disagreement = nn.Linear(3, hidden_size, bias=False, device=device, dtype=dtype)
        self.grid_prior = _build_band_projectors(config.mec_shape, hidden_size, device=device, dtype=dtype)
        self.sensory_query = _build_band_projectors(config.hpc.shape, hidden_size, device=device, dtype=dtype)
        self.sensory_recall = _build_band_projectors(config.hpc.shape, hidden_size, device=device, dtype=dtype)
        self.grid_post = _build_band_projectors(config.mec_shape, hidden_size, device=device, dtype=dtype)
        self.cue_proposal = _build_band_projectors(config.hpc.shape, hidden_size, device=device, dtype=dtype)
        self.cue_memory = _build_band_projectors(config.hpc.shape, hidden_size, device=device, dtype=dtype)

    def reset_parameters(self, *, init_std: float) -> None:
        """Initialize the workspace tokenizers owned by EHC."""
        trunc_normal_init_(self.previous_action.weight, std=init_std)
        for module in (
            self.external_context,
            self.episode_start,
            self.landmark,
            self.disagreement,
        ):
            _reset_linear_module(module, init_std=init_std)
        for modules in (
            self.grid_prior,
            self.sensory_query,
            self.sensory_recall,
            self.grid_post,
            self.cue_proposal,
            self.cue_memory,
        ):
            _reset_linear_modules(modules, init_std=init_std)


class _EHCHeadBundle(nn.Module):
    """Own the task-facing learned heads exposed by EHC v1."""

    def __init__(
        self,
        config: ModelSettings_V1,
        *,
        device: Optional[Device] = None,
        dtype: Optional[Dtype] = None,
    ) -> None:
        super().__init__()
        self.motor_action = nn.Linear(config.hidden_size, config.action_count, device=device, dtype=dtype)

    def reset_parameters(self, *, init_std: float) -> None:
        """Initialize the task-facing heads owned directly by EHC."""
        _reset_linear_module(self.motor_action, init_std=init_std)


@dataclass(frozen=True)
class _EHCNormalizedInputs:
    """Canonical EHC batch payload after model-local normalization.

    This object stores only caller-provided EHC inputs after strict validation,
    shape coercion, dtype normalization, and device alignment to the
    observation tensor. Derived features such as sequence summaries remain
    outside this contract boundary.
    """

    observation: Tensor
    previous_action: Tensor
    episode_start: Tensor
    landmark_id: Tensor
    external_context: Tensor
    input_ids: Optional[Tensor]

    @property
    def batch_size(self) -> int:
        """Return the authoritative batch size derived from observation."""
        return int(self.observation.shape[0])

    @property
    def device(self) -> torch.device:
        """Return the authoritative runtime device derived from observation."""
        return self.observation.device


class EHCModelV1(nn.Module):
    """EHC v1 backbone with a structured step output and no environment stepping."""

    def __init__(  # ------------------------------------------------------------------------------
        self, config: ModelSettings_V1, *,
        device: Optional[Device] = None, dtype: Optional[Dtype] = None,
    ) -> None:  # fmt: skip
        """Construct the EHC backbone from the resolved EHC v1 model settings.

        The staged TEM memory path is preserved exactly. Cortical context is
        introduced through a slot-derived cue proposal, a reinstated target-bank
        cortical trace, and a transient routed control cue that biases replay
        without being written back into hippocampal memory.
        """
        super().__init__()
        self._config = config
        n_freq, n_actions = config.n_total_freq, config.action_count
        f_initial = config.f_initial

        # Sensory transducer plus region modules / dynamical cores.
        self.autoencoder = Autoencoder(config.observation_dim, config.lec.feature_dim, config.autoencoder)
        self.pfc = PFCModel(config.pfc, device=device, dtype=dtype)  # Reasoning module with embedded inputs
        self.str = STRModelLinear(config.str, device=device, dtype=dtype)  # Reward estimator
        self.hpc = HPCAttention(n_freq, f_initial, config.hpc, device=device, dtype=dtype)
        self.mec = MECModel(n_actions, config.hpc.shape, f_initial, config.mec, device=device, dtype=dtype)
        self.lec = LECModel(f_initial, config.lec, device=device, dtype=dtype)

        # Model-local grouped owners for projections, pathways, workspace packing, and heads.
        self.projections = _EHCProjectionBundle(mec=self.mec, lec=self.lec, hpc=self.hpc, config=config)
        self.pathways = _EHCPathwayBundle(config, device=device, dtype=dtype)
        self.workspace = _EHCWorkspaceBundle(config, device=device, dtype=dtype)
        self.heads = _EHCHeadBundle(config, device=device, dtype=dtype)

        self.reset_parameters()

    @property
    def config(self) -> ModelSettings_V1:
        """Return the parsed EHC v1 model settings."""
        return self._config

    def reset_parameters(self) -> None:
        """Initialize the grouped learnable surfaces owned directly by EHC."""
        init_std = self.config.init_std
        self.pathways.reset_parameters(init_std=init_std)
        self.workspace.reset_parameters(init_std=init_std)
        self.heads.reset_parameters(init_std=init_std)

    def init_state(  # ----------------------------------------------------------------------------
        self, batch_size: int, *, memory: Optional[MemoryState] = None,
        device: Optional[Device] = None, dtype: Optional[Dtype] = None,
    ) -> EHCState:  # fmt: skip
        """Create an initial recurrent EHC state."""
        memory = memory if memory is not None else self.hpc.init_memory(batch_size=batch_size, device=device)
        return EHCState(
            pfc=self.pfc.init_state(batch_size),
            str=self.str.init_state(batch_size),
            lec=self.lec.init_state(batch_size, device=device),
            mec=self.mec.init_state(batch_size, device=device),
            hpc=self.hpc.init_state(batch_size, device=device, memory=memory),
        )

    def reset_state(  # ---------------------------------------------------------------------------
        self, reset_flag: Tensor, state: EHCState,
    ) -> EHCState:  # fmt: skip
        """Reset flagged rows to a fresh episode state while preserving active rows."""
        device = state.lec.features[0].device
        reset_flag = reset_flag.to(device=device, dtype=torch.bool).view(-1)
        if not torch.any(reset_flag):
            return state

        return EHCState(
            pfc=self.pfc.reset_state(state.pfc, reset_flag),
            str=self.str.reset_state(state.str, reset_flag),
            lec=self.lec.reset_state(state.lec, reset_flag),
            mec=self.mec.reset_state(state.mec, reset_flag),
            hpc=self.hpc.reset_state(state.hpc, reset_flag),
        )

    def set_runtime(  # ---------------------------------------------------------------------------
        self, eta: float, hebbian_decay: float, p2g_uncertainty_offset: float,
    ) -> None:  # fmt: skip
        """Apply runtime parameters to MEC and HPC without touching cortical adapters."""
        self.mec.set_runtime(p2g_uncertainty_offset=p2g_uncertainty_offset)
        self.hpc.set_runtime(eta=eta, hebbian_decay=hebbian_decay)

    def _prepare_forward_inputs(self, batch: Batch) -> _EHCNormalizedInputs:
        """Validate and normalize one caller-provided EHC batch payload.

        Observation is the authoritative batch and device anchor. Every other
        field is validated against that anchor and normalized into explicit
        batch-first tensors with semantic dtypes.
        """
        observation = batch.get("observation")
        if observation is None:
            raise ValueError("batch must include required field 'observation'.")
        if observation.ndim != 2 or int(observation.shape[1]) != self.config.observation_dim:
            raise ValueError("observation must have shape " f"(B, {self.config.observation_dim}), got {tuple(observation.shape)}.")
        observation = observation.to(dtype=torch.float32)

        previous_action_value = batch.get("previous_action")
        if previous_action_value is None:
            raise ValueError("batch must include required field 'previous_action'.")

        batch_size = int(observation.shape[0])
        device = observation.device
        previous_action = as_batch_column(
            previous_action_value,
            batch_size=batch_size,
            device=device,
            dtype=torch.int64,
            name="previous_action",
        )
        episode_start = as_optional_batch_column(
            batch.get("episode_start"),
            batch_size=batch_size,
            device=device,
            dtype=torch.bool,
            name="episode_start",
            fill_value=False,
        )
        landmark_id = as_optional_batch_column(
            batch.get("landmark_id"),
            batch_size=batch_size,
            device=device,
            dtype=torch.int64,
            name="landmark_id",
            fill_value=0,
        )
        external_context = as_optional_feature_matrix(
            batch.get("external_context"),
            batch_size=batch_size,
            width=self.config.external_context_dim,
            device=device,
            dtype=torch.float32,
            name="external_context",
        )
        input_ids = self._prepare_input_ids(batch.get("input_ids"), batch_size, device)

        return _EHCNormalizedInputs(
            observation=observation,
            previous_action=previous_action,
            episode_start=episode_start,
            landmark_id=landmark_id,
            external_context=external_context,
            input_ids=input_ids,
        )

    def forward(  # -------------------------------------------------------------------------------
        self, batch: Batch, state: Optional[EHCState] = None,
    ) -> EHCOutput:  # fmt: skip
        """Run one EHC step from the current payload and recurrent state.

        ``state`` is assumed to have already been reset for any fresh episode
        rows by the caller. When ``state`` is ``None``, a fresh full-batch state
        is allocated and the normal single-step EHC transition is executed. The
        method stops at emitting a structured step object; it never advances an
        environment.

        Args:
            batch: Map-style batch with current-step ``observation`` and
                ``previous_action`` required, plus optional ``episode_start``,
                ``landmark_id``, ``external_context``, and ``input_ids``.
            state: Optional recurrent EHC state carried across controller steps.
        """

        inputs = self._prepare_forward_inputs(batch)
        observation = inputs.observation
        previous_action = inputs.previous_action
        episode_start = inputs.episode_start
        landmark_id = inputs.landmark_id
        external_context = inputs.external_context
        sequence_summary = self._summarize_input_ids(inputs.input_ids, inputs.batch_size, inputs.device)

        if state is None:
            state = self.init_state(inputs.batch_size, memory=None, device=inputs.device)

        # 1. Compute the structural prior and sensory query from current inputs.
        observation_embedding = self.autoencoder.encode(observation)
        grid_prior, state.mec = self.mec.generative(previous_action, episode_start, landmark_id, state.mec)
        structural_query_prior = self.projections.mec_to_hpc_g(grid_prior)
        lec_features_post, state.lec = self.lec.inference(observation_embedding, state.lec)
        sensory_query = self.projections.lec_to_hpc_x(lec_features_post)

        # 2. Summarize current pre-HPC context and propose a cortical cue from prior PFC slots.
        cue_context_tokens = [
            *apply_per_band(grid_prior, self.workspace.grid_prior, dtype=torch.float32),
            *apply_per_band(sensory_query, self.workspace.sensory_query, dtype=torch.float32),
            self.workspace.external_context(external_context),
            sequence_summary.to(torch.float32),
            self.workspace.previous_action(previous_action.squeeze(-1)),
            self.workspace.episode_start(episode_start.to(torch.float32)),
            self.workspace.landmark(landmark_id.to(torch.float32)),
        ]
        cue_context_summary = self.pathways.pfc_hpc_cue.summarizer(torch.stack(cue_context_tokens, dim=1))
        cue_proposal = self._propose_cortical_cue(state.pfc, cue_context_summary)

        # 3. Reinstate the cortical target-bank trace and route proposal versus memory.
        cue_read_cues = ReadCues(families={"x": sensory_query, "g": structural_query_prior, "c": cue_proposal})
        cue_reinstated = self.hpc.recall(
            read_cues=cue_read_cues,
            state=state.hpc,
            role="generative",
            read=self.pathways.hpc_pfc_cortical.read,
        )
        cue_gate_features = torch.cat(
            [
                external_context,
                multiscale_mean_abs(structural_query_prior).unsqueeze(-1),
                multiscale_mean_abs(sensory_query).unsqueeze(-1),
                multiscale_row_mse(cue_proposal, cue_reinstated).unsqueeze(-1),
                multiscale_mean_abs(cue_proposal).unsqueeze(-1),
            ],
            dim=1,
        )
        cue_gate = torch.sigmoid(self.pathways.pfc_hpc_cue.gate(cue_gate_features.to(torch.float32)))
        cue_routed = [torch.lerp(proposal, reinstated, cue_gate) for proposal, reinstated in zip(cue_proposal, cue_reinstated, strict=True)]

        # 4. Preserve TEM ordering: sensory recall, posterior correction, then phase-2 HPC transition.
        sensory = self.hpc.read_sensory(
            HPCSensoryRead(
                state=state.hpc,
                read_cues=ReadCues(families={"x": sensory_query, "g": structural_query_prior, "c": cue_routed}),
                read=self.pathways.hpc_pfc_sensory.read,
                enable_sensory_recall=self.config.enable_sensory_recall,
            )
        )

        grid_post, state.mec = self.mec.inference(sensory.recall, landmark_id=landmark_id, state=state.mec)
        structural_query_post = self.projections.mec_to_hpc_g(grid_post)
        transition = EHCTransitionPlan(
            sensory=sensory,
            grid_prior=grid_prior,
            grid_query_prior=structural_query_prior,
            grid_post=grid_post,
            grid_query_posterior=structural_query_post,
            generative_read=self.pathways.pfc_hpc_replay.read,
            named_writes={"c": cue_reinstated},
        )

        # Bank c stores reinstated memory, not the transient routed cue.
        hpc_step = self.hpc.transition(transition.to_hpc_transition(state.hpc))
        state.hpc = hpc_step.state
        place_sensory = hpc_step.sensory.recall

        # 5. Decode observations, update cortical control, and pack the structured output.
        logits_inference = self._decode_observation_logits_from_place(hpc_step.place_post)
        logits_retrieved = self._decode_observation_logits_from_place(hpc_step.place_retrieved)
        logits_ancestral = self._decode_observation_logits_from_place(hpc_step.place_prior)

        workspace_tokens = self._build_cortical_workspace_tokens(
            grid_prior=grid_prior,
            place_query_from_obs=sensory_query,
            place_sensory=place_sensory,
            grid_post=grid_post,
            c_prop=cue_proposal,
            c_mem=cue_reinstated,
            external_context_features=external_context,
            sequence_summary=sequence_summary,
            previous_action=previous_action,
            episode_start=episode_start,
            landmark_id=landmark_id,
        )
        state_pfc, z_H, internal_control_logits = self.pfc(workspace_tokens, state=state.pfc)
        theta_summary = z_H[:, 0]
        motor_logits = self.heads.motor_action(theta_summary.to(torch.float32))
        state_str, reward_logits = self.str(theta_summary.detach(), internal_control_logits, state.str)

        new_state = EHCState(pfc=state_pfc, str=state_str, lec=state.lec, mec=state.mec, hpc=state.hpc)
        memory = EHCMemoryStep(
            c_prop=cue_proposal,
            c_mem=cue_reinstated,
            c_use=cue_routed,
            c_gate=cue_gate,
            grid_prior=grid_prior,
            grid_post=grid_post,
            place_query_from_obs=sensory_query,
            place_query_from_grid_prior=structural_query_prior,
            place_query_from_grid_post=structural_query_post,
            place_sensory=place_sensory,
            place_prior=hpc_step.place_prior,
            place_retrieved=hpc_step.place_retrieved,
            place_post=hpc_step.place_post,
            named_writes={"c": cue_reinstated},
        )
        control = EHCControlStep(
            z_H=z_H,
            theta_summary=theta_summary,
            internal_control_logits=internal_control_logits,
            motor_logits=motor_logits,
            reward_logits=reward_logits,
        )
        return EHCOutput(
            state=new_state,
            obs_logits=(logits_inference, logits_retrieved, logits_ancestral),
            memory=memory,
            control=control,
        )

    def _prepare_input_ids(self, input_ids: Optional[Tensor], batch_size: int, device: torch.device) -> Optional[Tensor]:
        """Validate and move optional serialized ``input_ids`` onto the model device."""
        if input_ids is None:
            return None
        if input_ids.ndim == 1:
            if batch_size != 1:
                raise ValueError(f"input_ids must have shape ({batch_size}, S), got {tuple(input_ids.shape)}.")
            input_ids = input_ids.unsqueeze(0)
        elif input_ids.ndim != 2 or int(input_ids.shape[0]) != batch_size:
            raise ValueError(f"input_ids must have shape ({batch_size}, S), got {tuple(input_ids.shape)}.")

        input_ids = input_ids.to(device=device, dtype=torch.int64)
        min_id = int(input_ids.min().item())
        max_id = int(input_ids.max().item())
        if min_id < 0 or max_id >= self.config.sequence_vocab_size:
            raise ValueError("input_ids must lie within " f"[0, {self.config.sequence_vocab_size - 1}], got min={min_id}, max={max_id}.")
        return input_ids

    def _embed_input_ids(self, input_ids: Tensor) -> Tensor:
        """Embed prepared input ids into cortical sequence features.

        EHC v1 keeps token-id semantics local to the model and currently uses a
        RoPE-only front-end for the optional serialized sequence path.
        """
        token_embeddings = self.pathways.sequence.token_embedding(input_ids.to(torch.int32))

        if self.config.pfc.cortex.pos_encodings == "rope":
            return self.pathways.sequence.embedding_scale * token_embeddings

        raise ValueError("EHC v1 cortical sequence summary path currently requires pfc.cortex.pos_encodings='rope'.")

    def _summarize_input_ids(self, input_ids: Optional[Tensor], batch_size: int, device: torch.device) -> Tensor:
        """Summarize optional prepared input_ids into one cortical summary vector.

        ``input_ids`` is the whole serialized discrete sequence, not the current
        step-local observation. Validation, device movement, and vocabulary
        checks belong to ``_prepare_input_ids(...)`` and the EHC batch parser.
        Missing sequences map to a zero summary so the dedicated slot remains
        shape-stable.
        """
        summary_dtype = self.pathways.sequence.summarizer.cls_token.dtype
        if input_ids is None:
            return torch.zeros((batch_size, self.config.hidden_size), dtype=summary_dtype, device=device)
        return self.pathways.sequence.summarizer(self._embed_input_ids(input_ids))

    def _propose_cortical_cue(
        self,
        pfc_state: PFCState,
        cue_context_summary: Tensor,
    ) -> MultiScaleCode:
        """Project current cue context over prior PFC workspace to form one cortical cue proposal.

        The previous PFC workspace remains the primary memory source, but the
        current pre-HPC cortical context is exposed as one extra slot so cue
        formation can depend on present evidence even on cold-start steps.
        """
        workspace_slots = pfc_state.memory.z_H[:, 1:]
        cue_query = cue_context_summary.to(dtype=workspace_slots.dtype)
        cue_context_slot = cue_query.unsqueeze(1)
        attended_slots = torch.cat([cue_context_slot, workspace_slots], dim=1)
        scale = math.sqrt(float(self.config.hidden_size))
        cue_proposal: MultiScaleCode = []
        for query_proj, key_proj, output_proj in zip(
            self.pathways.pfc_hpc_cue.query,
            self.pathways.pfc_hpc_cue.key,
            self.pathways.pfc_hpc_cue.output,
            strict=True,
        ):
            query = query_proj(cue_query).unsqueeze(1)
            keys = key_proj(attended_slots)
            scores = torch.einsum("bid,bjd->bij", query, keys).squeeze(1) / scale
            weights = torch.softmax(scores, dim=1)
            summary = torch.einsum("bj,bjd->bd", weights, attended_slots)
            cue_proposal.append(output_proj(summary))
        return cue_proposal

    def _decode_observation_logits_from_place(self, place_code: MultiScaleCode) -> Tensor:
        """Decode one place-like replay code back into observation logits through the LEC branch."""
        lec_features = self.projections.lec_to_hpc_x.inverse(place_code)
        return self.autoencoder.decode(self.lec.generative(lec_features))

    def _build_cortical_workspace_tokens(
        self,
        *,
        grid_prior: MultiScaleCode,
        place_query_from_obs: MultiScaleCode,
        place_sensory: Optional[MultiScaleCode],
        grid_post: MultiScaleCode,
        c_prop: MultiScaleCode,
        c_mem: MultiScaleCode,
        external_context_features: Tensor,
        sequence_summary: Tensor,
        previous_action: Tensor,
        episode_start: Tensor,
        landmark_id: Tensor,
    ) -> Tensor:
        """Build the fixed semantic cortical workspace token layout consumed by PFC."""
        if place_sensory is None:
            place_sensory = [torch.zeros_like(code) for code in place_query_from_obs]

        cue_delta = multiscale_row_mse(c_prop, c_mem)
        prior_post_delta = multiscale_row_mse(grid_prior, grid_post)
        sensory_support = multiscale_mean_abs(place_sensory)
        disagreement_features = torch.stack([prior_post_delta, cue_delta, sensory_support], dim=1)

        tokens = [
            *apply_per_band(grid_prior, self.workspace.grid_prior, dtype=torch.float32),
            *apply_per_band(place_query_from_obs, self.workspace.sensory_query, dtype=torch.float32),
            *apply_per_band(place_sensory, self.workspace.sensory_recall, dtype=torch.float32),
            *apply_per_band(grid_post, self.workspace.grid_post, dtype=torch.float32),
            *apply_per_band(c_prop, self.workspace.cue_proposal, dtype=torch.float32),
            *apply_per_band(c_mem, self.workspace.cue_memory, dtype=torch.float32),
            self.workspace.external_context(external_context_features),
            sequence_summary,
            self.workspace.previous_action(previous_action.squeeze(-1)),
            self.workspace.episode_start(episode_start.to(torch.float32)),
            self.workspace.landmark(landmark_id.to(torch.float32)),
            self.workspace.disagreement(disagreement_features),
        ]
        return torch.stack(tokens, dim=1)


# =================================================================================================
__all__ = [
    "ModelSettings_V1", "EHCState", "EHCModelV1",
    "Batch", "ObsLogits", "GridCodes", "PlaceCodes",
]  # fmt: skip
