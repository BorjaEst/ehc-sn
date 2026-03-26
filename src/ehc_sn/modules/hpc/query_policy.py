"""Policies and evidence composers for HPC retrieval cues."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Annotated, Literal, Optional, TypeAlias

import torch
from pydantic import BaseModel, Field
from torch import Tensor, nn

from ehc_sn import utils
from ehc_sn.types import DEFAULT_FACTOR_BANK_NAME, Activation, Device, Dtype, FactorMemoryStore, MemoryEntry, RetrievalRole

MissingCueBehavior: TypeAlias = Literal["error", "use_available", "zeros"]
RetrievalTarget: TypeAlias = Literal["grounded", "sensory", "structural"]
RetrievalEvidenceMode: TypeAlias = Literal["anchor_query", "factor_logits", "anchor_refine"]
RefinementCompose: TypeAlias = Literal["additive", "multiplicative"]
RefinementReference: TypeAlias = Literal["anchor", "current"]
RefinementSource: TypeAlias = Literal["retrieved_value"]
RefinementBankField: TypeAlias = Literal["keys", "values"]


# =================================================================================================
@dataclass(frozen=True)
class CueBundle:
    """Available multi-scale cues supplied to retrieval composition."""

    x: Optional[list[Tensor]] = None
    g: Optional[list[Tensor]] = None
    extras: dict[str, list[Tensor]] = field(default_factory=dict)


# =================================================================================================
@dataclass(frozen=True)
class ScoreTerm:
    """Named logit contribution used to compose factor-memory retrieval."""

    name: str
    logits: Tensor
    source_cues: tuple[str, ...]
    bank_name: str = DEFAULT_FACTOR_BANK_NAME
    normalization: Literal["none", "softmax", "temperature_scaled"] = "none"


# =================================================================================================
@dataclass(frozen=True)
class RetrievalRefinementStep:
    """One explicit refinement instruction executed by factor retrieval."""

    name: str
    bank_name: str = DEFAULT_FACTOR_BANK_NAME
    bank_field: RefinementBankField = "values"
    source: RefinementSource = "retrieved_value"
    compose: RefinementCompose = "multiplicative"
    reference: RefinementReference = "anchor"


# =================================================================================================
@dataclass(frozen=True)
class RetrievalEvidence:
    """Structured retrieval evidence emitted before backend-specific memory read."""

    mode: RetrievalEvidenceMode
    target: RetrievalTarget
    role: RetrievalRole
    score_terms: list[ScoreTerm]
    composed_logits: Optional[Tensor]
    anchor_logits: Optional[Tensor]
    fallback_query: Optional[Tensor]
    anchor_query: Optional[Tensor]
    read_bank: str = DEFAULT_FACTOR_BANK_NAME
    iterations: int = 1
    refinement_steps: tuple[RetrievalRefinementStep, ...] = ()
    metadata: dict[str, Tensor | float | int | str] = field(default_factory=dict)


# =================================================================================================
class RoleQueryPolicySettings(BaseModel, extra="forbid"):
    """Role-aware compatibility policy matching the legacy TEM query flow."""

    kind: Literal["by_role"] = Field(
        default="by_role",
        description="Use x for inference recall and g for generative recall, with optional fallback.",
    )
    missing_behavior: MissingCueBehavior = Field(
        default="use_available",
        description="Fallback behavior when the role-preferred cue is unavailable.",
    )


# =================================================================================================
class XOnlyQueryPolicySettings(BaseModel, extra="forbid"):
    """Always use the sensory cue as the retrieval query."""

    kind: Literal["x_only"] = Field(
        default="x_only",
        description="Use only the observation-derived cue as the retrieval query.",
    )
    missing_behavior: MissingCueBehavior = Field(
        default="use_available",
        description="Fallback behavior when the sensory cue is unavailable.",
    )


# =================================================================================================
class GOnlyQueryPolicySettings(BaseModel, extra="forbid"):
    """Always use the grid cue as the retrieval query."""

    kind: Literal["g_only"] = Field(
        default="g_only",
        description="Use only the grid-derived cue as the retrieval query.",
    )
    missing_behavior: MissingCueBehavior = Field(
        default="use_available",
        description="Fallback behavior when the grid cue is unavailable.",
    )


# =================================================================================================
class GatedMixQueryPolicySettings(BaseModel, extra="forbid"):
    """Learn a per-frequency mixture between x and g cues."""

    kind: Literal["gated_mix"] = Field(
        default="gated_mix",
        description="Learn a per-frequency gate between sensory and grid cues.",
    )
    missing_behavior: MissingCueBehavior = Field(
        default="use_available",
        description="Fallback behavior when one cue is unavailable.",
    )


# =================================================================================================
class ConjunctiveQueryPolicySettings(BaseModel, extra="forbid"):
    """Build a learned conjunctive query from sensory and grid cues."""

    kind: Literal["conjunctive"] = Field(
        default="conjunctive",
        description="Combine sensory and grid cues through a learned conjunctive projection.",
    )
    missing_behavior: MissingCueBehavior = Field(
        default="use_available",
        description="Fallback behavior when one cue is unavailable.",
    )
    activation: Activation = Field(
        default="leaky_relu",
        description="Activation applied after the conjunctive projection.",
    )
    clamp_min: float = Field(
        default=-1.0,
        description="Minimum clamp value applied before conjunctive activation.",
    )
    clamp_max: float = Field(
        default=1.0,
        description="Maximum clamp value applied before conjunctive activation.",
    )


# =================================================================================================
class _FactorScoreQueryPolicySettings(BaseModel, extra="forbid"):
    """Shared settings for cue-scored factor-memory composition."""

    missing_behavior: MissingCueBehavior = Field(
        default="use_available",
        description="Fallback behavior when one cue is unavailable.",
    )
    temperature: float = Field(
        default=1.0,
        gt=0.0,
        description="Temperature divisor used when projecting cues against factor-memory keys.",
    )
    x_bank: str = Field(
        default="sensory",
        description="Factor-memory bank used when scoring sensory cue queries.",
    )
    g_bank: str = Field(
        default="structural",
        description="Factor-memory bank used when scoring grid cue queries.",
    )
    read_bank: str = Field(
        default=DEFAULT_FACTOR_BANK_NAME,
        description="Factor-memory bank providing retrieved values.",
    )


# =================================================================================================
class AdditiveQueryPolicySettings(_FactorScoreQueryPolicySettings):
    """Compose x- and g-cued factor-memory scores additively."""

    kind: Literal["additive"] = Field(
        default="additive",
        description="Compose factor-memory logits from x and g cues using a sum in score space.",
    )


# =================================================================================================
class MultiplicativeQueryPolicySettings(_FactorScoreQueryPolicySettings):
    """Compose x- and g-cued factor-memory scores multiplicatively."""

    kind: Literal["multiplicative"] = Field(
        default="multiplicative",
        description="Compose factor-memory logits from x and g cues using an elementwise product.",
    )


# =================================================================================================
class AnchorRefineQueryPolicySettings(_FactorScoreQueryPolicySettings):
    """Compose anchor-query retrieval with explicit refinement steps."""

    kind: Literal["anchor_refine"] = Field(
        default="anchor_refine",
        description="Use a role-selected anchor query and explicit retrieval refinement steps.",
    )
    iterations: int = Field(
        default=2,
        ge=2,
        description="Number of retrieval passes executed by the factor retriever.",
    )
    refinement_source: RefinementSource = Field(
        default="retrieved_value",
        description="Signal used to compute refinement logits after the anchor pass.",
    )
    refinement_combine: RefinementCompose = Field(
        default="multiplicative",
        description="How anchor and refinement logits are combined.",
    )
    refinement_reference: RefinementReference = Field(
        default="anchor",
        description="Whether refinement logits combine with the anchor logits or the running logits.",
    )
    refinement_bank: str = Field(
        default=DEFAULT_FACTOR_BANK_NAME,
        description="Bank used when computing refinement logits.",
    )
    refinement_bank_field: RefinementBankField = Field(
        default="values",
        description="Bank tensor compared against the refinement source.",
    )


# =================================================================================================
QueryPolicySettings: TypeAlias = Annotated[
    RoleQueryPolicySettings
    | XOnlyQueryPolicySettings
    | GOnlyQueryPolicySettings
    | GatedMixQueryPolicySettings
    | ConjunctiveQueryPolicySettings
    | AdditiveQueryPolicySettings
    | MultiplicativeQueryPolicySettings
    | AnchorRefineQueryPolicySettings,
    Field(discriminator="kind"),
]


# =================================================================================================
class QueryPolicy(nn.Module, ABC):
    """Resolve retrieval cues and, when possible, compose retrieval evidence."""

    def __init__(  # ------------------------------------------------------------------------------
        self, shape: list[int], config: QueryPolicySettings, *,
        device: Optional[Device] = None, dtype: Optional[Dtype] = None,
    ) -> None:  # fmt: skip
        """Initialize shared query-policy state for the configured cue widths."""
        del device, dtype
        super().__init__()
        self._shape = list(shape)
        self._config = config

    @property
    def shape(self) -> list[int]:
        """Return the multi-frequency query widths expected by this policy."""
        return self._shape

    @property
    def config(self) -> QueryPolicySettings:
        """Return query-policy settings."""
        return self._config

    def forward(  # -------------------------------------------------------------------------------
        self, *, 
        x_query: Optional[list[Tensor]], g_query: Optional[list[Tensor]], role: RetrievalRole,
    ) -> list[Tensor]:  # fmt: skip
        """Return a resolved multi-scale anchor query for memory retrieval."""
        self._validate_query(x_query, name="x_query")
        self._validate_query(g_query, name="g_query")
        return self._forward(x_query=x_query, g_query=g_query, role=role)

    def compose_evidence(  # ----------------------------------------------------------------------
        self, *,
        x_query: Optional[list[Tensor]], g_query: Optional[list[Tensor]], role: RetrievalRole,
        target: RetrievalTarget, memory: MemoryEntry,
    ) -> RetrievalEvidence:  # fmt: skip
        """Return structured retrieval evidence for the configured backend.

        The default implementation emits only an anchor query, which is the
        dense-attractor-compatible special case.
        """
        del memory
        anchor = self._flatten_query(self.forward(x_query=x_query, g_query=g_query, role=role))
        return RetrievalEvidence(
            mode="anchor_query",
            target=target,
            role=role,
            score_terms=[],
            composed_logits=None,
            anchor_logits=None,
            fallback_query=anchor,
            anchor_query=anchor,
        )

    @abstractmethod
    def _forward(  # ------------------------------------------------------------------------------
        self, *,
        x_query: Optional[list[Tensor]], g_query: Optional[list[Tensor]], role: RetrievalRole,
    ) -> list[Tensor]:  # fmt: skip
        """Implement policy-specific query resolution."""

    def _validate_query(  # -----------------------------------------------------------------------
        self, query: Optional[list[Tensor]], *, name: str,
    ) -> None:  # fmt: skip
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

    def _flatten_query(  # ------------------------------------------------------------------------
        self, query: list[Tensor],
    ) -> Tensor:  # fmt: skip
        """Flatten a validated multi-scale query into shape ``(B, S)``."""
        return torch.cat(query, dim=1)

    def _fallback(  # -----------------------------------------------------------------------------
        self, *,
        preferred: Optional[list[Tensor]], preferred_name: str, other: Optional[list[Tensor]],
        missing_behavior: MissingCueBehavior,
    ) -> list[Tensor]:  # fmt: skip
        """Return the configured fallback query when the preferred cue is missing."""
        if preferred is not None:
            return preferred
        if missing_behavior == "use_available" and other is not None:
            return other
        if missing_behavior == "zeros" and other is not None:
            return [torch.zeros_like(tensor) for tensor in other]
        raise ValueError(f"{preferred_name} is required for query policy '{self.config.kind}'.")


# =================================================================================================
class RoleQueryPolicy(QueryPolicy):
    """Compatibility policy that preserves the legacy TEM role-based query selection."""

    @property
    def config(self) -> RoleQueryPolicySettings:
        """Return typed settings for the role-aware query policy."""
        return super().config  # type: ignore[return-value]

    def _forward(  # ------------------------------------------------------------------------------
        self, *,
        x_query: Optional[list[Tensor]], g_query: Optional[list[Tensor]], role: RetrievalRole,
    ) -> list[Tensor]:  # fmt: skip
        """Select the legacy role-preferred cue, with configured fallback behavior."""
        if role == "inference":
            return self._fallback(
                preferred=x_query,
                other=g_query,
                missing_behavior=self.config.missing_behavior,
                preferred_name="x_query",
            )
        if role == "generative":
            return self._fallback(
                preferred=g_query,
                other=x_query,
                missing_behavior=self.config.missing_behavior,
                preferred_name="g_query",
            )
        raise ValueError(f"Unrecognized retrieval role '{role}' for query policy '{self.config.kind}'.")


# =================================================================================================
class XOnlyQueryPolicy(QueryPolicy):
    """Policy that always prefers the sensory cue."""

    @property
    def config(self) -> XOnlyQueryPolicySettings:
        """Return typed settings for the x-only query policy."""
        return super().config  # type: ignore[return-value]

    def _forward(  # ------------------------------------------------------------------------------
        self, *,
        x_query: Optional[list[Tensor]], g_query: Optional[list[Tensor]], role: RetrievalRole,
    ) -> list[Tensor]:  # fmt: skip
        """Resolve retrieval queries using only sensory cues when available."""
        del role
        return self._fallback(
            preferred=x_query,
            other=g_query,
            missing_behavior=self.config.missing_behavior,
            preferred_name="x_query",
        )


# =================================================================================================
class GOnlyQueryPolicy(QueryPolicy):
    """Policy that always prefers the grid cue."""

    @property
    def config(self) -> GOnlyQueryPolicySettings:
        """Return typed settings for the g-only query policy."""
        return super().config  # type: ignore[return-value]

    def _forward(  # ------------------------------------------------------------------------------
        self, *,
        x_query: Optional[list[Tensor]], g_query: Optional[list[Tensor]], role: RetrievalRole,
    ) -> list[Tensor]:  # fmt: skip
        """Resolve retrieval queries using only grid cues when available."""
        del role
        return self._fallback(
            preferred=g_query,
            other=x_query,
            missing_behavior=self.config.missing_behavior,
            preferred_name="g_query",
        )


# =================================================================================================
class GatedMixQueryPolicy(QueryPolicy):
    """Learn a per-frequency convex interpolation between x and g cues."""

    def __init__(  # ------------------------------------------------------------------------------
        self, shape: list[int], config: GatedMixQueryPolicySettings, *,
        device: Optional[Device] = None, dtype: Optional[Dtype] = None,
    ) -> None:  # fmt: skip
        """Initialize one gate per frequency module for x/g cue mixing."""
        super().__init__(shape, config, device=device, dtype=dtype)
        self._gates = nn.ModuleList([nn.Linear(2 * width, width, device=device, dtype=dtype) for width in shape])
        self._reset_parameters()

    @property
    def config(self) -> GatedMixQueryPolicySettings:
        """Return typed settings for the gated-mix query policy."""
        return super().config  # type: ignore[return-value]

    def _forward(  # ------------------------------------------------------------------------------
        self, *,
        x_query: Optional[list[Tensor]], g_query: Optional[list[Tensor]], role: RetrievalRole,
    ) -> list[Tensor]:  # fmt: skip
        """Blend sensory and grid cues with a learned convex gate per frequency."""
        del role
        if x_query is None or g_query is None:
            return self._fallback(
                preferred=x_query,
                other=g_query,
                missing_behavior=self.config.missing_behavior,
                preferred_name="x_query and g_query",
            )

        mixed: list[Tensor] = []
        for gate_layer, x_tensor, g_tensor in zip(self._gates, x_query, g_query, strict=True):
            gate = torch.sigmoid(gate_layer(torch.cat((x_tensor, g_tensor), dim=1)))
            mixed.append(gate * x_tensor + (1.0 - gate) * g_tensor)
        return mixed

    def _reset_parameters(  # ---------------------------------------------------------------------
        self,
    ) -> None:  # fmt: skip
        """Initialize gates to an even mixture before training."""
        for layer in self._gates:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)


# =================================================================================================
class ConjunctiveQueryPolicy(QueryPolicy):
    """Build a learned conjunctive query from sensory and grid cues."""

    def __init__(  # ------------------------------------------------------------------------------
        self, shape: list[int], config: ConjunctiveQueryPolicySettings, *,
        device: Optional[Device] = None, dtype: Optional[Dtype] = None,
    ) -> None:  # fmt: skip
        """Initialize per-frequency conjunctive projections over x, g, and x*g."""
        super().__init__(shape, config, device=device, dtype=dtype)
        self._activation_fn = utils.activation_from_str(config.activation)
        self._projections = nn.ModuleList([nn.Linear(3 * width, width, device=device, dtype=dtype) for width in shape])
        self._reset_parameters()

    @property
    def config(self) -> ConjunctiveQueryPolicySettings:
        """Return typed settings for the conjunctive query policy."""
        return super().config  # type: ignore[return-value]

    def _forward(  # ------------------------------------------------------------------------------
        self, *,
        x_query: Optional[list[Tensor]], g_query: Optional[list[Tensor]], role: RetrievalRole,
    ) -> list[Tensor]:  # fmt: skip
        """Project sensory and grid cues into a learned conjunctive query."""
        del role
        if x_query is None or g_query is None:
            return self._fallback(
                preferred=x_query,
                other=g_query,
                missing_behavior=self.config.missing_behavior,
                preferred_name="x_query and g_query",
            )

        conjunctive: list[Tensor] = []
        for projection, x_tensor, g_tensor in zip(self._projections, x_query, g_query, strict=True):
            joined = torch.cat((x_tensor, g_tensor, x_tensor * g_tensor), dim=1)
            tensor = projection(joined)
            tensor = torch.clamp(tensor, min=self.config.clamp_min, max=self.config.clamp_max)
            conjunctive.append(self._activation_fn(tensor))
        return conjunctive

    def _reset_parameters(  # ---------------------------------------------------------------------
        self,
    ) -> None:  # fmt: skip
        """Initialize projections to start from the classic x*g conjunction."""
        for width, projection in zip(self.shape, self._projections, strict=True):
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)
            identity = torch.eye(width, dtype=projection.weight.dtype, device=projection.weight.device)
            projection.weight.data[:, 2 * width : 3 * width] = identity


# =================================================================================================
class _FactorScoreQueryPolicy(QueryPolicy, ABC):
    """Base class for factor-memory cue-score composers."""

    @property
    def config(self) -> _FactorScoreQueryPolicySettings:
        return super().config  # type: ignore[return-value]

    def _forward(  # ------------------------------------------------------------------------------
        self, *,
        x_query: Optional[list[Tensor]], g_query: Optional[list[Tensor]], role: RetrievalRole,
    ) -> list[Tensor]:  # fmt: skip
        if role == "inference":
            return self._fallback(
                preferred=x_query,
                other=g_query,
                missing_behavior=self.config.missing_behavior,
                preferred_name="x_query",
            )
        return self._fallback(
            preferred=g_query,
            other=x_query,
            missing_behavior=self.config.missing_behavior,
            preferred_name="g_query",
        )

    def compose_evidence(  # ----------------------------------------------------------------------
        self, *,
        x_query: Optional[list[Tensor]], g_query: Optional[list[Tensor]], role: RetrievalRole,
        target: RetrievalTarget, memory: MemoryEntry,
    ) -> RetrievalEvidence:  # fmt: skip
        anchor = self._flatten_query(self.forward(x_query=x_query, g_query=g_query, role=role))
        fallback = anchor
        if not isinstance(memory, FactorMemoryStore):
            return self._anchor_only_evidence(target=target, role=role, anchor=anchor)

        score_terms = self._collect_score_terms(memory=memory, x_query=x_query, g_query=g_query)
        if not score_terms:
            return self._anchor_only_evidence(target=target, role=role, anchor=anchor)

        composed_logits = self._compose_score_terms(score_terms)
        return RetrievalEvidence(
            mode="factor_logits",
            target=target,
            role=role,
            score_terms=score_terms,
            composed_logits=composed_logits,
            anchor_logits=composed_logits,
            fallback_query=fallback,
            anchor_query=anchor,
            read_bank=self.config.read_bank,
            metadata={"temperature": float(self.config.temperature)},
        )

    @abstractmethod
    def _compose_score_terms(self, score_terms: list[ScoreTerm]) -> Tensor:
        """Compose the per-cue score terms into one logit tensor."""

    def _anchor_only_evidence(self, *, target: RetrievalTarget, role: RetrievalRole, anchor: Tensor) -> RetrievalEvidence:
        """Return dense-compatible anchor-query evidence."""
        return RetrievalEvidence(
            mode="anchor_query",
            target=target,
            role=role,
            score_terms=[],
            composed_logits=None,
            anchor_logits=None,
            fallback_query=anchor,
            anchor_query=anchor,
        )

    def _collect_score_terms(
        self,
        *,
        memory: FactorMemoryStore,
        x_query: Optional[list[Tensor]],
        g_query: Optional[list[Tensor]],
    ) -> list[ScoreTerm]:
        """Collect cue-scored logits against the configured banks."""
        score_terms = [
            term
            for term in (
                self._cue_score_term("x", x_query, memory, self.config.x_bank),
                self._cue_score_term("g", g_query, memory, self.config.g_bank),
            )
            if term is not None
        ]
        if score_terms:
            return self._fill_missing_terms(score_terms)
        if self.config.missing_behavior == "error":
            raise ValueError(f"Both x_query and g_query are required for query policy '{self.config.kind}'.")
        return []

    def _fill_missing_terms(self, score_terms: list[ScoreTerm]) -> list[ScoreTerm]:
        """Apply missing-cue behavior to any absent factor score terms."""
        if self.config.missing_behavior != "zeros":
            return score_terms
        present = {term.name for term in score_terms}
        if len(present) == 2:
            return score_terms
        template = score_terms[0]
        completed = list(score_terms)
        if "x" not in present:
            completed.append(
                ScoreTerm(
                    name="x",
                    logits=torch.zeros_like(template.logits),
                    source_cues=("x",),
                    bank_name=self.config.x_bank,
                    normalization="temperature_scaled",
                )
            )
        if "g" not in present:
            completed.append(
                ScoreTerm(
                    name="g",
                    logits=torch.zeros_like(template.logits),
                    source_cues=("g",),
                    bank_name=self.config.g_bank,
                    normalization="temperature_scaled",
                )
            )
        return completed

    def _cue_score_term(
        self,
        cue_name: str,
        query: Optional[list[Tensor]],
        memory: FactorMemoryStore,
        bank_name: str,
    ) -> Optional[ScoreTerm]:
        """Return one cue-specific score term when that cue is available."""
        if query is None:
            if self.config.missing_behavior == "error":
                raise ValueError(f"{cue_name}_query is required for query policy '{self.config.kind}'.")
            return None
        bank = memory.bank(bank_name, fallback_to_default=True)
        logits = self._compute_logits(self._flatten_query(query), bank.keys)
        return ScoreTerm(
            name=cue_name,
            logits=logits,
            source_cues=(cue_name,),
            bank_name=bank_name,
            normalization="temperature_scaled",
        )

    def _compute_logits(self, query: Tensor, keys: Tensor) -> Tensor:
        """Project a flattened query against one key bank."""
        scale = math.sqrt(max(query.shape[1], 1)) * self.config.temperature
        return torch.einsum("bs,bts->bt", query.to(dtype=keys.dtype), keys) / scale


# =================================================================================================
class AdditiveQueryPolicy(_FactorScoreQueryPolicy):
    """Compose x- and g-cued factor-memory scores with a sum in logit space."""

    @property
    def config(self) -> AdditiveQueryPolicySettings:
        return super().config  # type: ignore[return-value]

    def _compose_score_terms(self, score_terms: list[ScoreTerm]) -> Tensor:
        return sum((term.logits for term in score_terms[1:]), score_terms[0].logits)


# =================================================================================================
class MultiplicativeQueryPolicy(_FactorScoreQueryPolicy):
    """Compose x- and g-cued factor-memory scores using an elementwise product."""

    @property
    def config(self) -> MultiplicativeQueryPolicySettings:
        return super().config  # type: ignore[return-value]

    def _compose_score_terms(self, score_terms: list[ScoreTerm]) -> Tensor:
        product = score_terms[0].logits
        for term in score_terms[1:]:
            product = product * term.logits
        return product


# =================================================================================================
class AnchorRefineQueryPolicy(QueryPolicy):
    """Compose a role-selected anchor query with explicit refinement metadata."""

    @property
    def config(self) -> AnchorRefineQueryPolicySettings:
        return super().config  # type: ignore[return-value]

    def _forward(  # ------------------------------------------------------------------------------
        self, *,
        x_query: Optional[list[Tensor]], g_query: Optional[list[Tensor]], role: RetrievalRole,
    ) -> list[Tensor]:  # fmt: skip
        if role == "inference":
            return self._fallback(
                preferred=x_query,
                other=g_query,
                missing_behavior=self.config.missing_behavior,
                preferred_name="x_query",
            )
        return self._fallback(
            preferred=g_query,
            other=x_query,
            missing_behavior=self.config.missing_behavior,
            preferred_name="g_query",
        )

    def compose_evidence(  # ----------------------------------------------------------------------
        self, *,
        x_query: Optional[list[Tensor]], g_query: Optional[list[Tensor]], role: RetrievalRole,
        target: RetrievalTarget, memory: MemoryEntry,
    ) -> RetrievalEvidence:  # fmt: skip
        anchor_query = self._flatten_query(self.forward(x_query=x_query, g_query=g_query, role=role))
        if not isinstance(memory, FactorMemoryStore):
            return RetrievalEvidence(
                mode="anchor_query",
                target=target,
                role=role,
                score_terms=[],
                composed_logits=None,
                anchor_logits=None,
                fallback_query=anchor_query,
                anchor_query=anchor_query,
            )

        anchor_bank_name = self.config.x_bank if role == "inference" else self.config.g_bank
        anchor_bank = memory.bank(anchor_bank_name, fallback_to_default=True)
        anchor_logits = self._compute_logits(anchor_query, anchor_bank.keys)
        return RetrievalEvidence(
            mode="anchor_refine",
            target=target,
            role=role,
            score_terms=[
                ScoreTerm(
                    name="anchor",
                    logits=anchor_logits,
                    source_cues=(("x",) if role == "inference" else ("g",)),
                    bank_name=anchor_bank_name,
                    normalization="temperature_scaled",
                )
            ],
            composed_logits=None,
            anchor_logits=anchor_logits,
            fallback_query=anchor_query,
            anchor_query=anchor_query,
            read_bank=self.config.read_bank,
            iterations=self.config.iterations,
            refinement_steps=(
                RetrievalRefinementStep(
                    name="retrieved_value",
                    bank_name=self.config.refinement_bank,
                    bank_field=self.config.refinement_bank_field,
                    source=self.config.refinement_source,
                    compose=self.config.refinement_combine,
                    reference=self.config.refinement_reference,
                ),
            ),
            metadata={"temperature": float(self.config.temperature)},
        )

    def _compute_logits(self, query: Tensor, keys: Tensor) -> Tensor:
        scale = math.sqrt(max(query.shape[1], 1)) * self.config.temperature
        return torch.einsum("bs,bts->bt", query.to(dtype=keys.dtype), keys) / scale


# =================================================================================================
def build_query_policy(  # ------------------------------------------------------------------------
    shape: list[int], config: QueryPolicySettings, *,
    device: Optional[Device] = None, dtype: Optional[Dtype] = None,
) -> QueryPolicy:  # fmt: skip
    """Construct the configured query policy."""
    if config.kind == "by_role":
        return RoleQueryPolicy(shape, config, device=device, dtype=dtype)
    if config.kind == "x_only":
        return XOnlyQueryPolicy(shape, config, device=device, dtype=dtype)
    if config.kind == "g_only":
        return GOnlyQueryPolicy(shape, config, device=device, dtype=dtype)
    if config.kind == "gated_mix":
        return GatedMixQueryPolicy(shape, config, device=device, dtype=dtype)
    if config.kind == "conjunctive":
        return ConjunctiveQueryPolicy(shape, config, device=device, dtype=dtype)
    if config.kind == "additive":
        return AdditiveQueryPolicy(shape, config, device=device, dtype=dtype)
    if config.kind == "multiplicative":
        return MultiplicativeQueryPolicy(shape, config, device=device, dtype=dtype)
    if config.kind == "anchor_refine":
        return AnchorRefineQueryPolicy(shape, config, device=device, dtype=dtype)
    raise ValueError(f"Unsupported query policy '{config.kind}'.")


# =================================================================================================
__all__ = [
    "AdditiveQueryPolicySettings", "AnchorRefineQueryPolicySettings",
    "ConjunctiveQueryPolicySettings", "GOnlyQueryPolicySettings", "GatedMixQueryPolicySettings",
    "MultiplicativeQueryPolicySettings", "QueryPolicySettings", "RoleQueryPolicySettings",
    "XOnlyQueryPolicySettings", "CueBundle", "MissingCueBehavior", "QueryPolicy",
    "RetrievalEvidence", "RetrievalRefinementStep", "RetrievalTarget", "ScoreTerm", "build_query_policy",
]  # fmt: skip
