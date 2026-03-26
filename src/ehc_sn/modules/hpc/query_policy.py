"""Policies and evidence composers for HPC retrieval cues."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Annotated, Iterable, Literal, Optional, TypeAlias

import torch
from pydantic import BaseModel, Field
from torch import Tensor, nn

from ehc_sn import utils
from ehc_sn.types import DEFAULT_FACTOR_BANK_NAME, Activation, Device, Dtype, FactorMemoryStore, MemoryEntry, RetrievalRole

MissingCueBehavior: TypeAlias = Literal["error", "use_available", "zeros"]
CueFamily: TypeAlias = str
RetrievalEvidenceMode: TypeAlias = Literal["anchor_query", "factor_logits", "anchor_refine"]
PairwiseMode: TypeAlias = Literal["gated_sum", "product_projection"]
RefinementCompose: TypeAlias = Literal["additive", "multiplicative"]
RefinementReference: TypeAlias = Literal["anchor", "current"]
RefinementSource: TypeAlias = Literal["retrieved_value"]
RefinementBankField: TypeAlias = Literal["keys", "values"]


# =================================================================================================
@dataclass(frozen=True)
class CueBundle:
    """Available multi-scale cues supplied to retrieval composition."""

    families: dict[CueFamily, list[Tensor]] = field(default_factory=dict)

    def get(self, family: CueFamily) -> Optional[list[Tensor]]:
        """Return one named cue family when present."""
        return self.families.get(family)

    def require(self, family: CueFamily) -> list[Tensor]:
        """Return one named cue family or raise when it is absent."""
        query = self.get(family)
        if query is None:
            raise ValueError(f"Cue family '{family}' is required.")
        return query

    def items(self) -> Iterable[tuple[CueFamily, list[Tensor]]]:
        """Iterate over named cue families and their multi-scale codes."""
        return self.families.items()

    def names(self) -> tuple[CueFamily, ...]:
        """Return the available cue-family names in insertion order."""
        return tuple(self.families.keys())

    def with_family(self, family: CueFamily, query: list[Tensor]) -> "CueBundle":
        """Return a new cue bundle with one family inserted or replaced."""
        families = dict(self.families)
        families[family] = query
        return CueBundle(families=families)


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
class AnchorQueryPolicySettings(BaseModel, extra="forbid"):
    """Resolve one named cue family as the dense retrieval anchor."""

    kind: Literal["anchor"] = Field(
        default="anchor",
        description="Resolve one named cue family as the retrieval anchor query.",
    )
    missing_behavior: MissingCueBehavior = Field(
        default="use_available",
        description="Fallback behavior when the requested anchor family is unavailable.",
    )


# =================================================================================================
class PairwiseQueryPolicySettings(BaseModel, extra="forbid"):
    """Compose a dense query from one ordered pair of cue families."""

    kind: Literal["pairwise"] = Field(
        default="pairwise",
        description="Compose a dense retrieval query from an ordered pair of cue families.",
    )
    families: tuple[str, str] = Field(
        ...,
        description="Ordered cue families consumed by the pairwise dense composer.",
    )
    mode: PairwiseMode = Field(
        ...,
        description="Pairwise dense-composition mode.",
    )
    missing_behavior: MissingCueBehavior = Field(
        default="use_available",
        description="Fallback behavior when one cue family in the ordered pair is unavailable.",
    )
    activation: Activation = Field(
        default="leaky_relu",
        description="Activation applied after product-projection composition.",
    )
    clamp_min: float = Field(
        default=-1.0,
        description="Minimum clamp applied before product-projection activation.",
    )
    clamp_max: float = Field(
        default=1.0,
        description="Maximum clamp applied before product-projection activation.",
    )


# =================================================================================================
class _FactorScoreQueryPolicySettings(BaseModel, extra="forbid"):
    """Shared settings for cue-scored factor-memory composition."""

    missing_behavior: MissingCueBehavior = Field(
        default="use_available",
        description="Fallback behavior when one cue family is unavailable.",
    )
    temperature: float = Field(
        default=1.0,
        gt=0.0,
        description="Temperature divisor used when projecting cues against factor-memory keys.",
    )
    cue_to_score_bank: dict[str, str] = Field(
        default_factory=dict,
        description="Optional cue-family to score-bank mapping for factor-memory score terms.",
    )
    read_bank: str = Field(
        default=DEFAULT_FACTOR_BANK_NAME,
        description="Factor-memory bank providing retrieved values.",
    )


# =================================================================================================
class AdditiveQueryPolicySettings(_FactorScoreQueryPolicySettings):
    """Compose factor-memory scores additively across named cue families."""

    kind: Literal["additive"] = Field(
        default="additive",
        description="Compose factor-memory logits from named cue families using a sum in score space.",
    )


# =================================================================================================
class MultiplicativeQueryPolicySettings(_FactorScoreQueryPolicySettings):
    """Compose factor-memory scores multiplicatively across named cue families."""

    kind: Literal["multiplicative"] = Field(
        default="multiplicative",
        description="Compose factor-memory logits from named cue families using an elementwise product.",
    )


# =================================================================================================
class AnchorRefineQueryPolicySettings(_FactorScoreQueryPolicySettings):
    """Compose anchor-query retrieval with explicit refinement steps."""

    kind: Literal["anchor_refine"] = Field(
        default="anchor_refine",
        description="Use an explicit anchor query and refinement metadata for factor retrieval.",
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
    AnchorQueryPolicySettings
    | PairwiseQueryPolicySettings
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

    def resolve_query(  # -------------------------------------------------------------------------
        self, *, cues: CueBundle, anchor_family: Optional[CueFamily] = None,
    ) -> list[Tensor]:  # fmt: skip
        """Return a resolved multi-scale anchor query for memory retrieval."""
        self._validate_cues(cues)
        return self._resolve_query(cues=cues, anchor_family=anchor_family)

    def compose(  # -------------------------------------------------------------------------------
        self, *, cues: CueBundle, memory: MemoryEntry, role: RetrievalRole,
        anchor_family: Optional[CueFamily] = None,
    ) -> RetrievalEvidence:  # fmt: skip
        """Compose structured retrieval evidence from a generic cue bundle."""
        del memory
        anchor = self._flatten_query(self.resolve_query(cues=cues, anchor_family=anchor_family))
        return RetrievalEvidence(
            mode="anchor_query",
            role=role,
            score_terms=[],
            composed_logits=None,
            anchor_logits=None,
            fallback_query=anchor,
            anchor_query=anchor,
        )

    @abstractmethod
    def _resolve_query(  # ------------------------------------------------------------------------
        self, *, cues: CueBundle,
        anchor_family: Optional[CueFamily] = None,
    ) -> list[Tensor]:  # fmt: skip
        """Implement policy-specific query resolution from named cue families."""

    def _validate_cues(  # ------------------------------------------------------------------------
        self, cues: CueBundle,
    ) -> None:  # fmt: skip
        """Validate all populated cue families in one cue bundle."""
        for family, query in cues.items():
            self._validate_query(query, name=f"cues[{family!r}]")

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

    def _resolve_family_query(  # -----------------------------------------------------------------
        self, *,
        cues: CueBundle, preferred_family: Optional[CueFamily], missing_behavior: MissingCueBehavior,
        label: str,
    ) -> list[Tensor]:  # fmt: skip
        """Resolve one cue family from the available bundle with fallback semantics."""
        preferred = cues.get(preferred_family) if preferred_family is not None else None
        if preferred is not None:
            return preferred
        available = next(iter(cues.families.values()), None)
        if missing_behavior == "use_available" and available is not None:
            return available
        if missing_behavior == "zeros" and available is not None:
            return [torch.zeros_like(tensor) for tensor in available]
        if preferred_family is None:
            raise ValueError(f"{label} must be provided for query policy '{self.config.kind}'.")
        raise ValueError(f"Cue family '{preferred_family}' is required for query policy '{self.config.kind}'.")

    def _score_bank_for_family(  # ----------------------------------------------------------------
        self, family: CueFamily,
    ) -> str:  # fmt: skip
        """Return the configured score bank for one cue family when available."""
        cue_to_score_bank = getattr(self.config, "cue_to_score_bank", {})
        return cue_to_score_bank.get(family, family)

    def _resolve_pair_family_queries(  # ----------------------------------------------------------
        self, *,
        cues: CueBundle, families: tuple[CueFamily, CueFamily], missing_behavior: MissingCueBehavior,
    ) -> tuple[list[Tensor], list[Tensor]]:  # fmt: skip
        """Resolve the ordered pair of cue families used by a dense pairwise composer."""
        left_family, right_family = families
        left_query = cues.get(left_family)
        right_query = cues.get(right_family)
        if left_query is not None and right_query is not None:
            return left_query, right_query
        if missing_behavior == "use_available":
            if left_query is not None:
                return left_query, left_query
            if right_query is not None:
                return right_query, right_query
        if missing_behavior == "zeros":
            template = left_query if left_query is not None else right_query
            if template is not None:
                zeros = [torch.zeros_like(tensor) for tensor in template]
                return (left_query or zeros), (right_query or zeros)
        raise ValueError(
            f"Cue families {families!r} are required for query policy '{self.config.kind}' "
            f"with missing_behavior='{missing_behavior}'."
        )  # fmt: skip


# =================================================================================================
class AnchorQueryPolicy(QueryPolicy):
    """Resolve one named cue family as the dense retrieval anchor."""

    @property
    def config(self) -> AnchorQueryPolicySettings:
        """Return typed settings for the anchor query policy."""
        return super().config  # type: ignore[return-value]

    def _resolve_query(  # ------------------------------------------------------------------------
        self, *, cues: CueBundle, anchor_family: Optional[CueFamily] = None,
    ) -> list[Tensor]:  # fmt: skip
        """Resolve the caller-selected anchor family."""
        return self._resolve_family_query(
            cues=cues,
            preferred_family=anchor_family,
            missing_behavior=self.config.missing_behavior,
            label="anchor_family",
        )


# =================================================================================================
class PairwiseQueryPolicy(QueryPolicy):
    """Compose one dense retrieval query from an ordered pair of cue families."""

    def __init__(  # ------------------------------------------------------------------------------
        self, shape: list[int], config: PairwiseQueryPolicySettings, *,
        device: Optional[Device] = None, dtype: Optional[Dtype] = None,
    ) -> None:  # fmt: skip
        """Initialize pairwise dense composition modules."""
        super().__init__(shape, config, device=device, dtype=dtype)
        if config.mode == "gated_sum":
            self._gates = nn.ModuleList([nn.Linear(2 * width, width, device=device, dtype=dtype) for width in shape])
            self._projections = None
            self._activation_fn = None
            self._reset_gates()
            return

        self._gates = None
        self._projections = nn.ModuleList([nn.Linear(3 * width, width, device=device, dtype=dtype) for width in shape])
        self._activation_fn = utils.activation_from_str(config.activation)
        self._reset_projections()

    @property
    def config(self) -> PairwiseQueryPolicySettings:
        """Return typed settings for the pairwise query policy."""
        return super().config  # type: ignore[return-value]

    def _resolve_query(  # ------------------------------------------------------------------------
        self, *, cues: CueBundle, anchor_family: Optional[CueFamily] = None,
    ) -> list[Tensor]:  # fmt: skip
        """Compose a dense query from an ordered cue-family pair."""
        del anchor_family
        left_query, right_query = self._resolve_pair_family_queries(
            cues=cues,
            families=self.config.families,
            missing_behavior=self.config.missing_behavior,
        )
        if self.config.mode == "gated_sum":
            assert self._gates is not None
            mixed: list[Tensor] = []
            for gate_layer, left_tensor, right_tensor in zip(self._gates, left_query, right_query, strict=True):
                gate = torch.sigmoid(gate_layer(torch.cat((left_tensor, right_tensor), dim=1)))
                mixed.append(gate * left_tensor + (1.0 - gate) * right_tensor)
            return mixed

        assert self._projections is not None
        assert self._activation_fn is not None
        composed: list[Tensor] = []
        for projection, left_tensor, right_tensor in zip(self._projections, left_query, right_query, strict=True):
            joined = torch.cat((left_tensor, right_tensor, left_tensor * right_tensor), dim=1)
            tensor = projection(joined)
            tensor = torch.clamp(tensor, min=self.config.clamp_min, max=self.config.clamp_max)
            composed.append(self._activation_fn(tensor))
        return composed

    def _reset_gates(self) -> None:
        """Initialize gating layers to an even mixture before training."""
        assert self._gates is not None
        for layer in self._gates:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def _reset_projections(self) -> None:
        """Initialize pairwise product-projection layers to the elementwise product path."""
        assert self._projections is not None
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

    def _resolve_query(  # ------------------------------------------------------------------------
        self, *, cues: CueBundle, anchor_family: Optional[CueFamily] = None,
    ) -> list[Tensor]:  # fmt: skip
        return self._resolve_family_query(
            cues=cues,
            preferred_family=anchor_family,
            missing_behavior=self.config.missing_behavior,
            label="anchor_family",
        )

    def compose(  # -------------------------------------------------------------------------------
        self, *, cues: CueBundle, memory: MemoryEntry, role: RetrievalRole,
        anchor_family: Optional[CueFamily] = None,
    ) -> RetrievalEvidence:  # fmt: skip
        anchor = self._flatten_query(self.resolve_query(cues=cues, anchor_family=anchor_family))
        if not isinstance(memory, FactorMemoryStore):
            return self._anchor_only_evidence(role=role, anchor=anchor)

        score_terms = self._collect_score_terms(memory=memory, cues=cues)
        if not score_terms:
            return self._anchor_only_evidence(role=role, anchor=anchor)

        composed_logits = self._compose_score_terms(score_terms)
        return RetrievalEvidence(
            mode="factor_logits",
            role=role,
            score_terms=score_terms,
            composed_logits=composed_logits,
            anchor_logits=composed_logits,
            fallback_query=anchor,
            anchor_query=anchor,
            read_bank=self.config.read_bank,
            metadata={"temperature": float(self.config.temperature)},
        )

    @abstractmethod
    def _compose_score_terms(self, score_terms: list[ScoreTerm]) -> Tensor:
        """Compose the per-cue score terms into one logit tensor."""

    def _anchor_only_evidence(self, *, role: RetrievalRole, anchor: Tensor) -> RetrievalEvidence:
        """Return dense-compatible anchor-query evidence."""
        return RetrievalEvidence(
            mode="anchor_query",
            role=role,
            score_terms=[],
            composed_logits=None,
            anchor_logits=None,
            fallback_query=anchor,
            anchor_query=anchor,
        )

    def _collect_score_terms(  # ------------------------------------------------------------------
        self, *, memory: FactorMemoryStore, cues: CueBundle,
    ) -> list[ScoreTerm]:  # fmt: skip
        """Collect cue-scored logits against the configured banks."""
        score_terms = [
            term
            for family, query in cues.items()
            for term in [self._cue_score_term(family, query, memory, self._score_bank_for_family(family))]
            if term is not None
        ]
        if score_terms:
            return self._fill_missing_terms(score_terms, cues=cues)
        if self.config.missing_behavior == "error":
            raise ValueError(f"At least one cue family is required for query policy '{self.config.kind}'.")
        return []

    def _fill_missing_terms(  # -------------------------------------------------------------------
        self, score_terms: list[ScoreTerm], *, cues: CueBundle,
    ) -> list[ScoreTerm]:  # fmt: skip
        """Apply missing-cue behavior to any absent factor score terms."""
        if self.config.missing_behavior != "zeros":
            return score_terms
        present = {term.name for term in score_terms}
        expected = set(cues.names()) | set(self.config.cue_to_score_bank)
        if present == expected:
            return score_terms
        template = score_terms[0]
        completed = list(score_terms)
        for family in sorted(expected - present):
            completed.append(
                ScoreTerm(
                    name=family,
                    logits=torch.zeros_like(template.logits),
                    source_cues=(family,),
                    bank_name=self._score_bank_for_family(family),
                    normalization="temperature_scaled",
                )
            )
        return completed

    def _cue_score_term(  # -----------------------------------------------------------------------
        self, cue_name: str, query: list[Tensor], memory: FactorMemoryStore, bank_name: str,
    ) -> Optional[ScoreTerm]:  # fmt: skip
        """Return one cue-specific score term when that cue is available."""
        bank = memory.bank(bank_name, fallback_to_default=True)
        logits = self._compute_logits(self._flatten_query(query), bank.keys)
        return ScoreTerm(
            name=cue_name,
            logits=logits,
            source_cues=(cue_name,),
            bank_name=bank_name,
            normalization="temperature_scaled",
        )

    def _compute_logits(  # -----------------------------------------------------------------------
        self, query: Tensor, keys: Tensor,
    ) -> Tensor:  # fmt: skip
        """Project a flattened query against one key bank."""
        scale = math.sqrt(max(query.shape[1], 1)) * self.config.temperature
        return torch.einsum("bs,bts->bt", query.to(dtype=keys.dtype), keys) / scale


# =================================================================================================
class AdditiveQueryPolicy(_FactorScoreQueryPolicy):
    """Compose factor-memory scores with a sum in logit space."""

    @property
    def config(self) -> AdditiveQueryPolicySettings:
        return super().config  # type: ignore[return-value]

    def _compose_score_terms(self, score_terms: list[ScoreTerm]) -> Tensor:
        return sum((term.logits for term in score_terms[1:]), score_terms[0].logits)


# =================================================================================================
class MultiplicativeQueryPolicy(_FactorScoreQueryPolicy):
    """Compose factor-memory scores using an elementwise product."""

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
    """Compose an explicit anchor query with explicit refinement metadata."""

    @property
    def config(self) -> AnchorRefineQueryPolicySettings:
        return super().config  # type: ignore[return-value]

    def _resolve_query(  # ------------------------------------------------------------------------
        self, *, cues: CueBundle, anchor_family: Optional[CueFamily] = None,
    ) -> list[Tensor]:  # fmt: skip
        return self._resolve_family_query(
            cues=cues,
            preferred_family=anchor_family,
            missing_behavior=self.config.missing_behavior,
            label="anchor_family",
        )

    def compose(  # -------------------------------------------------------------------------------
        self, *,
        cues: CueBundle, memory: MemoryEntry, role: RetrievalRole,
        anchor_family: Optional[CueFamily] = None,
    ) -> RetrievalEvidence:  # fmt: skip
        anchor_query = self._flatten_query(self.resolve_query(cues=cues, anchor_family=anchor_family))
        if not isinstance(memory, FactorMemoryStore):
            return RetrievalEvidence(
                mode="anchor_query",
                role=role,
                score_terms=[],
                composed_logits=None,
                anchor_logits=None,
                fallback_query=anchor_query,
                anchor_query=anchor_query,
            )

        if anchor_family is None:
            raise ValueError("anchor_family is required for query policy 'anchor_refine'.")
        anchor_bank_name = self._score_bank_for_family(anchor_family)
        anchor_bank = memory.bank(anchor_bank_name, fallback_to_default=True)
        anchor_logits = self._compute_logits(anchor_query, anchor_bank.keys)
        return RetrievalEvidence(
            mode="anchor_refine",
            role=role,
            score_terms=[
                ScoreTerm(
                    name="anchor",
                    logits=anchor_logits,
                    source_cues=(anchor_family,),
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

    def _compute_logits(  # -----------------------------------------------------------------------
        self, query: Tensor, keys: Tensor,
    ) -> Tensor:  # fmt: skip
        """Project a flattened query against one refinement key bank."""
        scale = math.sqrt(max(query.shape[1], 1)) * self.config.temperature
        return torch.einsum("bs,bts->bt", query.to(dtype=keys.dtype), keys) / scale


# =================================================================================================
def build_query_policy(  # ------------------------------------------------------------------------
    shape: list[int], config: QueryPolicySettings, *,
    device: Optional[Device] = None, dtype: Optional[Dtype] = None,
) -> QueryPolicy:  # fmt: skip
    """Construct the configured query policy."""
    if config.kind == "anchor":
        return AnchorQueryPolicy(shape, config, device=device, dtype=dtype)
    if config.kind == "pairwise":
        return PairwiseQueryPolicy(shape, config, device=device, dtype=dtype)
    if config.kind == "additive":
        return AdditiveQueryPolicy(shape, config, device=device, dtype=dtype)
    if config.kind == "multiplicative":
        return MultiplicativeQueryPolicy(shape, config, device=device, dtype=dtype)
    if config.kind == "anchor_refine":
        return AnchorRefineQueryPolicy(shape, config, device=device, dtype=dtype)
    raise ValueError(f"Unsupported query policy '{config.kind}'.")


# =================================================================================================
__all__ = [
    "AdditiveQueryPolicySettings", "AnchorQueryPolicySettings", "AnchorRefineQueryPolicySettings",
    "CueBundle", "MissingCueBehavior", "MultiplicativeQueryPolicySettings",
    "PairwiseQueryPolicySettings", "QueryPolicy", "QueryPolicySettings", "RetrievalEvidence",
    "RetrievalRefinementStep", "ScoreTerm", "build_query_policy",
]  # fmt: skip
