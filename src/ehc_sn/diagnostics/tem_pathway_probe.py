"""TEM v1 place-code-to-logit pathway probe.

Localises where retrieved/ancestral content is lost between the HPC place code
and the final observation logits.

The decoder chain is::

    p_*  →  lec_to_hpc.inverse(p_*)  →  prediction_freq slice  →  w_x * code + b_x  →  MLPDecoder  →  logits

Stage 0: Place code (p_post / p_recall / p_path)
Stage 1: Inverse-projected sensory code (lec_to_hpc.inverse)
Stage 2: Single-frequency slice (prediction_freq)
Stage 3: Decoder input after w_x/b_x transform
Stage 4: Logits

Usage::

    from ehc_sn.diagnostics.tem_pathway_probe import produce_tem_pathway_probe, persist_pathway_probe

    result = produce_tem_pathway_probe(model, tem_input, observation_ids)
    path = persist_pathway_probe(result, Path("artifacts/evaluation/tem_v1/diagnostics/"))
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from pydantic import BaseModel, Field
from torch import Tensor

from ehc_sn.models.tem.tem_v1 import TEMInputV1, TEMModelV1
from ehc_sn.types import DenseMemoryStore

# =============================================================================
# Stage schema
# =============================================================================


class StageMetrics(BaseModel, extra="forbid"):
    """Representation statistics at one stage of the decode chain.

    Attributes:
        norm_mean: Mean L2 norm over timesteps.
        feature_std_mean: Mean per-feature standard deviation over time.
        pairwise_cosine_mean: Mean pairwise cosine similarity across timesteps.
        effective_rank: Effective (nuclear) rank of the code matrix ``(T, D)``.
    """

    norm_mean: float = Field(default=float("nan"))
    feature_std_mean: float = Field(default=float("nan"))
    pairwise_cosine_mean: float = Field(default=float("nan"))
    effective_rank: float = Field(default=float("nan"))


class LogitMetrics(BaseModel, extra="forbid"):
    """Logit-level statistics for one pathway.

    Attributes:
        entropy_mean: Mean softmax entropy over timesteps.
        top1_margin_mean: Mean margin between top-1 and top-2 logits.
        gt_prob_mean: Mean softmax probability assigned to the ground-truth class.
        class2_prob_mean: Mean softmax probability assigned to class 2.
        accuracy: Argmax accuracy vs ground-truth observation IDs.
        prediction_histogram: Map from predicted class ID to count (top 20).
    """

    entropy_mean: float = Field(default=float("nan"))
    top1_margin_mean: float = Field(default=float("nan"))
    gt_prob_mean: float = Field(default=float("nan"))
    class2_prob_mean: float = Field(default=float("nan"))
    accuracy: float = Field(default=float("nan"))
    prediction_histogram: dict[str, int] = Field(default_factory=dict)


class PathwayProbeResult(BaseModel, extra="forbid"):
    """Complete pathway-probe result for one TEM v1 episode.

    Attributes:
        model_family: Always ``"tem-v1"``.
        episode_steps: Number of probed timesteps.
        n_freq: Number of HPC frequency modules.
        prediction_freq: Which frequency band is used for single-scale decode.
        w_x: Learned decoder scale parameter.
        b_x_norm: Norm of the decoder bias vector.
        inference: Per-stage metrics for the inference (sensory-conditioned) pathway.
        retrieved: Per-stage metrics for the retrieved (corrected-grid) pathway.
        ancestral: Per-stage metrics for the ancestral (prior) pathway.
        contrast_post_vs_recall: Ratio ``retrieved / inference`` for
            each stage.  Values near 1.0 indicate the pathway is as strong as
            inference; near 0.0 indicates complete content loss.
        contrast_post_vs_path: Same ratio for ancestral vs inference.
    """

    model_family: str = "tem-v1"
    episode_steps: int = 0
    n_freq: int = 0
    prediction_freq: int = 0
    w_x: float = float("nan")
    b_x_norm: float = float("nan")

    inference: list[StageMetrics] = Field(default_factory=list)
    retrieved: list[StageMetrics] = Field(default_factory=list)
    ancestral: list[StageMetrics] = Field(default_factory=list)

    logits_post: LogitMetrics = Field(default_factory=LogitMetrics)
    logits_recall: LogitMetrics = Field(default_factory=LogitMetrics)
    logits_path: LogitMetrics = Field(default_factory=LogitMetrics)

    contrast_post_vs_recall: list[float] = Field(default_factory=list)
    contrast_post_vs_path: list[float] = Field(default_factory=list)


# =============================================================================
# Stage indices
# =============================================================================

STAGE_PLACE_CODE = 0
STAGE_INVERSE = 1
STAGE_SINGLE_FREQ = 2
STAGE_DECODER_INPUT = 3
STAGE_NAMES = [
    "place_code",
    "inverse_projection",
    "single_freq_slice",
    "decoder_input",
]

N_STAGES = len(STAGE_NAMES)


# =============================================================================
# Helpers
# =============================================================================


def _pairwise_cosine_mean(z: Tensor) -> float:
    """Mean pairwise cosine similarity over the first dimension of *z*."""
    T = int(z.shape[0])
    if T <= 1:
        return float("nan")
    z_norm = z / (z.norm(dim=-1, keepdim=True) + 1e-8)
    sim = z_norm @ z_norm.T
    triu = torch.triu_indices(T, T, offset=1, device=sim.device)
    return float(sim[triu[0], triu[1]].mean().item())


def _effective_rank(z: Tensor) -> float:
    """Effective rank via singular-value entropy, ``(T, D)`` -> float."""
    if z.ndim != 2 or z.shape[0] < 2:
        return float("nan")
    s = torch.linalg.svdvals(z.to(dtype=torch.float64))
    s = s[s > 1e-12]
    if s.numel() <= 1:
        return float(s.numel())
    p = s / s.sum()
    entropy = -(p * p.log()).sum()
    return float(torch.exp(entropy).item())


def _stage_metrics(code: Tensor) -> StageMetrics:
    """Compute StageMetrics from a ``(T, D)`` code tensor."""
    if code.ndim != 2:
        raise ValueError(
            f"Expected rank-2 (T, D) tensor, got shape {tuple(code.shape)}."
        )
    T = int(code.shape[0])
    if T < 2:
        return StageMetrics()
    return StageMetrics(
        norm_mean=float(code.norm(dim=-1).mean().item()),
        feature_std_mean=float(code.std(dim=0).mean().item()),
        pairwise_cosine_mean=_pairwise_cosine_mean(code),
        effective_rank=_effective_rank(code),
    )


def _logit_metrics(logits: Tensor, obs_ids: Tensor) -> LogitMetrics:
    """Compute LogitMetrics from logits ``(T, C)`` and ground-truth ``(T,)``."""
    T = int(logits.shape[0])
    if T < 1:
        return LogitMetrics()
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * probs.log()).sum(dim=-1)
    sorted_logits, _ = logits.sort(dim=-1, descending=True)
    top1_margin = sorted_logits[:, 0] - sorted_logits[:, 1]
    gt_probs = probs[torch.arange(T), obs_ids]
    class2_probs = probs[:, 2]
    pred = logits.argmax(dim=-1)
    acc = float((pred == obs_ids).float().mean().item())

    # Histogram: top 20 predicted classes
    hist: dict[str, int] = {}
    for idx in pred.tolist():
        key = str(idx)
        hist[key] = hist.get(key, 0) + 1
    top20 = dict(sorted(hist.items(), key=lambda x: -x[1])[:20])

    return LogitMetrics(
        entropy_mean=float(entropy.mean().item()),
        top1_margin_mean=float(top1_margin.mean().item()),
        gt_prob_mean=float(gt_probs.mean().item()),
        class2_prob_mean=float(class2_probs.mean().item()),
        accuracy=acc,
        prediction_histogram=top20,
    )


def _contrast(
    retrieved: list[StageMetrics],
    inference: list[StageMetrics],
) -> list[float]:
    """Compute per-stage ratio ``retrieved / inference``."""
    ratios: list[float] = []
    for r, i in zip(retrieved, inference, strict=True):
        pc = r.pairwise_cosine_mean
        ic = i.pairwise_cosine_mean
        if ic is not None and abs(ic) > 1e-8 and pc is not None:
            ratios.append(min(pc / ic, 10.0))
        else:
            ratios.append(float("nan"))
    return ratios


# =============================================================================
# Public API
# =============================================================================


def produce_tem_pathway_probe(  # -------------------------------------------
    model: TEMModelV1,
    input_batch: TEMInputV1,
    observation_ids: Tensor,
    *,
    prediction_freq: int = 0,
    w_x_val: float = 1.0,
    b_x: Tensor | None = None,
    decoder: torch.nn.Module | None = None,
) -> PathwayProbeResult:
    """Run the place-code-to-logit pathway probe for one TEM v1 episode.

    Args:
        model: TEM v1 model in evaluation mode.
        input_batch: Complete episode TEM input (T steps).
        observation_ids: Ground-truth observation IDs, ``(T,)`` integer.
        prediction_freq: Frequency band index used for single-scale decode.
        w_x_val: Learned ``w_x`` scale from the adapter's ``ArenaOutputsDecoderV1``.
        b_x: Learned ``b_x`` bias vector from the adapter's decoder.
        decoder: Optional ``MLPDecoder`` from the adapter for logit computation.
            When ``None``, logits are the raw ``w_x * code + b_x`` output.

    Returns:
        Filled ``PathwayProbeResult``.

    Raises:
        TypeError: If HPC backend is not ``HPCAttractor``.
        RuntimeError: If model is in training mode.
    """
    if model.training:
        raise RuntimeError("Pathway probe requires model.eval().")

    n_freq = len(model._config.hpc.shape)
    device = next(model.parameters()).device
    T = int(input_batch.observation_embedding[0].shape[0])

    state = model.init_state(1, device=device)

    # ---- Forward pass: collect intermediate codes per step ----
    all_p_post: list[list[Tensor]] = []
    all_p_recall: list[list[Tensor]] = []
    all_p_path: list[list[Tensor]] = []

    with torch.no_grad():
        for t in range(T):
            step_input = TEMInputV1(
                observation_embedding=[
                    freq_t[t : t + 1]
                    for freq_t in input_batch.observation_embedding
                ],
                previous_action=input_batch.previous_action[t : t + 1],
                episode_start=(
                    input_batch.episode_start[t : t + 1]
                    if input_batch.episode_start is not None
                    else None
                ),
                landmark_id=None,
            )
            output, state = model(step_input, state=state)

            all_p_post.append(
                [p.clone().cpu() for p in output.place_codes.post]
            )
            all_p_path.append(
                [p.clone().cpu() for p in output.place_codes.path]
            )
            if output.place_codes.recall is not None:
                all_p_recall.append(
                    [p.clone().cpu() for p in output.place_codes.recall]
                )

    # Stack per-step codes: list of (T, D_f) per frequency
    def _stack(bundle: list[list[Tensor]]) -> list[Tensor]:
        return [
            torch.cat([step[f] for step in bundle], dim=0)
            for f in range(n_freq)
        ]

    p_post = _stack(all_p_post)
    p_path = _stack(all_p_path)
    has_recall = len(all_p_recall) == T
    p_recall = _stack(all_p_recall) if has_recall else p_post

    obs_ids = observation_ids.to(device="cpu", dtype=torch.long)

    # ---- Stage 0: Place code ----
    def _flatten(bundle: list[Tensor]) -> Tensor:
        return torch.cat(bundle, dim=-1)  # (T, S)

    s0_post = _stage_metrics(_flatten(p_post))
    s0_ret = _stage_metrics(_flatten(p_recall))
    s0_pri = _stage_metrics(_flatten(p_path))

    # ---- Stage 1: Inverse projection (lec_to_hpc.inverse) ----
    inv_post = model.lec_to_hpc.inverse(p_post)  # list of (T, D_f) in LEC space
    inv_ret = model.lec_to_hpc.inverse(p_recall)
    inv_pri = model.lec_to_hpc.inverse(p_path)

    s1_post = _stage_metrics(torch.cat(inv_post, dim=-1))
    s1_ret = _stage_metrics(torch.cat(inv_ret, dim=-1))
    s1_pri = _stage_metrics(torch.cat(inv_pri, dim=-1))

    # ---- Stage 2: Single-frequency slice ----
    sf_post = inv_post[prediction_freq]  # (T, D_f0)
    sf_ret = inv_ret[prediction_freq]
    sf_pri = inv_pri[prediction_freq]

    s2_post = _stage_metrics(sf_post)
    s2_ret = _stage_metrics(sf_ret)
    s2_pri = _stage_metrics(sf_pri)

    # ---- Stage 3: Decoder input (w_x * code + b_x) ----
    b_x_vec = torch.zeros(sf_post.shape[-1]) if b_x is None else b_x
    code_post = w_x_val * sf_post + b_x_vec.to(dtype=sf_post.dtype)
    code_ret = w_x_val * sf_ret + b_x_vec.to(dtype=sf_ret.dtype)
    code_pri = w_x_val * sf_pri + b_x_vec.to(dtype=sf_pri.dtype)

    s3_post = _stage_metrics(code_post)
    s3_ret = _stage_metrics(code_ret)
    s3_pri = _stage_metrics(code_pri)

    # ---- Stage 4: Logits ----
    if decoder is not None:
        logits_post = decoder(code_post)
        logits_ret = decoder(code_ret)
        logits_pri = decoder(code_pri)
    else:
        logits_post = code_post
        logits_ret = code_ret
        logits_pri = code_pri

    # Compute logit metrics from the actual logit tensors.
    logit_metrics_inf = _logit_metrics(logits_post, obs_ids)
    logit_metrics_ret = _logit_metrics(logits_ret, obs_ids)
    logit_metrics_pri = _logit_metrics(logits_pri, obs_ids)

    return PathwayProbeResult(
        model_family="tem-v1",
        episode_steps=T,
        n_freq=n_freq,
        prediction_freq=prediction_freq,
        w_x=w_x_val,
        b_x_norm=float(b_x_vec.norm().item()),
        inference=[s0_post, s1_post, s2_post, s3_post],
        retrieved=[s0_ret, s1_ret, s2_ret, s3_ret],
        ancestral=[s0_pri, s1_pri, s2_pri, s3_pri],
        logits_post=logit_metrics_inf,
        logits_recall=logit_metrics_ret,
        logits_path=logit_metrics_pri,
        contrast_post_vs_recall=_contrast(
            [s0_ret, s1_ret, s2_ret, s3_ret],
            [s0_post, s1_post, s2_post, s3_post],
        ),
        contrast_post_vs_path=_contrast(
            [s0_pri, s1_pri, s2_pri, s3_pri],
            [s0_post, s1_post, s2_post, s3_post],
        ),
    )


# =============================================================================
def persist_pathway_probe(  # -------------------------------------------------
    result: PathwayProbeResult,
    output_dir: Path,
    *,
    filename: str = "pathway_probe.json",
) -> Path:
    """Persist a pathway probe result to JSON.

    Args:
        result: Probe result to persist.
        output_dir: Output directory (created if needed).
        filename: Output filename.

    Returns:
        Absolute path to the written JSON file.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    out_path.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, allow_nan=True),
        encoding="utf-8",
    )
    return out_path


def load_pathway_probe(  # ----------------------------------------------------
    path: str | Path,
) -> PathwayProbeResult:
    """Load a pathway probe result from a persisted JSON file.

    Args:
        path: Path to a ``pathway_probe.json`` file.

    Returns:
        Deserialised ``PathwayProbeResult``.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return PathwayProbeResult.model_validate(raw)


# =============================================================================
def format_pathway_probe_table(  # --------------------------------------------
    result: PathwayProbeResult,
) -> str:
    """Format a pathway probe result as a human-readable table.

    Shows per-stage metrics for each pathway and contrast ratios.
    """
    lines = [
        f"Pathway probe: {result.model_family}",
        f"  Steps: {result.episode_steps}  Freq: {result.n_freq}  pred_freq: {result.prediction_freq}",
        f"  w_x: {_fmt(result.w_x)}  b_x_norm: {_fmt(result.b_x_norm)}",
        "",
    ]

    def _row(
        stage: int,
        name: str,
        inf: StageMetrics,
        ret: StageMetrics,
        pri: StageMetrics,
    ) -> str:
        pc_inf = _fmt(inf.pairwise_cosine_mean)
        pc_ret = _fmt(ret.pairwise_cosine_mean)
        pc_pri = _fmt(pri.pairwise_cosine_mean)
        return f"| {stage} | {name:<20s} | {pc_inf:>8s} | {pc_ret:>8s} | {pc_pri:>8s} |"

    lines.append(
        "| Stage | Name                 | inference | retrieved | ancestral |"
    )
    lines.append(
        "| ----- | -------------------- | --------- | --------- | --------- |"
    )
    for s in range(N_STAGES):
        lines.append(
            _row(
                s,
                STAGE_NAMES[s],
                result.inference[s],
                result.retrieved[s],
                result.ancestral[s],
            )
        )
    lines.append("")
    lines.append(
        "| Stage | Name                 | contrast inf/ret | contrast inf/pri |"
    )
    lines.append(
        "| ----- | -------------------- | ---------------- | ---------------- |"
    )
    for s in range(N_STAGES):
        cr = _fmt(result.contrast_post_vs_recall[s])
        ca = _fmt(result.contrast_post_vs_path[s])
        lines.append(f"| {s} | {STAGE_NAMES[s]:<20s} | {cr:>16s} | {ca:>16s} |")

    lines.append("")
    lines.append("--- Logits ---")
    for label, lm in [
        ("inference", result.logits_post),
        ("retrieved", result.logits_recall),
        ("ancestral", result.logits_path),
    ]:
        lines.append(
            f"  {label}: acc={_fmt(lm.accuracy)} "
            f"entropy={_fmt(lm.entropy_mean)} "
            f"gt_prob={_fmt(lm.gt_prob_mean)} "
            f"class2_prob={_fmt(lm.class2_prob_mean)} "
            f"margin={_fmt(lm.top1_margin_mean)}"
        )

    # Prediction histogram for retrieved
    lines.append("")
    lines.append("Prediction histograms (top 5 classes):")
    for label, lm in [
        ("inference", result.logits_post),
        ("retrieved", result.logits_recall),
        ("ancestral", result.logits_path),
    ]:
        top5 = dict(list(lm.prediction_histogram.items())[:5])
        lines.append(f"  {label}: {top5}")

    return "\n".join(lines)


def _fmt(v: float | None, decimals: int = 4) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    return f"{v:.{decimals}f}"


# =============================================================================
__all__ = [
    "PathwayProbeResult",
    "StageMetrics",
    "LogitMetrics",
    "produce_tem_pathway_probe",
    "persist_pathway_probe",
    "load_pathway_probe",
    "format_pathway_probe_table",
]
