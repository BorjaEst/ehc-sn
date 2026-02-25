""" """  # TODO: We need a better name for this module and probably split it into multiple files.

# TODO: This is currently a dumping ground for all loss-related code, but we can organize it better as we go.

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Normal

from ehc_sn import utils
from ehc_sn.data.world import WorldStep
from ehc_sn.model import TEMOutput, TEMState
from ehc_sn.settings import AbstractLocationSettings  # fmt: skip
from ehc_sn.settings import GroundedLocationSettings  # fmt: skip
from ehc_sn.settings import LossSettings  # fmt: skip
from ehc_sn.settings import RegularizationSettings  # fmt: skip
from ehc_sn.settings import SensoryReconstructionSettings  # fmt: skip
from ehc_sn.types import AbstractLocation, GroundedLocation, Prediction, Reduction, Scalar


@dataclass
class LossX:
    """Sensory reconstruction loss container ($L_x$).

    Holds cross-entropy losses for the three TEM sensory prediction pathways.
    Supports arithmetic operations for accumulation during rollouts.

    Attributes:
        infer: Inference pathway loss ($p_{inf} \to x$). Predicts sensory
            observation directly from inferred grounded location.
        retrieved: Retrieved pathway loss ($g_{inf} \to p \to x$). Predicts via
            grounded location retrieved from inferred abstract location.
        ancestral: Ancestral/generative pathway loss ($g_{gen} \to p \to x$).
            Predicts via one-state transition in abstract space.

    Note:
        Shape is scalar (reduced) or ``(B,)`` depending on reduction mode.
    """

    infer: Tensor
    retrieved: Tensor
    ancestral: Tensor

    @classmethod
    def zero(cls, *, device, dtype=torch.float32) -> "LossX":
        """Create a zero-initialized instance.

        Args:
            device: Torch device.
            dtype: Torch dtype.

        Returns:
            A `LossX` instance with all fields set to zero scalars.
        """
        z = torch.zeros((), device=device, dtype=dtype)
        return cls(infer=z.clone(), retrieved=z.clone(), ancestral=z.clone())

    def __add__(self, other: "LossX") -> "LossX":
        return LossX(
            infer=self.infer + other.infer,
            retrieved=self.retrieved + other.retrieved,
            ancestral=self.ancestral + other.ancestral,
        )

    def __truediv__(self, divisor: int | float) -> "LossX":
        return LossX(
            infer=self.infer / divisor,
            retrieved=self.retrieved / divisor,
            ancestral=self.ancestral / divisor,
        )

    def __mul__(self, scalar: Scalar) -> "LossX":
        return LossX(
            infer=self.infer * scalar,
            retrieved=self.retrieved * scalar,
            ancestral=self.ancestral * scalar,
        )

    def __rmul__(self, scalar: Scalar) -> "LossX":
        return self.__mul__(scalar)

    @property
    def total(self) -> Tensor:
        """Total reconstruction loss (sum of pathways)."""
        return self.infer + self.retrieved + self.ancestral


class SensoryReconstructionLoss(nn.Module):
    """Compute sensory reconstruction losses ($L_x$).

    Evaluates the model's ability to predict sensory observations through three
    parallel pathways, each testing different aspects of TEM's spatial memory:

    1. **Inference pathway**: Direct prediction from infered place cells
    2. **Retrieved pathway**: Prediction via grid→place retrieval
    3. **Ancestral pathway**: Prediction via grid transition and retrieval

    All pathways use cross-entropy loss with the ground-truth observation.
    """

    def __init__(self, settings: Optional[SensoryReconstructionSettings] = None):
        """Initialize the loss module.

        Args:
            settings: Optional configuration. If omitted, defaults are used.
        """
        super().__init__()
        self.settings = settings or SensoryReconstructionSettings()

    @property
    def reduction(self) -> Reduction:
        """Map reduction from settings to PyTorch string."""
        return self.settings.reduction

    @property
    def weight(self) -> float:
        """Return weight multiplier applied to all $L_x$ components."""
        return self.settings.weight

    def forward(
        self, y_p_inf: Prediction, y_gen_gi: Prediction, y_gen_gg: Prediction, o_labels: Tensor
    ) -> LossX:
        """Compute `LossX` from logits and ground-truth observations.

        Args:
            y_p_inf: Prediction from inference pathway.
            y_gen_gi: Prediction from retrieved pathway.
            y_gen_gg: Prediction from ancestral pathway.
            o_labels: Ground-truth observation, one-hot encoded `(B, n_classes)`.

        Returns:
            LossX: Per-pathway cross-entropy losses.

        Raises:
            ValueError: If `o_logits` does not contain exactly 3 tensors.
        """
        # Convert one-hot to class indices (legacy: labels = argmax(o, 1))
        labels = torch.argmax(o_labels, dim=1)

        # Pathway order is fixed to match the legacy implementation.
        loss_infer = F.cross_entropy(y_p_inf.logits, labels, reduction=self.reduction)
        loss_retrieved = F.cross_entropy(y_gen_gi.logits, labels, reduction=self.reduction)
        loss_ancestral = F.cross_entropy(y_gen_gg.logits, labels, reduction=self.reduction)

        return LossX(infer=loss_infer, retrieved=loss_retrieved, ancestral=loss_ancestral) * self.weight


@dataclass
class LossG:
    """Abstract location transition loss container ($L_g$).

    Enforces consistency between inferred and predicted abstract locations (grid cells).

    Attributes:
        transition: Transition consistency ($||g_{inf} - g_{gen}||^2$ or NLL divergence).
            Compares inference with one-state prediction from previous timestate.
    """

    transition: Tensor

    @classmethod
    def zero(cls, *, device, dtype=torch.float32) -> "LossG":
        """Create a zero-initialized instance.

        Args:
            device: Torch device.
            dtype: Torch dtype.

        Returns:
            A `LossG` instance with the field set to a zero scalar.
        """
        return cls(transition=torch.zeros((), device=device, dtype=dtype))

    def __add__(self, other: "LossG") -> "LossG":
        return LossG(transition=self.transition + other.transition)

    def __truediv__(self, divisor: int | float) -> "LossG":
        return LossG(transition=self.transition / divisor)

    def __mul__(self, scalar: Scalar) -> "LossG":
        return LossG(transition=self.transition * scalar)

    def __rmul__(self, scalar: Scalar) -> "LossG":
        return self.__mul__(scalar)

    @property
    def total(self) -> Tensor:
        """Total abstract-location loss."""
        return self.transition


class AbstractLocationLoss(nn.Module):
    """Compute abstract location transition consistency ($L_g$).

        Enforces that the inferred abstract location (grid cells from path integration
        and sensory input) matches the predicted abstract location (from previous
        grid cells via transition model).

        Modes:
            mse:
                Surrogate objective using squared error (legacy):
                $0.5 sum_f ||g_{inf}^f - g_{gen}^f||^2$

            nll:
                Uncertainty-weighted nll divergence. Requires :class:`LocationBelief`
                input with mean and uncertainty:
    <<<<<<< Updated upstream
                $sum_f D_{KL}(g_{inf}^f || mathcal{N}(g_{gen}^f, sigma_{gen}^f))$
    =======
                $\sum_f D_{nll}(g_{inf}^f || \mathcal{N}(g_{gen}^f, \sigma_{gen}^f))$
    >>>>>>> Stashed changes
    """

    def __init__(self, settings: Optional[AbstractLocationSettings] = None):
        """Initialize the loss module.

        Args:
            settings: Optional configuration. If omitted, defaults are used.
        """
        super().__init__()
        self.settings = settings or AbstractLocationSettings()

    @property
    def mode(self) -> str:
        return self.settings.mode

    @property
    def reduction(self) -> Reduction:
        return self.settings.reduction

    @property
    def weight(self) -> float:
        return self.settings.weight

    def forward(
        self, g_inf: AbstractLocation, g_gen: AbstractLocation, uncertainty: Optional[list[Tensor]] = None
    ) -> LossG:
        """Compute `LossG` from inferred and generated abstract locations.

        Args:
            g_inf: Inferred abstract location (posterior from sensory + path integration).
                List of tensors, one per frequency module. Shape: [(B, n_g_f1), (B, n_g_f2), ...]
            g_gen: Generated abstract location (from previous state via transition).
                List of tensors, one per frequency module. Shape: [(B, n_g_f1), (B, n_g_f2), ...]
            uncertainty: Optional uncertainty tensors for nll mode.
                List of tensors, one per frequency module. Shape: [(B, n_g_f1), (B, n_g_f2), ...]

        Returns:
            LossG dataclass with transition loss.
        """
        if self.mode == "mse":
            transition_loss = self._loss_mse(g_inf, g_gen)
        elif self.mode == "nll" and uncertainty is None:
            raise ValueError("Uncertainty tensors must be provided for nll loss mode.")
        elif self.mode == "nll":
            transition_loss = self._loss_kl(g_inf, g_gen, uncertainty)
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Expected 'mse' or 'nll'.")
        transition_loss = utils.reduce_per_env(transition_loss, self.reduction)
        return LossG(transition=transition_loss) * self.weight

    # We can move _loss_mse and _loss_kl outside the class as standalone functions if desired.

    def _loss_mse(self, g_inf: AbstractLocation, g_gen: AbstractLocation) -> Tensor:
        """Compute MSE loss between inferred and generated abstract locations.

        Args:
            g_inf: Inferred abstract location (posterior from sensory + path integration).
                List of tensors, one per frequency module. Shape: [(B, n_g_f1), (B, n_g_f2), ...]
            g_gen: Generated abstract location (from previous state via transition).
                List of tensors, one per frequency module. Shape: [(B, n_g_f1), (B, n_g_f2), ...]

        Returns:
            Tensor: Per-environment MSE loss (shape: (B,)).
        """
        loss = torch.zeros(g_inf[0].shape[0], device=g_inf[0].device, dtype=g_inf[0].dtype)
        for g_i, g_g in zip(g_inf, g_gen):
            loss += 0.5 * (g_i - g_g).pow(2).sum(dim=-1)
        return loss

    def _loss_kl(self, g_inf: AbstractLocation, g_gen: AbstractLocation, uncertainty: list[Tensor]) -> Tensor:
        """Compute uncertainty-weighted NLL loss (negative log-likelihood).

        Computes the negative log-likelihood of the inferred abstract location under
        a Gaussian prior defined by the transition model. This is equivalent to a
        precision-weighted MSE loss.

        Note:
            This is NOT a true nll divergence (which would require posterior uncertainty).
            It computes $-\log p(g_{inf} | g_{gen}, \sigma_{gen})$ where the prior
            is $\mathcal{N}(g_{gen}, \sigma_{gen})$.

        Args:
            g_inf: Inferred abstract location (posterior point estimate from sensory + path integration).
                List of tensors, one per frequency module. Shape: [(B, n_g_f1), (B, n_g_f2), ...]
            g_gen: Generated abstract location mean (from previous state via transition).
                List of tensors, one per frequency module. Shape: [(B, n_g_f1), (B, n_g_f2), ...]
            uncertainty: Standard deviation tensors for generated abstract locations.
                List of tensors, one per frequency module. Shape: [(B, n_g_f1), (B, n_g_f2), ...]

        Returns:
            Tensor: Per-environment NLL loss (shape: (B,)).
        """

        loss = torch.zeros(g_inf[0].shape[0], device=g_inf[0].device, dtype=g_inf[0].dtype)
        for mu_post, mu_prior, sigma_prior in zip(g_inf, g_gen, uncertainty):
            # Clamp uncertainty to avoid numerical issues
            sigma_prior = torch.clamp(sigma_prior, min=1e-6)
            # Create prior distribution and compute negative log-likelihood
            prior = Normal(loc=mu_prior, scale=sigma_prior)
            nll = -prior.log_prob(mu_post).sum(dim=-1)
            loss += nll
        return loss


@dataclass
class LossP:
    """Grounded location consistency loss container ($L_p$).

    Enforces consistency between inferred grounded location (place cells) and
    memory retrieval pathways using squared-error terms.

    Attributes:
        abstract: Abstract pathway consistency ($||p_{inf} - p_{gen}||^2$).
            Compares inference with retrieval via abstract location ($g \to p$).
        sensory: Sensory pathway consistency ($||p_{inf} - p_{inf_x}||^2$).
            Compares inference with retrieval via sensory input ($x \to p$).
    """

    abstract: Tensor
    sensory: Tensor

    @classmethod
    def zero(cls, *, device, dtype=torch.float32) -> "LossP":
        """Create a zero-initialized instance.

        Args:
            device: Torch device.
            dtype: Torch dtype.

        Returns:
            A `LossP` instance with all fields set to zero scalars.
        """
        z = torch.zeros((), device=device, dtype=dtype)
        return cls(abstract=z.clone(), sensory=z.clone())

    def __add__(self, other: "LossP") -> "LossP":
        return LossP(
            abstract=self.abstract + other.abstract,
            sensory=self.sensory + other.sensory,
        )

    def __truediv__(self, divisor: int | float) -> "LossP":
        return LossP(
            abstract=self.abstract / divisor,
            sensory=self.sensory / divisor,
        )

    def __mul__(self, scalar: Scalar) -> "LossP":
        return LossP(
            abstract=self.abstract * scalar,
            sensory=self.sensory * scalar,
        )

    def __rmul__(self, scalar: Scalar) -> "LossP":
        return self.__mul__(scalar)

    @property
    def total(self) -> Tensor:
        """Total grounded-location loss."""
        return self.abstract + self.sensory


class GroundedLocationLoss(nn.Module):
    """Compute grounded location consistency ($L_p$).

    Ensures that inferred place cell representations remain consistent with
    memory retrieval through both abstract and sensory pathways.

    Components:
        Abstract ($L_{p,g}$):
            $0.5 \sum_f ||p_{inf}^f - p_{gen,gi}^f||^2$
            where $p_{gen,gi}$ is retrieved via inferred grid cells ($g_{inf} to p$)

        Sensory ($L_{p,x}$):
            $0.5 \sum_f ||p_{inf}^f - p_{xi}^f||^2$
            where $p_{xi}$ is retrieved via sensory input ($x to p$)
            (optional, controlled by ``use_x_cued_recall`` flag)
    """

    def __init__(self, settings: Optional[GroundedLocationSettings] = None):
        """Initialize the loss module.

        Args:
            settings: Optional configuration. If omitted, defaults are used.
        """
        super().__init__()
        self.settings = settings or GroundedLocationSettings()

    @property
    def reduction(self) -> Reduction:
        return self.settings.reduction

    @property
    def weight(self) -> float:
        return self.settings.weight

    def forward(
        self, p_inf: GroundedLocation, p_gen_gi: GroundedLocation, p_xi: Optional[GroundedLocation] = None
    ) -> LossP:
        """Compute `LossP` from grounded location codes.

        Args:
            p_inf: Inferred grounded location (conjunctive place cells from x_ and g_).
                List of tensors, one per frequency module. Shape: [(B, n_p_f1), (B, n_p_f2), ...]
            p_gen_gi: Grounded location retrieved via abstract location (g_inf → p).
                Required for L_p_g computation.
            p_xi: Grounded location from sensory memory retrieval (x → p).
                Used for L_p_x if settings.use_x_cued_recall=True.

        Returns:
            LossP dataclass with abstract and sensory components.

        Notes:
            Squared errors include a 0.5 factor by convention. When `use_x_cued_recall`
            is False, the sensory term is zeroed.
        """
        batch_size = p_inf[0].shape[0]
        loss_abstract = torch.zeros(batch_size, device=p_inf[0].device, dtype=p_inf[0].dtype)
        for p_i, p_g in zip(p_inf, p_gen_gi):
            loss_abstract += 0.5 * (p_i - p_g).pow(2).sum(dim=-1)

        loss_sensory = torch.zeros(batch_size, device=p_inf[0].device, dtype=p_inf[0].dtype)
        if self.settings.use_x_cued_recall and p_xi is not None:
            for p_i, p_x in zip(p_inf, p_xi):
                loss_sensory += 0.5 * (p_i - p_x).pow(2).sum(dim=-1)

        abstract = utils.reduce_per_env(loss_abstract, self.reduction)
        sensory = utils.reduce_per_env(loss_sensory, self.reduction)
        return LossP(abstract=abstract, sensory=sensory) * self.weight


@dataclass
class LossReg:
    """Regularization loss container.

    Auxiliary penalties to encourage sparse/bounded representations.

    Attributes:
        g_l2: L2 penalty on abstract location codes (grid cells).
        p_l1: L1 penalty on grounded location codes (place cells).
    """

    g_l2: Tensor
    p_l1: Tensor

    @classmethod
    def zero(cls, *, device, dtype=torch.float32) -> "LossReg":
        """Create a zero-initialized instance.

        Args:
            device: Torch device.
            dtype: Torch dtype.

        Returns:
            A `LossReg` instance with all fields set to zero scalars.
        """
        z = torch.zeros((), device=device, dtype=dtype)
        return cls(g_l2=z.clone(), p_l1=z.clone())

    def __add__(self, other: "LossReg") -> "LossReg":
        return LossReg(
            g_l2=self.g_l2 + other.g_l2,
            p_l1=self.p_l1 + other.p_l1,
        )

    def __truediv__(self, divisor: int | float) -> "LossReg":
        return LossReg(
            g_l2=self.g_l2 / divisor,
            p_l1=self.p_l1 / divisor,
        )

    def __mul__(self, scalar: Scalar) -> "LossReg":
        """Uniformly scale all regularization components."""
        return LossReg(g_l2=self.g_l2 * scalar, p_l1=self.p_l1 * scalar)

    def __rmul__(self, scalar: Scalar) -> "LossReg":
        return self.__mul__(scalar)

    @property
    def total(self) -> Tensor:
        """Total regularization loss."""
        return self.g_l2 + self.p_l1


class RegularizationLoss(nn.Module):
    """Compute auxiliary regularization penalties.

    Encourages sparse and bounded neural representations:

    Components:
        g_l2:
            $\lambda_g \sum_f ||g^f||_2^2$
            Bounds magnitude of grid cell activations

        p_l1:
            $\lambda_p \sum_f ||p^f||_1$
            Encourages sparse place cell activations
    """

    def __init__(self, settings: Optional[RegularizationSettings] = None):
        """Initialize the loss module.

        Args:
            settings: Optional configuration. If omitted, defaults are used.
        """
        super().__init__()
        self.settings = settings or RegularizationSettings()

    @property
    def reduction(self) -> Reduction:
        return self.settings.reduction

    def forward(self, g_inf: AbstractLocation, p_inf: GroundedLocation) -> LossReg:
        """Compute `LossReg`.

        Args:
            g: Abstract location (grid cells). List of tensors per frequency.
            p: Grounded location (place cells). List of tensors per frequency.

        Returns:
            LossReg dataclass with g_l2 and p_l1 components.

        Notes:
            This matches legacy behaviour (sum across frequencies and features).
        """
        loss_g_l2 = torch.zeros(g_inf[0].shape[0], device=g_inf[0].device, dtype=g_inf[0].dtype)
        for g_f in g_inf:
            loss_g_l2 += (g_f**2).sum(dim=-1)

        loss_p_l1 = torch.zeros(p_inf[0].shape[0], device=p_inf[0].device, dtype=p_inf[0].dtype)
        for p_f in p_inf:
            loss_p_l1 += torch.abs(p_f).sum(dim=-1)

        g_l2 = utils.reduce_per_env(loss_g_l2, self.reduction)
        p_l1 = utils.reduce_per_env(loss_p_l1, self.reduction)
        return LossReg(g_l2=g_l2 * self.settings.weight_g_l2, p_l1=p_l1 * self.settings.weight_p_l1)


@dataclass
class LossOutput:
    """Complete hierarchical loss output for TEM.

    Primary container for all TEM loss components. Supports arithmetic operations
    to enable efficient accumulation during streaming rollouts.

    The total loss is computed as the sum of all section totals, where individual
    components are already weighted by their respective configuration parameters.

    Attributes:
        x: Sensory reconstruction losses (:class:`LossX`).
        p: Grounded location consistency losses (:class:`LossP`).
        g: Abstract location transition losses (:class:`LossG`).
        reg: Regularization penalties (:class:`LossReg`).

    Example:
        Accumulating losses over a rollout::

            accum = LossOutput.zero(device=device)
            for state in rollout:
                state_loss = loss_fn(state)
                accum = accum + state_loss
            mean_loss = accum / len(rollout)
    """

    x: LossX
    p: LossP
    g: LossG
    reg: LossReg

    @classmethod
    def zero(cls, *, device, dtype=torch.float32) -> "LossOutput":
        """Create a zero-initialized instance.

        Args:
            device: Torch device.
            dtype: Torch dtype.

        Returns:
            A `LossOutput` instance with all fields set to zero scalars.
        """
        return cls(
            x=LossX.zero(device=device, dtype=dtype),
            p=LossP.zero(device=device, dtype=dtype),
            g=LossG.zero(device=device, dtype=dtype),
            reg=LossReg.zero(device=device, dtype=dtype),
        )

    def __add__(self, other: "LossOutput") -> "LossOutput":
        return LossOutput(
            x=self.x + other.x,
            p=self.p + other.p,
            g=self.g + other.g,
            reg=self.reg + other.reg,
        )

    def __truediv__(self, divisor: int | float) -> "LossOutput":
        return LossOutput(
            x=self.x / divisor,
            p=self.p / divisor,
            g=self.g / divisor,
            reg=self.reg / divisor,
        )

    def __mul__(self, scalar: Scalar) -> "LossOutput":
        return LossOutput(
            x=self.x * scalar,
            p=self.p * scalar,
            g=self.g * scalar,
            reg=self.reg * scalar,
        )

    def __rmul__(self, scalar: Scalar) -> "LossOutput":
        return self.__mul__(scalar)

    @property
    def total(self) -> Tensor:
        """Return total loss as a sum of section totals."""
        return self.x.total + self.p.total + self.g.total + self.reg.total


#: Type alias for single-timestep loss (identical to LossOutput).
StepLoss = LossOutput

#: Type alias for accumulated rollout losses (identical to LossOutput).
AccumLoss = LossOutput


class TEMLoss(nn.Module):
    """Top-level TEM loss computation module.

    Orchestrates all loss components for a single timestate, applying configured
    weights and reduction modes. This is the primary interface used by the
    training loop.

    The module computes:
        - Sensory reconstruction ($L_x$)
        - Grounded location consistency ($L_p$)
        - Abstract location transition ($L_g$)
        - Regularization penalties

    Returns :class:`LossOutput` with pre-weighted components ready for
    optimization or visit-masked accumulation.

    Note:
        Default ``reduction="none"`` produces per-environment vectors ``(B,)``
        to support visit masking in :class:`~ehc_sn.training.TrainingLoop`.
    """

    def __init__(self, config: LossSettings):
        """Initialize the composite TEM loss.

        Args:
            settings: Loss configuration tree.
        """
        super().__init__()
        self._config = config

        # Per-env outputs are required by the training loop's visited mask,
        self.loss_x_fn = SensoryReconstructionLoss(config.x)
        self.loss_p_fn = GroundedLocationLoss(config.p)
        self.loss_g_fn = AbstractLocationLoss(config.g)
        self.loss_reg_fn = RegularizationLoss(config.reg)

    def forward(self, output: TEMOutput, label: WorldStep, state: TEMState) -> LossOutput:
        # Move use_x_cued_recall to GroundedLocationConfig
        """Compute weighted loss components for one timestep.

        Args:
            output: TEMOutput at current timestep.
            state: TEMState at current timestep.
            label: WorldStep with ground-truth data for current timestep.

        Returns:
            LossOutput where each component is already multiplied by settings weights.
        """
        reconstruction, observation = output.reconstruction, label.observation
        y_p_inf, y_gen_gi, y_gen_gg = reconstruction.y_p_inf, reconstruction.y_gen_gi, reconstruction.y_gen_gg
        g_inf, g_gen = output.inference.g_inf, output.generative.g_gen
        p_inf, p_gen_gi, p_xi = output.inference.p_inf, output.generative.p_gen_gi, output.inference.p_xi

        # Raw (possibly per-env) losses
        lx: LossX = self.loss_x_fn(y_p_inf, y_gen_gi, y_gen_gg, observation)
        lg: LossG = self.loss_g_fn(g_inf, g_gen, state.mec.uncertainty)
        lp: LossP = self.loss_p_fn(p_inf, p_gen_gi, p_xi)
        lreg: LossReg = self.loss_reg_fn(g_inf, p_inf)

        return LossOutput(x=lx, p=lp, g=lg, reg=lreg)
        return LossOutput(x=lx, p=lp, g=lg, reg=lreg)
