"""
Backpropagation Baseline for Entorhinal-Hippocampal Circuit

Simple script for running baseline autoencoder experiments using backpropagation
training on cognitive maps. Uses TOML configuration files for parameter management.

Usage:
    python bp_baseline.py
    python bp_baseline.py --config experiments/baseline.toml
    python bp_baseline.py --epochs 300 --samples 8000
"""

import argparse
from functools import partial
from typing import Any, Dict, Optional

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, RichModelSummary, RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from pydantic import BaseModel, Field

from ehc_sn.data import cognitive_maps as data
from ehc_sn.figures import cognitive_maps as figures
from ehc_sn.models import decoders, encoders
from ehc_sn.models.autoencoders import Autoencoder, AutoencoderParams
from ehc_sn.models.decoders import DecoderParams
from ehc_sn.models.encoders import EncoderParams
from ehc_sn.utils import load_settings

# -------------------------------------------------------------------------------------------
# Configuration Settings
# -------------------------------------------------------------------------------------------


class ExperimentSettings(BaseModel):
    """
    Configuration settings for the autoencoder baseline experiment.

    Loads parameters from TOML configuration files with validation.
    """

    # Data Generation Settings
    grid_width: int = Field(default=32, ge=8, le=128, description="Width of the cognitive map grid")
    grid_height: int = Field(default=16, ge=8, le=128, description="Height of the cognitive map grid")
    diffusion_iterations: int = Field(default=0, ge=0, le=10, description="Number of diffusion iterations")
    diffusion_strength: float = Field(default=0.0, ge=0.0, le=1.0, description="Diffusion strength")
    noise_level: float = Field(default=0.0, ge=0.0, le=1.0, description="Base noise level")

    # Dataset Settings
    num_samples: int = Field(default=4000, ge=100, le=50000, description="Total number of samples")
    batch_size: int = Field(default=32, ge=1, le=512, description="Training batch size")
    num_workers: int = Field(default=8, ge=0, le=16, description="Number of data loading workers")
    val_split: float = Field(default=0.1, ge=0.05, le=0.3, description="Validation split fraction")
    test_split: float = Field(default=0.1, ge=0.05, le=0.3, description="Test split fraction")

    # Model Architecture Settings
    latent_dim: int = Field(default=256, ge=32, le=1024, description="Latent space dimensionality")

    # Training Settings
    max_epochs: int = Field(default=200, ge=1, le=1000, description="Maximum training epochs")
    learning_rate: float = Field(default=1e-3, ge=1e-6, le=1e-1, description="Learning rate for optimizer")

    # Split Training Loss Settings
    gramian_center: bool = Field(default=True, description="Center activations before Gramian computation")
    gramian_weight: float = Field(default=1.0, ge=0.0, le=10.0, description="Weight for Gramian orthogonality loss")
    rate_target: float = Field(
        default=0.05, ge=0.0, le=1.0, description="Target mean firing rate for homeostatic regulation"
    )
    min_active: int = Field(default=8, ge=1, le=64, description="Minimum number of active neurons per sample")
    homeo_weight: float = Field(default=1.0, ge=0.0, le=10.0, description="Weight for homeostatic activity loss")

    # Logging and Output Settings
    log_dir: str = Field(default="logs", description="Directory for experiment logs")
    experiment_name: str = Field(default="baseline", description="Experiment name")
    checkpoint_every_n_epochs: int = Field(default=5, ge=1, le=50, description="Checkpoint frequency")
    progress_refresh_rate: int = Field(default=5, ge=1, le=20, description="Progress bar refresh rate")

    # Visualization Settings
    figure_width: int = Field(default=10, ge=6, le=20, description="Figure width in inches")
    figure_height: int = Field(default=10, ge=6, le=20, description="Figure height in inches")
    figure_dpi: int = Field(default=100, ge=50, le=300, description="Figure resolution (DPI)")
    num_visualization_samples: int = Field(default=5, ge=1, le=20, description="Number of samples to visualize")

    # -----------------------------------------------------------------------------------
    # Utility Properties
    # -----------------------------------------------------------------------------------

    @property
    def grid_size(self) -> tuple[int, int]:
        """Get grid size as tuple (height, width)."""
        return (self.grid_height, self.grid_width)

    @property
    def input_shape(self) -> tuple[int, int, int]:
        """Get input shape for model (channels, height, width)."""
        return (1, self.grid_height, self.grid_width)

    @property
    def figsize(self) -> tuple[int, int]:
        """Get figure size as tuple (width, height)."""
        return (self.figure_width, self.figure_height)

    # -----------------------------------------------------------------------------------
    # Component Configuration Methods
    # -----------------------------------------------------------------------------------

    def create_generator_params(self) -> data.BlockMapParams:
        """Create data generator parameters from settings."""
        return data.BlockMapParams(
            grid_size=self.grid_size,
            diffusion_iterations=self.diffusion_iterations,
            diffusion_strength=self.diffusion_strength,
            noise_level=self.noise_level,
        )

    def create_datamodule_params(self) -> data.DataModuleParams:
        """Create data module parameters from settings."""
        return data.DataModuleParams(
            num_samples=self.num_samples,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            val_split=self.val_split,
            test_split=self.test_split,
        )

    def create_encoder_params(self) -> EncoderParams:
        """Create encoder parameters from settings."""
        return EncoderParams(
            input_shape=self.input_shape,
            latent_dim=self.latent_dim,
            activation_fn=torch.nn.GELU,
        )

    def create_decoder_params(self) -> DecoderParams:
        """Create decoder parameters from settings."""
        return DecoderParams(
            output_shape=self.input_shape,
            latent_dim=self.latent_dim,
            activation_fn=torch.nn.GELU,
        )

    def create_autoencoder_params(self) -> AutoencoderParams:
        """Create autoencoder parameters from settings."""
        encoder = encoders.Linear(self.create_encoder_params())
        decoder = decoders.Linear(self.create_decoder_params())
        return AutoencoderParams(
            encoder=encoder,
            decoder=decoder,
            gramian_center=self.gramian_center,
            gramian_weight=self.gramian_weight,
            rate_target=self.rate_target,
            min_active=self.min_active,
            homeo_weight=self.homeo_weight,
            detach_gradients=False,  # Enable standard autoencoder training with full gradient flow
            optimizer_init=partial(torch.optim.Adam, lr=self.learning_rate),
        )

    def create_figure_params(self) -> figures.CompareMapsFigParam:
        """Create figure parameters from settings."""
        return figures.CompareMapsFigParam(
            figsize=self.figsize,
            dpi=self.figure_dpi,
            tight_layout=True,
            constrained_layout=False,
            title_fontsize=14,
        )


# -------------------------------------------------------------------------------------------
# Experiment Pipeline
# -------------------------------------------------------------------------------------------


class ExperimentPipeline:
    """Orchestrates the complete experiment workflow."""

    def __init__(self, settings: ExperimentSettings):
        """
        Initialize experiment pipeline with settings.

        Args:
            settings: CLI configuration settings
        """
        self.settings = settings
        self.components: Optional[Dict[str, Any]] = None

    # -----------------------------------------------------------------------------------

    def setup_components(self) -> None:
        """Initialize all experiment components based on settings."""
        print("🔧 Setting up experiment components...")

        # Initialize components directly from settings
        generator = data.BlockMapGenerator(self.settings.create_generator_params())
        datamodule = data.DataModule(generator, self.settings.create_datamodule_params())
        model = Autoencoder(self.settings.create_autoencoder_params())

        # Create standard Lightning trainer
        trainer = pl.Trainer(
            max_epochs=self.settings.max_epochs,
            callbacks=[
                RichModelSummary(),
                RichProgressBar(refresh_rate=self.settings.progress_refresh_rate),
                ModelCheckpoint(
                    every_n_epochs=self.settings.checkpoint_every_n_epochs,
                    save_weights_only=True,
                ),
            ],
            logger=TensorBoardLogger(self.settings.log_dir, name=self.settings.experiment_name),
            profiler="simple",
        )

        fig_generator = figures.CompareCognitiveMaps(self.settings.create_figure_params())

        self.components = {
            "datamodule": datamodule,
            "model": model,
            "trainer": trainer,
            "fig_generator": fig_generator,
        }

        print("✅ Experiment setup completed!")

    # -----------------------------------------------------------------------------------

    def display_configuration(self) -> None:
        """Display experiment configuration summary."""
        print("\n📋 EXPERIMENT CONFIGURATION")
        print("-" * 50)
        print(f"Grid Size: {self.settings.grid_height}×{self.settings.grid_width}")
        print(f"Samples: {self.settings.num_samples:,}")
        print(f"Batch Size: {self.settings.batch_size}")
        print(f"Latent Dim: {self.settings.latent_dim}")
        print(f"Max Epochs: {self.settings.max_epochs}")
        print(f"Learning Rate: {self.settings.learning_rate:.1e}")
        print("\n🧠 Split Training Loss Weights:")
        print(f"  • Gramian Weight: {self.settings.gramian_weight:.3f}")
        print(f"  • Homeostatic Weight: {self.settings.homeo_weight:.3f}")
        print(f"  • Rate Target: {self.settings.rate_target:.3f}")
        print(f"  • Min Active: {self.settings.min_active}")
        print(f"Log Directory: {self.settings.log_dir}")
        print(f"Experiment: {self.settings.experiment_name}")

    # -----------------------------------------------------------------------------------

    def train(self) -> None:
        """Execute the training pipeline."""
        if not self.components:
            raise RuntimeError("Components not initialized. Call setup_components() first.")

        print("\n🚀 Starting model training...")

        # Setup data
        self.components["datamodule"].setup()

        # Display training info
        train_samples = int(self.settings.num_samples * (1 - self.settings.val_split - self.settings.test_split))
        val_samples = int(self.settings.num_samples * self.settings.val_split)
        test_samples = int(self.settings.num_samples * self.settings.test_split)

        print(f"   • Training samples: {train_samples:,}")
        print(f"   • Validation samples: {val_samples:,}")
        print(f"   • Test samples: {test_samples:,}")

        # Execute training using standard Lightning trainer
        self.components["trainer"].fit(model=self.components["model"], datamodule=self.components["datamodule"])

        print("🎉 Training completed successfully!")

    # -----------------------------------------------------------------------------------

    def evaluate_and_visualize(self) -> None:
        """Evaluate model and generate visualizations."""
        if not self.components:
            raise RuntimeError("Components not initialized. Call setup_components() first.")

        print("\n📊 Evaluating model performance...")

        # Get test data
        test_batch = next(iter(self.components["datamodule"].test_dataloader()))

        # Handle batch input - extract tensor from tuple if needed
        if isinstance(test_batch, (list, tuple)):
            test_data = test_batch[0]  # Extract the first element (cognitive map tensor)
        else:
            test_data = test_batch

        model = self.components["model"]

        # Ensure device compatibility
        model.to(device=test_data.device)

        # Generate predictions
        model.eval()
        with torch.no_grad():
            reconstructed, latent = model(test_data)

            # Calculate metrics
            mse_loss = torch.nn.functional.mse_loss(reconstructed, test_data)
            sparsity = (latent.abs() < 1e-6).float().mean()

            print(f"   • Test MSE Loss: {mse_loss:.6f}")
            print(f"   • Latent Sparsity: {sparsity:.1%}")

        # Generate visualization
        print("\n🎨 Generating visualization...")
        sample_pairs = list(
            zip(
                reconstructed[: self.settings.num_visualization_samples].cpu(),
                test_data[: self.settings.num_visualization_samples].squeeze(1).cpu(),
            )
        )

        self.components["fig_generator"].plot(sample_pairs)
        self.components["fig_generator"].show()

        print("✅ Evaluation completed!")

    # -----------------------------------------------------------------------------------

    def run_complete_experiment(self) -> None:
        """Execute the complete experiment pipeline."""
        self.display_configuration()
        self.setup_components()
        self.train()
        self.evaluate_and_visualize()


# -------------------------------------------------------------------------------------------
# Main Experiment Function
# -------------------------------------------------------------------------------------------


def run_experiment(
    config: Optional[str] = None,
    epochs: Optional[int] = None,
    samples: Optional[int] = None,
    batch_size: Optional[int] = None,
    latent_dim: Optional[int] = None,
    learning_rate: Optional[float] = None,
    gramian_weight: Optional[float] = None,
    homeo_weight: Optional[float] = None,
) -> None:
    """
    Run the complete backpropagation baseline experiment.

    Args:
        config: Path to TOML configuration file
        epochs: Maximum training epochs (overrides config)
        samples: Number of training samples (overrides config)
        batch_size: Training batch size (overrides config)
        latent_dim: Latent space dimension (overrides config)
        learning_rate: Learning rate for optimizer (overrides config)
        gramian_weight: Weight for Gramian orthogonality loss (overrides config)
        homeo_weight: Weight for homeostatic activity loss (overrides config)
    """
    print("=" * 80)
    print("🧠 ENTORHINAL-HIPPOCAMPAL CIRCUIT: BACKPROPAGATION BASELINE")
    print("=" * 80)

    # Prepare overrides
    overrides = {}
    if epochs is not None:
        overrides["max_epochs"] = epochs
    if samples is not None:
        overrides["num_samples"] = samples
    if batch_size is not None:
        overrides["batch_size"] = batch_size
    if latent_dim is not None:
        overrides["latent_dim"] = latent_dim
    if learning_rate is not None:
        overrides["learning_rate"] = learning_rate
    if gramian_weight is not None:
        overrides["gramian_weight"] = gramian_weight
    if homeo_weight is not None:
        overrides["homeo_weight"] = homeo_weight

    # Load settings
    print(f"📁 Loading configuration from: {config}")
    settings_dict = load_settings(config) if config else {}
    settings_dict.update(overrides)
    settings = ExperimentSettings(**settings_dict)

    # Run experiment pipeline
    pipeline = ExperimentPipeline(settings)
    pipeline.run_complete_experiment()

    print("\n" + "=" * 80)
    print("🎯 EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("=" * 80)


# -------------------------------------------------------------------------------------------
# CLI Argument Parsing
# -------------------------------------------------------------------------------------------


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Backpropagation baseline for Entorhinal-Hippocampal Circuit",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--config", "-c", type=str, help="Path to TOML configuration file")
    parser.add_argument("--epochs", "-e", type=int, help="Maximum training epochs")
    parser.add_argument("--samples", "-s", type=int, help="Number of training samples")
    parser.add_argument("--batch-size", "-b", type=int, help="Training batch size")
    parser.add_argument("--latent-dim", "-l", type=int, help="Latent space dimension")
    parser.add_argument("--learning-rate", "-lr", type=float, help="Learning rate for optimizer")
    parser.add_argument("--gramian-weight", "-gw", type=float, help="Weight for Gramian orthogonality loss")
    parser.add_argument("--homeo-weight", "-hw", type=float, help="Weight for homeostatic activity loss")
    parser.add_argument("--l1-weight", "-l1w", type=float, help="Weight for L1 sparsity loss")

    return parser.parse_args()


# -------------------------------------------------------------------------------------------
# Main Entry Point
# -------------------------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        args = parse_arguments()
        run_experiment(
            config=args.config,
            epochs=args.epochs,
            samples=args.samples,
            batch_size=args.batch_size,
            latent_dim=args.latent_dim,
            learning_rate=args.learning_rate,
            gramian_weight=args.gramian_weight,
            homeo_weight=args.homeo_weight,
        )
    except Exception as e:
        print(f"\n❌ EXPERIMENT FAILED: {e}")
        raise
