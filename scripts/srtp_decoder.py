"""
SRTP Decoder for Entorhinal-Hippocampal Circuit

Simple script for training only the decoder part of a pretrained autoencoder
using SRTP (Selective Random Target Projection). Uses TOML configuration files
for parameter management.

Usage:
    python srtp_decoder.py
    python srtp_decoder.py --config experiments/decoder.toml
    python srtp_decoder.py --pretrained models/baseline.ckpt --epochs 100
"""

import argparse
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional, Union

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, RichModelSummary, RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from pydantic import BaseModel, Field
from torch import nn

from ehc_sn.data import cognitive_maps as data
from ehc_sn.figures import cognitive_maps as figures
from ehc_sn.models.ann import decoders, encoders
from ehc_sn.models.ann.autoencoders import Autoencoder, AutoencoderParams
from ehc_sn.models.ann.decoders import DecoderParams
from ehc_sn.models.ann.decoders import SRTPLinear as SRTPDecoder
from ehc_sn.models.ann.encoders import EncoderParams
from ehc_sn.models.ann.encoders import Linear as LinearEncoder
from ehc_sn.utils import load_settings

# -------------------------------------------------------------------------------------------
# Configuration Settings
# -------------------------------------------------------------------------------------------


class SRTPDecoderTrainingSettings(BaseModel):
    """
    Configuration settings for SRTP decoder-only training experiment.

    Loads parameters from TOML configuration files with validation.
    """

    # Model Loading Settings
    pretrained_path: str = Field(default="autoencoder.ckpt", description="Path to pretrained autoencoder checkpoint")
    freeze_encoder: bool = Field(default=True, description="Whether to freeze encoder weights during training")
    reinit_decoder: bool = Field(default=True, description="Whether to reinitialize decoder weights before training")

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
    gramian_center: bool = Field(default=True, description="Whether to center Gramian loss")
    gramian_weight: float = Field(default=0.2, ge=0.0, le=1.0, description="Weight for Gramian loss component")
    rate_target: float = Field(default=0.15, ge=0.01, le=0.5, description="Target activation rate")
    min_active: int = Field(default=8, ge=1, le=64, description="Minimum active units in latent space")
    homeo_weight: float = Field(default=0.2, ge=0.0, le=1.0, description="Weight for homeostatic loss component")

    # Logging and Output Settings
    log_dir: str = Field(default="logs", description="Directory for experiment logs")
    experiment_name: str = Field(default="srtp_decoder", description="Experiment name")
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

    def create_encoder_params(self) -> EncoderParams:
        """Create encoder parameters from settings."""
        return EncoderParams(
            input_shape=self.input_shape,
            latent_dim=self.latent_dim,
            activation_fn=nn.GELU,  # Used GELU for encoder
        )

    def create_decoder_params(self) -> DecoderParams:
        """Create decoder parameters from settings."""
        return DecoderParams(
            output_shape=self.input_shape,
            latent_dim=self.latent_dim,
            activation_fn=nn.GELU,  # Used GELU for decoder
        )

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

    def create_autoencoder_params(self, encoder, decoder) -> AutoencoderParams:
        """Create autoencoder parameters from settings."""
        return AutoencoderParams(
            encoder=encoder,
            decoder=decoder,
            gramian_center=self.gramian_center,
            gramian_weight=self.gramian_weight,
            rate_target=self.rate_target,
            min_active=self.min_active,
            homeo_weight=self.homeo_weight,
            detach_gradients=True,  # Detach gradients for SRTP decoder training
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
# SRTP Decoder Training Pipeline
# -------------------------------------------------------------------------------------------


class SRTPDecoderTrainingPipeline:
    """Orchestrates the SRTP decoder-only training workflow."""

    def __init__(self, settings: SRTPDecoderTrainingSettings):
        """
        Initialize SRTP decoder training pipeline with settings.

        Args:
            settings: Configuration settings
        """
        self.settings = settings
        self.components: Optional[Dict[str, Any]] = None

    # -----------------------------------------------------------------------------------

    def setup_components(self) -> None:
        """Initialize all experiment components and load pretrained model."""
        print("🔧 Setting up SRTP decoder training components...")

        # Create initial encoder and decoder for loading pretrained model
        encoder = LinearEncoder(self.settings.create_encoder_params())
        temp_decoder = decoders.Linear(self.settings.create_decoder_params())  # Temporary decoder for loading

        # Create autoencoder with initial parameters to load pretrained weights
        autoencoder_params = self.settings.create_autoencoder_params(encoder, temp_decoder)
        model = Autoencoder(autoencoder_params)

        # Load pretrained weights if available
        if Path(self.settings.pretrained_path).exists():
            print(f"📥 Loading pretrained model from: {self.settings.pretrained_path}")
            checkpoint = torch.load(self.settings.pretrained_path, map_location="cpu")
            if "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint)
        else:
            print(f"⚠️  Warning: Pretrained model not found at {self.settings.pretrained_path}")
            print("   Proceeding with randomly initialized weights")

        # Configure model for decoder-only training
        if self.settings.freeze_encoder:
            print("🔒 Freezing encoder parameters")
            for param in model.encoder.parameters():
                param.requires_grad = False

        print("🔄 Replacing decoder with SRTPDecoder")
        # Create SRTP decoder parameters with Tanh activation
        srtp_decoder_params = DecoderParams(
            output_shape=self.settings.input_shape,
            latent_dim=self.settings.latent_dim,
            activation_fn=nn.Tanh,  # Use Tanh for SRTP
        )
        model.decoder = SRTPDecoder(srtp_decoder_params)

        # Update optimizer to include new decoder parameters
        model.optimizer_init = self.settings.create_autoencoder_params(model.encoder, model.decoder).optimizer_init

        # Initialize data components
        generator = data.BlockMapGenerator(self.settings.create_generator_params())
        datamodule = data.DataModule(generator, self.settings.create_datamodule_params())

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

        # Initialize visualization
        fig_generator = figures.CompareCognitiveMaps(self.settings.create_figure_params())

        self.components = {
            "model": model,
            "datamodule": datamodule,
            "trainer": trainer,
            "fig_generator": fig_generator,
        }

        print("✅ Component setup completed!")

    # -----------------------------------------------------------------------------------

    def display_configuration(self) -> None:
        """Display experiment configuration summary."""
        print("\n📋 SRTP DECODER TRAINING CONFIGURATION")
        print("-" * 50)
        print(f"Pretrained Path: {self.settings.pretrained_path}")
        print(f"Freeze Encoder: {self.settings.freeze_encoder}")
        print(f"Reinit Decoder: {self.settings.reinit_decoder}")
        print(f"Grid Size: {self.settings.grid_height}×{self.settings.grid_width}")
        print(f"Samples: {self.settings.num_samples:,}")
        print(f"Batch Size: {self.settings.batch_size}")
        print(f"Latent Dim: {self.settings.latent_dim}")
        print(f"Max Epochs: {self.settings.max_epochs}")
        print(f"Learning Rate: {self.settings.learning_rate:.1e}")
        print(f"Log Directory: {self.settings.log_dir}")
        print(f"Experiment: {self.settings.experiment_name}")

    # -----------------------------------------------------------------------------------

    def train_decoder(self) -> None:
        """Execute the SRTP decoder training pipeline."""
        if not self.components:
            raise RuntimeError("Components not initialized. Call setup_components() first.")

        print("\n🚀 Starting SRTP decoder-only training...")

        # Setup data
        self.components["datamodule"].setup()

        # Display training info
        train_samples = int(self.settings.num_samples * (1 - self.settings.val_split - self.settings.test_split))
        val_samples = int(self.settings.num_samples * self.settings.val_split)
        test_samples = int(self.settings.num_samples * self.settings.test_split)

        print(f"   • Training samples: {train_samples:,}")
        print(f"   • Validation samples: {val_samples:,}")
        print(f"   • Test samples: {test_samples:,}")

        # Count trainable parameters
        total_params = sum(p.numel() for p in self.components["model"].parameters())
        trainable_params = sum(p.numel() for p in self.components["model"].parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        print(f"   • Total parameters: {total_params:,}")
        print(f"   • Trainable parameters: {trainable_params:,}")
        print(f"   • Frozen parameters: {frozen_params:,}")

        # Execute training using standard Lightning trainer
        self.components["trainer"].fit(model=self.components["model"], datamodule=self.components["datamodule"])

        print("🎉 SRTP decoder training completed successfully!")

    # -----------------------------------------------------------------------------------

    def evaluate_and_visualize(self) -> None:
        """Evaluate SRTP decoder performance and generate visualizations."""
        if not self.components:
            raise RuntimeError("Components not initialized. Call setup_components() first.")

        print("\n📊 Evaluating SRTP decoder performance...")

        # Get test data
        test_batch = next(iter(self.components["datamodule"].test_dataloader()))

        # Handle batch input - extract tensor from tuple if needed
        if isinstance(test_batch, (list, tuple)):
            inputs = test_batch[0]  # Extract the first element (cognitive map tensor)
        else:
            inputs = test_batch

        original_targets = inputs.squeeze(1) if inputs.dim() > 3 else inputs
        model = self.components["model"]

        # Ensure device compatibility
        device = inputs.device
        model.to(device)

        # Generate predictions
        model.eval()
        with torch.no_grad():
            reconstructed, latent = model(inputs)

            # Calculate metrics
            mse_loss = torch.nn.functional.mse_loss(reconstructed, inputs)
            sparsity = (latent.abs() < 1e-6).float().mean()

            print(f"   • Test MSE Loss: {mse_loss:.6f}")
            print(f"   • Latent Sparsity: {sparsity:.1%}")

        # Generate visualization
        print("\n🎨 Generating visualization...")
        comparison_pairs = list(
            zip(
                reconstructed[: self.settings.num_visualization_samples].cpu(),
                original_targets[: self.settings.num_visualization_samples].cpu(),
            )
        )

        self.components["fig_generator"].plot(comparison_pairs)
        self.components["fig_generator"].show()

        print("✅ Evaluation completed!")

    # -----------------------------------------------------------------------------------

    def run_complete_experiment(self) -> None:
        """Execute the complete SRTP decoder training pipeline."""
        self.display_configuration()
        self.setup_components()
        self.train_decoder()
        self.evaluate_and_visualize()


# -------------------------------------------------------------------------------------------
# Main Experiment Function
# -------------------------------------------------------------------------------------------


def run_srtp_decoder_experiment(
    config: Optional[str] = None,
    pretrained: Optional[str] = None,
    epochs: Optional[int] = None,
    samples: Optional[int] = None,
) -> None:
    """
    Run the complete SRTP decoder training experiment.

    Args:
        config: Path to TOML configuration file
        pretrained: Path to pretrained autoencoder checkpoint (overrides config)
        epochs: Maximum training epochs (overrides config)
        samples: Number of training samples (overrides config)
    """
    print("=" * 80)
    print("🧠 ENTORHINAL-HIPPOCAMPAL CIRCUIT: SRTP DECODER TRAINING")
    print("=" * 80)

    # Prepare overrides
    overrides: Dict[str, Union[str, int, bool]] = {}
    if pretrained is not None:
        overrides["pretrained_path"] = pretrained
    if epochs is not None:
        overrides["max_epochs"] = epochs
    if samples is not None:
        overrides["num_samples"] = samples

    # Load settings
    if config:
        print(f"📁 Loading configuration from: {config}")
    settings_dict = load_settings(config) if config else {}
    settings_dict.update(overrides)
    settings = SRTPDecoderTrainingSettings(**settings_dict)

    # Run experiment pipeline
    pipeline = SRTPDecoderTrainingPipeline(settings)
    pipeline.run_complete_experiment()

    print("\n" + "=" * 80)
    print("🎯 SRTP DECODER TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 80)


# -------------------------------------------------------------------------------------------
# CLI Argument Parsing
# -------------------------------------------------------------------------------------------


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="SRTP Decoder training for Entorhinal-Hippocampal Circuit",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--config", "-c", type=str, help="Path to TOML configuration file")
    parser.add_argument("--pretrained", "-p", type=str, help="Path to pretrained autoencoder checkpoint")
    parser.add_argument("--epochs", "-e", type=int, help="Maximum training epochs")
    parser.add_argument("--samples", "-s", type=int, help="Number of training samples")

    return parser.parse_args()


# -------------------------------------------------------------------------------------------
# Main Entry Point
# -------------------------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        args = parse_arguments()

        run_srtp_decoder_experiment(
            config=args.config,
            pretrained=args.pretrained,
            epochs=args.epochs,
            samples=args.samples,
        )
    except Exception as e:
        print(f"\n❌ SRTP DECODER TRAINING FAILED: {e}")
        raise
