"""
Backpropagation Baseline CLI for Entorhinal-Hippocampal Circuit

This script provides a command-line interface for running baseline autoencoder
experiments using backpropagation training on cognitive maps. It serves as a
reference point for comparing more sophisticated EHC circuit models.

Usage:
    python bp_baseline.py --help
    python bp_baseline.py --grid-size 16 32 --num-samples 5000 --max-epochs 300
    python bp_baseline.py --config config.json
"""

import json
from pathlib import Path
from typing import Annotated, Any, Dict, Optional

import torch
from lightning.pytorch.callbacks import ModelCheckpoint, RichModelSummary, RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typer import Argument, Option, Typer

from ehc_sn.data import cognitive_maps as data
from ehc_sn.figures import cognitive_maps as figures
from ehc_sn.models import autoencoders as models
from ehc_sn.models import decoders, encoders
from ehc_sn.trainers.backprop import sparsity as trainers

# -------------------------------------------------------------------------------------------
# CLI Configuration Settings
# -------------------------------------------------------------------------------------------


class ExperimentSettings(BaseSettings):
    """
    CLI-configurable settings for the autoencoder baseline experiment.

    Supports configuration via:
    - Command line arguments
    - Environment variables (with EHC_ prefix)
    - Configuration files (JSON/TOML)
    - Default values
    """

    model_config = SettingsConfigDict(
        env_prefix="EHC_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="forbid",
    )

    # -----------------------------------------------------------------------------------
    # Data Generation Settings
    # -----------------------------------------------------------------------------------

    # Grid configuration
    grid_width: Annotated[int, Field(
        description="Width of the cognitive map grid",
        ge=8, le=128
    )] = 32  # fmt: skip

    grid_height: Annotated[int, Field(
        description="Height of the cognitive map grid",
        ge=8, le=128
    )] = 16  # fmt: skip

    # Map generation parameters
    diffusion_iterations: Annotated[int, Field(
        description="Number of diffusion iterations for map generation",
        ge=0, le=10
    )] = 0  # fmt: skip

    diffusion_strength: Annotated[float, Field(
        description="Strength of diffusion effect (0.0-1.0)",
        ge=0.0, le=1.0
    )] = 0.0  # fmt: skip

    noise_level: Annotated[float, Field(
        description="Base noise level throughout the map (0.0-1.0)",
        ge=0.0, le=1.0
    )] = 0.0  # fmt: skip

    # -----------------------------------------------------------------------------------
    # Dataset Settings
    # -----------------------------------------------------------------------------------

    num_samples: Annotated[int, Field(
        description="Total number of samples to generate",
        ge=100, le=50000
    )] = 4000  # fmt: skip

    batch_size: Annotated[int, Field(
        description="Training batch size",
        ge=1, le=512
    )] = 32  # fmt: skip

    num_workers: Annotated[int, Field(
        description="Number of data loading workers",
        ge=0, le=16
    )] = 8  # fmt: skip

    val_split: Annotated[float, Field(
        description="Fraction of data for validation",
        ge=0.05, le=0.3
    )] = 0.1  # fmt: skip

    test_split: Annotated[float, Field(
        description="Fraction of data for testing",
        ge=0.05, le=0.3
    )] = 0.1  # fmt: skip

    # -----------------------------------------------------------------------------------
    # Model Architecture Settings
    # -----------------------------------------------------------------------------------

    latent_dim: Annotated[int, Field(
        description="Latent space dimensionality",
        ge=32, le=1024
    )] = 256  # fmt: skip

    # -----------------------------------------------------------------------------------
    # Training Settings
    # -----------------------------------------------------------------------------------

    max_epochs: Annotated[int, Field(
        description="Maximum training epochs",
        ge=1, le=1000
    )] = 200  # fmt: skip

    sparsity_target: Annotated[float, Field(
        description="Target sparsity level (fraction of active neurons)",
        ge=0.01, le=0.5
    ) ] = 0.05  # fmt: skip

    sparsity_weight: Annotated[float, Field(
        description="Weight for sparsity regularization term",
        ge=0.0, le=1.0
    )] = 0.01  # fmt: skip

    # -----------------------------------------------------------------------------------
    # Logging and Output Settings
    # -----------------------------------------------------------------------------------

    log_dir: Annotated[str, Field(
        description="Directory for experiment logs"
    )] = "logs"  # fmt: skip

    experiment_name: Annotated[str, Field(
        description="Name for this experiment"
    )] = "autoencoder_baseline"  # fmt: skip

    checkpoint_every_n_epochs: Annotated[int, Field(
        description="Save checkpoint every N epochs",
        ge=1, le=50
    )] = 5  # fmt: skip

    progress_refresh_rate: Annotated[int, Field(
        description="Progress bar refresh rate",
        ge=1, le=20
    )] = 5  # fmt: skip

    # -----------------------------------------------------------------------------------
    # Visualization Settings
    # -----------------------------------------------------------------------------------

    figure_width: Annotated[int, Field(
        description="Figure width in inches",
        ge=6, le=20
    )] = 10  # fmt: skip

    figure_height: Annotated[int, Field(
        description="Figure height in inches",
        ge=6, le=20
    )] = 10  # fmt: skip

    figure_dpi: Annotated[int, Field(
        description="Figure resolution (DPI)",
        ge=50, le=300
    )] = 100  # fmt: skip

    num_visualization_samples: Annotated[int, Field(
        description="Number of samples to visualize",
        ge=1, le=20
    )] = 5  # fmt: skip

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

    def create_encoder_params(self) -> encoders.EncoderParams:
        """Create encoder parameters from settings."""
        return encoders.EncoderParams(
            input_shape=self.input_shape,
            latent_dim=self.latent_dim,
        )

    def create_decoder_params(self) -> decoders.DecoderParams:
        """Create decoder parameters from settings."""
        return decoders.DecoderParams(
            input_shape=self.input_shape,
            latent_dim=self.latent_dim,
        )

    def create_trainer_params(self) -> trainers.SparsityBPTrainerParams:
        """Create trainer parameters from settings."""
        return trainers.SparsityBPTrainerParams(
            max_epochs=self.max_epochs,
            sparsity_target=self.sparsity_target,
            sparsity_weight=self.sparsity_weight,
            callbacks=[
                RichModelSummary(),
                RichProgressBar(refresh_rate=self.progress_refresh_rate),
                ModelCheckpoint(
                    every_n_epochs=self.checkpoint_every_n_epochs,
                    save_weights_only=True,
                ),
            ],
            logger=TensorBoardLogger(self.log_dir, name=self.experiment_name),
            profiler="simple",
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
        self.components = None

    # -----------------------------------------------------------------------------------

    def setup_components(self) -> None:
        """Initialize all experiment components based on settings."""
        print("🔧 Setting up experiment components...")

        # Initialize components directly from settings
        generator = data.BlockMapGenerator(self.settings.create_generator_params())
        datamodule = data.DataModule(generator, self.settings.create_datamodule_params())
        encoder = encoders.LinearEncoder(self.settings.create_encoder_params())
        decoder = decoders.LinearDecoder(self.settings.create_decoder_params())
        model = models.Autoencoder(encoder, decoder)
        trainer = trainers.SparsityBPTrainer(model, self.settings.create_trainer_params())
        fig_generator = figures.CompareCognitiveMaps(self.settings.create_figure_params())

        self.components: Dict[str, Any] = {
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
        print(f"Sparsity Target: {self.settings.sparsity_target:.1%}")
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

        # Execute training
        self.components["trainer"].fit(datamodule=self.components["datamodule"])

        print("🎉 Training completed successfully!")

    # -----------------------------------------------------------------------------------

    def evaluate_and_visualize(self) -> None:
        """Evaluate model and generate visualizations."""
        if not self.components:
            raise RuntimeError("Components not initialized. Call setup_components() first.")

        print(f"\n📊 Evaluating model performance...")

        # Get test data
        (test_batch,) = next(iter(self.components["datamodule"].test_dataloader()))
        model = self.components["model"]

        # Ensure device compatibility
        model.to(device=test_batch.device)

        # Generate predictions
        model.eval()
        with torch.no_grad():
            reconstructed, latent = model(test_batch)

            # Calculate metrics
            mse_loss = torch.nn.functional.mse_loss(reconstructed, test_batch)
            sparsity = (latent.abs() < 1e-6).float().mean()

            print(f"   • Test MSE Loss: {mse_loss:.6f}")
            print(f"   • Latent Sparsity: {sparsity:.1%}")

        # Generate visualization
        print(f"\n🎨 Generating visualization...")
        sample_pairs = list(
            zip(
                reconstructed[: self.settings.num_visualization_samples].cpu(),
                test_batch[: self.settings.num_visualization_samples].squeeze(1).cpu(),
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
# CLI Application
# -------------------------------------------------------------------------------------------

app = Typer(
    name="bp-baseline",
    help="Backpropagation baseline for Entorhinal-Hippocampal Circuit modeling",
    add_completion=False,
    rich_markup_mode="rich",
)

# -----------------------------------------------------------------------------------


@app.command()
def run(
    # Grid configuration
    grid_width: Annotated[int, Option("--grid-width", "-w", help="Grid width")] = 32,
    grid_height: Annotated[int, Option("--grid-height", "-h", help="Grid height")] = 16,
    # Dataset parameters
    num_samples: Annotated[int, Option("--num-samples", "-n", help="Number of samples")] = 4000,
    batch_size: Annotated[int, Option("--batch-size", "-b", help="Batch size")] = 32,
    # Model parameters
    latent_dim: Annotated[int, Option("--latent-dim", "-l", help="Latent dimension")] = 256,
    # Training parameters
    max_epochs: Annotated[int, Option("--max-epochs", "-e", help="Maximum epochs")] = 200,
    sparsity_target: Annotated[float, Option("--sparsity", "-s", help="Sparsity target")] = 0.05,
    # Output parameters
    log_dir: Annotated[str, Option("--log-dir", help="Log directory")] = "logs",
    experiment_name: Annotated[str, Option("--name", help="Experiment name")] = "autoencoder_baseline",
    # Configuration file
    config: Annotated[Optional[Path], Option("--config", "-c", help="Configuration file path")] = None,
):
    """
    Run the backpropagation baseline experiment.

    This command trains a sparse autoencoder on cognitive maps using backpropagation
    and evaluates its reconstruction performance.
    """
    print("=" * 80)
    print("🧠 ENTORHINAL-HIPPOCAMPAL CIRCUIT: BACKPROPAGATION BASELINE")
    print("=" * 80)

    try:
        # Load settings
        if config and config.exists():
            print(f"📁 Loading configuration from: {config}")
            with open(config) as f:
                config_data = json.load(f)
            settings = ExperimentSettings(**config_data)
        else:
            # Create settings from CLI arguments
            settings = ExperimentSettings(
                grid_width=grid_width,
                grid_height=grid_height,
                num_samples=num_samples,
                batch_size=batch_size,
                latent_dim=latent_dim,
                max_epochs=max_epochs,
                sparsity_target=sparsity_target,
                log_dir=log_dir,
                experiment_name=experiment_name,
            )

        # Run experiment
        pipeline = ExperimentPipeline(settings)
        pipeline.run_complete_experiment()

        print("\n" + "=" * 80)
        print("🎯 EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ EXPERIMENT FAILED: {e}")
        raise


# -----------------------------------------------------------------------------------


@app.command()
def save_config(
    output: Annotated[Path, Argument(help="Output configuration file path")],
    grid_width: Annotated[int, Option("--grid-width", help="Grid width")] = 32,
    grid_height: Annotated[int, Option("--grid-height", help="Grid height")] = 16,
    num_samples: Annotated[int, Option("--num-samples", help="Number of samples")] = 4000,
    max_epochs: Annotated[int, Option("--max-epochs", help="Maximum epochs")] = 200,
):
    """
    Generate a configuration file with specified parameters.

    This creates a JSON configuration file that can be used with the --config option.
    """
    settings = ExperimentSettings(
        grid_width=grid_width,
        grid_height=grid_height,
        num_samples=num_samples,
        max_epochs=max_epochs,
    )

    config_dict = settings.model_dump()

    with open(output, "w") as f:
        json.dump(config_dict, f, indent=2)

    print(f"✅ Configuration saved to: {output}")


# -----------------------------------------------------------------------------------


@app.command()
def show_config():
    """Display default configuration parameters."""
    settings = ExperimentSettings()
    config_dict = settings.model_dump()

    print("📋 DEFAULT CONFIGURATION")
    print("=" * 50)
    print(json.dumps(config_dict, indent=2))


# -------------------------------------------------------------------------------------------
# Main Entry Point
# -------------------------------------------------------------------------------------------

if __name__ == "__main__":
    app()
