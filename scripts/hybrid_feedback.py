"""
Hybrid Feedback Autoencoder Training Script.

This script demonstrates the training and evaluation of a hybrid feedback autoencoder
model for pattern separation and completion tasks. The model combines forward and
feedback learning mechanisms to learn efficient representations of input data.

The script supports:
- CLI argument parsing via Pydantic Settings
- Environment variable configuration
- Comprehensive visualization of results
- Configurable model, data, and training parameters

Usage:
    python scripts/hybrid_feedback.py [OPTIONS]

Examples:
    # Run with default settings
    python scripts/hybrid_feedback.py

    # Override specific parameters
    python scripts/hybrid_feedback.py --trainer.max_epochs=50 --model.latent_units=64
"""

import matplotlib.pyplot as plt
import torch
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from ehc_sn.core.datamodule import BaseDataModule, DataModuleParams
from ehc_sn.core.trainer import TrainerParams
from ehc_sn.data.simple_example import DataGenerator, DataParams
from ehc_sn.figures.reconstruction_1d import ReconstructionTraceFigure
from ehc_sn.figures.reconstruction_1d import ReconstructionTraceParams as Figure1Params
from ehc_sn.figures.sparsity import SparsityFigure
from ehc_sn.figures.sparsity import SparsityParams as Figure2Params
from ehc_sn.models.ann.autoencoders.hybrid_feedback import Autoencoder, AutoencoderParams
from ehc_sn.trainers.feed_forward import FeedbackTainer


# -------------------------------------------------------------------------------------------
class Experiment(BaseSettings):
    """
    Configuration settings for the hybrid feedback autoencoder experiment.

    This class defines all configurable parameters for the experiment, including:
    - Data generation parameters (input/latent dimensions, sparsity, noise)
    - Data module settings (batch size, splits, workers)
    - Model architecture (layer sizes, learning rates)
    - Training configuration (epochs, logging, checkpointing)
    - Visualization parameters (figure dimensions, styling)

    Settings can be configured via:
    - CLI arguments: --trainer.max_epochs=100
    - Default values defined in the parameter classes

    Attributes:
        data: Parameters for synthetic data generation
        datamodule: Data loading and preprocessing configuration
        model: Autoencoder architecture and optimization settings
        trainer: Training loop and experiment management
        figure_1: Reconstruction visualization parameters
        figure_2: Sparsity analysis visualization parameters
    """

    model_config = SettingsConfigDict(extra="forbid", cli_parse_args=True)

    data: DataParams = Field(default_factory=DataParams, description="Data generation parameters")
    datamodule: DataModuleParams = Field(default_factory=DataModuleParams, description="Data module parameters")
    model: AutoencoderParams = Field(default_factory=AutoencoderParams, description="Autoencoder parameters")
    trainer: TrainerParams = Field(default_factory=TrainerParams, description="Trainer parameters")
    figure_1: Figure1Params = Field(default_factory=Figure1Params, description="Reconstruction figure parameters")
    figure_2: Figure2Params = Field(default_factory=Figure2Params, description="Sparsity figure parameters")


# -------------------------------------------------------------------------------------------
def main(experiment: Experiment) -> None:
    """
    Run the complete hybrid feedback autoencoder experiment.

    This function orchestrates the entire experimental pipeline:
    1. Data generation and preparation
    2. Model initialization and training
    3. Evaluation and visualization

    The hybrid feedback autoencoder combines:
    - Forward pass: Standard autoencoder reconstruction
    - Feedback mechanism: Direct random target projection (DRTP) for learning

    Args:
        experiment: Configured experiment settings containing all parameters

    Workflow:
        1. Generate synthetic sparse data with configurable parameters
        2. Create PyTorch Lightning data module for efficient loading
        3. Initialize hybrid feedback autoencoder with specified architecture
        4. Train model using the feedback training algorithm
        5. Evaluate trained model on test data
        6. Generate reconstruction and sparsity visualizations
    """
    # Generate synthetic sparse latent data for pattern separation tasks
    data_gen = DataGenerator(experiment.data)

    # Create Lightning data module for efficient batching and GPU transfer
    datamodule = BaseDataModule(data_gen, experiment.datamodule)

    # Initialize feedback trainer with specified parameters
    trainer = FeedbackTainer(experiment.trainer)

    # Create hybrid feedback autoencoder model
    # Combines forward reconstruction with feedback learning mechanism
    model = Autoencoder(experiment.model, trainer)

    # Train the model using hybrid feedback algorithm
    # This combines standard backpropagation with direct feedback learning
    trainer.fit(model, datamodule)

    # Switch model to evaluation mode (disables dropout, etc.)
    model.eval()

    # Prepare test data for evaluation
    datamodule.setup("test")
    test_dataloader = datamodule.test_dataloader()

    # Get a batch of test data for visualization
    inputs, _ = next(iter(test_dataloader))

    # Generate model predictions without gradient computation
    with torch.inference_mode():
        outputs, activations = model(inputs)

    # Create reconstruction trace figure showing input vs output comparison
    # This helps assess the quality of pattern completion
    fig_reconstruction = ReconstructionTraceFigure(experiment.figure_1)
    _ = fig_reconstruction.plot(inputs, outputs)
    plt.show()

    # Create sparsity analysis figure showing activation distributions
    # This helps evaluate pattern separation capabilities
    sparsity_figure = SparsityFigure(experiment.figure_2)
    _ = sparsity_figure.plot(activations)
    plt.show()


if __name__ == "__main__":
    """
    Main entry point for the hybrid feedback autoencoder experiment.

    This script can be run with various configuration options:

    CLI Examples:
        # Basic run with defaults
        python scripts/hybrid_feedback.py

        # Override training parameters
        python scripts/hybrid_feedback.py --trainer.max_epochs=100 --trainer.experiment_name=my_exp

        # Modify model architecture
        python scripts/hybrid_feedback.py --model.latent_units=64 --model.layer1_units=2048

        # Adjust data generation
        python scripts/hybrid_feedback.py --data.input_dim=512 --data.sparsity=0.1
    """
    # Parse CLI arguments and environment variables into experiment configuration
    experiment = Experiment()

    # Execute the complete experimental pipeline
    main(experiment)
