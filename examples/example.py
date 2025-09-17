"""
Hybrid Feedback Autoencoder Training Script.

Trains a hybrid feedback autoencoder combining forward and feedback learning
mechanisms for pattern separation and completion tasks. Supports CLI configuration
and visualization of results.

Usage:
    python scripts/hybrid_feedback.py [OPTIONS]
"""

from functools import partial

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
from ehc_sn.models.ann.temp_model1 import Autoencoder, AutoencoderParams
from ehc_sn.trainers.feed_forward import FeedbackTainer


# -------------------------------------------------------------------------------------------
class Experiment(BaseSettings):
    """Configuration settings for the hybrid feedback autoencoder experiment."""

    model_config = SettingsConfigDict(extra="forbid", cli_parse_args=True)

    data: DataParams = Field(default_factory=DataParams, description="Data generation parameters")
    datamodule: DataModuleParams = Field(default_factory=DataModuleParams, description="Data module parameters")
    model: AutoencoderParams = Field(default_factory=AutoencoderParams, description="Autoencoder parameters")

    figure_1: Figure1Params = Field(default_factory=Figure1Params, description="Reconstruction figure parameters")
    figure_2: Figure2Params = Field(default_factory=Figure2Params, description="Sparsity figure parameters")

    trainer: TrainerParams = Field(
        default_factory=partial(TrainerParams, experiment_name="example1"),
        description="Trainer parameters",
    )


# -------------------------------------------------------------------------------------------
def main(experiment: Experiment) -> None:
    """Run the complete hybrid feedback autoencoder experiment."""
    # Generate data and initialize components
    data_gen = DataGenerator(experiment.data)
    datamodule = BaseDataModule(data_gen, experiment.datamodule)
    trainer = FeedbackTainer(experiment.trainer)
    model = Autoencoder(experiment.model, trainer)

    # Train model using hybrid feedback algorithm
    trainer.fit(model, datamodule)

    # Evaluate on test data
    model.eval()
    datamodule.setup("test")
    test_dataloader = datamodule.test_dataloader()
    inputs, _ = next(iter(test_dataloader))

    with torch.inference_mode():
        outputs, activations = model(inputs)

    # Generate visualizations
    fig_reconstruction = ReconstructionTraceFigure(experiment.figure_1)
    _ = fig_reconstruction.plot(inputs, outputs)
    plt.show()

    sparsity_figure = SparsityFigure(experiment.figure_2)
    _ = sparsity_figure.plot(activations)
    plt.show()


if __name__ == "__main__":
    """Main entry point for the hybrid feedback autoencoder experiment."""
    experiment = Experiment()
    main(experiment)
