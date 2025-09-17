"""
Backpropagation Autoencoder Training Script.

Trains a sparse autoencoder using standard backpropagation for pattern separation
and completion tasks. Supports CLI configuration and visualization of results.

Usage:
    python scripts/bp_autoencoder.py [OPTIONS]
"""

import matplotlib.pyplot as plt
import torch
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from ehc_sn.core.datamodule import BaseDataModule, DataModuleParams
from ehc_sn.core.trainer import TrainerParams
from ehc_sn.data.obstacle_maps import DataGenerator, DataParams
from ehc_sn.figures.reconstruction_map import ReconstructionMapFigure
from ehc_sn.figures.reconstruction_map import ReconstructionMapParams as Figure1Params
from ehc_sn.figures.sparsity import SparsityFigure
from ehc_sn.figures.sparsity import SparsityParams as Figure2Params
from ehc_sn.models.ann.sparse_autoencoder import Autoencoder, AutoencoderParams
from ehc_sn.trainers.back_propagation import BackwardTrainer


# -------------------------------------------------------------------------------------------
class Experiment(BaseSettings):
    """Configuration settings for the backpropagation autoencoder experiment."""

    model_config = SettingsConfigDict(extra="forbid", cli_parse_args=True)

    data: DataParams = Field(default_factory=DataParams, description="Data generation parameters")
    datamodule: DataModuleParams = Field(default_factory=DataModuleParams, description="Data module parameters")
    model: AutoencoderParams = Field(default_factory=AutoencoderParams, description="Autoencoder parameters")
    trainer: TrainerParams = Field(default_factory=TrainerParams, description="Trainer parameters")
    figure_1: Figure1Params = Field(default_factory=Figure1Params, description="Reconstruction figure parameters")
    figure_2: Figure2Params = Field(default_factory=Figure2Params, description="Sparsity figure parameters")


# -------------------------------------------------------------------------------------------
def main(experiment: Experiment) -> None:
    """Run the complete backpropagation autoencoder experiment."""
    # Generate data and initialize components
    data_gen = DataGenerator(experiment.data)
    datamodule = BaseDataModule(data_gen, experiment.datamodule)
    trainer = BackwardTrainer(experiment.trainer)
    model = Autoencoder(experiment.model, trainer)

    # Train model using backpropagation
    trainer.fit(model, datamodule)

    # Evaluate on test data
    model.eval()
    datamodule.setup("test")
    test_dataloader = datamodule.test_dataloader()
    inputs, _ = next(iter(test_dataloader))

    with torch.inference_mode():
        outputs, activations = model(inputs)

    # Generate visualizations
    fig_reconstruction = ReconstructionMapFigure(experiment.figure_1)
    _ = fig_reconstruction.plot(inputs, outputs)
    plt.show()

    sparsity_figure = SparsityFigure(experiment.figure_2)
    _ = sparsity_figure.plot(activations)
    plt.show()


if __name__ == "__main__":
    """Main entry point for the backpropagation autoencoder experiment."""
    experiment = Experiment()
    main(experiment)
