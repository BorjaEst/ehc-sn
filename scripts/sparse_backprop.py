"""Sparse Autoencoder training script with backpropagation."""

import matplotlib.pyplot as plt
import torch
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from ehc_sn.core.datamodule import BaseDataModule, DataModuleParams
from ehc_sn.core.trainer import TrainerParams
from ehc_sn.data.obstacle_maps import DataGenerator, DataParams
from ehc_sn.figures.decoder_montage import DecoderMontageFigure
from ehc_sn.figures.decoder_montage import DecoderMontageParams as Figure3Params
from ehc_sn.figures.reconstruction_map import ReconstructionMapFigure
from ehc_sn.figures.reconstruction_map import ReconstructionMapParams as Figure1Params
from ehc_sn.figures.sparsity import SparsityFigure
from ehc_sn.figures.sparsity import SparsityParams as Figure2Params
from ehc_sn.models.ann.sparse_autoencoder import Autoencoder, ModelParams
from ehc_sn.trainers.back_propagation import BackwardTrainer


# -------------------------------------------------------------------------------------------
class Experiment(BaseSettings):
    """Configuration settings for the sparse autoencoder experiment."""

    model_config = SettingsConfigDict(extra="forbid", cli_parse_args=True)

    data: DataParams = Field(default_factory=DataParams, description="Data generation parameters")
    datamodule: DataModuleParams = Field(default_factory=DataModuleParams, description="Data module parameters")
    model: ModelParams = Field(default_factory=ModelParams, description="Autoencoder parameters")

    figure_1: Figure1Params = Field(default_factory=Figure1Params, description="Reconstruction figure parameters")
    figure_2: Figure2Params = Field(default_factory=Figure2Params, description="Sparsity figure parameters")
    figure_3: Figure3Params = Field(default_factory=Figure3Params, description="Decoder montage figure parameters")

    trainer: TrainerParams = Field(
        default_factory=lambda: TrainerParams(experiment_name="sparse_backprop"),
        description="Trainer parameters",
    )


# -------------------------------------------------------------------------------------------
def main(experiment: Experiment) -> None:
    """Run the sparse autoencoder experiment."""
    # Generate data and initialize components
    data_gen = DataGenerator(experiment.data)
    datamodule = BaseDataModule(data_gen, experiment.datamodule)
    trainer = BackwardTrainer(experiment.trainer)
    model = Autoencoder(experiment.model, trainer)

    # Train till end of training or keyboard interup
    try:
        trainer.fit(model, datamodule)
    except KeyboardInterrupt:
        print("Training interrupted by user. Generating figures...")
    finally:
        gen_figures(model, datamodule, experiment)


# -------------------------------------------------------------------------------------------
def gen_figures(model: Autoencoder, datamodule: BaseDataModule, experiment: Experiment) -> None:
    """Generate reconstruction, sparsity, and decoder montage figures from model outputs."""
    model.eval()
    datamodule.setup("test")
    test_dataloader = datamodule.test_dataloader()

    try:
        inputs, _ = next(iter(test_dataloader))
    except StopIteration:
        print("No test data available for plotting.")
        return

    activations = model.encode(inputs)
    reconstructions = model.decode(activations)

    # Figure 1: Reconstruction comparison
    fig_reconstruction = ReconstructionMapFigure(experiment.figure_1)
    _ = fig_reconstruction.plot(inputs, reconstructions)
    plt.show()

    # Figure 2: Sparsity analysis
    sparsity_figure = SparsityFigure(experiment.figure_2)
    _ = sparsity_figure.plot(activations)
    plt.show()

    latent_dim = experiment.model.latent_units  # Number of latent units
    latents = torch.eye(latent_dim)  # One-hot encoding for each unit

    # Generate decoder outputs for one-hot latents
    reconstructions = model.decode(latents)

    # Figure 3: Decoder montage showing individual latent unit reconstructions
    decoder_montage_figure = DecoderMontageFigure(experiment.figure_3)
    _ = decoder_montage_figure.plot(latents, reconstructions)
    plt.show()


if __name__ == "__main__":
    """Main entry point for sparse autoencoder experiment."""
    experiment = Experiment()
    main(experiment)
