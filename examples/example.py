import matplotlib.pyplot as plt
import torch
from pydantic import BaseModel, Field

from ehc_sn.core.datamodule import BaseDataModule, DataModuleParams
from ehc_sn.core.trainer import TrainerParams
from ehc_sn.data.simple_example import DataGenerator, DataParams
from ehc_sn.figures.reconstruction_1d import ReconstructionTraceFigure
from ehc_sn.figures.reconstruction_1d import ReconstructionTraceParams as Figure1Params
from ehc_sn.figures.sparsity import SparsityFigure
from ehc_sn.figures.sparsity import SparsityParams as Figure2Params
from ehc_sn.models.ann.autoencoders.hybrid_feedback import Autoencoder, AutoencoderParams
from ehc_sn.trainers.feed_forward import FeedbackTainer


class Experiment(BaseModel):
    """Configuration settings for the hybrid feedback autoencoder experiment."""

    model_config = {"extra": "forbid", "arbitrary_types_allowed": True}

    data: DataParams = Field(default_factory=DataParams, description="Data generation parameters")
    datamodule: DataModuleParams = Field(default_factory=DataModuleParams, description="Data module parameters")
    model: AutoencoderParams = Field(default_factory=AutoencoderParams, description="Autoencoder parameters")
    trainer: TrainerParams = Field(default_factory=TrainerParams, description="Trainer parameters")
    figure_1: Figure1Params = Field(default_factory=Figure1Params, description="Reconstruction figure parameters")
    figure_2: Figure2Params = Field(default_factory=Figure2Params, description="Sparsity figure parameters")

    def run(self):
        """Run the experiment with the specified settings."""
        # Setup data and training
        data_gen = DataGenerator(self.data)
        datamodule = BaseDataModule(data_gen, self.datamodule)
        trainer = FeedbackTainer(self.trainer)
        trainer.params.max_epochs = 5  # Short training for example
        model = Autoencoder(self.model, trainer)

        # Train the model
        trainer.fit(model, datamodule)

        # Generate visualization
        model.eval()
        datamodule.setup("test")
        test_dataloader = datamodule.test_dataloader()

        # Get a batch for visualization
        inputs, _ = next(iter(test_dataloader))

        with torch.no_grad():
            outputs, activations = model(inputs)

        # Create and show reconstruction figure
        fig_reconstruction = ReconstructionTraceFigure(self.figure_1)
        fig = fig_reconstruction.plot(inputs, outputs)
        plt.show()

        # Create and show sparsity figure
        sparsity_figure = SparsityFigure(self.figure_2)
        fig2 = sparsity_figure.plot(activations)
        plt.show()


if __name__ == "__main__":
    """Run the hybrid feedback autoencoder experiment."""
    Experiment().run()
