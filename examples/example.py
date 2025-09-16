import argparse
from typing import Any, Dict, Iterable, List, Optional, Tuple

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, RichModelSummary, RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from pydantic import BaseModel, Field
from torch.utils.data import DataLoader, TensorDataset

from ehc_sn.core.datamodule import BaseDataModule, DataModuleParams
from ehc_sn.core.trainer import TrainerParams
from ehc_sn.data import cognitive_maps as data
from ehc_sn.data.simple_example import DataGenerator, DataParams, make_figure
from ehc_sn.figures import cognitive_maps as figures
from ehc_sn.models.ann.autoencoders.hybrid_feedback import Autoencoder, AutoencoderParams
from ehc_sn.trainers.feed_forward import FeedbackTainer
from ehc_sn.utils import load_settings


class Experiment(BaseModel):
    """Configuration settings for the hybrid feedback autoencoder experiment."""

    model_config = {"extra": "forbid", "arbitrary_types_allowed": True}

    # # Data Generation Settings
    # grid_width: int = Field(default=32, ge=8, le=128, description="Width of the cognitive map grid")
    # grid_height: int = Field(default=16, ge=8, le=128, description="Height of the cognitive map grid")
    # diffusion_iterations: int = Field(default=0, ge=0, le=10, description="Number of diffusion iterations")
    # diffusion_strength: float = Field(default=0.0, ge=0.0, le=1.0, description="Diffusion strength")
    # noise_level: float = Field(default=0.0, ge=0.0, le=1.0, description="Base noise level")

    data: DataParams = Field(default_factory=DataParams, description="Data generation parameters")
    datamodule: DataModuleParams = Field(default_factory=DataModuleParams, description="Data module parameters")
    model: AutoencoderParams = Field(default_factory=AutoencoderParams, description="Autoencoder parameters")
    trainer: TrainerParams = Field(default_factory=TrainerParams, description="Trainer parameters")

    # Visualization Settings
    fig_width: int = Field(default=10, ge=6, le=20, description="Figure width in inches")
    fig_height: int = Field(default=10, ge=6, le=20, description="Figure height in inches")
    fig_dpi: int = Field(default=100, ge=50, le=300, description="Figure resolution (DPI)")
    vis_samples: int = Field(default=5, ge=1, le=20, description="Number of samples to visualize")

    @property
    def input_units(self) -> int:
        """Input dimensionality derived from model parameters."""
        return self.model.output_units

    @property
    def output_units(self) -> int:
        """Output dimensionality derived from model parameters."""
        return self.model.output_units

    @property
    def latent_units(self) -> int:
        """Latent dimensionality derived from model parameters."""
        return self.model.latent_units

    def run(self):
        """Run the experiment with the specified settings."""
        data_gen = DataGenerator(self.data)
        datamodule = BaseDataModule(data_gen, self.datamodule)
        trainer = FeedbackTainer(self.trainer)
        model = Autoencoder(self.model, trainer)
        trainer.fit(model, datamodule)

        # model.eval()
        # with torch.no_grad():
        #     output, _ = model(sensors)
        # make_figure(sensors, output)


if __name__ == "__main__":
    """Run the hybrid feedback autoencoder experiment."""
    Experiment().run()
