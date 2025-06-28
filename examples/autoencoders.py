import torch
from lightning.pytorch.callbacks import ModelCheckpoint, RichModelSummary, RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from pydantic import BaseModel, Field

from ehc_sn.data import cognitive_maps as data
from ehc_sn.figures import cognitive_maps as figures
from ehc_sn.models import autoencoders as models
from ehc_sn.models import decoders, encoders
from ehc_sn.trainers.backprop import sparsity as trainers


class Parameters(BaseModel):
    """Parameters for the autoencoder example."""

    model_config = {"extra": "forbid"}  # Forbid extra fields not defined in the model

    generator: data.BlockMapParams = Field(
        description="Parameters for the grid map generator",
        default_factory=lambda: data.BlockMapParams(
            grid_size=(16, 32),  # Size of the grid map
            diffusion_iterations=0,  # Number of diffusion iterations (0 for no diffusion)
            diffusion_strength=0.0,  # Strength of the diffusion effect (0.0
            noise_level=0.0,  # Base noise level throughout the map (0.0-1.0)
        ),
    )
    datamodule: data.DataModuleParams = Field(
        description="Parameters for the data module",
        default_factory=lambda: data.DataModuleParams(
            num_samples=4000,  # Total number of samples to generate
            num_workers=8,  # Number of workers to generate the data
            batch_size=32,  # Increased batch size for better efficiency
            val_split=0.1,  # Fraction of data to use for validation
            test_split=0.1,  # Fraction of data to use for testing
        ),
    )
    encoder: encoders.EncoderParams = Field(
        description="Parameters for the encoder",
        default_factory=lambda: encoders.EncoderParams(
            input_shape=(1, 16, 32),  # 1 channel
            latent_dim=256,
        ),
    )
    decoder: decoders.DecoderParams = Field(
        description="Parameters for the decoder",
        default_factory=lambda: decoders.DecoderParams(
            input_shape=(1, 16, 32),  # 1 channel
            latent_dim=256,
        ),
    )
    trainer: trainers.SparsityBPTrainerParams = Field(
        description="Parameters for the backpropagation trainer",
        default_factory=lambda: trainers.SparsityBPTrainerParams(
            max_epochs=200,  # Reduced epochs for faster testing
            sparsity_target=0.05,  # Target activation (95% neurons inactive)
            sparsity_weight=0.01,  # Much lower weight for sparsity loss term
            callbacks=[
                RichModelSummary(),  # Summary of model architecture
                RichProgressBar(refresh_rate=5),  # Reduce progress bar update frequency
                ModelCheckpoint(every_n_epochs=5, save_weights_only=True),
            ],  # Optimized callbacks for training
            logger=TensorBoardLogger("logs", name="autoencoder"),  # Logger for training
            profiler="simple",  # Profiler for performance monitoring
        ),
    )
    figure: figures.CompareMapsFigParam = Field(
        description="Parameters for the cognitive map figure",
        default_factory=lambda: figures.CompareMapsFigParam(
            figsize=(10, 10),  # Size of the figure
            dpi=100,  # Dots per inch for the figure
            tight_layout=True,  # Use tight layout for better spacing
            constrained_layout=False,  # Do not use constrained layout
            title_fontsize=14,  # Font size for the title
        ),
    )


# Generate parameters
params = Parameters()

# Initialize components
generator = data.BlockMapGenerator(params.generator)
datamodule = data.DataModule(generator, params.datamodule)
encoder = encoders.LinearEncoder(params.encoder)
decoder = decoders.LinearDecoder(params.decoder)
model = models.Autoencoder(encoder, decoder)
trainer = trainers.SparsityBPTrainer(model, params.trainer)
fig_generator = figures.CompareCognitiveMaps(params.figure)

# Set up data module
datamodule.setup()

# Train the model
print("Training autoencoder...")
trainer.fit(datamodule=datamodule)
print("Training completed!")

# Run and get some predictions
print("Running predictions...")
# Get a batch of test data
(test_batch,) = next(iter(datamodule.test_dataloader()))
model.to(device=test_batch.device)  # Ensure model is on the same device as the data
with torch.no_grad():
    reconstructed, _ = model(test_batch)

# Visualize the results
print("Visualizing results...")
fig_generator.plot(list(zip(reconstructed, test_batch.squeeze(1)))[:5])
fig_generator.show()
