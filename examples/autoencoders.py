import torch
from pydantic import BaseModel, Field
from torch import optim

from ehc_sn.data import cognitive_maps as data
from ehc_sn.figures import cognitive_maps as figures
from ehc_sn.losses import autoencoders as losses
from ehc_sn.models import autoencoders as models
from ehc_sn.models import decoders, encoders
from ehc_sn.trainers.backprop import sparsity as trainers


class Parameters(BaseModel):
    """Parameters for the autoencoder example."""

    model_config = {"extra": "forbid"}  # Forbid extra fields not defined in the model

    generator: data.GeneratorParams = Field(
        description="Parameters for the grid map generator",
        default_factory=lambda: data.GeneratorParams(
            grid_size=(16, 32),  # Size of the grid map
            obstacle_density=0.4,  # Density of obstacles in the grid
            diffusion_iterations=0,  # Number of diffusion iterations (0 for no diffusion)
            diffusion_strength=0.0,  # Strength of the diffusion effect (0.0
            noise_level=0.0,  # Base noise level throughout the map (0.0-1.0)
        ),
    )
    data_module: data.DataModuleParams = Field(
        description="Parameters for the data module",
        default_factory=lambda: data.DataModuleParams(
            num_samples=1000,  # Total number of samples to generate
            num_workers=14,  # Number of workers for data loading
            batch_size=4,  # Batch size for training and testing
            val_split=0.2,  # Fraction of data to use for validation
            test_split=0.1,  # Fraction of data to use for testing
        ),
    )
    encoder: encoders.EncoderParams = Field(
        description="Parameters for the encoder",
        default_factory=lambda: encoders.EncoderParams(
            input_shape=(1, 16, 32),  # 1 channel
            latent_dim=128,
        ),
    )
    decoder: decoders.DecoderParams = Field(
        description="Parameters for the decoder",
        default_factory=lambda: decoders.DecoderParams(
            input_shape=(1, 16, 32),  # 1 channel
            latent_dim=128,
        ),
    )
    trainer: trainers.SparsityBPTrainerParams = Field(
        description="Parameters for the backpropagation trainer",
        default_factory=lambda: trainers.SparsityBPTrainerParams(
            max_epochs=20,  # Maximum number of epochs for training
            sparsity_target=0.05,  # Target activation rate for sparsity
            sparsity_weight=0.1,  # Weight for sparsity loss term (beta)
            # callbacks=[],
            # loggers=[],
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
generator = data.Generator(params.generator)
datamodule = data.DataModule(generator, params.data_module)
encoder = encoders.Encoder(params.encoder)
decoder = decoders.Decoder(params.decoder)
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
