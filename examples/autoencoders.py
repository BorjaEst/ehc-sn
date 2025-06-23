import torch
from pydantic import BaseModel, Field
from torch import optim

from ehc_sn.data import cognitive_maps as data
from ehc_sn.figures import cognitive_maps as figures
from ehc_sn.hooks import TensorBoardLogger, TqdmProgress, datahooks
from ehc_sn.losses import autoencoders as losses
from ehc_sn.models import autoencoders as models
from ehc_sn.models import decoders, encoders
from ehc_sn.trainers import backprop as trainers


class Parameters(BaseModel):
    """Parameters for the autoencoder example."""

    model_config = {"extra": "forbid"}  # Forbid extra fields not defined in the model

    generator: data.GeneratorParams = Field(
        description="Parameters for the grid map generator",
        default_factory=lambda: data.GeneratorParams(
            grid_size=(10, 10),  # Size of the grid map
            obstacle_density=0.3,  # Density of obstacles in the grid
            diffusion_iterations=0,  # Number of diffusion iterations (0 for no diffusion)
            diffusion_strength=0.0,  # Strength of the diffusion effect (0.0
            noise_level=0.0,  # Base noise level throughout the map (0.0-1.0)
        ),
    )
    datamodule: data.DataModuleParams = Field(
        description="Parameters for the data module",
        default_factory=lambda: data.DataModuleParams(
            num_samples=1000,  # Total number of samples to generate
            batch_size=4,  # Batch size for training and testing
            val_split=0.2,  # Fraction of data to use for validation
            test_split=0.1,  # Fraction of data to use for testing
        ),
    )
    encoder: encoders.EncoderParams = Field(
        description="Parameters for the encoder",
        default_factory=lambda: encoders.EncoderParams(
            feature_dims=(10, 10),  # Input is a 10x10 grid
            embedding_dim=12,  # Latent space dimension
            dims=[100, 32, 12],  # Network architecture
        ),
    )
    decoder: decoders.DecoderParams = Field(
        description="Parameters for the decoder",
        default_factory=lambda: decoders.DecoderParams(
            feature_dims=(10, 10),  # Input is a 10x10 grid
            embedding_dim=12,  # Latent space dimension
            dims=[12, 32, 100],  # Network architecture
        ),
    )
    trainer: trainers.BPTrainerParams = Field(
        description="Parameters for the backpropagation trainer",
        default_factory=lambda: trainers.BPTrainerParams(
            max_epochs=50,  # Maximum number of epochs for training
            accelerator="auto",  # Use available hardware (CPU/GPU)
            callbacks=[
                datahooks.AppendToBatch(),
                TqdmProgress(),
            ],
            loggers=[
                TensorBoardLogger("logs"),
            ],
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
data_module = data.DataModule(generator, params.datamodule)
encoder = encoders.Encoder(params.encoder)
decoder = decoders.Decoder(params.decoder)
model = models.Autoencoder(encoder, decoder)
trainer = trainers.BPTrainer(params.trainer)
fig_generator = figures.CompareCognitiveMaps(params.figure)

# Set up data module
data_module.setup()


# Define loss functions, optimizer and scheduler
loss_functions = {
    "reconstruction": losses.ReconstructionLoss(),
    "sparsity": losses.SparsityLoss(
        losses.SparseLossParams(
            sparsity_target=0.05,  # Target activation rate for sparsity
            sparsity_weight=0.1,  # Weight for sparsity loss term (beta)
        )
    ),
}
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",  # Reduce learning rate when validation loss plateaus
    factor=0.1,  # Factor by which to reduce the learning rate
    patience=5,  # Number of epochs with no improvement after which learning rate will be reduced
)

# Train the model
print("Training autoencoder...")
trainer.fit(model, data_module, loss_functions, optimizer, scheduler)
print("Training completed!")

# Run and get some predictions
print("Running predictions...")
# Get a batch of test data
(test_batch,) = next(iter(data_module.test_dataloader()))
model.to(device=test_batch.device)  # Ensure model is on the same device as the data
with torch.no_grad():
    reconstructed, _ = model(test_batch)

# Visualize the results
print("Visualizing results...")
fig_generator.plot(list(zip(reconstructed, test_batch))[:5])
fig_generator.show()
