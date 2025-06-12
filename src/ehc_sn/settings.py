import importlib.resources

import torch
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings for the EHC spatial navigation model."""

    parameters_file: str = Field(
        str(importlib.resources.files("ehc_sn").joinpath("config", "parameters.toml")),
        description="Path to the TOML file containing model parameters located inside the package.",
    )

    device: str = Field(
        "cuda" if torch.cuda.is_available() else "cpu",
        description="Device to run the model on, either 'cpu' or 'cuda'.",
        validation_alias=AliasChoices("cpu", "gpu", "cuda", "device"),
    )

    float_matmul_precision: str = Field(
        "medium",
        description="Floating point precision for matrix multiplication.",
        validation_alias=AliasChoices("high", "medium", "low", "float_matmul_precision"),
    )

    cudnn_benchmark: bool = Field(
        False,
        description="Enable CuDNN benchmark mode for performance optimization on fixed input sizes.",
        validation_alias=AliasChoices("cudnn_benchmark", "cudnn", "benchmark"),
    )


config = Settings()
torch.set_float32_matmul_precision(config.float_matmul_precision)
torch.backends.cudnn.benchmark = config.cudnn_benchmark
