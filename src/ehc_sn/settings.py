import torch
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    device: str = Field(
        "cuda" if torch.cuda.is_available() else "cpu",
        validation_alias=AliasChoices("cpu", "gpu", "cuda", "device"),
    )

    float_matmul_precision: str = Field(
        "medium", validation_alias=AliasChoices("high", "medium", "low", "float_matmul_precision")
    )

    cudnn_benchmark: bool = Field(
        False,
        validation_alias=AliasChoices("cudnn_benchmark", "cudnn", "benchmark"),
    )


config = Settings()
torch.set_float32_matmul_precision(config.float_matmul_precision)
torch.backends.cudnn.benchmark = config.cudnn_benchmark
