import importlib.resources
from typing import Optional

import torch
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings for the EHC spatial navigation model."""

    parameters_file: str = Field(
        str(importlib.resources.files("ehc_sn").joinpath("config", "parameters.toml")),
        description="Path to the TOML file containing model parameters located inside the package.",
    )


config = Settings()
