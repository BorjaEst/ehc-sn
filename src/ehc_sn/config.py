"""The configuration module for the EHC-SN model."""

from pydantic import Field, NonNegativeFloat
from pydantic_settings import BaseSettings

# pylint: disable=too-few-public-methods
# pylint: disable=non-ascii-name


class HGMSettings(BaseSettings):
    """The param set class for model configuration."""

    δ: NonNegativeFloat = Field(0.7, ge=0, le=1)  # Discount factor sequence
    τ: NonNegativeFloat = Field(0.9, ge=0, le=1)  # Exp. decay mixing cat
    c: NonNegativeFloat = Field(0.4, ge=0, le=1)  # Velocity rate item code


class LearningSettings(BaseSettings):
    """The param set class for learning configuration."""

    γ: NonNegativeFloat = Field(0.1, ge=0, le=1)  # π hyperparameter decay
    λ: NonNegativeFloat = Field(0.1, ge=0, le=1)  # Priority maps decay


class GenSettings(HGMSettings, LearningSettings):
    """The param set class for model generation configuration."""
