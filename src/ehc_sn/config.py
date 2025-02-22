"""The configuration module for the EHC-SN model."""

from typing import Any

from pydantic import Field, NonNegativeFloat, field_validator
from pydantic_settings import BaseSettings


class HGModelSettings(BaseSettings):
    """The param set class example for model configuration."""

    δ: NonNegativeFloat = Field(0.7, ge=0, le=1)  # Discount factor sequence
    τ: NonNegativeFloat = Field(0.9, ge=0, le=1)  # Exp. decay mixing cat
    c: NonNegativeFloat = Field(0.4, ge=0, le=1)  # Velocity rate item code

    @field_validator("δ", "τ", "c")
    @classmethod
    def check_smaller_than_1(cls, v: str, info: Any) -> str:
        """Check if the value is smaller than 1."""
        if isinstance(v, float):
            assert v <= 1, f"{info.field_name} must be smaller than 1"
        return v
