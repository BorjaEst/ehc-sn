from pydantic import BaseModel, Field
from dataclasses import dataclass


class BGSettings(BaseModel, extra="forbid"):
    """Basal Ganglia Settings"""


@dataclass
class BGState:
    """Basal Ganglia State"""


class BGModel:
    """Basal Ganglia Model"""
