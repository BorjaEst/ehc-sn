"""Arena spatial navigation experiments.

Each module wires a (model family, version) into its Lightning regime.
"""

from ehc_sn.experiments.arena import tem_v1, tem_v2

__all__ = ["tem_v1", "tem_v2"]
