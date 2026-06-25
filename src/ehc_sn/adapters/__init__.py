"""Adapter package namespace.

This package only exports model-family subpackages. Canonical public adapter
symbols live in family barrels such as ``ehc_sn.adapters.ehp``,
``ehc_sn.adapters.hrm``, and ``ehc_sn.adapters.tem``.
"""

# from . import ehp, hrm, tem
from . import hrm, tem

__all__ = ["ehp", "hrm", "tem"]
