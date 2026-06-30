"""Adapter package namespace.

This package only exports model-family subpackages. Canonical public adapter
symbols live in family barrels such as ``ehc_sn.adapters.ehp``,
``ehc_sn.adapters.hrm``, and ``ehc_sn.adapters.tem``.

Note: ``ehp`` is omitted from the export list because the EHP adapter module
depends on a model path (``models.ehp.ehp_v1``) that does not currently exist.
The commented import is preserved for when the model module is created.
"""

# from . import ehp, hrm, tem
from . import hrm, tem

__all__ = ["hrm", "tem"]
