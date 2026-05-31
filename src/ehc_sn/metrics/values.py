"""Contract for metric scalar values — finite int/float, not bool.

A **metric value** is any scalar that can appear in a summary dict,
``MetricRecord.value``, or an eval-artifact manifest summary entry.

Valid values: ``int``, ``float`` (finite).
Invalid: ``bool``, ``str``, ``None``, sequences, ``NaN``, ``inf``, ``-inf``.
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

MetricValue = int | float
"""A valid metric scalar: finite ``int`` or ``float`` (not ``bool``)."""


# ---------------------------------------------------------------------------
# Value validator
# ---------------------------------------------------------------------------


def validate_metric_value(*, key: str, value: object) -> MetricValue:
    """Validate and type-narrow a single metric value.

    Accepts ``int`` and ``float``.  Rejects ``bool``, non-numeric types,
    ``NaN``, and ``inf`` (positive and negative).

    Parameters
    ----------
    key:
        Human-readable name for error messages (e.g. the summary key
        or metric name being validated).
    value:
        Candidate value to validate.

    Returns
    -------
    MetricValue
        ``int`` or ``float``, guaranteed finite.

    Raises
    ------
    TypeError
        If *value* is ``bool`` or not ``int`` or ``float``.
    ValueError
        If *value* is ``NaN``, ``inf``, or ``-inf``.
    """
    if isinstance(value, bool):
        raise TypeError(
            f"Metric {key!r} is bool ({value!r}); bool is not allowed."
        )
    if not isinstance(value, (int, float)):
        raise TypeError(
            f"Metric {key!r} must be int or float, "
            f"got {type(value).__name__} ({value!r})."
        )
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        raise ValueError(
            f"Metric {key!r} is {value!r}; NaN and inf are not allowed."
        )
    return value


# ---------------------------------------------------------------------------
__all__ = ["MetricValue", "validate_metric_value"]
