"""Shared type aliases, identical to ``tacular.types`` so the tacular-omics packages agree.

tdfpy does not depend on tacular, so it keeps its own copies with the same names,
values and order.
"""

from typing import Literal

__all__ = ["Polarity", "ToleranceUnit"]

Polarity = Literal["positive", "negative"]
"""Ion polarity of a frame. Fields typed ``Polarity | None`` are ``None`` when unknown or mixed."""

ToleranceUnit = Literal["da", "ppm"]
"""Unit of an m/z tolerance: ``"da"`` (absolute, daltons) or ``"ppm"`` (parts per million)."""
