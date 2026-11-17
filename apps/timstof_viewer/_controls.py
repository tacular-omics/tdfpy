"""Pure sidebar-control decisions for the viewer, with no Streamlit import.

Kept Streamlit-free for the same reason as :mod:`_sql`: ``apps/`` is internal
dev tooling and Streamlit is not a dependency of this package, so a decision
worth testing has to live where the tests can import it. ``tests/
test_apps_controls.py`` loads this file by path.
"""

from __future__ import annotations

#: Shown when the ion mobility axis is not 1/K0, where an absolute tolerance has no meaning.
ABSOLUTE_IS_OOK0_HINT = (
    "An absolute IM tolerance is a width in 1/K0 (V·s/cm²), because centroiding always "
    "clusters in 1/K0. Switch the ion mobility axis to 1/K0 to use it."
)


def im_tolerance_unit_options(mobility_type: str) -> tuple[list[str], str | None]:
    """The IM tolerance units to offer for a mobility axis, and the hint to show.

    ``MergePeaksCentroider`` raises ``TdfpyError`` for ``im_tolerance_unit=
    "absolute"`` with ``mobility_type`` ``"ccs"`` or ``"voltage"``, because
    clustering runs in 1/K0 and an absolute width in volts or Å² is not one
    number there. The control therefore must not offer that combination: the
    dashboard should never be able to reach that error from its own widgets.

    Args:
        mobility_type (str): The ion mobility axis the page is reporting.

    Returns:
        tuple[list[str], str | None]: The units to offer, last one first-choice,
            and the hint to display, or ``None`` when every unit is available.
    """
    if mobility_type == "ook0":
        return ["relative", "absolute"], None
    return ["relative"], ABSOLUTE_IS_OOK0_HINT
