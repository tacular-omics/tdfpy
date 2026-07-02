"""Region exclusions in (m/z, 1/K0) space.

A region is a known area of the (m/z, 1/K0) plane that the user wants to
drop wholesale — typically based on physical/chemical knowledge of the
acquisition rather than from estimating noise. The canonical example is
the singly-charged / polymer contamination band in timsTOF MS1.

Conceptually distinct from :mod:`tdfpy.noise`: region exclusion answers
"which part of the data plane are we even interested in?", while noise
filtering answers "of what's left, what's real signal?".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .timsdata import TimsData


@dataclass(frozen=True)
class ChargeStateRegion:
    """Drop peaks above a line in (m/z, 1/K0) space, capped at the line's upper endpoint.

    The line is defined by two ``(m/z, 1/K0)`` points. A peak is *in the
    region* (and therefore dropped) iff

        ``1/K0 > line(m/z)``     OR (if ``cap_at_upper_endpoint``)
        ``1/K0 > max(point_1[1], point_2[1])``.

    The default endpoints target the singly-charged region in typical
    timsTOF MS1 data.
    """

    line: tuple[tuple[float, float], tuple[float, float]] = (
        (350.0, 0.7),
        (1200.0, 1.4),
    )
    cap_at_upper_endpoint: bool = True

    def __post_init__(self) -> None:
        (mz_1, ook0_1), (mz_2, ook0_2) = self.line
        if mz_1 == mz_2:
            raise ValueError(
                "ChargeStateRegion line endpoints must differ in m/z; got both at "
                f"m/z={mz_1}. Provide two points with distinct m/z, e.g. "
                "line=((350.0, 0.7), (1200.0, 1.4))."
            )
        if ook0_1 == ook0_2:
            raise ValueError(
                "ChargeStateRegion line endpoints must differ in 1/K0; got both at "
                f"1/K0={ook0_1}. Provide two points with distinct 1/K0."
            )
        # The exclusion mask assumes 1/K0 increases with m/z (positive slope),
        # as real timsTOF charge-state bands do. A negative slope would silently
        # exclude the opposite half-plane, so reject it rather than mislead.
        if (mz_2 - mz_1) * (ook0_2 - ook0_1) < 0:
            raise ValueError(
                "ChargeStateRegion requires a non-negative m/z-vs-1/K0 slope "
                f"(1/K0 must increase with m/z), but line={self.line} has 1/K0 "
                "decreasing as m/z increases. Order the points so the higher-m/z "
                "endpoint also has the higher 1/K0."
            )

    def index_cutoff_per_scan(
        self, td: TimsData, frame_id: int, num_scans: int
    ) -> np.ndarray:
        """Per-scan TOF-index cutoff implementing this region exclusion.

        For each scan, ``mz_indices < cutoff[scan]`` lies above the line
        (in the region) and should be dropped. Scans whose 1/K0 is above
        the cap get cutoff ``+inf`` so all their peaks are dropped.

        Performing the comparison in integer-index space is much cheaper
        than converting every peak to m/z and 1/K0.
        """
        (mz_1, ook0_1), (mz_2, ook0_2) = self.line
        mz_per_ook0 = (mz_2 - mz_1) / (ook0_2 - ook0_1)
        ook0_cap = max(ook0_1, ook0_2)

        ook0_per_scan = np.asarray(
            td.scanNumToOneOverK0(frame_id, np.arange(num_scans))  # type: ignore[call-arg]
        )
        mz_cutoff = mz_1 + (ook0_per_scan - ook0_1) * mz_per_ook0
        mz_cutoff_clipped = np.clip(mz_cutoff, a_min=1e-6, a_max=None)
        index_cutoff = np.asarray(
            td.mzToIndex(frame_id, mz_cutoff_clipped)
        ).astype(np.float64, copy=True)
        index_cutoff = np.where(mz_cutoff > 0, index_cutoff, 0.0)
        if self.cap_at_upper_endpoint:
            index_cutoff[ook0_per_scan > ook0_cap] = np.inf
        return index_cutoff


__all__ = ["ChargeStateRegion"]
