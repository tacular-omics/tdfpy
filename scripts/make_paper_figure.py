"""Generate the JOSS paper figure (papers/pipeline.png).

Renders tdfpy's own ``plot_centroiding`` 2x2 view on the bundled example DDA
data, zoomed to the informative region and using the structural noise filters
the paper describes. Reproducible:

    uv run python scripts/make_paper_figure.py
"""

from __future__ import annotations

from pathlib import Path

import tdfpy
from tdfpy import GaussianNoiseFilter, VerticalNoiseFilter
from tdfpy.viz import plot_centroiding

DATA = "tests/data/example_dda.d"
OUT = Path("papers/pipeline.png")

MZ_RANGE = (400.0, 1200.0)
OOK0_RANGE = (0.6, 1.4)


def _busiest_ms1_frame(td) -> int:
    """Pick the MS1 frame with the most raw peaks — the most illustrative."""
    cur = td.conn.cursor()
    cur.execute(
        "SELECT Id FROM Frames WHERE MsMsType = 0 "
        "ORDER BY NumPeaks DESC, Id LIMIT 1"
    )
    return int(cur.fetchone()[0])


def main() -> None:
    with tdfpy.timsdata_connect(DATA) as td:
        frame_id = _busiest_ms1_frame(td)
        fig = plot_centroiding(
            td,
            frame_id=frame_id,
            noise=[VerticalNoiseFilter(), GaussianNoiseFilter()],
            mz_range=MZ_RANGE,
            im_range=OOK0_RANGE,
        )
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT} (frame {frame_id})")


if __name__ == "__main__":
    main()
