"""Generate the JOSS paper figure (papers/pipeline.png).

Builds a two-panel figure from the bundled example DDA data, using the
structural noise filters the paper describes. Reproducible:

    uv run python scripts/make_paper_figure.py

Left: the (m/z, 1/K0) ion map, with peaks rejected by the noise filters drawn
in grey *behind* the retained peaks (coloured by log-intensity). Right: the
1-D m/z spectrum, with centroids drawn as stem lines (no markers).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import tdfpy
from tdfpy import GaussianNoiseFilter, VerticalNoiseFilter
from tdfpy.centroiding import merge_peaks
from tdfpy.pipeline import apply_noise, convert, read_spectrum

DATA = "tests/data/example_dda.d"
OUT = Path("papers/pipeline.png")

MZ_RANGE = (400.0, 1200.0)
OOK0_RANGE = (0.6, 1.4)
FILTERS = (VerticalNoiseFilter(), GaussianNoiseFilter())


def _busiest_ms1_frame(td) -> int:
    """Pick the MS1 frame with the most raw peaks — the most illustrative."""
    cur = td.conn.cursor()
    cur.execute(
        "SELECT Id FROM Frames WHERE MsMsType = 0 "
        "ORDER BY NumPeaks DESC, Id LIMIT 1"
    )
    return int(cur.fetchone()[0])


def _zoom(mz: np.ndarray, im: np.ndarray) -> np.ndarray:
    return (
        (mz >= MZ_RANGE[0]) & (mz <= MZ_RANGE[1])
        & (im >= OOK0_RANGE[0]) & (im <= OOK0_RANGE[1])
    )


def main() -> None:
    import matplotlib.pyplot as plt

    with tdfpy.timsdata_connect(DATA) as td:
        frame_id = _busiest_ms1_frame(td)
        spectrum = read_spectrum(td, frame_id)
        raw_all = convert(spectrum, td, frame_id)

        kept_spec = apply_noise(spectrum, FILTERS, td=td, frame_id=frame_id)
        kept = convert(kept_spec, td, frame_id)

        # Rejected = raw peaks not in the kept set (matched on integer indices).
        kept_keys = set(
            zip(kept_spec.scan_indices.tolist(), kept_spec.mz_indices.tolist())
        )
        rej_mask = np.array(
            [
                (int(s), int(m)) not in kept_keys
                for s, m in zip(spectrum.scan_indices, spectrum.mz_indices)
            ],
            dtype=bool,
        )
        rejected = raw_all[rej_mask]

        centroids = merge_peaks(
            kept[:, 0], kept[:, 1], kept[:, 2],
            mz_tolerance=8.0, mz_tolerance_type="ppm",
            im_tolerance=0.01, im_tolerance_type="relative", min_peaks=3,
        )

    # Restrict everything to the zoom window.
    k = _zoom(kept[:, 0], kept[:, 2])
    r = _zoom(rejected[:, 0], rejected[:, 2]) if rejected.size else np.zeros(0, bool)
    c = _zoom(centroids[:, 0], centroids[:, 2]) if centroids.size else np.zeros(0, bool)
    kept, rejected, centroids = kept[k], rejected[r], centroids[c]

    fig, (ax_map, ax_spec) = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- Left: 2D ion map, rejected (grey) behind retained (coloured) --------
    ax_map.scatter(
        rejected[:, 0], rejected[:, 2],
        s=2, c="0.8", alpha=0.5, linewidths=0, rasterized=True, zorder=1,
        label=f"rejected as noise (n={len(rejected):,})",
    )
    sc = ax_map.scatter(
        kept[:, 0], kept[:, 2],
        s=3, c=np.log1p(kept[:, 1]), cmap="viridis",
        linewidths=0, rasterized=True, zorder=2,
        label=f"retained (n={len(kept):,})",
    )
    cb = fig.colorbar(sc, ax=ax_map, pad=0.02)
    cb.set_label("log(intensity + 1)")
    ax_map.set_xlim(MZ_RANGE)
    ax_map.set_ylim(OOK0_RANGE)
    ax_map.set_xlabel("m/z")
    ax_map.set_ylabel("1/K₀ (V·s/cm²)")
    ax_map.set_title("Raw ion map — retained vs. rejected")
    ax_map.legend(loc="upper left", fontsize=8, framealpha=0.9, markerscale=3)

    # --- Right: 1-D spectrum, centroids as stem lines (no markers) -----------
    mz_lo, mz_hi = MZ_RANGE
    n_bins = int((mz_hi - mz_lo) / 0.05)
    edges = np.linspace(mz_lo, mz_hi, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    raw_profile, _ = np.histogram(kept[:, 0], bins=edges, weights=kept[:, 1])
    ax_spec.fill_between(
        centres, raw_profile, color="0.8", lw=0, label="retained raw (summed)"
    )
    ax_spec.vlines(
        centroids[:, 0], 0, centroids[:, 1],
        color="tomato", lw=0.8, label=f"centroids (n={len(centroids):,})",
    )
    ax_spec.set_xlim(MZ_RANGE)
    ax_spec.set_ylim(bottom=0)
    ax_spec.set_xlabel("m/z")
    ax_spec.set_ylabel("Intensity")
    ax_spec.set_title("Centroided spectrum")
    ax_spec.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT} (frame {frame_id})")


if __name__ == "__main__":
    main()
