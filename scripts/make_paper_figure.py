"""Generate the JOSS paper figure (papers/pipeline.png).

Builds a four-panel figure from the bundled example DDA data, zoomed to the
informative region, using the structural noise filters the paper describes.
Reproducible:

    uv run python scripts/make_paper_figure.py

Panels (clockwise from top-left):
  1. Raw (m/z, 1/K0) ion map — *all* raw peaks, with the peaks rejected by the
     noise filters drawn in grey *behind* the retained peaks (coloured by
     log-intensity), so signal sits in front of noise.
  2. The retained cloud (faded) with the resulting centroids as stars.
  3. 1-D m/z spectrum — retained raw (summed) with centroids as stem lines.
  4. The peaks rejected as noise, on their own.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import tdfpy
from tdfpy import HorizontalHaloFilter, VerticalNoiseFilter
from tdfpy.centroiding import merge_peaks
from tdfpy.pipeline import apply_noise, convert, read_spectrum

DATA = "tests/data/example_dda.d"
OUT = Path("papers/pipeline.png")

MZ_RANGE = (400.0, 1200.0)
OOK0_RANGE = (0.6, 1.4)
FILTERS = (VerticalNoiseFilter(), HorizontalHaloFilter())


def _busiest_ms1_frame(td) -> int:
    cur = td.conn.cursor()
    cur.execute(
        "SELECT Id FROM Frames WHERE MsMsType = 0 "
        "ORDER BY NumPeaks DESC, Id LIMIT 1"
    )
    return int(cur.fetchone()[0])


def _zoom(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    m = (
        (arr[:, 0] >= MZ_RANGE[0]) & (arr[:, 0] <= MZ_RANGE[1])
        & (arr[:, 2] >= OOK0_RANGE[0]) & (arr[:, 2] <= OOK0_RANGE[1])
    )
    return arr[m]


def main() -> None:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    with tdfpy.timsdata_connect(DATA) as td:
        frame_id = _busiest_ms1_frame(td)
        spectrum = read_spectrum(td, frame_id)
        raw_all = convert(spectrum, td, frame_id)

        kept_spec = apply_noise(spectrum, FILTERS, td=td, frame_id=frame_id)
        kept = convert(kept_spec, td, frame_id)

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

    kept, rejected, centroids = _zoom(kept), _zoom(rejected), _zoom(centroids)

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.28, wspace=0.22)
    ax_raw = fig.add_subplot(gs[0, 0])
    ax_cent = fig.add_subplot(gs[0, 1], sharex=ax_raw, sharey=ax_raw)
    ax_spec = fig.add_subplot(gs[1, 0])
    ax_rej = fig.add_subplot(gs[1, 1], sharex=ax_raw, sharey=ax_raw)

    # 1 — raw ion map: rejected (grey) behind, retained (colour) in front
    ax_raw.scatter(rejected[:, 0], rejected[:, 2], s=2, c="0.78", alpha=0.6,
                   linewidths=0, rasterized=True, zorder=1)
    sc = ax_raw.scatter(kept[:, 0], kept[:, 2], s=3, c=np.log1p(kept[:, 1]),
                        cmap="viridis", linewidths=0, rasterized=True, zorder=2)
    fig.colorbar(sc, ax=ax_raw, pad=0.02).set_label("log(intensity + 1)", fontsize=8)
    ax_raw.set_ylabel("1/K₀ (V·s/cm²)")
    ax_raw.set_title(f"Raw peaks (n={len(kept) + len(rejected):,}) — "
                     f"retained over rejected (grey)")

    # 2 — centroids over the faded retained cloud
    ax_cent.scatter(kept[:, 0], kept[:, 2], s=2, c="0.8", alpha=0.5,
                    linewidths=0, rasterized=True, zorder=1)
    if len(centroids):
        s_c = 20 + 180 * (centroids[:, 1] - centroids[:, 1].min()) / (
            np.ptp(centroids[:, 1]) or 1.0)
        ax_cent.scatter(centroids[:, 0], centroids[:, 2], s=s_c,
                        c=np.log1p(centroids[:, 1]), cmap="plasma", marker="*",
                        edgecolors="white", linewidths=0.3, zorder=3)
    ax_cent.set_title(f"Centroids (n={len(centroids):,})")

    # 3 — 1-D spectrum: retained raw (summed) + centroid stem lines
    edges = np.linspace(*MZ_RANGE, int((MZ_RANGE[1] - MZ_RANGE[0]) / 0.05) + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    profile, _ = np.histogram(kept[:, 0], bins=edges, weights=kept[:, 1])
    ax_spec.fill_between(centres, profile, color="0.8", lw=0,
                         label="retained raw (summed)")
    ax_spec.vlines(centroids[:, 0], 0, centroids[:, 1], color="tomato", lw=0.8,
                   label="centroids")
    ax_spec.set_xlim(MZ_RANGE); ax_spec.set_ylim(bottom=0)
    ax_spec.set_xlabel("m/z"); ax_spec.set_ylabel("intensity")
    ax_spec.set_title("Centroided spectrum")
    ax_spec.legend(loc="upper right", fontsize=8)

    # 4 — rejected peaks on their own
    ax_rej.scatter(rejected[:, 0], rejected[:, 2], s=3, c="crimson", alpha=0.45,
                   linewidths=0, rasterized=True)
    ax_rej.set_xlabel("m/z"); ax_rej.set_ylabel("1/K₀ (V·s/cm²)")
    ax_rej.set_title(f"Rejected as noise (n={len(rejected):,})", color="crimson")

    for ax in (ax_raw, ax_cent, ax_rej):
        ax.set_xlim(MZ_RANGE); ax.set_ylim(OOK0_RANGE)
    ax_cent.set_xlabel("m/z")

    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT} (frame {frame_id})")


if __name__ == "__main__":
    main()
