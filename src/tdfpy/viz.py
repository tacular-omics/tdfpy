"""
Visualization utilities for inspecting centroiding quality.

Requires matplotlib (not installed by default):
    pip install matplotlib
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from .noise import NoiseSpec
    from .timsdata import TimsData



def plot_centroiding(
    td: TimsData,
    frame_id: int,
    ion_mobility_type: Literal["ook0", "ccs", "voltage"] = "ook0",
    mz_tolerance: float = 8.0,
    mz_tolerance_type: Literal["ppm", "da"] = "ppm",
    im_tolerance: float = 0.01,
    im_tolerance_type: Literal["relative", "absolute"] = "relative",
    min_peaks: int = 3,
    max_peaks: int | None = None,
    noise: "NoiseSpec" = None,
    mz_range: tuple[float, float] | None = None,
    im_range: tuple[float, float] | None = None,
) -> Figure:
    """Visualize centroiding quality for a single frame.

    Produces a 2×2 figure:

    - **Top-left** — 2D ion map of raw peaks (m/z vs ion mobility), coloured
      by log-intensity.
    - **Top-right** — centroided peaks (stars, sized by intensity) overlaid on
      the faded raw cloud.
    - **Bottom-left** — peaks that were *discarded* by centroiding (raw points
      that fall outside the tolerance of every centroid), coloured by
      log-intensity.  These are typically low-intensity singletons rejected by
      ``min_peaks``.
    - **Bottom-right** — 1D mass spectrum: raw summed projection vs centroided
      stems, plus a discarded-intensity fill so you can judge how much signal
      was lost.

    A stats box reports raw / centroided / discarded counts and the fraction of
    total intensity retained.

    Args:
        td: Open TimsData instance.
        frame_id: Frame to inspect.
        ion_mobility_type: Ion mobility axis — ``"ook0"``, ``"ccs"``, or
            ``"voltage"``.
        mz_tolerance: m/z tolerance used for centroiding.
        mz_tolerance_type: ``"ppm"`` or ``"da"``.
        im_tolerance: Ion mobility tolerance used for centroiding.
        im_tolerance_type: ``"relative"`` or ``"absolute"``.
        min_peaks: Minimum raw peaks required to form a centroid.
        max_peaks: Maximum centroids to return (``None`` = unlimited).
        noise: Pre-centroiding noise filter pipeline — see
            :func:`tdfpy.noise.coerce_filters` for accepted forms.
        mz_range: Optional ``(min_mz, max_mz)`` to restrict the plot axes.
        im_range: Optional ``(min_im, max_im)`` to restrict the plot axes.

    Returns:
        matplotlib Figure.  Call ``fig.savefig(...)`` or ``plt.show()`` as
        needed.

    Raises:
        ImportError: If matplotlib is not installed.

    Example:
        ```python
        import tdfpy
        from tdfpy.viz import plot_centroiding

        with tdfpy.timsdata_connect("experiment.d") as td:
            fig = plot_centroiding(td, frame_id=100, noise="mad")
            fig.savefig("centroiding_check.png", dpi=150)
        ```
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for visualization: pip install matplotlib"
        ) from exc

    from .centroiding import merge_peaks
    from .noise import coerce_filters
    from .pipeline import apply_noise, convert, read_spectrum

    # --- collect data ----------------------------------------------------------
    # Read raw peaks once. If a noise filter is given, apply it pre-centroid
    # and record which raw peaks it rejected so the third panel can show them.
    spectrum = read_spectrum(td, frame_id)
    if len(spectrum) == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"Frame {frame_id}: no peaks found", ha="center", va="center")
        return fig

    raw_all = convert(spectrum, td, frame_id, ion_mobility_type=ion_mobility_type)

    filters = coerce_filters(noise)
    if filters:
        kept_spectrum = apply_noise(spectrum, filters, td=td, frame_id=frame_id)
    else:
        kept_spectrum = spectrum

    raw = convert(kept_spectrum, td, frame_id, ion_mobility_type=ion_mobility_type)

    # The "rejected" raw peaks are the set-difference between raw_all and raw.
    # Since both come from the same RawSpectrum in the same order, the kept
    # rows are a strict subset — compute the rejected rows via a row-mask.
    if filters and len(raw) < len(raw_all):
        kept_keys = set(zip(kept_spectrum.scan_indices.tolist(), kept_spectrum.mz_indices.tolist()))
        rejected_mask = np.array(
            [(int(s), int(m)) not in kept_keys for s, m in zip(spectrum.scan_indices, spectrum.mz_indices)],
            dtype=bool,
        )
        rejected_raw = raw_all[rejected_mask]
    else:
        rejected_raw = np.empty((0, 3), dtype=np.float64)

    mz_raw = raw[:, 0]
    int_raw = raw[:, 1]
    im_raw = raw[:, 2]

    # Centroid the (possibly filtered) raw peaks
    centroided = merge_peaks(
        mz_array=mz_raw,
        intensity_array=int_raw,
        ion_mobility_array=im_raw,
        mz_tolerance=mz_tolerance,
        mz_tolerance_type=mz_tolerance_type,
        im_tolerance=im_tolerance,
        im_tolerance_type=im_tolerance_type,
        min_peaks=min_peaks,
        max_peaks=max_peaks,
    )

    mz_c = centroided[:, 0]
    int_c = centroided[:, 1]
    im_c = centroided[:, 2]

    mz_nr = rejected_raw[:, 0] if rejected_raw.size else np.empty(0)
    int_nr = rejected_raw[:, 1] if rejected_raw.size else np.empty(0)
    im_nr = rejected_raw[:, 2] if rejected_raw.size else np.empty(0)

    # --- stats ----------------------------------------------------------------
    total_raw_int = float(int_raw.sum())
    total_c_int = float(int_c.sum())
    total_nr_int = float(int_nr.sum())
    retention_pct = 100.0 * total_c_int / total_raw_int if total_raw_int > 0 else 0.0
    lost_pct = 100.0 * total_nr_int / total_raw_int if total_raw_int > 0 else 0.0

    # --- optional axis clipping -----------------------------------------------
    def _mask(mz: np.ndarray, im: np.ndarray) -> np.ndarray:
        m = np.ones(len(mz), dtype=bool)
        if mz_range is not None:
            m &= (mz >= mz_range[0]) & (mz <= mz_range[1])
        if im_range is not None:
            m &= (im >= im_range[0]) & (im <= im_range[1])
        return m

    mz_raw_p, int_raw_p, im_raw_p = mz_raw[_mask(mz_raw, im_raw)], int_raw[_mask(mz_raw, im_raw)], im_raw[_mask(mz_raw, im_raw)]
    mz_c_p, int_c_p, im_c_p = mz_c[_mask(mz_c, im_c)], int_c[_mask(mz_c, im_c)], im_c[_mask(mz_c, im_c)]
    mz_nr_p, int_nr_p, im_nr_p = (mz_nr[_mask(mz_nr, im_nr)], int_nr[_mask(mz_nr, im_nr)], im_nr[_mask(mz_nr, im_nr)]) if len(mz_nr) > 0 else (mz_nr, int_nr, im_nr)

    # --- layout ---------------------------------------------------------------
    im_label = {"ook0": "1/K₀ (V·s/cm²)", "ccs": "CCS (Å²)", "voltage": "Voltage (V)"}[
        ion_mobility_type
    ]

    fig = plt.figure(figsize=(15, 11))
    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.3)
    ax_raw = fig.add_subplot(gs[0, 0])
    ax_cent = fig.add_subplot(gs[0, 1], sharex=ax_raw, sharey=ax_raw)
    ax_lost = fig.add_subplot(gs[1, 0], sharex=ax_raw, sharey=ax_raw)
    ax_spec = fig.add_subplot(gs[1, 1])

    log_int_raw = np.log1p(int_raw_p)
    log_int_c = np.log1p(int_c_p)

    # Panel 1 — raw 2D ion map
    sc_raw = ax_raw.scatter(
        mz_raw_p, im_raw_p,
        c=log_int_raw, s=1, cmap="viridis", alpha=0.4, rasterized=True,
    )
    cb = fig.colorbar(sc_raw, ax=ax_raw, pad=0.02)
    cb.set_label("log(intensity + 1)", fontsize=8)
    ax_raw.set_xlabel("m/z")
    ax_raw.set_ylabel(im_label)
    ax_raw.set_title(f"Raw  (n={len(mz_raw_p):,})")

    # Panel 2 — centroided overlaid on faded raw
    ax_cent.scatter(
        mz_raw_p, im_raw_p,
        c=log_int_raw, s=1, cmap="viridis", alpha=0.15, rasterized=True,
    )
    if len(mz_c_p) > 0:
        s_min, s_max = 20, 200
        if int_c_p.max() > int_c_p.min():
            s_c = s_min + (s_max - s_min) * (int_c_p - int_c_p.min()) / (int_c_p.max() - int_c_p.min())
        else:
            s_c = np.full(len(int_c_p), (s_min + s_max) / 2)
        sc_cent = ax_cent.scatter(
            mz_c_p, im_c_p,
            c=log_int_c, s=s_c, cmap="plasma",
            marker="*", edgecolors="white", linewidths=0.3, zorder=3,
        )
        cb2 = fig.colorbar(sc_cent, ax=ax_cent, pad=0.02)
        cb2.set_label("log(intensity + 1)", fontsize=8)
    ax_cent.set_xlabel("m/z")
    ax_cent.set_title(f"Centroided  (n={len(mz_c_p):,})")

    # Panel 3 — noise-rejected raw peaks (filters operate pre-centroid now)
    nr_title = (
        f"Noise-rejected raw peaks  (n={len(mz_nr_p):,},  {lost_pct:.1f}% of intensity)"
        if filters
        else "Noise-rejected raw peaks  (no `noise` set)"
    )
    if len(mz_nr_p) > 0:
        ax_lost.scatter(
            mz_nr_p, im_nr_p,
            s=6, c="crimson", alpha=0.5, linewidths=0, rasterized=True,
        )
    else:
        ax_lost.text(0.5, 0.5, "No noise-rejected raw peaks" if filters else "Set `noise=` to see\nrejected peaks",
                     ha="center", va="center", transform=ax_lost.transAxes, fontsize=10, color="grey")
    ax_lost.set_xlabel("m/z")
    ax_lost.set_ylabel(im_label)
    ax_lost.set_title(nr_title, color="firebrick")

    # Panel 4 — 1D mass spectrum: raw vs kept centroids vs noise-rejected centroids
    if len(mz_raw_p) > 0:
        mz_lo, mz_hi = mz_raw_p.min(), mz_raw_p.max()
        n_bins = min(2000, max(200, int((mz_hi - mz_lo) / 0.05)))
        bin_edges = np.linspace(mz_lo, mz_hi, n_bins + 1)
        bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        hist_raw, _ = np.histogram(mz_raw_p, bins=bin_edges, weights=int_raw_p)
        ax_spec.fill_between(bin_centres, hist_raw, alpha=0.25, color="steelblue", label="Raw (summed)")
        ax_spec.plot(bin_centres, hist_raw, lw=0.4, color="steelblue", alpha=0.5)

        if len(mz_nr_p) > 0:
            markerline_nr, stemlines_nr, _ = ax_spec.stem(
                mz_nr_p, int_nr_p, linefmt="firebrick", markerfmt="x", basefmt=" ",
            )
            markerline_nr.set_markersize(4)
            markerline_nr.set_color("firebrick")
            plt.setp(stemlines_nr, linewidth=0.6, alpha=0.6)
            ax_spec.plot([], [], color="firebrick", marker="x", markersize=4, label="Noise-rejected")

    if len(mz_c_p) > 0:
        markerline, stemlines, _ = ax_spec.stem(
            mz_c_p, int_c_p, linefmt="tomato", markerfmt="D", basefmt=" ",
        )
        markerline.set_markersize(3)
        markerline.set_color("tomato")
        plt.setp(stemlines, linewidth=0.8, alpha=0.85)
        ax_spec.plot([], [], color="tomato", marker="D", markersize=3, label="Centroided (kept)")

    ax_spec.set_xlabel("m/z")
    ax_spec.set_ylabel("Intensity")
    ax_spec.set_title("1D spectrum — raw / kept / noise-rejected")
    ax_spec.legend(fontsize=8)
    if mz_range is not None:
        ax_spec.set_xlim(mz_range)

    # --- stats box ------------------------------------------------------------
    stats = (
        f"Frame {frame_id}  |  "
        f"raw={len(mz_raw):,}  centroided={len(mz_c):,}  noise-rejected={len(rejected_raw):,}\n"
        f"Intensity retained: {retention_pct:.1f}%   noise-rejected: {lost_pct:.1f}%\n"
        f"mz_tol={mz_tolerance} {mz_tolerance_type}   "
        f"im_tol={im_tolerance} {im_tolerance_type}   "
        f"min_peaks={min_peaks}   noise={noise}"
    )
    fig.text(
        0.5, 0.97, stats,
        ha="center", va="top", fontsize=8.5, family="monospace",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#f5f5f5", "edgecolor": "#cccccc"},
    )

    return fig
