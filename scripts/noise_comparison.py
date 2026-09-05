import csv
import sys
from pathlib import Path

import tdfpy
from tdfpy import DDA
from tdfpy.centroiding import get_raw_peaks, merge_peaks
from tdfpy.noise import estimate_noise_level
from tdfpy.viz import plot_centroiding

REPO_ROOT = Path(__file__).parent.parent
DATA_PATH = REPO_ROOT / "tests" / "data" / "example_dda.d"
PLOTS_DIR = REPO_ROOT / "plots"

METHODS = [None, "mad", "percentile", "histogram", "baseline", "iterative_median"]


def main():
    PLOTS_DIR.mkdir(exist_ok=True)

    frame_id = None
    with DDA(DATA_PATH) as dda:
        first_ms1 = next(iter(dda.ms1))
        frame_id = first_ms1.frame_id
        print(f"First MS1 frame ID: {frame_id}")

    if frame_id is None:
        print("No MS1 frames found.")
        sys.exit(1)

    stats_rows = []

    with tdfpy.timsdata_connect(DATA_PATH) as td:
        raw = get_raw_peaks(td, frame_id)
        raw_count = len(raw)
        raw_intensity = float(raw[:, 1].sum()) if raw_count > 0 else 0.0

        all_centroids = merge_peaks(
            mz_array=raw[:, 0],
            intensity_array=raw[:, 1],
            ion_mobility_array=raw[:, 2],
            im_tolerance=0.1,
        )

        for method in METHODS:
            label = str(method) if method is not None else "none"

            # stats
            if method is not None and len(all_centroids) > 0:
                threshold = estimate_noise_level(all_centroids[:, 1], method=method)
                kept_mask = all_centroids[:, 1] >= threshold
                kept = all_centroids[kept_mask]
                rejected = all_centroids[~kept_mask]
            else:
                kept = all_centroids
                rejected = all_centroids[:0]  # empty

            kept_count = len(kept)
            rejected_count = len(rejected)
            kept_intensity = float(kept[:, 1].sum()) if kept_count > 0 else 0.0
            rejected_intensity = (
                float(rejected[:, 1].sum()) if rejected_count > 0 else 0.0
            )

            stats_rows.append(
                {
                    "method": label,
                    "raw_peaks": raw_count,
                    "centroided_peaks": len(all_centroids),
                    "kept_peaks": kept_count,
                    "rejected_peaks": rejected_count,
                    "raw_intensity": raw_intensity,
                    "kept_intensity": kept_intensity,
                    "rejected_intensity": rejected_intensity,
                    "pct_intensity_kept": round(
                        100.0 * kept_intensity / raw_intensity, 2
                    )
                    if raw_intensity > 0
                    else 0.0,
                    "pct_intensity_rejected": round(
                        100.0 * rejected_intensity / raw_intensity, 2
                    )
                    if raw_intensity > 0
                    else 0.0,
                }
            )

            # plot
            out = PLOTS_DIR / f"noise_{label}.png"
            print(f"Generating {out} ...")
            fig = plot_centroiding(
                td,
                frame_id=frame_id,
                im_tolerance=0.1,
                noise_filter=method,
                mz_range=(400, 1200),
            )
            fig.savefig(out, dpi=150)
            print(f"  Saved {out}")

    stats_path = PLOTS_DIR / "noise_stats.csv"
    fieldnames = list(stats_rows[0].keys())
    with open(stats_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(stats_rows)
    print(f"Stats written to {stats_path}")


if __name__ == "__main__":
    main()
