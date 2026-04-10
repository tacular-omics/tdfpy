"""Benchmark centroiding: pure-Python/NumPy vs Numba JIT.

Collects raw peaks from up to N_FRAMES MS1 frames in the example DDA dataset,
times both merge_peaks() backends (N_REPS repetitions, best-of-N), validates
that outputs agree, and prints a per-frame table plus aggregate summary.

Usage:
    python scripts/benchmark_centroiding.py
"""

import sys
import time
from pathlib import Path

import numpy as np

import tdfpy
from tdfpy import DDA
from tdfpy.centroiding import (
    _HAS_NUMBA,
    _merge_peaks_numba,
    _merge_peaks_python,
    get_raw_peaks,
    merge_peaks,
)

REPO_ROOT = Path(__file__).parent.parent
DATA_PATH = REPO_ROOT / "tests" / "data" / "example_dda.d"

N_FRAMES = 10
N_REPS = 5
PARAMS = {
    "mz_tolerance": 8.0,
    "mz_tolerance_type": "ppm",
    "im_tolerance": 0.1,
    "im_tolerance_type": "relative",
    "min_peaks": 3,
    "max_peaks": None,
}


def _warm_up_numba() -> None:
    print("Warming up Numba JIT (first-call compilation)...")
    dummy_mz = np.array([100.0, 100.0001, 200.0], dtype=np.float64)
    dummy_int = np.array([1000.0, 900.0, 500.0], dtype=np.float64)
    dummy_im = np.array([1.0, 1.0001, 1.5], dtype=np.float64)
    _merge_peaks_numba(dummy_mz, dummy_int, dummy_im, **PARAMS)
    print("  Done.\n")


def _time_fn(fn, *args, **kwargs) -> tuple[float, object]:
    """Return (best_time_seconds, result) over N_REPS calls."""
    best = float("inf")
    result = None
    for _ in range(N_REPS):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        best = min(best, time.perf_counter() - t0)
    return best, result


def main() -> None:
    if not DATA_PATH.exists():
        print(f"ERROR: data not found at {DATA_PATH}")
        print("Tests require real Bruker .d data — skipping benchmark.")
        sys.exit(1)

    if not _HAS_NUMBA:
        print("WARNING: Numba not installed. Only the Python path is available.")
        sys.exit(1)

    try:
        import numba
        print(f"numba {numba.__version__}")
    except Exception:
        pass

    # -- collect frame IDs --------------------------------------------------
    frame_ids: list[int] = []
    with DDA(DATA_PATH) as dda:
        for frame in dda.ms1:
            frame_ids.append(frame.frame_id)
            if len(frame_ids) >= N_FRAMES:
                break

    print(f"Collected {len(frame_ids)} MS1 frame(s) from {DATA_PATH.name}\n")

    # -- warm up Numba BEFORE timing ----------------------------------------
    _warm_up_numba()

    # -- per-frame benchmark ------------------------------------------------
    col_w = (8, 12, 14, 14, 10)  # column widths
    header = (
        f"{'frame_id':>{col_w[0]}} "
        f"{'raw_peaks':>{col_w[1]}} "
        f"{'python_ms':>{col_w[2]}} "
        f"{'numba_ms':>{col_w[3]}} "
        f"{'speedup':>{col_w[4]}}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    py_times: list[float] = []
    nb_times: list[float] = []
    validated = False

    with tdfpy.timsdata_connect(DATA_PATH) as td:
        for frame_id in frame_ids:
            raw = get_raw_peaks(td, frame_id)
            if len(raw) == 0:
                continue

            mz = raw[:, 0]
            intensity = raw[:, 1]
            im = raw[:, 2]

            py_t, py_result = _time_fn(_merge_peaks_python, mz, intensity, im, **PARAMS)
            nb_t, nb_result = _time_fn(_merge_peaks_numba, mz, intensity, im, **PARAMS)

            # validate once
            if not validated and py_result is not None and nb_result is not None:
                if np.allclose(
                    py_result[np.argsort(py_result[:, 0])],
                    nb_result[np.argsort(nb_result[:, 0])],
                    rtol=1e-5, atol=1e-8,
                ):
                    print(f"  [OK] Python and Numba outputs match on frame {frame_id}\n")
                else:
                    print(f"  [WARN] Outputs differ on frame {frame_id}\n")
                validated = True

            speedup = py_t / nb_t if nb_t > 0 else float("nan")
            py_times.append(py_t)
            nb_times.append(nb_t)

            print(
                f"{frame_id:>{col_w[0]}} "
                f"{len(raw):>{col_w[1]},} "
                f"{py_t * 1000:>{col_w[2]}.3f} "
                f"{nb_t * 1000:>{col_w[3]}.3f} "
                f"{speedup:>{col_w[4]}.2f}x"
            )

    if not py_times:
        print("No frames with peaks found.")
        sys.exit(1)

    # -- aggregate summary --------------------------------------------------
    mean_py = np.mean(py_times) * 1000
    mean_nb = np.mean(nb_times) * 1000
    mean_speedup = np.mean(py_times) / np.mean(nb_times)

    print(sep)
    print(
        f"\nMean over {len(py_times)} frame(s): "
        f"Python {mean_py:.3f} ms | Numba {mean_nb:.3f} ms | "
        f"Speedup {mean_speedup:.2f}x"
    )


if __name__ == "__main__":
    main()
