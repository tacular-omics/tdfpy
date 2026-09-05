"""Record reproducible reader and processing timings as JSON.

Usage: uv run python scripts/benchmark_reader.py tests/data/example_dia.d
Run in a fresh process with an empty NUMBA_CACHE_DIR to measure compilation.
RSS is the process high-water mark, including imports and all benchmark stages.
"""

import argparse
import hashlib
from importlib.metadata import version
from itertools import islice
import json
from pathlib import Path
import platform
from statistics import median
import sys
from time import perf_counter

import tdfpy


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def measure(fn, repeats: int) -> dict:
    samples = []
    for _ in range(repeats):
        start = perf_counter()
        fn()
        samples.append(perf_counter() - start)
    return {"seconds": samples, "median_seconds": median(samples)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("analysis_directory", type=Path)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-peaks", type=int, default=100)
    args = parser.parse_args()
    if args.frames < 1 or args.repeats < 1:
        parser.error("frames and repeats must be positive")
    mode = tdfpy.get_acquisition_type(args.analysis_directory)
    reader_cls = {"DDA": tdfpy.DDA, "DIA": tdfpy.DIA, "PRM": tdfpy.PRM}.get(mode)
    if reader_cls is None:
        parser.error("An acquisition with supported MS2 metadata is required")

    def opening():
        with reader_cls(args.analysis_directory):
            pass

    result = {
        "package_version": tdfpy.__version__,
        "python": sys.version,
        "platform": platform.platform(),
        "dependencies": {name: version(name) for name in ("numpy", "pandas", "numba")},
        "input_checksums": {
            name: checksum(args.analysis_directory / name)
            for name in ("analysis.tdf", "analysis.tdf_bin")
        },
        "opening": measure(opening, args.repeats),
        "max_peaks": args.max_peaks,
    }
    with reader_cls(args.analysis_directory) as reader:
        frames = list(islice(reader.ms1, args.frames))
        result["frame_ids"] = [f.frame_id for f in frames]
        result["raw_peak_count"] = sum(f.num_peaks for f in frames)
        result["decode"] = measure(
            lambda: [reader.timsdata.read_frame_arrays(f.frame_id) for f in frames],
            args.repeats,
        )
        cfg = tdfpy.MergePeaksCentroider(max_peaks=args.max_peaks)

        def run():
            return [f.centroid(centroid=cfg) for f in frames]

        result["first_centroid_call"] = measure(run, 1)
        result["warm_centroid"] = measure(run, args.repeats)
        result["raw_peaks_per_second"] = (
            result["raw_peak_count"] / result["decode"]["median_seconds"]
        )
        windows = list(islice(reader.windows, 20)) if mode == "DIA" else []
        if windows:
            result["window_count"] = len(windows)
            result["individual_windows"] = measure(
                lambda: [w.centroid(centroid=cfg) for w in windows], args.repeats
            )
            if hasattr(tdfpy, "iter_window_spectra"):
                result["batched_windows"] = measure(
                    lambda: list(tdfpy.iter_window_spectra(windows, centroid=cfg)),
                    args.repeats,
                )
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result["process_peak_rss_bytes"] = rss * (
            1 if sys.platform == "darwin" else 1024
        )
    except ImportError:
        result["process_peak_rss_bytes"] = None
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
