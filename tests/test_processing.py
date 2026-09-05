"""Scientific accounting and streaming behavior for analysis helpers."""

from dataclasses import asdict
from itertools import islice
import json
import subprocess
import sys

import numpy as np
import pytest

from tdfpy import (
    DDA,
    DIA,
    PRM,
    AbsoluteThreshold,
    MergePeaksCentroider,
    Smooth,
    get_centroided_spectrum,
    iter_window_spectra,
    read_spectrum,
    validate_acquisition,
)


def test_diagnostics_reconcile_with_pipeline():
    from tdfpy._diagnostics import process_frame

    with DDA("tests/data/example_dda.d") as reader:
        frame = next(iter(reader.ms1))
        noise = AbsoluteThreshold(100)
        cfg = MergePeaksCentroider(min_peaks=1, max_peaks=10)
        result = process_frame(
            reader.timsdata, frame.frame_id, noise=noise, centroid=cfg
        )
        expected = get_centroided_spectrum(
            reader.timsdata, frame.frame_id, noise=noise, centroid=cfg
        )
        np.testing.assert_array_equal(result.peaks, expected)
        original = read_spectrum(reader.timsdata, frame.frame_id)
        assert result.stages[0].num_peaks == len(original)
        assert result.stages[0].intensity_sum == original.intensities.sum()
        filtered = original.intensities[original.intensities >= 100]
        assert result.stages[1].num_peaks == len(filtered)
        assert result.stages[1].intensity_sum == filtered.sum()
        assert result.stages[-1].intensity_sum == result.peaks[:, 1].sum()
        json.dumps(asdict(result.provenance))
        smoothed = process_frame(
            reader.timsdata, frame.frame_id, smooth=Smooth(0, 1), centroid=cfg
        )
        assert smoothed.stages[0].intensity_basis == "original"
        assert all(s.intensity_basis == "smoothed" for s in smoothed.stages[1:])


@pytest.mark.parametrize(
    "reader_cls, mode, attribute",
    [(DIA, "dia", "windows"), (PRM, "prm", "transitions")],
)
def test_batch_matches_windows_and_decodes_each_frame_once(
    reader_cls, mode, attribute, monkeypatch
):
    from tdfpy import processing

    original = processing.read_spectrum
    calls = []

    def record(td, fid):
        calls.append(fid)
        return original(td, fid)

    monkeypatch.setattr(processing, "read_spectrum", record)
    with reader_cls(f"tests/data/example_{mode}.d") as reader:
        windows = list(islice(getattr(reader, attribute), 8))
        config = MergePeaksCentroider(min_peaks=1, max_peaks=10)
        results = list(iter_window_spectra(windows, centroid=config))
        assert calls == list(dict.fromkeys(w.frame_id for w in windows))
        for window, result in zip(windows, results):
            assert result[0] is window
            np.testing.assert_array_equal(result[1], window.centroid(centroid=config))


def test_metadata_and_full_validation(single_frame):
    metadata = validate_acquisition(single_frame)
    assert metadata.valid and metadata.frames_checked == 1
    binary = single_frame / "analysis.tdf_bin"
    binary.write_bytes(binary.read_bytes()[:8])
    result = validate_acquisition(single_frame, full=True)
    assert not result.valid
    assert result.issues[0].frame_id == 1
    assert "truncated" in result.issues[0].message


@pytest.fixture
def single_frame(tmp_path):
    from tdfpy import slice_d_folder

    return slice_d_folder("tests/data/example_dda.d", tmp_path / "one.d", 1, 1)


def test_validation_cli_returns_json_and_failure_status(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "tdfpy", "validate", str(tmp_path / "missing.d")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    data = json.loads(result.stdout)
    assert not data["valid"]
    assert data["issues"]


def test_batch_iterator_rejects_cached_spectral_access_after_close():
    reader = DIA("tests/data/example_dia.d")
    try:
        windows = list(islice(reader.windows, 2))
        assert windows[0].frame_id == windows[1].frame_id
        iterator = iter_window_spectra(
            windows, centroid=MergePeaksCentroider(max_peaks=5)
        )
        window, peaks = next(iterator)
        assert window is windows[0]
        assert peaks.shape[1] == 3
    finally:
        reader.close()
    with pytest.raises(RuntimeError):
        next(iterator)
