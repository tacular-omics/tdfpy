"""Capture calibration reference values from Bruker's native ``timsdata`` library.

The output (``tests/data/calibration_golden.json``) pins the index/mz, scan/mobility,
scan/voltage and CCS conversions so they can be regression-tested *after* the native
library is removed. Run this only while ``libtimsdata.so`` / ``timsdata.dll`` is still
present; the committed JSON is the durable artifact.

    uv run python scripts/generate_calibration_golden.py
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

from tdfpy.timsdata import TimsData, oneOverK0ToCCSforMz

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "tests" / "data" / "calibration_golden.json"
FIXTURES = ["example_dda.d", "example_dia.d", "example_prm.d"]

# CCS probe grid — (1/K0, charge, m/z) triples spanning the useful range.
CCS_PROBES = [
    (0.60, 1, 200.0),
    (0.90, 1, 622.0),
    (1.25, 1, 1200.0),
    (0.75, 2, 400.0),
    (1.10, 2, 850.0),
    (1.60, 2, 1500.0),
    (0.95, 3, 700.0),
    (1.40, 3, 1100.0),
    (2.00, 4, 1900.0),
]


def pick_frames(conn: sqlite3.Connection) -> list[int]:
    """A small frame set covering the calibration variation that matters.

    Frames differ in which ``MzCalibration`` / ``TimsCalibration`` row they point
    at, and in their ``T1`` / ``T2`` temperatures, which drive the ``dC1`` / ``dC2``
    compensation. Both extremes of each temperature are included so that path is
    genuinely exercised rather than evaluated at a single point.
    """
    rows = conn.execute(
        "SELECT Id, MsMsType, MzCalibration, TimsCalibration, T1, T2 FROM Frames ORDER BY Id"
    ).fetchall()
    chosen: list[int] = [rows[0][0], rows[-1][0]]

    seen: set[tuple[int, int, int]] = set()
    for fid, msms, mzcal, tcal, _t1, _t2 in rows:
        key = (msms, mzcal, tcal)
        if key not in seen:
            seen.add(key)
            chosen.append(fid)

    for col in (4, 5):  # T1, T2
        chosen.append(min(rows, key=lambda r: r[col])[0])
        chosen.append(max(rows, key=lambda r: r[col])[0])

    return sorted(set(chosen))


def main() -> None:
    golden: dict[str, object] = {
        "_comment": (
            "Reference values captured from Bruker's native timsdata library. "
            "Regenerate with scripts/generate_calibration_golden.py (requires the "
            "native library). See tests/test_calibration_golden.py for tolerances."
        ),
        "fixtures": {},
    }

    for name in FIXTURES:
        d = REPO / "tests" / "data" / name
        if not d.is_dir():
            raise SystemExit(f"missing fixture: {d}")
        conn = sqlite3.connect(d / "analysis.tdf")
        meta = dict(conn.execute("SELECT Key, Value FROM GlobalMetadata").fetchall())
        n_samples = int(meta["DigitizerNumSamples"])
        frames = pick_frames(conn)

        entries = []
        with TimsData(str(d)) as td:
            for fid in frames:
                num_scans = int(
                    conn.execute(
                        "SELECT NumScans FROM Frames WHERE Id = ?", (fid,)
                    ).fetchone()[0]
                )
                # Endpoints plus an irregular interior spread; avoids only sampling
                # a grid the model might happen to be exact on.
                idx = np.unique(
                    np.concatenate(
                        [
                            [0, 1, 2, n_samples - 1, n_samples],
                            np.linspace(0, n_samples, 29)[1:-1],
                            [n_samples * 0.137, n_samples * 0.618, n_samples * 0.977],
                        ]
                    ).round()
                )
                mz = td.indexToMz(fid, idx)

                scans = np.unique(
                    np.concatenate(
                        [
                            [0, 1, 2, num_scans - 1, num_scans],
                            np.linspace(0, num_scans, 23)[1:-1],
                            [num_scans * 0.137, num_scans * 0.618],
                        ]
                    ).round()
                )
                ook0 = td.scanNumToOneOverK0(fid, scans)

                entries.append(
                    {
                        "frame_id": fid,
                        "num_scans": num_scans,
                        "tof_indices": idx.tolist(),
                        "mz": mz.tolist(),
                        # Round-trip: feed the m/z back to get the index again.
                        "mz_to_index": td.mzToIndex(fid, mz).tolist(),
                        "scans": scans.tolist(),
                        "one_over_k0": ook0.tolist(),
                        "one_over_k0_to_scan": td.oneOverK0ToScanNum(
                            fid, ook0
                        ).tolist(),
                        "voltage": td.scanNumToVoltage(fid, scans).tolist(),
                    }
                )

        golden["fixtures"][name] = {  # type: ignore[index]
            "digitizer_num_samples": n_samples,
            "frames": entries,
        }
        conn.close()
        print(f"{name}: {len(entries)} frames captured")

    golden["ccs"] = [
        {"one_over_k0": k, "charge": z, "mz": m, "ccs": oneOverK0ToCCSforMz(k, z, m)}
        for k, z, m in CCS_PROBES
    ]

    OUT.write_text(json.dumps(golden, indent=1) + "\n")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
