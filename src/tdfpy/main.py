from typing import Literal
from collections.abc import Generator
from dataclasses import dataclass

from .spectra import Ms1Spectrum, Ms2Spectrum, get_centroided_ms1_spectra
from .timsdata import timsdata_connect

MS1_MSMSTYPE = 0
MS2_MSMSTYPE = 8

@dataclass
class DFolder:
    analysis_tdf_path: str
    ion_mobility_type: Literal["ook0", "ccs"] = "ook0"

    def ms1_frame_ids(self) -> list[int]:
        with timsdata_connect(self.analysis_tdf_path) as td:

            if td.conn is None:
                raise RuntimeError("Database connection is not established.")
            
            cursor = td.conn.cursor()
            cursor.execute(f"SELECT Id FROM Frames WHERE MsMsType = {MS1_MSMSTYPE} ORDER BY Id")
            frame_ids: list[int] = [row[0] for row in cursor.fetchall()]
            return frame_ids
        
    def ms2_frame_ids(self) -> list[int]:
        with timsdata_connect(self.analysis_tdf_path) as td:

            if td.conn is None:
                raise RuntimeError("Database connection is not established.")
            
            cursor = td.conn.cursor()
            cursor.execute(f"SELECT Id FROM Frames WHERE MsMsType = {MS2_MSMSTYPE} ORDER BY Id")
            frame_ids: list[int] = [row[0] for row in cursor.fetchall()]
            return frame_ids
        
    def frame_ids(self) -> list[int]:
        with timsdata_connect(self.analysis_tdf_path) as td:

            if td.conn is None:
                raise RuntimeError("Database connection is not established.")
            
            cursor = td.conn.cursor()
            cursor.execute("SELECT Id FROM Frames ORDER BY Id")
            frame_ids: list[int] = [row[0] for row in cursor.fetchall()]
            return frame_ids

    def ms1_spectra(self, 
                    frame_ids: list[int] | None = None,
                    mz_tolerance: float = 8.0,
                    mz_tolerance_type: Literal["ppm", "da"] = "ppm",
                    im_tolerance: float = 0.05,
                    im_tolerance_type: Literal["relative", "absolute"] = "relative",
                    min_peaks: int = 3,
                    max_peaks: int | None = None,
                    noise_filter: None | (
                            Literal["mad", "percentile", "histogram", "baseline", "iterative_median"] |
                            float
                    ) = None,   
                    use_rust: bool = True,                 
                    ) -> Generator[Ms1Spectrum, None, None]:
        with timsdata_connect(self.analysis_tdf_path) as td:

            if td.conn is None:
                raise RuntimeError("Database connection is not established.")
            
            yield from get_centroided_ms1_spectra(
                td=td,
                frame_ids=frame_ids,
                ion_mobility_type=self.ion_mobility_type,
                mz_tolerance=mz_tolerance,
                mz_tolerance_type=mz_tolerance_type,
                im_tolerance=im_tolerance,
                im_tolerance_type=im_tolerance_type,
                min_peaks=min_peaks,
                max_peaks=max_peaks,
                noise_filter=noise_filter,
                use_rust=use_rust,
            )

    def ms2_spectra(self) -> Generator[Ms2Spectrum, None, None]:
        raise NotImplementedError("MS2 spectra extraction is not yet implemented.")
