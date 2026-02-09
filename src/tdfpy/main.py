from pathlib import Path
from typing import Literal
from collections.abc import Generator
from dataclasses import dataclass
import pandas as pd
from .spectra import get_centroided_ms1_spectra
from .timsdata import timsdata_connect
from .pandas_tdf import PandasTdf

MS1_MSMSTYPE = 0
MS2_MSMSTYPE = 8
MS1_FRAME_QUERY = f"SELECT Id FROM Frames WHERE MsMsType = {MS1_MSMSTYPE} ORDER BY Id"
MS2_FRAME_QUERY = f"SELECT Id FROM Frames WHERE MsMsType = {MS2_MSMSTYPE} ORDER BY Id"
ALL_FRAME_QUERY = "SELECT Id FROM Frames ORDER BY Id"

@dataclass
class DFolder:
    analysis_dir: str

    @property
    def analysis_tdf_path(self) -> Path:
        return Path(self.analysis_dir) / "analysis.tdf"

    @property
    def analysis_tdf_bin_path(self) -> Path:
        return Path(self.analysis_dir) / "analysis.tdf_bin"

    @property
    def analysis_path(self) -> Path:
        return Path(self.analysis_dir)

    def _get_frame_ids(self, ms_level: int | None) -> list[int]:
        query = {
            1: MS1_FRAME_QUERY,
            2: MS2_FRAME_QUERY,
            None: ALL_FRAME_QUERY
        }.get(ms_level)

        if query is None:
            raise ValueError(f"Invalid ms_level: {ms_level}. Must be 1, 2, or None.")

        with timsdata_connect(str(self.analysis_tdf_path)) as td:
            if td.conn is None:
                raise RuntimeError("Database connection is not established.")

            cursor = td.conn.cursor()
            cursor.execute(query)
            frame_ids: list[int] = [row[0] for row in cursor.fetchall()]
            return frame_ids


    @property
    def ms1_frame_ids(self) -> list[int]:
        return self._get_frame_ids(ms_level=1)

    @property
    def ms2_frame_ids(self) -> list[int]:
        return self._get_frame_ids(ms_level=2)

    @property
    def frame_ids(self) -> list[int]:
        return self._get_frame_ids(ms_level=None)

    def get_dda_df(
        self,
        ) -> pd.DataFrame:

        pd_tdf = PandasTdf(str(self.analysis_tdf_path))

        precursors_df = pd_tdf.precursors

        merged_df = pd.merge(
            precursors_df,
            pd_tdf.frames,
            left_on="Parent",
            right_on="Id",
            suffixes=("_Precursor", "_Frame"),
        )

        pasef_frame_msms_info_df = pd_tdf.pasef_frame_msms_info.drop(["Frame"], axis=1)

        # count the number of items in each group
        pasef_frame_msms_info_df["count"] = pasef_frame_msms_info_df.groupby("Precursor")[
            "Precursor"
        ].transform("count")

        # keep only the row for each group
        pasef_frame_msms_info_df = pasef_frame_msms_info_df.drop_duplicates(
            subset="Precursor", keep="first"
        )
        assert len(pasef_frame_msms_info_df) == len(merged_df)

        merged_df = pd.merge(
            merged_df,
            pasef_frame_msms_info_df,
            left_on="Id_Precursor",
            right_on="Precursor",
            suffixes=("_Precursor", "_PasefFrameMsmsInfo"),
        ).drop("Precursor", axis=1)


        return merged_df