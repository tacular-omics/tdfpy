"""
This module contains PandasTdf which is a utility class to easily retrieve table information from the analysis.tdf
file in the format of pandas dataframes
"""

import sqlite3
import logging
from dataclasses import dataclass

import pandas as pd

from .constants import TableNames

logger = logging.getLogger(__name__)


def convert_table_to_df(db_path: str, table_name: str) -> pd.DataFrame:
    """
    Converts a table in an SQLite database to a pandas DataFrame.

    Args:
        db_path (str): The path to the SQLite database.
        table_name (str): The name of the table to convert.

    Returns:
        pd.DataFrame: The converted table as a pandas DataFrame.
    """
    logger.debug("Fetching " + table_name + " from " + db_path)
    try:
        with sqlite3.connect(str(db_path)) as conn:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)  # type: ignore[arg-type]
            return df
    except Exception as e:
        logger.error(f"Error fetching table {table_name} from {db_path}: {e}")
        raise


@dataclass
class PandasTdf:
    """
    A class for working with TDF (Bruker Data File) using pandas DataFrames.
    """

    db_path: str

    @property
    def calibration_info(self) -> pd.DataFrame:
        """
        The 'CALIBRATION_INFO' table as a pandas DataFrame.
        :return: table as a pandas DataFrame
        """
        return convert_table_to_df(self.db_path, TableNames.CALIBRATION_INFO)

    @property
    def dia_frame_msms_info(self) -> pd.DataFrame:
        """
        The 'DIA_FRAME_MSMS_INFO' table as a pandas DataFrame.
        :return: table as a pandas DataFrame
        """
        return convert_table_to_df(self.db_path, TableNames.DIA_FRAME_MSMS_INFO)

    @property
    def dia_frame_msms_window_groups(self) -> pd.DataFrame:
        """
        The 'DIA_FRAME_MSMS_WINDOW_GROUPS' table as a pandas DataFrame.
        :return: table as a pandas DataFrame
        """
        return convert_table_to_df(
            self.db_path, TableNames.DIA_FRAME_MSMS_WINDOW_GROUPS
        )

    @property
    def dia_frame_msms_windows(self) -> pd.DataFrame:
        """
        The 'DIA_FRAME_MSMS_WINDOWS' table as a pandas DataFrame.
        :return: table as a pandas DataFrame
        """
        return convert_table_to_df(self.db_path, TableNames.DIA_FRAME_MSMS_WINDOWS)

    @property
    def error_log(self) -> pd.DataFrame:
        """
        The 'ERROR_LOG' table as a pandas DataFrame.
        :return: table as a pandas DataFrame
        """
        return convert_table_to_df(self.db_path, TableNames.ERROR_LOG)

    @property
    def frame_msms_info(self) -> pd.DataFrame:
        """
        The 'FRAME_MSMS_WINDOW' table as a pandas DataFrame.
        :return: table as a pandas DataFrame
        """
        return convert_table_to_df(self.db_path, TableNames.FRAME_MSMS_WINDOW)

    @property
    def frame_properties(self) -> pd.DataFrame:
        """
        The 'FRAME_PROPERTIES' table as a pandas DataFrame.
        :return: table as a pandas DataFrame
        """
        return convert_table_to_df(self.db_path, TableNames.FRAME_PROPERTIES)

    @property
    def frames(self) -> pd.DataFrame:
        """
        The 'FRAMES' table as a pandas DataFrame.
        :return: table as a pandas DataFrame
        """
        return convert_table_to_df(self.db_path, TableNames.FRAMES)

    @property
    def global_metadata(self) -> pd.DataFrame:
        """
        The 'GLOBAL_METADATA' table as a pandas DataFrame.
        :return: table as a pandas DataFrame
        """
        return convert_table_to_df(self.db_path, TableNames.GLOBAL_METADATA)

    @property
    def group_properties(self) -> pd.DataFrame:
        """
        The 'GROUP_PROPERTIES' table as a pandas DataFrame.
        :return: table as a pandas DataFrame
        """
        return convert_table_to_df(self.db_path, TableNames.GROUP_PROPERTIES)

    @property
    def mz_calibration(self) -> pd.DataFrame:
        """
        The 'MZ_CALIBRATION' table as a pandas DataFrame.
        :return: table as a pandas DataFrame
        """
        return convert_table_to_df(self.db_path, TableNames.MZ_CALIBRATION)

    @property
    def pasef_frame_msms_info(self) -> pd.DataFrame:
        """
        The 'PASEF_FRAMES_MSMS_INFO' table as a pandas DataFrame.
        :return: table as a pandas DataFrame
        """
        return convert_table_to_df(self.db_path, TableNames.PASEF_FRAME_MSMS_INFO)

    @property
    def precursors(self) -> pd.DataFrame:
        """
        The 'PRECURSORS' table as a pandas DataFrame.
        :return: table as a pandas DataFrame
        """
        return convert_table_to_df(self.db_path, TableNames.PRECURSORS)

    @property
    def properties(self) -> pd.DataFrame:
        """
        The 'PROPERTIES' table as a pandas DataFrame.
        :return: table as a pandas DataFrame
        """
        return convert_table_to_df(self.db_path, TableNames.PROPERTIES)

    @property
    def property_definitions(self) -> pd.DataFrame:
        """
        The 'PROPERTY_DEFINITIONS' table as a pandas DataFrame.
        :return: table as a pandas DataFrame
        """
        return convert_table_to_df(self.db_path, TableNames.PROPERTY_DEFINITIONS)

    @property
    def property_groups(self) -> pd.DataFrame:
        """
        The 'PROPERTY_GROUPS' table as a pandas DataFrame.
        :return: table as a pandas DataFrame
        """
        return convert_table_to_df(self.db_path, TableNames.PROPERTY_GROUPS)

    @property
    def segments(self) -> pd.DataFrame:
        """
        The 'SEGMENTS' table as a pandas DataFrame.
        :return: table as a pandas DataFrame
        """
        return convert_table_to_df(self.db_path, TableNames.SEGMENTS)

    @property
    def tims_calibration(self) -> pd.DataFrame:
        """
        The 'TIMS_CALIBRATION' table as a pandas DataFrame.
        :return: table as a pandas DataFrame
        """
        return convert_table_to_df(self.db_path, TableNames.TIMS_CALIBRATION)

    @property
    def prm_frame_measurement_mode(self) -> pd.DataFrame:
        """
        The 'PRM_FRAME_MEASUREMENT_MODE' table as a pandas DataFrame.
        :return: table as a pandas DataFrame
        """
        return convert_table_to_df(self.db_path, TableNames.PRM_FRAME_MEASUREMENT_MODE)

    @property
    def prm_frame_msms_info(self) -> pd.DataFrame:
        """
        The 'PRM_FRAME_MSMS_INFO' table as a pandas DataFrame.
        :return: table as a pandas DataFrame
        """
        return convert_table_to_df(self.db_path, TableNames.PRM_FRAME_MSMS_INFO)

    @property
    def prm_targets(self) -> pd.DataFrame:
        """
        The 'PRM_TARGETS' table as a pandas DataFrame.
        :return: table as a pandas DataFrame
        """
        return convert_table_to_df(self.db_path, TableNames.PRM_TARGETS)

    def get_table_names(self) -> list[str]:
        """
        Retrieves the names of all tables in the SQLite database.

        Returns:
            list[str]: A list of table names in the database.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            table_names = [table[0] for table in cursor.fetchall()]
        return table_names

    @property
    def is_dda(self) -> bool:
        """
        Checks if the database contains DDA (Data-Dependent Acquisition) data.

        Returns:
            bool: True if DDA data is present, False otherwise.
        """
        return (
            TableNames.PRECURSORS.value in self.get_table_names()
            and len(self.precursors) > 0
        )

    @property
    def is_prm(self) -> bool:
        """
        Checks if the database contains PRM (Parallel Reaction Monitoring) data.

        Returns:
            bool: True if PRM data is present, False otherwise.
        """
        return (
            TableNames.PRM_FRAME_MSMS_INFO.value in self.get_table_names()
            and len(self.prm_frame_msms_info) > 0
        )
