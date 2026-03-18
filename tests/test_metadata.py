import datetime
import pathlib

import pytest

from tdfpy import DDA

D_PATH = "tests/data/200ngHeLaPASEF_1min.d"


@pytest.fixture(scope="module")
def dda():
    if not pathlib.Path(D_PATH).exists():
        pytest.skip("Test data not found")
    with DDA(D_PATH) as d:
        yield d


# ---------------------------------------------------------------------------
# MetaData — type tests
# ---------------------------------------------------------------------------


def test_metadata_schema_type_is_str(dda) -> None:
    assert isinstance(dda.metadata.schema_type, str)


def test_metadata_schema_version_major_is_int(dda) -> None:
    assert isinstance(dda.metadata.schema_version_major, int)


def test_metadata_schema_version_minor_is_int(dda) -> None:
    assert isinstance(dda.metadata.schema_version_minor, int)


def test_metadata_tims_compression_type_is_int(dda) -> None:
    assert isinstance(dda.metadata.tims_compression_type, int)


def test_metadata_max_num_peaks_per_scan_is_int(dda) -> None:
    assert isinstance(dda.metadata.max_num_peaks_per_scan, int)


def test_metadata_closed_properly_is_bool(dda) -> None:
    assert isinstance(dda.metadata.closed_properly, bool)


def test_metadata_mz_acq_range_lower_is_float(dda) -> None:
    assert isinstance(dda.metadata.mz_acq_range_lower, float)


def test_metadata_mz_acq_range_upper_is_float(dda) -> None:
    assert isinstance(dda.metadata.mz_acq_range_upper, float)


def test_metadata_one_over_k0_acq_range_lower_is_float(dda) -> None:
    assert isinstance(dda.metadata.one_over_k0_acq_range_lower, float)


def test_metadata_one_over_k0_acq_range_upper_is_float(dda) -> None:
    assert isinstance(dda.metadata.one_over_k0_acq_range_upper, float)


def test_metadata_acquisition_datetime_is_datetime(dda) -> None:
    assert isinstance(dda.metadata.acquisition_datetime, datetime.datetime)


def test_metadata_acquisition_software_vendor_is_str(dda) -> None:
    assert isinstance(dda.metadata.acquisition_software_vendor, str)


def test_metadata_instrument_vendor_is_str(dda) -> None:
    assert isinstance(dda.metadata.instrument_vendor, str)


def test_metadata_analysis_id_is_str(dda) -> None:
    assert isinstance(dda.metadata.analysis_id, str)


# ---------------------------------------------------------------------------
# MetaData — value tests
# ---------------------------------------------------------------------------


def test_metadata_schema_type_value(dda) -> None:
    assert dda.metadata.schema_type == "TDF"


def test_metadata_schema_version_major_value(dda) -> None:
    assert dda.metadata.schema_version_major == 3


def test_metadata_schema_version_minor_value(dda) -> None:
    assert dda.metadata.schema_version_minor == 1


def test_metadata_acquisition_software_vendor_value(dda) -> None:
    assert dda.metadata.acquisition_software_vendor == "Bruker"


def test_metadata_instrument_vendor_value(dda) -> None:
    assert dda.metadata.instrument_vendor == "Bruker"


def test_metadata_tims_compression_type_value(dda) -> None:
    assert dda.metadata.tims_compression_type == 2


def test_metadata_closed_properly_is_true(dda) -> None:
    assert dda.metadata.closed_properly is True


def test_metadata_max_num_peaks_per_scan_value(dda) -> None:
    assert dda.metadata.max_num_peaks_per_scan == 1412


def test_metadata_analysis_id_value(dda) -> None:
    assert dda.metadata.analysis_id == "00000000-0000-0000-0000-000000000000"


def test_metadata_mz_acq_range_lower_value(dda) -> None:
    assert dda.metadata.mz_acq_range_lower == pytest.approx(100.0)


def test_metadata_mz_acq_range_upper_value(dda) -> None:
    assert dda.metadata.mz_acq_range_upper == pytest.approx(1700.0)


def test_metadata_one_over_k0_acq_range_lower_value(dda) -> None:
    assert dda.metadata.one_over_k0_acq_range_lower == pytest.approx(0.578703)


def test_metadata_one_over_k0_acq_range_upper_value(dda) -> None:
    assert dda.metadata.one_over_k0_acq_range_upper == pytest.approx(1.524471)


def test_metadata_acquisition_software_value(dda) -> None:
    assert dda.metadata.acquisition_software == "Bruker otofControl"


def test_metadata_acquisition_software_version_contains_5_1(dda) -> None:
    assert "5.1" in dda.metadata.acquisition_software_version


def test_metadata_instrument_name_value(dda) -> None:
    assert dda.metadata.instrument_name == "timsTOF Pro"


def test_metadata_instrument_serial_number_value(dda) -> None:
    assert dda.metadata.instrument_serial_number == "1844426.34"


def test_metadata_operator_name_value(dda) -> None:
    assert dda.metadata.operator_name == "Demo User"


def test_metadata_sample_name_contains_hela(dda) -> None:
    assert "HeLa" in dda.metadata.sample_name or "Hela" in dda.metadata.sample_name


# ---------------------------------------------------------------------------
# MetaData — tuple / range tests
# ---------------------------------------------------------------------------


def test_metadata_mz_acq_range_is_tuple(dda) -> None:
    assert isinstance(dda.metadata.mz_acq_range, tuple)
    assert len(dda.metadata.mz_acq_range) == 2


def test_metadata_mz_acq_range_lower_less_than_upper(dda) -> None:
    lower, upper = dda.metadata.mz_acq_range
    assert lower < upper


def test_metadata_mz_acq_range_matches_individual_properties(dda) -> None:
    md = dda.metadata
    assert md.mz_acq_range == (md.mz_acq_range_lower, md.mz_acq_range_upper)


def test_metadata_one_over_k0_acq_range_is_tuple(dda) -> None:
    assert isinstance(dda.metadata.one_over_k0_acq_range, tuple)
    assert len(dda.metadata.one_over_k0_acq_range) == 2


def test_metadata_one_over_k0_acq_range_lower_less_than_upper(dda) -> None:
    lower, upper = dda.metadata.one_over_k0_acq_range
    assert lower < upper


def test_metadata_one_over_k0_acq_range_matches_individual_properties(dda) -> None:
    md = dda.metadata
    assert md.one_over_k0_acq_range == (
        md.one_over_k0_acq_range_lower,
        md.one_over_k0_acq_range_upper,
    )


# ---------------------------------------------------------------------------
# MetaData — acquisition_datetime tests
# ---------------------------------------------------------------------------


def test_metadata_acquisition_datetime_year(dda) -> None:
    assert dda.metadata.acquisition_datetime.year == 2018


def test_metadata_acquisition_datetime_has_tzinfo(dda) -> None:
    assert dda.metadata.acquisition_datetime.tzinfo is not None


# ---------------------------------------------------------------------------
# MetaData — unknown key raises KeyError
# ---------------------------------------------------------------------------


def test_metadata_unknown_key_raises_key_error(dda) -> None:
    with pytest.raises(KeyError, match="NotARealKey"):
        _ = dda.metadata["NotARealKey"]


# ---------------------------------------------------------------------------
# Calibration — type tests
# ---------------------------------------------------------------------------


def test_calibration_date_is_datetime(dda) -> None:
    assert isinstance(dda.calibration.date, datetime.datetime)


def test_calibration_user_is_str(dda) -> None:
    assert isinstance(dda.calibration.user, str)


def test_calibration_software_is_str(dda) -> None:
    assert isinstance(dda.calibration.software, str)


def test_calibration_software_version_is_str(dda) -> None:
    assert isinstance(dda.calibration.software_version, str)


def test_calibration_mobility_calibration_date_is_datetime(dda) -> None:
    assert isinstance(dda.calibration.mobility_calibration_date, datetime.datetime)


def test_calibration_mobility_calibration_user_is_str(dda) -> None:
    assert isinstance(dda.calibration.mobility_calibration_user, str)


def test_calibration_mobility_standard_deviation_percent_is_float(dda) -> None:
    assert isinstance(dda.calibration.mobility_standard_deviation_percent, float)


def test_calibration_reference_mobility_list_is_str(dda) -> None:
    assert isinstance(dda.calibration.reference_mobility_list, str)


def test_calibration_reference_masses_is_str(dda) -> None:
    assert isinstance(dda.calibration.reference_masses, str)


# ---------------------------------------------------------------------------
# Calibration — value tests
# ---------------------------------------------------------------------------


def test_calibration_date_year(dda) -> None:
    assert dda.calibration.date.year == 2018


def test_calibration_date_has_tzinfo(dda) -> None:
    assert dda.calibration.date.tzinfo is not None


def test_calibration_user_value(dda) -> None:
    assert dda.calibration.user == "Demo User"


def test_calibration_software_value(dda) -> None:
    assert dda.calibration.software == "Bruker otofControl"


def test_calibration_software_version_contains_5_1(dda) -> None:
    assert "5.1" in dda.calibration.software_version


def test_calibration_mobility_calibration_date_year(dda) -> None:
    assert dda.calibration.mobility_calibration_date.year == 2018


def test_calibration_mobility_calibration_user_value(dda) -> None:
    assert dda.calibration.mobility_calibration_user == "Demo User"


def test_calibration_mobility_standard_deviation_percent_value(dda) -> None:
    assert dda.calibration.mobility_standard_deviation_percent == pytest.approx(0.000932)


def test_calibration_reference_mobility_list_value(dda) -> None:
    assert dda.calibration.reference_mobility_list == "Tuning Mix ES-TOF (ESI)"


def test_calibration_reference_masses_value(dda) -> None:
    assert dda.calibration.reference_masses == "Tuning Mix ES-TOF (ESI)"


# ---------------------------------------------------------------------------
# Calibration — unknown key raises KeyError
# ---------------------------------------------------------------------------


def test_calibration_unknown_key_raises_key_error(dda) -> None:
    with pytest.raises(KeyError, match="NotARealKey"):
        _ = dda.calibration["NotARealKey"]
