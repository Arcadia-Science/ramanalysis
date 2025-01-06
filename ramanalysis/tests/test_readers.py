from io import StringIO

import numpy as np
import pytest

from ramanalysis.readers import (
    read_horiba_txt,
    read_openraman_csv,
    read_renishaw_csv,
    read_wasatch_csv,
)


def test_read_openraman_csv_basic(valid_openraman_csv_filepath):
    intensities = read_openraman_csv(valid_openraman_csv_filepath)
    first_five_intensities = [0.661781, 0.661274, 0.664067, 0.658976, 0.660716]
    np.testing.assert_allclose(intensities[:5], first_five_intensities, rtol=1e-6)


def test_read_openraman_csv_noncsv(valid_horiba_txt_filepath):
    with pytest.raises(UnicodeDecodeError):
        read_openraman_csv(valid_horiba_txt_filepath)


def test_read_openraman_csv_dtype():
    csv_float_data = StringIO(
        """
        Pixels #,Intensity (a.u.)
        0.00000e+00,6.61781e-01
        1.00000e+00,6.61274e-01
        2.00000e+00,6.64067e-01
        """
    )
    csv_int_data = StringIO(
        """
        Pixels #,Intensity (a.u.)
        0.00000e+00,3
        1.00000e+00,4
        2.00000e+00,5
        """
    )
    csv_str_data = StringIO(
        """
        Pixels #,Intensity (a.u.)
        0.00000e+00,abc
        1.00000e+00,xyz
        2.00000e+00,puppy
        """
    )
    # test float --> float
    intensities = read_openraman_csv(csv_float_data)  # type: ignore
    assert intensities.dtype == np.float64
    # test int --> float
    intensities = read_openraman_csv(csv_int_data)  # type: ignore
    assert intensities.dtype == np.float64
    # test str --> ValueError
    with pytest.raises(ValueError):
        read_openraman_csv(csv_str_data)  # type: ignore


def test_read_openraman_csv_nans():
    csv_nan_data = StringIO(
        """
        Pixels #,Intensity (a.u.)
        0.00000e+00,3
        1.00000e+00,
        2.00000e+00,5
        """
    )
    intensities = read_openraman_csv(csv_nan_data)  # type: ignore
    expected_intensities = np.array([3, np.nan, 5], dtype=np.float64)
    np.testing.assert_allclose(intensities, expected_intensities, rtol=1e-6)


def test_read_openraman_csv_missing_column():
    csv_data_botched_header = StringIO(
        """
        Pixels #,Llama (W)
        0.00000e+00,6.72077e-01
        1.00000e+00,6.71534e-01
        2.00000e+00,6.74614e-01
        """
    )
    with pytest.raises(KeyError):
        read_openraman_csv(csv_data_botched_header)  # type: ignore


def test_read_horiba_txt_basic(valid_horiba_txt_filepath):
    wavenumbers_cm1, intensities, _metadata = read_horiba_txt(valid_horiba_txt_filepath)
    first_five_wavenumbers = [87.8957, 90.0299, 92.1642, 94.2969, 96.4298]
    first_five_intensities = [620.5, 680.5, 704.5, 705.0, 741.0]
    np.testing.assert_allclose(wavenumbers_cm1[:5], first_five_wavenumbers, rtol=1e-2)
    np.testing.assert_allclose(intensities[:5], first_five_intensities, rtol=1e-2)


def test_read_renishaw_csv_basic(valid_renishaw_csv_filepath):
    wavenumbers_cm1, intensities = read_renishaw_csv(valid_renishaw_csv_filepath)
    first_five_wavenumbers = [717.258789, 716.048828, 714.837891, 713.626953, 712.416016]
    first_five_intensities = [839088.625, 841384.125, 839254.75, 841847.625, 839767.875]
    np.testing.assert_allclose(wavenumbers_cm1[:5], first_five_wavenumbers, rtol=1e-2)
    np.testing.assert_allclose(intensities[:5], first_five_intensities, rtol=1e-2)


def test_read_wasatch_csv_basic(valid_wasatch_csv_filepath):
    wavenumbers_cm1, intensities, _metadata = read_wasatch_csv(valid_wasatch_csv_filepath)
    first_five_wavenumbers = [260.19, 262.64, 265.08, 267.53, 269.97]
    first_five_intensities = [2111.50, 2021.50, 2072.50, 2003.00, 1977.00]
    np.testing.assert_allclose(wavenumbers_cm1[:5], first_five_wavenumbers, rtol=1e-2)
    np.testing.assert_allclose(intensities[:5], first_five_intensities, rtol=1e-2)


def test_read_horiba_metadata(valid_horiba_txt_filepath):
    *_, metadata = read_horiba_txt(valid_horiba_txt_filepath)
    assert metadata["Acq. time (s)"] == "10"
    assert metadata["Dark correction"] == "Off"
    assert metadata["AxisUnit[1]"] == "1/cm"


def test_read_wasatch_metadata(valid_wasatch_csv_filepath):
    *_, metadata = read_wasatch_csv(valid_wasatch_csv_filepath)
    assert metadata["ENLIGHTEN Version"] == "4.1.6"
    assert metadata["Laser Power mW"] == "100.0"
    assert metadata["Pixel Count"] == "2048"
