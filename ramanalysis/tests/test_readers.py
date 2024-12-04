from io import StringIO

import numpy as np
import pytest

from ramanalysis.readers import read_horiba_txt, read_openraman_csv


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
    wavenumbers_cm1, intensities = read_horiba_txt(valid_horiba_txt_filepath)
    first_five_wavenumbers = [87.8957, 90.0299, 92.1642, 94.2969, 96.4298]
    first_five_intensities = [620.5, 680.5, 704.5, 705.0, 741.0]
    np.testing.assert_allclose(wavenumbers_cm1[:5], first_five_wavenumbers, rtol=1e-2)
    np.testing.assert_allclose(intensities[:5], first_five_intensities, rtol=1e-2)
