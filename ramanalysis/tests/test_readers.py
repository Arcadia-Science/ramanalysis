from io import StringIO

import numpy as np
import pytest

from ramanalysis.readers import (
    read_horiba_txt,
    read_openraman_csv,
    read_renishaw_multipoint_txt,
    read_renishaw_singlepoint_txt,
    read_wasatch_csv,
)


def test_read_openraman_csv_basic(valid_openraman_csv_filepath):
    intensities = read_openraman_csv(valid_openraman_csv_filepath)
    first_five_intensities = [0.661781, 0.661274, 0.664067, 0.658976, 0.660716]
    np.testing.assert_allclose(intensities[:5], first_five_intensities)


def test_read_horiba_txt_basic(valid_horiba_singlepoint_txt_filepath):
    wavenumbers_cm1, intensities, _metadata = read_horiba_txt(valid_horiba_singlepoint_txt_filepath)
    first_five_wavenumbers = [87.8957, 90.0299, 92.1642, 94.2969, 96.4298]
    first_five_intensities = [620.5, 680.5, 704.5, 705.0, 741.0]
    np.testing.assert_allclose(wavenumbers_cm1[:5], first_five_wavenumbers)
    np.testing.assert_allclose(intensities[:5], first_five_intensities)


def test_read_renishaw_singlepoint_txt_basic(valid_renishaw_singlepoint_filepath):
    wavenumbers_cm1, intensities = read_renishaw_singlepoint_txt(
        valid_renishaw_singlepoint_filepath
    )
    first_five_wavenumbers = [600.934570, 602.157227, 603.380859, 604.603516, 605.826172]
    first_five_intensities = [41363.480469, 41879.410156, 41838.308594, 41764.992188, 41643.339844]
    np.testing.assert_allclose(wavenumbers_cm1[:5], first_five_wavenumbers)
    np.testing.assert_allclose(intensities[:5], first_five_intensities)


def test_read_renishaw_multipoint_txt_basic(valid_renishaw_multipoint_filepath):
    wavenumbers_cm1, intensities, positions = read_renishaw_multipoint_txt(
        valid_renishaw_multipoint_filepath
    )
    first_five_wavenumbers = [712.416016, 713.626953, 714.837891, 716.048828, 717.258789]
    first_five_intensities = [
        [748389.1250, 749051.4375, 750075.6875, 749563.8750, 749238.7500],  # position 1
        [839767.8750, 841847.6250, 839254.7500, 841384.1250, 839088.6250],  # position 2
        [797822.7500, 798641.0625, 799250.5000, 798585.7500, 799925.1250],  # position 3
        [675062.3125, 674205.8125, 674375.0000, 674599.1250, 673573.0625],  # position 4
    ]
    known_positions = [
        [-10.722373, 21.898673],
        [-19.589040, 8.375501],
        [-26.935707, 9.396118],
        [-32.255707, -9.485292]
    ]
    np.testing.assert_allclose(wavenumbers_cm1[:5], first_five_wavenumbers)
    np.testing.assert_allclose(intensities[:, :5], first_five_intensities)
    np.testing.assert_allclose(positions, known_positions)


def test_read_wasatch_csv_basic(valid_wasatch_csv_filepath):
    wavenumbers_cm1, intensities, _metadata = read_wasatch_csv(valid_wasatch_csv_filepath)
    first_five_wavenumbers = [260.19, 262.64, 265.08, 267.53, 269.97]
    first_five_intensities = [2111.50, 2021.50, 2072.50, 2003.00, 1977.00]
    np.testing.assert_allclose(wavenumbers_cm1[:5], first_five_wavenumbers)
    np.testing.assert_allclose(intensities[:5], first_five_intensities)


def test_read_openraman_csv_noncsv(valid_horiba_singlepoint_txt_filepath):
    with pytest.raises(UnicodeDecodeError):
        read_openraman_csv(valid_horiba_singlepoint_txt_filepath)


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
    np.testing.assert_allclose(intensities, expected_intensities)


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


def test_read_renishaw_singlepoint_txt_corrupt():
    txt_data_botched_header = StringIO(
        """
        #Wav3		#Intensity
        3399.408203	1016.205444
        3398.738281	1008.868225
        3398.067383	1016.025818
        3397.396484	979.173096
        """
    )
    txt_data_extra_header = StringIO(
        """
        #Wave		#Intensity	#Intensity
        3399.408203	1016.205444	-38.7
        3398.738281	1008.868225	-38.7
        3398.067383	1016.025818	-38.7
        3397.396484	979.173096	-38.7
        """
    )
    with pytest.raises(KeyError):
        read_renishaw_singlepoint_txt(txt_data_botched_header)  # type: ignore
    with pytest.raises(KeyError):
        read_renishaw_singlepoint_txt(txt_data_extra_header)  # type: ignore


def test_read_horiba_metadata(valid_horiba_singlepoint_txt_filepath):
    *_, metadata = read_horiba_txt(valid_horiba_singlepoint_txt_filepath)
    assert metadata["Acq. time (s)"] == "10"
    assert metadata["Dark correction"] == "Off"
    assert metadata["AxisUnit[1]"] == "1/cm"


def test_read_wasatch_metadata(valid_wasatch_csv_filepath):
    *_, metadata = read_wasatch_csv(valid_wasatch_csv_filepath)
    assert metadata["ENLIGHTEN Version"] == "4.1.6"
    assert metadata["Laser Power mW"] == "100.0"
    assert metadata["Pixel Count"] == "2048"
