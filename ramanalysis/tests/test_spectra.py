import numpy as np
import pytest

from ramanalysis.spectra import RamanSpectrum


def test_conversions_to_numpy_float():
    spectrum = RamanSpectrum(
        wavenumbers_cm1=["0.68"],  # type: ignore
        intensities=[34],  # type: ignore
    )
    assert spectrum.wavenumbers_cm1 == np.array([0.68])
    assert spectrum.wavenumbers_cm1.dtype == np.float64
    assert spectrum.intensities.dtype == np.float64


def test_string_input_fails():
    with pytest.raises(ValueError):
        RamanSpectrum(
            wavenumbers_cm1=["fish"],  # type: ignore
            intensities=[34],  # type: ignore
        )


def test_nonarraylike_input_fails():
    with pytest.raises(ValueError):
        RamanSpectrum(
            wavenumbers_cm1=32,  # type: ignore
            intensities=np.array([34.3]),
        )
    with pytest.raises(ValueError):
        RamanSpectrum(
            wavenumbers_cm1=np.array([3.0]),
            intensities=91,  # type: ignore
        )
    with pytest.raises(ValueError):
        RamanSpectrum(
            wavenumbers_cm1="0.54",  # type: ignore
            intensities=np.array([34.3]),
        )


def test_null_input_fails():
    with pytest.raises(ValueError):
        RamanSpectrum(
            wavenumbers_cm1=None,  # type: ignore
            intensities=np.array([65]),
        )
    with pytest.raises(ValueError):
        RamanSpectrum(
            wavenumbers_cm1=np.array([45, 65]),
            intensities=[None, None],  # type: ignore
        )
    with pytest.raises(ValueError):
        RamanSpectrum(
            wavenumbers_cm1=np.array([]),
            intensities=np.array([54]),
        )
    with pytest.raises(ValueError):
        RamanSpectrum(
            wavenumbers_cm1=np.array([653]),
            intensities=np.array([]),
        )


def test_nan_input_fails():
    with pytest.raises(ValueError):
        RamanSpectrum(
            wavenumbers_cm1=np.array([653, 43, 60]),
            intensities=np.array([1, 2, np.nan]),
        )
    with pytest.raises(ValueError):
        RamanSpectrum(
            wavenumbers_cm1=np.array([np.nan, 0, 8]),
            intensities=np.array([5, 2, 8]),
        )


def test_different_size_input_fails():
    with pytest.raises(ValueError):
        RamanSpectrum(
            wavenumbers_cm1=np.array([43, 5]),
            intensities=np.array([34]),
        )


def test_raman_spectrum_from_openraman_csvfiles(
    valid_openraman_csv_filepath,
    valid_openraman_excitation_csv_filepath,
    valid_openraman_emission_csv_filepath,
):
    spectrum = RamanSpectrum.from_openraman_csvfiles(
        valid_openraman_csv_filepath,
        valid_openraman_excitation_csv_filepath,
        valid_openraman_emission_csv_filepath,
    )
    first_five_wavenumbers_cm1 = [402.923577, 404.783446, 406.642939, 408.502055, 410.360796]
    first_five_intensities = [0.661781, 0.661274, 0.664067, 0.658976, 0.660716]
    np.testing.assert_allclose(spectrum.wavenumbers_cm1[:5], first_five_wavenumbers_cm1)
    np.testing.assert_allclose(spectrum.intensities[:5], first_five_intensities)


def test_raman_spectrum_from_horiba_txtfile(valid_horiba_singlepoint_txt_filepath):
    spectrum = RamanSpectrum.from_horiba_txtfile(valid_horiba_singlepoint_txt_filepath)
    first_five_wavenumbers_cm1 = [87.8957, 90.0299, 92.1642, 94.2969, 96.4298]
    first_five_intensities = [620.5, 680.5, 704.5, 705.0, 741.0]
    np.testing.assert_allclose(spectrum.wavenumbers_cm1[:5], first_five_wavenumbers_cm1)
    np.testing.assert_allclose(spectrum.intensities[:5], first_five_intensities)


def test_raman_spectrum_from_renishaw_txtfile(valid_renishaw_singlepoint_filepath):
    spectrum = RamanSpectrum.from_renishaw_txtfile(valid_renishaw_singlepoint_filepath)
    first_five_wavenumbers = [600.934570, 602.157227, 603.380859, 604.603516, 605.826172]
    first_five_intensities = [41363.480469, 41879.410156, 41838.308594, 41764.992188, 41643.339844]
    np.testing.assert_allclose(spectrum.wavenumbers_cm1[:5], first_five_wavenumbers)
    np.testing.assert_allclose(spectrum.intensities[:5], first_five_intensities)


def test_raman_spectrum_from_wasatch_csvfile(valid_wasatch_csv_filepath):
    spectrum = RamanSpectrum.from_wasatch_csvfile(valid_wasatch_csv_filepath)
    first_five_wavenumbers = [260.19, 262.64, 265.08, 267.53, 269.97]
    first_five_intensities = [2111.50, 2021.50, 2072.50, 2003.00, 1977.00]
    np.testing.assert_allclose(spectrum.wavenumbers_cm1[:5], first_five_wavenumbers)
    np.testing.assert_allclose(spectrum.intensities[:5], first_five_intensities)


def test_raman_spectrum_from_synthetic_data(
    valid_synthetic_spectrum_wavenumbers, valid_synthetic_spectrum_intensities
):
    spectrum = RamanSpectrum(
        valid_synthetic_spectrum_wavenumbers, valid_synthetic_spectrum_intensities
    )
    assert spectrum.wavenumbers_cm1.size == 512
    assert spectrum.intensities.size == 512
    assert spectrum.wavenumbers_cm1[0] == 800


def test_between_out_of_range():
    spectrum = RamanSpectrum(
        wavenumbers_cm1=np.array([200, 201, 202, 203, 204]),
        intensities=np.array([0.2, 0.3, 1.0, 0.3, 0.2]),
    )
    with pytest.raises(ValueError):
        spectrum.between(-4, 3)


def test_interpolate_float():
    spectrum = RamanSpectrum(
        wavenumbers_cm1=np.array([200, 201, 202, 203, 204]),
        intensities=np.array([0.2, 0.3, 1.0, 0.3, 0.2]),
    )
    np.testing.assert_allclose(spectrum.interpolate(0), [200.0, 0.2])
    np.testing.assert_allclose(spectrum.interpolate(1.9), [201.9, 0.93])
    np.testing.assert_allclose(spectrum.interpolate(3.0), [203.0, 0.3])
    np.testing.assert_allclose(spectrum.interpolate(3.2), [203.2, 0.28])


def test_interpolate_float_array():
    spectrum = RamanSpectrum(
        wavenumbers_cm1=np.array([200, 201, 202, 203, 204]),
        intensities=np.array([0.2, 0.3, 1.0, 0.3, 0.2]),
    )
    float_array = np.array([-2, 1.9, 3.0, 3.2, 6])
    interpolated_wavenumbers_cm1, interpolated_intensities = spectrum.interpolate(float_array)
    np.testing.assert_allclose(interpolated_wavenumbers_cm1, [200.0, 201.9, 203.0, 203.2, 204.0])
    np.testing.assert_allclose(interpolated_intensities, [0.2, 0.93, 0.3, 0.28, 0.2])
