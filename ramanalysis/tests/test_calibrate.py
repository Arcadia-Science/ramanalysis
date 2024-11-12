import numpy as np
import pytest
from numpy.typing import NDArray

from ramanalysis.calibrate import (
    ACETONITRILE_PEAKS_CM1,
    _OpenRamanDataCalibrator,
    calculate_raman_shift,
)
from ramanalysis.peak_fitting import find_n_most_prominent_peaks

# type aliases
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int32]
ScalarArray = FloatArray | IntArray


def test_calculate_raman_shift_basic():
    """Test :func:`calculate_raman_shift` with a non-default excitation wavelength."""
    emission_wavelengths = np.array([800, 850, 900], dtype=float)
    excitation_wavelength = 785
    calculated_shifts = calculate_raman_shift(emission_wavelengths, excitation_wavelength)
    expected_shifts = [238.85, 974.15, 1627.74]
    np.testing.assert_allclose(calculated_shifts, expected_shifts, rtol=1e-4)


def test_calculate_raman_shift_invalid_input():
    """Test :func:`calculate_raman_shift` with invalid input types."""
    with pytest.raises(TypeError):
        calculate_raman_shift("not an array", 532)  # type: ignore
    with pytest.raises(TypeError):
        calculate_raman_shift(np.array([600, 650]), "invalid wavelength")  # type: ignore


def test_rough_calibration_batch(
    batch_neon_spectra_csv_filepaths, batch_acetonitrile_spectra_csv_filepaths
):
    """Test :func:`calibrate_rough` on a batch of excitation spectra."""
    for excitation_csv_filepath, emission_csv_filepath in zip(
        batch_neon_spectra_csv_filepaths, batch_acetonitrile_spectra_csv_filepaths
    ):
        calibrator = _OpenRamanDataCalibrator(
            excitation_csv_filepath,
            emission_csv_filepath,
            excitation_wavelength_nm=532,
            kernel_size=5,
            rough_calibration_residuals_threshold=1e0,
            fine_calibration_residuals_threshold=1e6,  # absurdly high so as not to cause failure
        )
        calibrator.calibrate_rough()


def test_fine_calibration_batch(
    batch_neon_spectra_csv_filepaths, batch_acetonitrile_spectra_csv_filepaths
):
    """Test :func:`calibrate_fine` on a batch of paired excitation and emission spectra."""
    for excitation_csv_filepath, emission_csv_filepath in zip(
        batch_neon_spectra_csv_filepaths, batch_acetonitrile_spectra_csv_filepaths
    ):
        calibrator = _OpenRamanDataCalibrator(
            excitation_csv_filepath,
            emission_csv_filepath,
            excitation_wavelength_nm=532,
            kernel_size=5,
            rough_calibration_residuals_threshold=1e0,
            fine_calibration_residuals_threshold=1e2,
        )
        wavenumbers_cm1_rough = calibrator.calibrate_rough()
        wavenumbers_cm1 = calibrator.calibrate_fine(wavenumbers_cm1_rough)

        # verify that calibrated wavenumbers are within 4 cm^-1 of reference wavenumbers
        emission_intensities = calibrator.emission_intensities
        detected_peak_indices = find_n_most_prominent_peaks(
            emission_intensities, ACETONITRILE_PEAKS_CM1.size
        )
        np.testing.assert_allclose(
            ACETONITRILE_PEAKS_CM1, wavenumbers_cm1[detected_peak_indices], atol=4
        )


def test_fine_calibration_with_refined_peaks(
    batch_neon_spectra_csv_filepaths, batch_acetonitrile_spectra_csv_filepaths
):
    """Test :func:`calibrate_fine` on a batch of paired excitation and emission spectra."""
    for excitation_csv_filepath, emission_csv_filepath in zip(
        batch_neon_spectra_csv_filepaths, batch_acetonitrile_spectra_csv_filepaths
    ):
        calibrator = _OpenRamanDataCalibrator(
            excitation_csv_filepath,
            emission_csv_filepath,
            excitation_wavelength_nm=532,
            kernel_size=5,
            rough_calibration_residuals_threshold=1e0,
            fine_calibration_residuals_threshold=1e2,
        )
        wavenumbers_cm1_rough = calibrator.calibrate_rough()
        wavenumbers_cm1 = calibrator.calibrate_fine_with_refined_peaks(
            wavenumbers_cm1_rough, method="parabolic"
        )

        # verify that calibrated wavenumbers are within 4 cm^-1 of reference wavenumbers
        emission_intensities = calibrator.emission_intensities
        detected_peak_indices = find_n_most_prominent_peaks(
            emission_intensities, ACETONITRILE_PEAKS_CM1.size
        )
        np.testing.assert_allclose(
            ACETONITRILE_PEAKS_CM1, wavenumbers_cm1[detected_peak_indices], atol=4
        )
